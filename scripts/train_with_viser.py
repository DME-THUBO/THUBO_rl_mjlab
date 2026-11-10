#!/usr/bin/env python3
"""Run THUBO PPO training alongside a Viser viewer with checkpoint hotloading."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
MAX_ITERATIONS = 4000


def wait_for_checkpoint(run_dir: Path, training: subprocess.Popen, timeout: float = 3600.0) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        checkpoints = sorted(
            run_dir.glob("model_*.pt"),
            key=lambda path: int(path.stem.removeprefix("model_"))
            if path.stem.removeprefix("model_").isdigit()
            else -1,
        )
        if checkpoints and time.time() - checkpoints[-1].stat().st_mtime >= 2.0:
            return checkpoints[-1]
        if training.poll() is not None:
            raise RuntimeError(f"Training exited with status {training.returncode} before saving a checkpoint")
        time.sleep(1.0)
    raise TimeoutError(f"No completed checkpoint appeared in {run_dir} within {timeout:g}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="THUBO-Rough-Velocity")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--log-root", type=Path, default=ROOT / "logs" / "rsl_rl")
    args = parser.parse_args()
    if args.iterations > MAX_ITERATIONS:
        parser.error(f"--iterations cannot exceed {MAX_ITERATIONS}")
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Project Python environment not found: {PYTHON}")

    run_name = f"thubo_hotload_{datetime.now():%Y%m%d_%H%M%S}"
    log_root = args.log_root.resolve()
    env = os.environ.copy()
    env.update(
        DME_THUBO_NUM_ENVS=str(args.num_envs),
        DME_THUBO_MAX_ITERATIONS=str(args.iterations),
        DME_THUBO_SAVE_INTERVAL=str(args.save_interval),
        DME_THUBO_RUN_NAME=run_name,
        DME_THUBO_LOG_ROOT=str(log_root),
    )
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    train_cmd = [str(PYTHON), str(ROOT / "scripts" / "train.py"), args.task]
    training = subprocess.Popen(train_cmd, cwd=ROOT, env=env)
    viewer: subprocess.Popen | None = None
    run_root = log_root / "dme_thubo_rough_velocity"
    run_dir = None
    print(f"[INFO] Training PID: {training.pid}", flush=True)
    print(
        f"[INFO] Task={args.task} environments={args.num_envs} "
        f"iterations={args.iterations} save_interval={args.save_interval}",
        flush=True,
    )

    def handle_shutdown(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_shutdown)
    try:
        while training.poll() is None:
            if run_dir is None:
                matches = sorted(run_root.glob(f"*_{run_name}"))
                if matches:
                    run_dir = matches[-1]
            if run_dir is not None and viewer is None:
                checkpoint = wait_for_checkpoint(run_dir, training)
                viewer_cmd = [
                    str(PYTHON),
                    str(ROOT / "scripts" / "play_hotload.py"),
                    args.task,
                    str(run_dir),
                    "--port",
                    str(args.viser_port),
                    "--device",
                    "cpu" if args.device == "cuda" else args.device,
                ]
                viewer_env = env.copy()
                if args.device == "cuda":
                    viewer_env["CUDA_VISIBLE_DEVICES"] = ""
                viewer = subprocess.Popen(viewer_cmd, cwd=ROOT, env=viewer_env)
                print(f"[INFO] Initial checkpoint: {checkpoint.name}", flush=True)
                print(f"[INFO] Viser PID: {viewer.pid}", flush=True)
                print(f"[INFO] Viser URL: http://localhost:{args.viser_port}", flush=True)
            if viewer is not None and viewer.poll() is not None:
                raise RuntimeError(f"Viser exited with status {viewer.returncode}")
            time.sleep(1.0)
        if viewer is None and training.returncode == 0:
            raise RuntimeError("Training completed without producing a checkpoint for Viser")
        if training.returncode != 0:
            raise RuntimeError(f"Training exited with status {training.returncode}")
        print("[INFO] Training completed; Viser remains available until interrupted.", flush=True)
        while viewer is not None and viewer.poll() is None:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping training and Viser", flush=True)
    finally:
        for process in (viewer, training):
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in (viewer, training):
            if process is not None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
