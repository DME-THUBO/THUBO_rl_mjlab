#!/usr/bin/env python3
"""Run THUBO or HoST PPO training alongside Viser with checkpoint hotloading."""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT))


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


def run_host_viewer(args: argparse.Namespace) -> None:
    import torch
    import viser
    import warp

    warp.config.kernel_cache_dir = str(ROOT / "logs/warp_cache")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import ViserPlayViewer
    from src.tasks.HoST_getUp.config import recovery_cfg
    from dme_thubo.mdp import host_state
    from src.tasks.HoST_getUp.ppo import load

    def latest():
        files = [path for path in args.run_dir.glob("model_*.pt")
                 if path.stem[6:].isdigit()]
        return max(files, key=lambda path: int(path.stem[6:])) if files else None

    current = latest()
    if current is None:
        raise FileNotFoundError("No atomic model_*.pt checkpoint available")
    policy, checkpoint = load(current, args.device)
    action_mode = checkpoint["metadata"].get("action_mode", "absolute")
    action_scale = checkpoint["metadata"].get("action_scale", 1.0)
    cfg = recovery_cfg(checkpoint["robot"], play=True, action_mode=action_mode)
    cfg.scene.num_envs = args.viewer_num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    last_poll = 0.0

    def infer(obs):
        nonlocal policy, current, last_poll, action_scale
        if time.monotonic() - last_poll > 2.0:
            candidate = latest()
            last_poll = time.monotonic()
            if candidate is not None and candidate != current:
                new_policy, data = load(candidate, args.device)
                if (data["robot"] != checkpoint["robot"]
                        or data["architecture"] != checkpoint["architecture"]
                        or data["metadata"].get("action_mode", "absolute") != action_mode):
                    raise ValueError("Cannot hotload an incompatible HoST checkpoint")
                policy = new_policy
                current = candidate
                action_scale = data["metadata"].get("action_scale", 1.0)
                with (args.run_dir / "hotload.jsonl").open("a") as output:
                    output.write(json.dumps(dict(checkpoint=candidate.name, time=time.time())) + "\n")
                print(f"Hotloaded {candidate.name}", flush=True)
        if not args.assist:
            host_state(env)["assist"].zero_()
        host_state(env)["scale"].fill_(action_scale)
        with torch.no_grad():
            return policy.inference(obs["actor"], host_state(env)["scale"])

    server = None
    try:
        print(f"[INFO] HoST viewer environments: {env.num_envs}", flush=True)
        server = viser.ViserServer(host="127.0.0.1", port=args.port)
        ViserPlayViewer(env, infer, viser_server=server).run()
    finally:
        if server is not None:
            server.stop()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="THUBO-Flat-Velocity")
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--log-root", type=Path, default=ROOT / "logs" / "rsl_rl")
    parser.add_argument("--viewer-only", type=Path, metavar="RUN_DIR",
                        help="View and hotload an existing native HoST run without training")
    parser.add_argument("--viewer-num-envs", type=int, default=4)
    parser.add_argument("--assist", action="store_true",
                        help="Enable curriculum assistance in the HoST viewer")
    args = parser.parse_args()
    if args.viewer_num_envs < 1 or args.num_envs < 1 or args.iterations < 1 or args.save_interval < 1:
        parser.error("Environment counts, iterations and save interval must be positive")
    if args.viewer_only is not None:
        args.run_dir = args.viewer_only.resolve()
        args.port = args.viser_port
        run_host_viewer(args)
        return
    if args.iterations > MAX_ITERATIONS:
        parser.error(f"--iterations cannot exceed {MAX_ITERATIONS}")
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Project Python environment not found: {PYTHON}")
    is_host = args.task.startswith("HoST-GetUp-")
    if is_host and args.task not in ("HoST-GetUp-THUBO", "HoST-GetUp-PI"):
        parser.error(f"Unknown HoST task: {args.task}")

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
    host_run_dir = log_root / "host_getup" / run_name
    if is_host:
        train_cmd = [
            str(PYTHON), str(ROOT / "scripts" / "host" / "train.py"),
            "--robot", args.task.removeprefix("HoST-GetUp-").lower(),
            "--num-envs", str(args.num_envs),
            "--iterations", str(args.iterations),
            "--save-interval", str(args.save_interval),
            "--device", args.device,
            "--run-dir", str(host_run_dir),
        ]
    training = subprocess.Popen(train_cmd, cwd=ROOT, env=env)
    viewer: subprocess.Popen | None = None
    run_root = log_root / "dme_thubo_rough_velocity"
    run_dir = host_run_dir if is_host else None
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
                    # Show a mixed getup demo: envs 0-1 stand, envs 2-3 start fallen.
                    "--num-envs",
                    str(args.viewer_num_envs),
                    "--fallen-env-ids",
                    "2",
                    "3",
                ]
                if is_host:
                    viewer_cmd = [
                        str(PYTHON), str(Path(__file__).resolve()),
                        "--viewer-only", str(run_dir),
                        "--viser-port", str(args.viser_port),
                        "--viewer-num-envs", str(args.viewer_num_envs),
                        "--device", "cpu",
                    ]
                if is_host and args.assist:
                    viewer_cmd.append("--assist")
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
