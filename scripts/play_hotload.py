#!/usr/bin/env python3
"""Play a local MJLab run in Viser and hotload new checkpoints automatically."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import viser

import src.tasks  # noqa: F401  # registers project tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def checkpoint_entries(checkpoint_dir: Path) -> list[tuple[str, str]]:
    now = time.time()
    entries: list[tuple[int, str, str]] = []
    for checkpoint in checkpoint_dir.glob("model_*.pt"):
        try:
            iteration = int(checkpoint.stem.removeprefix("model_"))
        except ValueError:
            continue
        age = max(0, int(now - checkpoint.stat().st_mtime))
        if age < 2:
            continue
        entries.append((iteration, checkpoint.name, format_time_ago(age)))
    entries.sort()
    return [(name, age) for _, name, age in entries]


class HotloadViewer(ViserPlayViewer):
    def __init__(self, *args, checkpoint_manager: CheckpointManager, **kwargs):
        super().__init__(*args, checkpoint_manager=checkpoint_manager, **kwargs)
        self._checkpoint_poll_at = 0.0

    def setup(self) -> None:
        # Rough heightfields cover a large world. Keep the robot centered in the
        # browser view instead of aiming at the terrain generator's world center.
        self.cfg.distance = 4.0
        self.cfg.azimuth = 135.0
        self.cfg.elevation = 25.0
        super().setup()
        self._scene.camera_tracking_enabled = True
        for client in self._server.get_clients().values():
            client.camera.position = np.array([2.8, -2.8, 1.9])
            client.camera.look_at = np.array([0.0, 0.0, 0.4])

    def _process_actions(self) -> None:
        super()._process_actions()
        now = time.monotonic()
        if now < self._checkpoint_poll_at:
            return
        self._checkpoint_poll_at = now + 2.0
        manager = self._ckpt_mgr
        if manager is None:
            return
        entries = manager.fetch_available()
        if entries and entries[-1][0] != manager.current_name:
            self._actions.append((ViewerAction.FETCH_CHECKPOINT, "latest"))
            super()._process_actions()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Registered MJLab task ID")
    parser.add_argument("run_dir", type=Path, help="Training run directory")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--fallen-env-ids",
        type=int,
        nargs="*",
        default=None,
        help="Pin these viewer env indices to the fallen reset pose (getup tasks)",
    )
    args = parser.parse_args()

    configure_torch_backends()
    run_dir = args.run_dir.resolve()
    available = checkpoint_entries(run_dir)
    if not available:
        raise RuntimeError(f"No completed model_*.pt checkpoint found in {run_dir}")
    checkpoint_name = available[-1][0]

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    if args.fallen_env_ids:
        fallen_term = env_cfg.events.get("reset_fallen_pose")
        if fallen_term is None:
            print(
                "[WARN] --fallen-env-ids ignored: task has no 'reset_fallen_pose' event",
                flush=True,
            )
        else:
            fallen_term.params["fallen_env_ids"] = tuple(args.fallen_env_ids)
            print(
                f"[INFO] Envs {args.fallen_env_ids} start fallen; the other "
                f"{args.num_envs - len(args.fallen_env_ids)} start standing",
                flush=True,
            )
    agent_cfg = load_rl_cfg(args.task)
    device = args.device
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)

    def load_checkpoint(name: str):
        path = run_dir / name
        runner.load(str(path), load_cfg={"actor": True}, strict=True, map_location=device)
        return runner.get_inference_policy(device=device)

    load_checkpoint(checkpoint_name)
    policy = runner.get_inference_policy(device=device)
    manager = CheckpointManager(
        current_name=checkpoint_name,
        fetch_available=lambda: checkpoint_entries(run_dir),
        load_checkpoint=load_checkpoint,
    )
    server = viser.ViserServer(host=args.host, port=args.port, label="THUBO PPO")
    print(f"[INFO] Viser ready at http://127.0.0.1:{args.port}", flush=True)
    print(f"[INFO] Watching {run_dir} for model_*.pt checkpoints", flush=True)
    try:
        HotloadViewer(env, policy, viser_server=server, checkpoint_manager=manager).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
