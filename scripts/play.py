#!/usr/bin/env python3
"""Replay a local RSL-RL or HoST checkpoint in the MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.tasks  # noqa: F401  # registers project tasks
from mjlab.scripts.play import PlayConfig, run_play


def play_host(task: str, checkpoint: Path, viewer: str, device: str | None, num_envs: int) -> None:
    import torch
    import warp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
    from src.tasks.HoST_getUp.config import recovery_cfg
    from src.tasks.HoST_getUp.mdp import state
    from src.tasks.HoST_getUp.ppo import load

    configure_torch_backends()
    warp.config.kernel_cache_dir = str(ROOT / "logs/warp_cache")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    policy, data = load(checkpoint, device)
    if task != f"HoST-GetUp-{data['robot'].upper()}":
        raise ValueError("Task does not match the HoST checkpoint embodiment")
    scale = data["metadata"].get("action_scale", 1.0)
    cfg = recovery_cfg(data["robot"], play=True,
                       action_mode=data["metadata"].get("action_mode", "absolute"))
    cfg.scene.num_envs = num_envs
    cfg.events.pop("assistance")

    def replay_condition(env, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        state(env)["assist"][ids] = 0.0
        state(env)["scale"][ids] = scale

    cfg.events["replay_condition"] = EventTermCfg(func=replay_condition, mode="reset")
    env = ManagerBasedRlEnv(cfg=cfg, device=device)

    def infer(obs):
        with torch.no_grad():
            return policy.inference(obs["actor"], state(env)["scale"])

    print(f"[INFO] HoST checkpoint={checkpoint.name} environments={num_envs} "
          f"assistance=0 action_scale={scale:g} viewer={viewer}", flush=True)
    try:
        viewer_cls = NativeMujocoViewer if viewer == "native" else ViserPlayViewer
        viewer_cls(env, infer).run()
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Registered task, for example THUBO-Flat-Velocity")
    parser.add_argument("checkpoint", type=Path, help="Local model_*.pt checkpoint")
    parser.add_argument(
        "--viewer",
        choices=("native", "viser"),
        default="native",
        help="MuJoCo native window or browser-based Viser viewer",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: CUDA when available)")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--log-root", default="logs/rsl_rl")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.task.startswith("HoST-GetUp-"):
        play_host(args.task, checkpoint, args.viewer, args.device, args.num_envs)
        return

    # MJLab's standard play path owns checkpoint loading and native/Viser replay.
    run_play(
        args.task,
        PlayConfig(
            checkpoint_file=str(checkpoint),
            viewer=args.viewer,
            device=args.device,
            num_envs=args.num_envs,
            log_root=args.log_root,
        ),
    )


if __name__ == "__main__":
    main()
