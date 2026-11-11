#!/usr/bin/env python3
"""Replay a local RSL-RL checkpoint in the MuJoCo viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.tasks  # noqa: F401  # registers project tasks
from mjlab.scripts.play import PlayConfig, run_play


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
