#!/usr/bin/env python3
"""Launch MJLab's RSL-RL PPO runner for THUBO tasks (flat velocity by default)."""

from dataclasses import replace
from pathlib import Path
import os
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
MAX_ITERATIONS = 4000
sys.path.insert(0, str(ROOT))

import src.tasks  # noqa: F401  # registers ManagerBased tasks
from mjlab.scripts.train import launch_training
from mjlab.scripts.train import TrainConfig


def build_train_config(task_id):
    """Apply launch overrides once, shared by the entry point and run recorder."""
    cfg = TrainConfig.from_task(task_id)
    if task_id == "THUBO-I23-Human-Walk":
        cfg = replace(cfg, gpu_ids=[0], enable_nan_guard=True)
        cfg.env.scene.num_envs = 512
    # Small overrides keep the same entry point usable for a quick smoke test.
    if "DME_THUBO_NUM_ENVS" in os.environ:
        cfg.env.scene.num_envs = int(os.environ["DME_THUBO_NUM_ENVS"])
    if "DME_THUBO_NUM_STEPS" in os.environ:
        cfg.agent.num_steps_per_env = int(os.environ["DME_THUBO_NUM_STEPS"])
    if "DME_THUBO_MAX_ITERATIONS" in os.environ:
        max_iterations = int(os.environ["DME_THUBO_MAX_ITERATIONS"])
        if max_iterations > MAX_ITERATIONS:
            raise ValueError(f"DME_THUBO_MAX_ITERATIONS cannot exceed {MAX_ITERATIONS}")
        cfg.agent.max_iterations = max_iterations
    if "DME_THUBO_SAVE_INTERVAL" in os.environ:
        cfg.agent.save_interval = int(os.environ["DME_THUBO_SAVE_INTERVAL"])
    if "DME_THUBO_RUN_NAME" in os.environ:
        cfg.agent.run_name = os.environ["DME_THUBO_RUN_NAME"]
    if "DME_THUBO_LOG_ROOT" in os.environ:
        cfg = replace(cfg, log_root=os.environ["DME_THUBO_LOG_ROOT"])
    if task_id == "THUBO-I23-Human-Walk" and "DME_I23_RESUME_CHECKPOINT" in os.environ:
        checkpoint = Path(os.environ["DME_I23_RESUME_CHECKPOINT"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        cfg.agent.resume = True
        cfg.agent.load_run = re.escape(checkpoint.parent.name)
        cfg.agent.load_checkpoint = re.escape(checkpoint.name)
        cfg.agent.max_iterations = MAX_ITERATIONS - int(checkpoint.stem[6:])
    return cfg


if __name__ == "__main__":
    task_id = os.environ.get("DME_THUBO_TASK", "THUBO-Flat-Velocity")
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        task_id = sys.argv[1]
    launch_training(task_id, build_train_config(task_id))
