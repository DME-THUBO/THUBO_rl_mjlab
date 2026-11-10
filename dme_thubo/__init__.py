"""12-DoF THUBO robot task for MJLab."""

from .env_cfg import TASK_ID, dme_thubo_standing_env_cfg
from .rl_cfg import dme_thubo_ppo_runner_cfg

__all__ = ["TASK_ID", "dme_thubo_standing_env_cfg", "dme_thubo_ppo_runner_cfg"]
