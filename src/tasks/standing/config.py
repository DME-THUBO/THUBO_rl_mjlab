"""Standing task registrations following Unitree's ManagerBased format."""

from mjlab.tasks.registry import register_mjlab_task

from dme_thubo.env_cfg import dme_thubo_standing_env_cfg
from dme_thubo.rl_cfg import dme_thubo_ppo_runner_cfg

THUBO_FLAT_STAND = "THUBO-Flat-Stand"
THUBO_ROUGH_STAND = "THUBO-Rough-Stand"

register_mjlab_task(
    task_id=THUBO_FLAT_STAND,
    env_cfg=dme_thubo_standing_env_cfg(terrain="flat"),
    play_env_cfg=dme_thubo_standing_env_cfg(play=True, terrain="flat"),
    rl_cfg=dme_thubo_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id=THUBO_ROUGH_STAND,
    env_cfg=dme_thubo_standing_env_cfg(terrain="rough"),
    play_env_cfg=dme_thubo_standing_env_cfg(play=True, terrain="rough"),
    rl_cfg=dme_thubo_ppo_runner_cfg(),
)
