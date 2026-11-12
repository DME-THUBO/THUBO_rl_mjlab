"""Thin registration of the shared velocity_getup preset."""

from mjlab.tasks.registry import register_mjlab_task
from dme_thubo.env_cfg import dme_thubo_env_cfg
from dme_thubo.rl_cfg import dme_thubo_ppo_runner_cfg

THUBO_FLAT_VELOCITY_GETUP = "THUBO-Flat-Velocity-Getup"


def thubo_velocity_getup_env_cfg(*, play=False, terrain="flat"):
    return dme_thubo_env_cfg(task="velocity_getup", play=play, terrain=terrain)


register_mjlab_task(
    task_id=THUBO_FLAT_VELOCITY_GETUP,
    env_cfg=thubo_velocity_getup_env_cfg(),
    play_env_cfg=thubo_velocity_getup_env_cfg(play=True),
    rl_cfg=dme_thubo_ppo_runner_cfg(),
)
