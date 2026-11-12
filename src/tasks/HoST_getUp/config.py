"""Thin HoST task registration; no environment parsing at module import."""

from mjlab.tasks.registry import register_mjlab_task
from dme_thubo.env_cfg import dme_thubo_env_cfg
from dme_thubo.embodiments import pi_assets_available
from dme_thubo.rl_cfg import dme_thubo_ppo_runner_cfg


def recovery_cfg(robot="thubo", play=False, action_mode=None, **kwargs):
    return dme_thubo_env_cfg(
        task="host_getup", robot=robot, play=play, action_mode=action_mode, **kwargs
    )


for name in ("thubo", "pi"):
    if name == "pi" and not pi_assets_available():
        continue
    register_mjlab_task(
        task_id=f"HoST-GetUp-{name.upper()}",
        env_cfg=recovery_cfg(name),
        play_env_cfg=recovery_cfg(name, play=True),
        rl_cfg=dme_thubo_ppo_runner_cfg(),
    )
