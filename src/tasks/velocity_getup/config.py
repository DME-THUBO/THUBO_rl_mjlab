"""Velocity tracking combined with fall-to-stand recovery.

A share of resets start from a supine lying pose. While an env is down it earns
getup rewards (root height recovery); once the root height recovers with
hysteresis, the velocity-tracking rewards take over. No AMP terms or
observations are involved.
"""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task

from dme_thubo import mdp as task_mdp
from dme_thubo.rl_cfg import dme_thubo_ppo_runner_cfg
from src.tasks.velocity.config import thubo_velocity_env_cfg

THUBO_FLAT_VELOCITY_GETUP = "THUBO-Flat-Velocity-Getup"

FALLEN_RATIO = 0.4


def thubo_velocity_getup_env_cfg(*, play: bool = False):
    """Velocity tracking plus fall-to-stand recovery from supine starts."""
    cfg = thubo_velocity_env_cfg(play=play, terrain="flat")

    cfg.events["reset_fallen_pose"] = EventTermCfg(
        func=task_mdp.reset_fallen_pose,
        mode="reset",
        params={"ratio": 1.0 if play else FALLEN_RATIO},
    )

    for group in cfg.observations.values():
        group.terms["root_height"] = ObservationTermCfg(
            func=task_mdp.root_height_observation,
        )

    cfg.rewards["getup_root_height"] = RewardTermCfg(
        func=task_mdp.gated_reward(task_mdp.getup_root_height_reward, down=True),
        weight=2.0,
    )
    for name in (
        "root_height",
        "track_linear_velocity",
        "track_angular_velocity",
        "feet_slide",
        "feet_clearance_swing",
        "feet_flat_orientation",
    ):
        term = cfg.rewards[name]
        cfg.rewards[name] = RewardTermCfg(
            func=task_mdp.gated_reward(term.func, down=False),
            weight=term.weight,
            params=term.params,
        )

    # Falling must not terminate: the robot is expected to get back up.
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    return cfg


register_mjlab_task(
    task_id=THUBO_FLAT_VELOCITY_GETUP,
    env_cfg=thubo_velocity_getup_env_cfg(play=False),
    play_env_cfg=thubo_velocity_getup_env_cfg(play=True),
    rl_cfg=dme_thubo_ppo_runner_cfg(),
)
