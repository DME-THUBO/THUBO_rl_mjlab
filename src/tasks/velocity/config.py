"""Velocity-tracking tasks staged after the standing task."""

from math import pi

from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from dme_thubo.env_cfg import dme_thubo_standing_env_cfg
from dme_thubo import mdp as task_mdp
from dme_thubo.rl_cfg import dme_thubo_ppo_runner_cfg

THUBO_FLAT_VELOCITY = "THUBO-Flat-Velocity"
THUBO_ROUGH_VELOCITY = "THUBO-Rough-Velocity"


def thubo_velocity_env_cfg(*, play: bool = False, terrain: str = "flat"):
    """Build a ManagerBased velocity task from the validated standing baseline."""
    cfg = dme_thubo_standing_env_cfg(play=play, terrain=terrain)  # type: ignore[arg-type]
    robot = SceneEntityCfg("robot", body_names=("pelvis",), joint_names=(".*",))
    cfg.commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            heading_command=True,
            heading_control_stiffness=0.5,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.4),
                lin_vel_y=(-0.1, 0.1),
                ang_vel_z=(-0.6, 0.6),
                heading=(-pi, pi),
            ),
        )
    }
    for group in cfg.observations.values():
        group.terms["command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        )
    cfg.rewards.update(
        {
            "track_linear_velocity": RewardTermCfg(
                func=velocity_mdp.track_linear_velocity,
                weight=2.0,
                params={"command_name": "twist", "std": 0.5, "asset_cfg": robot},
            ),
            "track_angular_velocity": RewardTermCfg(
                func=velocity_mdp.track_angular_velocity,
                weight=1.0,
                params={"command_name": "twist", "std": 0.5, "asset_cfg": robot},
            ),
            "feel_slide": RewardTermCfg(
                func=task_mdp.feel_slide,
                weight=-0.1,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=("left_ankle_roll", "right_ankle_roll")
                    )
                },
            ),
        }
    )
    cfg.episode_length_s = 20.0 if not play else 1.0e9
    return cfg


register_mjlab_task(
    task_id=THUBO_FLAT_VELOCITY,
    env_cfg=thubo_velocity_env_cfg(terrain="flat"),
    play_env_cfg=thubo_velocity_env_cfg(play=True, terrain="flat"),
    rl_cfg=dme_thubo_ppo_runner_cfg(),
)

register_mjlab_task(
    task_id=THUBO_ROUGH_VELOCITY,
    env_cfg=thubo_velocity_env_cfg(terrain="rough"),
    play_env_cfg=thubo_velocity_env_cfg(play=True, terrain="rough"),
    rl_cfg=dme_thubo_ppo_runner_cfg(),
)
