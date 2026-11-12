"""Velocity-tracking tasks staged after the standing task."""

from math import pi

from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
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
    gait_period = 0.8
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="feet_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=("left_sole_collision", "right_sole_collision"),
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
        ),
    )
    cfg.commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.05,
            heading_command=True,
            heading_control_stiffness=0.5,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.6),
                lin_vel_y=(-0.4, 0.4),
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
        group.terms["gait_phase"] = ObservationTermCfg(
            func=task_mdp.gait_phase_observation,
            params={"period": gait_period},
        )
    # Replace the standing baseline's uniform weak posture penalty with
    # Unitree-style speed-dependent pose shaping and a near-zero stop penalty.
    cfg.rewards.pop("posture")
    cfg.rewards.update(
        {
            "pose": RewardTermCfg(
                func=task_mdp.VariablePosture,
                weight=1.0,
                params={
                    "asset_cfg": robot,
                    "command_name": "twist",
                    "std_standing": {".*": 0.05},
                    "std_walking": {
                        ".*hip_pitch": 0.5, ".*hip_roll": 0.15,
                        ".*hip_yaw": 0.15, ".*knee_pitch": 0.5,
                        ".*ankle_pitch": 0.15, ".*ankle_roll": 0.1,
                    },
                    "std_running": {
                        ".*hip_pitch": 0.5, ".*hip_roll": 0.25,
                        ".*hip_yaw": 0.25, ".*knee_pitch": 0.5,
                        ".*ankle_pitch": 0.25, ".*ankle_roll": 0.1,
                    },
                    "walking_threshold": 0.1,
                    "running_threshold": 1.5,
                },
            ),
            "stand_still": RewardTermCfg(
                func=task_mdp.stand_still,
                weight=-1.0,
                params={"asset_cfg": robot, "command_name": "twist",
                        "command_threshold": 0.1},
            ),
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
            "feet_slide": RewardTermCfg(
                func=task_mdp.feet_slide,
                weight=-2.0,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", body_names=("left_ankle_roll", "right_ankle_roll")
                    ),
                    "command_name": "twist",
                },
            ),
            "feet_clearance_swing": RewardTermCfg(
                func=task_mdp.feet_clearance_swing_reward,
                weight=1.0,
                params={
                    "sensor_name": "feet_contact",
                    "asset_cfg": SceneEntityCfg(
                        "robot", geom_names=("left_sole_collision", "right_sole_collision")
                    ),
                    "command_name": "twist",
                    "period": gait_period,
                    "command_threshold": 0.1,
                    "swing_ratio": 0.6,
                    "target_height": 0.10,
                    "clearance_tolerance": 0.02,
                    "min_swing_height": 0.04,
                    "contact_penalty": 1.0,
                },
            ),
            "feet_flat_orientation": RewardTermCfg(
                func=task_mdp.feet_flat_orientation_reward,
                weight=1.0,
                params={
                    "sensor_name": "feet_contact",
                    "asset_cfg": SceneEntityCfg(
                        "robot", geom_names=("left_sole_collision", "right_sole_collision")
                    ),
                    "tilt_tolerance": 0.15,
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

# THUBO-Rough-Velocity registration disabled: the current rough-terrain setup
# is unsatisfactory (peak obstacles of 0.30 m against a blind policy, plus
# absolute-world-height reward/termination terms that misbehave on elevated
# ground). See AAAMigration/rough_terrain_disabled.md before re-enabling.
# register_mjlab_task(
#     task_id=THUBO_ROUGH_VELOCITY,
#     env_cfg=thubo_velocity_env_cfg(terrain="rough"),
#     play_env_cfg=thubo_velocity_env_cfg(play=True, terrain="rough"),
#     rl_cfg=dme_thubo_ppo_runner_cfg(),
# )
