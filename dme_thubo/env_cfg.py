"""Single side-effect-free ManagerBased factory for all THUBO task presets."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from . import mdp as task_mdp
from .robot import JOINT_NAMES

from copy import deepcopy
from math import pi
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from .embodiments import robot_cfg as embodiment_cfg

FALLEN_RATIO = 0.4


def _proprio_terms(robot):
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, params={"asset_cfg": robot}
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, params={"asset_cfg": robot}
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel, params={"asset_cfg": robot}
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, params={"asset_cfg": robot}
        ),
        "last_action": ObservationTermCfg(func=mdp.last_action),
    }
    return actor_terms


def _balance_rewards(robot, *, posture):
    rewards = {
        "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
        "upright": RewardTermCfg(func=task_mdp.upright_reward, weight=2.0),
        "root_height": RewardTermCfg(func=task_mdp.root_height_reward, weight=2.0),
        "posture": RewardTermCfg(func=task_mdp.joint_position_penalty, weight=-0.05),
        "joint_velocity": RewardTermCfg(
            func=mdp.joint_vel_l2, weight=-0.002, params={"asset_cfg": robot}
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    }
    if not posture:
        rewards.pop("posture")
    return rewards


def _balance_terminations(robot):
    terminations = {
        "fell": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": 0.9, "asset_cfg": robot}
        ),
        "low_root": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.2, "asset_cfg": robot},
        ),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    return terminations


def _velocity_preset(cfg, play):
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
            func=mdp.generated_commands, params={"command_name": "twist"}
        )
        group.terms["gait_phase"] = ObservationTermCfg(
            func=task_mdp.gait_phase_observation, params={"period": gait_period}
        )
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
                        ".*hip_pitch": 0.5,
                        ".*hip_roll": 0.15,
                        ".*hip_yaw": 0.15,
                        ".*knee_pitch": 0.5,
                        ".*ankle_pitch": 0.15,
                        ".*ankle_roll": 0.1,
                    },
                    "std_running": {
                        ".*hip_pitch": 0.5,
                        ".*hip_roll": 0.25,
                        ".*hip_yaw": 0.25,
                        ".*knee_pitch": 0.5,
                        ".*ankle_pitch": 0.25,
                        ".*ankle_roll": 0.1,
                    },
                    "walking_threshold": 0.1,
                    "running_threshold": 1.5,
                },
            ),
            "stand_still": RewardTermCfg(
                func=task_mdp.stand_still,
                weight=-1.0,
                params={
                    "asset_cfg": robot,
                    "command_name": "twist",
                    "command_threshold": 0.1,
                },
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
                        "robot",
                        geom_names=("left_sole_collision", "right_sole_collision"),
                    ),
                    "command_name": "twist",
                    "period": gait_period,
                    "command_threshold": 0.1,
                    "swing_ratio": 0.6,
                    "target_height": 0.1,
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
                        "robot",
                        geom_names=("left_sole_collision", "right_sole_collision"),
                    ),
                    "tilt_tolerance": 0.15,
                },
            ),
        }
    )
    cfg.episode_length_s = 20.0 if not play else 1000000000.0


def _velocity_getup_preset(cfg, play):
    cfg.events["reset_fallen_pose"] = EventTermCfg(
        func=task_mdp.reset_fallen_pose,
        mode="reset",
        params={"ratio": 1.0 if play else FALLEN_RATIO},
    )
    for group in cfg.observations.values():
        group.terms["root_height"] = ObservationTermCfg(
            func=task_mdp.root_height_observation
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
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}


def _host_preset(cfg, meta, action_mode, poses):
    if action_mode == "relative":
        cfg.actions = {
            "joint_pos": RelativeJointPositionActionCfg(
                entity_name="robot", actuator_names=(".*",), scale=1.0
            )
        }
    elif action_mode == "absolute":
        cfg.actions = {
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=2.5,
                use_default_offset=True,
            )
        }
    else:
        raise ValueError(action_mode)
    cfg.observations = {
        n: ObservationGroupCfg(
            terms={"proprio": ObservationTermCfg(func=task_mdp.host_proprioception)},
            history_length=6,
            enable_corruption=False,
        )
        for n in ("actor", "critic")
    }
    cfg.events = {
        "default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        "reset_fallen_pose": EventTermCfg(
            func=task_mdp.reset_host_recovery,
            mode="reset",
            params={
                "height": meta["height"],
                "reset_height": meta["reset_height"],
                "curriculum_mode": "terminal" if action_mode == "relative" else "peak",
                "poses": poses,
            },
        ),
        "assistance": EventTermCfg(
            func=task_mdp.apply_host_assistance,
            mode="step",
            params={"base": meta["base"], "mass": meta["mass"] or 6.0},
        ),
    }
    cfg.rewards = {
        "host_groups": RewardTermCfg(
            func=task_mdp.host_grouped_reward,
            weight=1.0,
            params={"height": meta["height"], "feet": meta["feet"]},
        )
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nonfinite": TerminationTermCfg(func=task_mdp.host_nonfinite),
    }
    cfg.episode_length_s = 10.0


def dme_thubo_env_cfg(
    *,
    task="standing",
    robot="thubo",
    terrain="flat",
    play=False,
    action_mode=None,
    num_envs=None,
    poses=("supine",),
) -> ManagerBasedRlEnvCfg:
    """Build a fresh task preset; environment variables are parsed by entry points.

    Rough tasks remain disabled pending independent height/obstacle acceptance.
    Pi is supported only for HoST and requires explicitly generated local assets.
    """
    task = task.lower().replace("-", "_")
    if task == "human_walk" and robot != "thubo_i23":
        raise ValueError("human_walk requires the explicit I23 embodiment")
    if task not in ("standing", "velocity", "velocity_getup", "host_getup", "human_walk"):
        raise ValueError(f"Unsupported task: {task}")
    if robot not in ("thubo", "thubo_i23", "pi") or (robot == "pi" and task != "host_getup"):
        raise ValueError(f"Unsupported task/robot combination: {task}/{robot}")
    if terrain != "flat":
        raise ValueError(
            "Rough terrain is disabled pending separate height and obstacle validation"
        )
    if action_mode is None:
        action_mode = "relative" if task == "host_getup" else "absolute"
    if action_mode not in ("relative", "absolute") or (
        task != "host_getup" and action_mode != "absolute"
    ):
        raise ValueError(
            f"Unsupported task/action_mode combination: {task}/{action_mode}"
        )
    poses = tuple(poses)
    if not poses or any(p not in ("supine", "prone", "left", "right") for p in poses):
        raise ValueError(f"Unsupported recovery poses: {poses}")
    if num_envs is not None and (
        isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1
    ):
        raise ValueError("num_envs must be positive")
    entity, meta = embodiment_cfg(robot)
    host = task == "host_getup"
    asset = SceneEntityCfg("robot", joint_names=(".*",), body_names=("pelvis",))
    observations = (
        {}
        if host
        else {
            "actor": ObservationGroupCfg(
                terms=_proprio_terms(asset), enable_corruption=True
            ),
            "critic": ObservationGroupCfg(
                terms=_proprio_terms(deepcopy(asset)), enable_corruption=False
            ),
        }
    )
    cfg = ManagerBasedRlEnvCfg(
        decimation=10,
        scene=SceneCfg(
            num_envs=(num_envs if num_envs is not None else (1 if play else 2048)),
            env_spacing=2.0,
            entities={"robot": entity},
            terrain=TerrainEntityCfg(terrain_type="plane"),
        ),
        observations=observations,
        actions=(
            {}
            if host
            else {
                "joint_pos": JointPositionActionCfg(
                    entity_name="robot",
                    actuator_names=JOINT_NAMES,
                    scale=0.35,
                    use_default_offset=True,
                )
            }
        ),
        rewards={} if host else _balance_rewards(asset, posture=task == "standing"),
        terminations={} if host else _balance_terminations(asset),
        events=(
            {}
            if host
            else {
                "reset_scene_to_default": EventTermCfg(
                    func=mdp.reset_scene_to_default, mode="reset"
                )
            }
        ),
        curriculum={},
        sim=SimulationCfg(
            nconmax=512,
            njmax=4096,
            mujoco=MujocoCfg(timestep=0.002, integrator="implicitfast", iterations=80),
        ),
        episode_length_s=10.0 if not play else 1.0e9,
        seed=42,
        scale_rewards_by_dt=False,
    )
    if task == "human_walk":
        _i23_walk_preset(cfg, play, meta)
    if task in ("velocity", "velocity_getup"):
        _velocity_preset(cfg, play)
    if task == "velocity_getup":
        _velocity_getup_preset(cfg, play)
    elif host:
        _host_preset(cfg, meta, action_mode, poses)
    # A term in one observation group must never be resolved or mutated by another.
    cfg.observations["critic"] = deepcopy(cfg.observations["critic"])
    return cfg


def dme_thubo_standing_env_cfg(*, play=False, terrain="flat"):
    """Compatibility wrapper; all construction belongs to dme_thubo_env_cfg."""
    return dme_thubo_env_cfg(task="standing", play=play, terrain=terrain)


def _i23_walk_preset(cfg, play, meta):
    """I23 forward walking with a procedural gait prior, not motion capture."""
    _velocity_preset(cfg, play)
    command = cfg.commands["twist"]
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.05
    command.ranges.lin_vel_x = (0.04, 0.16)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (-0.10, 0.10)
    cfg.actions["joint_pos"].scale = 0.25
    for group in cfg.observations.values():
        group.terms["gait_phase"].params["period"] = 1.0
    cfg.rewards["root_height"].params = {"target": meta["height"], "std": 0.03}
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.10
    cfg.rewards["track_angular_velocity"].weight = 0.5
    cfg.rewards["track_angular_velocity"].params["std"] = 0.3
    cfg.rewards["pose"].weight = 0.25
    cfg.rewards["pose"].params["walking_threshold"] = 0.02
    cfg.rewards["stand_still"].params["command_threshold"] = 0.02
    cfg.rewards["action_rate"].weight = -0.03
    cfg.rewards["feet_slide"].weight = -2.0
    cfg.rewards["feet_slide"].params["command_threshold"] = 0.02
    cfg.rewards["feet_slide"].params["default_offsets"] = (0.071, -0.071)
    cfg.rewards["feet_clearance_swing"].params.update(
        period=1.0, swing_ratio=0.4, target_height=0.02,
        min_swing_height=0.008, clearance_tolerance=0.012,
        command_threshold=0.02, contact_penalty=0.5,
    )
    cfg.rewards["gait_contact"] = RewardTermCfg(
        func=task_mdp.i23_gait_contact, weight=1.0,
        params={"sensor_name": "feet_contact", "period": 1.0, "swing_ratio": 0.4},
    )
    cfg.rewards["gait_reference"] = RewardTermCfg(
        func=task_mdp.i23_gait_reference, weight=0.4,
        params={"period": 1.0},
    )
    cfg.rewards["body_sway"] = RewardTermCfg(
        func=task_mdp.i23_body_sway, weight=-0.15,
    )
    feet = SceneEntityCfg("robot", body_names=("left_ankle_roll", "right_ankle_roll"))
    cfg.rewards["contact_slide"] = RewardTermCfg(
        func=task_mdp.i23_contact_slide, weight=-2.0, params={"asset_cfg": deepcopy(feet)},
    )
    cfg.terminations["feet_crossed"] = TerminationTermCfg(
        func=task_mdp.i23_feet_crossed, params={"asset_cfg": feet, "minimum_width": 0.09},
    )
    cfg.terminations["low_root"].params["minimum_height"] = 0.26
    cfg.terminations["fell"].params["limit_angle"] = 0.65
    cfg.episode_length_s = 20.0
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 512
