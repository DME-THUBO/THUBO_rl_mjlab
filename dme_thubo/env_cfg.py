"""MJLab manager-based PPO environment for standing and balance."""

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ALL_TERRAIN_PRESETS
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from . import mdp as task_mdp
from .robot import JOINT_NAMES, get_robot_cfg

def dme_thubo_standing_env_cfg(
    *, play: bool = False, terrain: Literal["flat", "rough"] = "flat"
) -> ManagerBasedRlEnvCfg:
    robot_cfg = get_robot_cfg()
    robot = SceneEntityCfg("robot", joint_names=(".*",), body_names=("pelvis",))

    if terrain == "flat":
        terrain_cfg = TerrainEntityCfg(terrain_type="plane")
    elif terrain == "rough":
        terrain_names = (
             "box_random_grid", "random_spread_boxes"
        )
        terrain_generator = TerrainGeneratorCfg(
            curriculum=True, size=(8.0, 8.0), border_width=20.0,
            num_rows=3, num_cols=len(terrain_names), difficulty_range=(0.0, 1.0),
            sub_terrains={name: ALL_TERRAIN_PRESETS[name](proportion=1.0) for name in terrain_names},
            add_lights=True,
        )
        terrain_cfg = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=terrain_generator,
            max_init_terrain_level=0,
        )
    else:
        raise ValueError(f"Unknown THUBO terrain: {terrain}")

    actor_terms = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": robot}),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": robot}),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, params={"asset_cfg": robot}),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, params={"asset_cfg": robot}),
        "last_action": ObservationTermCfg(func=mdp.last_action),
    }
    observations = {
        "actor": ObservationGroupCfg(terms=actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(terms=actor_terms, enable_corruption=False),
    }
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=JOINT_NAMES,
            scale=0.35, use_default_offset=True,
        )
    }
    rewards = {
        "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
        "upright": RewardTermCfg(func=task_mdp.upright_reward, weight=2.0),
        "root_height": RewardTermCfg(func=task_mdp.root_height_reward, weight=2.0),
        "posture": RewardTermCfg(func=task_mdp.joint_position_penalty, weight=-0.05),
        "joint_velocity": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.002, params={"asset_cfg": robot}),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    }
    terminations = {
        "fell": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": 0.9, "asset_cfg": robot}
        ),
        "low_root": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.20, "asset_cfg": robot},
        ),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    cfg = ManagerBasedRlEnvCfg(
        decimation=10,
        scene=SceneCfg(
            num_envs=1 if play else 2048,
            env_spacing=2.0,
            entities={"robot": robot_cfg},
            terrain=terrain_cfg,
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events={
            "reset_scene_to_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
        },
        curriculum=(
            {"terrain_levels": CurriculumTermCfg(func=task_mdp.terrain_curriculum)}
            if terrain == "rough" else {}
        ),
        sim=SimulationCfg(
            nconmax=512,
            njmax=4096,
            mujoco=MujocoCfg(timestep=0.002, integrator="implicitfast", iterations=80),
        ),
        episode_length_s=10.0 if not play else 1.0e9,
        seed=42,
        scale_rewards_by_dt=False,
    )
    return cfg
