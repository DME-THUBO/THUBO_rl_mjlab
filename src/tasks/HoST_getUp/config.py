"""Register both 12-DoF robot embodiments without changing existing tasks."""
import os
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.tasks.registry import register_mjlab_task
from dme_thubo.env_cfg import dme_thubo_standing_env_cfg
from dme_thubo.rl_cfg import dme_thubo_ppo_runner_cfg
from .robots import robot_cfg
from . import mdp

def recovery_cfg(robot="thubo",play=False,action_mode=None):
    cfg=dme_thubo_standing_env_cfg(play=play,terrain="flat")
    entity,meta=robot_cfg(robot)
    cfg.scene.entities={"robot":entity}
    cfg.scene.num_envs=1 if play else int(os.environ.get("HOST_NUM_ENVS","256"))
    action_mode=action_mode or os.environ.get('HOST_ACTION_MODE','relative')
    if action_mode=='relative':
        cfg.actions={'joint_pos':RelativeJointPositionActionCfg(
            entity_name='robot',actuator_names=('.*',),scale=1.)}
    elif action_mode=='absolute':
        cfg.actions["joint_pos"].actuator_names=(".*",)
        cfg.actions["joint_pos"].scale=2.5
    else:
        raise ValueError(action_mode)
    cfg.observations={n:ObservationGroupCfg(terms={"proprio":ObservationTermCfg(func=mdp.proprioception)},
        history_length=6,enable_corruption=False) for n in ("actor","critic")}
    cfg.events={"default":EventTermCfg(func=base_mdp.reset_scene_to_default,mode="reset"),
        "reset_fallen_pose":EventTermCfg(func=mdp.reset_recovery,mode="reset",
            params={"height":meta["height"],"reset_height":meta["reset_height"],
                    "curriculum_mode":"terminal" if action_mode=='relative' else 'peak',
                    "poses":tuple(os.environ.get("HOST_POSES","supine").split(","))}),
        "assistance":EventTermCfg(func=mdp.apply_assistance,mode="step",
            params={"base":meta["base"],"mass":meta["mass"] or 6.})}
    cfg.rewards={"host_groups":RewardTermCfg(func=mdp.grouped_reward,weight=1.,
        params={"height":meta["height"],"feet":meta["feet"]})}
    cfg.terminations={"time_out":TerminationTermCfg(func=base_mdp.time_out,time_out=True),
                      "nonfinite":TerminationTermCfg(func=mdp.nonfinite)}
    cfg.episode_length_s=10.
    return cfg

for name in ("thubo","pi"):
    if name=="pi" and not (__import__("pathlib").Path(__file__).resolve().parents[3]/"src/assets/robots/pi/pi.xml").exists():
        continue
    register_mjlab_task(task_id=f"HoST-GetUp-{name.upper()}",env_cfg=recovery_cfg(name),
        play_env_cfg=recovery_cfg(name,play=True),rl_cfg=dme_thubo_ppo_runner_cfg())
