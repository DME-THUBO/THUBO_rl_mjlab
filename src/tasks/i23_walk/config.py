"""Thin I23 registration; all environment and MDP ownership stays shared."""
from mjlab.tasks.registry import register_mjlab_task
from dme_thubo.env_cfg import dme_thubo_env_cfg
from dme_thubo.rl_cfg import dme_thubo_ppo_runner_cfg

def runner_cfg():
    cfg=dme_thubo_ppo_runner_cfg()
    cfg.experiment_name="thubo_i23_human_walk"
    cfg.num_steps_per_env=32
    cfg.max_iterations=4000
    cfg.save_interval=50
    return cfg

register_mjlab_task(task_id="THUBO-I23-Human-Walk",env_cfg=dme_thubo_env_cfg(task="human_walk",robot="thubo_i23"),play_env_cfg=dme_thubo_env_cfg(task="human_walk",robot="thubo_i23",play=True),rl_cfg=runner_cfg())
