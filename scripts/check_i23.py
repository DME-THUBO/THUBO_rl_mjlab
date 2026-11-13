"""I23 source mass/geometry, collision masks, CUDA rollout and PPO save/reload."""
from pathlib import Path
from dataclasses import asdict
import sys,json,hashlib,os,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco as mj,numpy as np,torch,warp
warp.config.kernel_cache_dir=str(ROOT/'logs/warp_cache')
from dme_thubo.model_validation import validate
from dme_thubo.env_cfg import dme_thubo_env_cfg
from src.tasks.i23_walk.config import runner_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,help='Also replay this trained native RSL-RL checkpoint')
    parser.add_argument('--output',type=Path,default=ROOT/'logs/i23_validation.json')
    args=parser.parse_args()
    settings=json.loads((ROOT/'i23_training_settings.json').read_text())
    report=validate(ROOT/'mjcf/i23/THUBO.xml',ROOT/'mjcf/i23/actuator_assumptions.json')
    assert abs(report['mass_kg']-settings['model_mass_kg'])<1e-9
    m=mj.MjModel.from_xml_path(str(ROOT/'mjcf/i23/THUBO.xml'));d=mj.MjData(m)
    d.qpos[:3]=settings['default_root_pos_m']
    for n,v in settings['default_joint_pose_rad'].items():d.qpos[m.jnt_qposadr[m.joint(n).id]]=v
    mj.mj_forward(m,d)
    penetration=[dict(a=m.geom(c.geom1).name,b=m.geom(c.geom2).name,distance=float(c.dist)) for c in d.contact if c.dist<-.0001]
    assert not penetration,penetration
    left=m.geom('left_sole_collision').id;right=m.geom('right_sole_collision').id
    assert m.geom_contype[left]&m.geom_conaffinity[right]
    assert abs(m.body_pos[m.body('left_hip_roll').id,1]+.016)<1e-9
    cfg=dme_thubo_env_cfg(task='human_walk',robot='thubo_i23',num_envs=8)
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    try:
        obs,_=env.reset();assert obs['actor'].shape==(8,47)
        rewards=[]
        for _ in range(100):
            obs,reward,*_=env.step(torch.zeros(8,12,device='cuda:0'))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            rewards.append(float(reward.mean()))
        agent=runner_cfg();agent.num_steps_per_env=8;agent.save_interval=1
        run=ROOT/'logs/i23_smoke';run.mkdir(parents=True,exist_ok=True)
        runner=MjlabOnPolicyRunner(RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions),asdict(agent),str(run),'cuda:0')
        runner.learn(num_learning_iterations=2,init_at_random_ep_len=True)
        checkpoint=run/'model_1.pt';assert checkpoint.is_file()
        replay_checkpoint=args.checkpoint or checkpoint
        runner.load(str(replay_checkpoint));policy=runner.get_inference_policy(device='cuda:0')
        obs,_=env.reset()
        for _ in range(16):
            with torch.no_grad():action=policy(obs)
            obs,reward,*_=env.step(action)
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
        report.update(cuda_device=torch.cuda.get_device_name(),neutral_penetrations=penetration,cross_leg_contact_enabled=True,rollout_steps=100,obs_dim=47,short_ppo_updates=2,checkpoint_reload=True,replayed_checkpoint=replay_checkpoint.name,replay_steps=16,reward_finite=True,smoke_is_not_walk_acceptance=True)
    finally:env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print('I23 VALIDATION PASS',flush=True)
if __name__=='__main__':main()
