"""Measure held, unassisted recovery; report each pose/action-scale separately."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--num-envs',type=int,default=8)
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--action-scales',type=float,nargs='+',default=None,
        help='Default: the action scale recorded in the checkpoint; pass multiple values for an explicit sweep')
    parser.add_argument('--poses',nargs='+',default=('supine','prone','left','right'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,default=ROOT/'logs/host_evaluation.json')
    args=parser.parse_args()
    import torch,warp
    warp.config.kernel_cache_dir=str(ROOT/'logs/warp_cache')
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from src.tasks.HoST_getUp.config import recovery_cfg
    from src.tasks.HoST_getUp.robots import robot_cfg
    from src.tasks.HoST_getUp.ppo import load
    from src.tasks.HoST_getUp.mdp import state
    policy,checkpoint=load(args.checkpoint,args.device)
    args.action_scales=args.action_scales or (checkpoint['metadata'].get('action_scale',1.),)
    _,meta=robot_cfg(checkpoint['robot'])
    rows=[]
    for scale in args.action_scales:
        for pose in args.poses:
            cfg=recovery_cfg(checkpoint['robot'],play=True,
                action_mode=checkpoint['metadata'].get('action_mode','absolute'))
            cfg.scene.num_envs=args.num_envs;cfg.episode_length_s=1e6;cfg.auto_reset=False
            cfg.events['reset_fallen_pose'].params['poses']=(pose,)
            def evaluation_condition(env,env_ids):
                ids=slice(None) if env_ids is None else env_ids
                state(env)['assist'][ids]=0.;state(env)['scale'][ids]=scale
            cfg.events['evaluation_condition']=EventTermCfg(func=evaluation_condition,mode='reset')
            env=ManagerBasedRlEnv(cfg=cfg,device=args.device)
            try:
                obs,_=env.reset()
                hold=torch.zeros(args.num_envs,device=args.device)
                best=torch.zeros_like(hold);finite=True
                for _ in range(round(args.seconds/env.step_dt)):
                    with torch.no_grad(): action=policy.inference(obs['actor'],state(env)['scale'])
                    obs,_,done,_,_=env.step(action)
                    robot=env.scene['robot']
                    z=robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
                    upright=-robot.data.projected_gravity_b[:,2]
                    speed=torch.linalg.vector_norm(robot.data.root_link_vel_w[:,:3],dim=-1)
                    turn=torch.linalg.vector_norm(robot.data.root_link_vel_w[:,3:],dim=-1)
                    stable=(z>.90*meta['height'])&(upright>.95)&(speed<.25)&(turn<.5)
                    hold=torch.where(stable,hold+env.step_dt,torch.zeros_like(hold));best=torch.maximum(best,hold)
                    if done.any() or not torch.isfinite(obs['actor']).all(): finite=False;break
                row=dict(pose=pose,action_scale=scale,episodes=args.num_envs,
                    assistance_fraction=0.,horizon_s=args.seconds,required_hold_s=.5,
                    successes=int((best>=.5).sum()) if finite else 0,
                    best_hold_s=best.cpu().tolist(),finite=finite)
                rows.append(row);print(json.dumps(row),flush=True)
            finally: env.close()
    report=dict(checkpoint=str(args.checkpoint),iteration=checkpoint['iteration'],robot=checkpoint['robot'],
        conditions='root height >90% target, upright >.95, linear speed <.25 m/s, angular speed <.5 rad/s, held .5s',
        results=rows,hardware_validation=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': main()
