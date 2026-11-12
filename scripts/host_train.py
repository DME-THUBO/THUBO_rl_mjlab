"""Train or smoke-test the HoST recovery port on either 12-DoF embodiment."""
from pathlib import Path
import sys,os,argparse,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',choices=('thubo','pi'),default='thubo')
    p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--save-interval',type=int,default=25)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.smoke: a.num_envs=4;a.iterations=2;a.steps=8;a.save_interval=1
    os.environ['HOST_NUM_ENVS']=str(a.num_envs)
    import torch
    import warp
    warp.config.kernel_cache_dir=str(ROOT/'logs/warp_cache')
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.HoST_getUp.config import recovery_cfg
    from src.tasks.HoST_getUp import mdp
    from src.tasks.HoST_getUp.ppo import Policy,advantages,update,save,load
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to silently run the training on CPU')
    torch.manual_seed(42)
    action_mode=os.environ.get('HOST_ACTION_MODE','relative')
    if a.resume:
        header=torch.load(a.resume,map_location='cpu',weights_only=True)
        if header['robot']!=a.robot: raise ValueError('Resume embodiment does not match --robot')
        action_mode=header['metadata'].get('action_mode','absolute')
    cfg=recovery_cfg(a.robot,action_mode=action_mode);cfg.auto_reset=False
    env=ManagerBasedRlEnv(cfg=cfg,device=a.device)
    run=a.run_dir or ROOT/'logs/host_getup'/f'{a.robot}_{time.strftime("%Y%m%d_%H%M%S")}'
    run.mkdir(parents=True,exist_ok=True)
    metadata=dict(robot=a.robot,device=a.device,smoke=a.smoke,seed=42,
        group_weights=mdp.GROUP_WEIGHTS,poses=os.environ.get('HOST_POSES','supine'),
        action_mode=action_mode,
        success_verified=False,num_envs=a.num_envs,environment='mjlab',assistance='curriculum, not evaluation')
    try:
        obs,_=env.reset();x=obs['actor']
        policy=Policy(x.shape[-1],12,hidden=64 if a.smoke else 256).to(a.device)
        start=0
        if a.resume:
            policy,checkpoint=load(a.resume,a.device);policy.train();start=checkpoint['iteration']+1
        opt=torch.optim.Adam(policy.parameters(),lr=3e-4)
        if a.resume: opt.load_state_dict(checkpoint['optimizer'])
        (run/'run.json').write_text(json.dumps(metadata,indent=2))
        with (run/'metrics.jsonl').open('a',buffering=1) as log:
            for iteration in range(start,start+a.iterations):
                rollout=[[] for _ in range(8)]
                for _ in range(a.steps):
                    with torch.no_grad(): raw,lp,v=policy.act(x)
                    control=torch.tanh(raw)*mdp.state(env)['scale'][:,None]
                    nxt,_,term,tout,_=env.step(control)
                    groups=mdp.state(env)['groups'].clone()
                    with torch.no_grad(): nv=policy.critic(nxt['actor'])
                    done=term|tout
                    for target,value in zip(rollout,(x,raw,lp,v,groups,nv,term,done)):
                        target.append(value.detach().clone())
                    if not torch.isfinite(nxt['actor']).all(): raise FloatingPointError('Nonfinite observation')
                    ids=done.nonzero().flatten()
                    if len(ids): nxt,_=env.reset(env_ids=ids)
                    x=nxt['actor']
                O,A,L,V,R,N,T,D=[torch.stack(z) for z in rollout]
                adv,ret=advantages(R,V,N,T,D)
                loss=update(policy,opt,(O,A,L,adv,ret),epochs=1 if a.smoke else 5)
                row=dict(iteration=iteration,loss=loss,group_rewards=R.mean((0,1)).tolist(),
                    assistance=float(mdp.state(env)['assist'].mean()),
                    action_scale=float(mdp.state(env)['scale'].mean()),
                    simulated_transitions=(iteration+1)*a.steps*a.num_envs)
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                if iteration%a.save_interval==0 or iteration==start+a.iterations-1:
                    snapshot_metadata=dict(metadata,action_scale=row['action_scale'],assistance=row['assistance'])
                    save(run/f'model_{iteration}.pt',policy,opt,iteration,a.robot,snapshot_metadata)
        (run/'completion.json').write_text(json.dumps(dict(status='smoke_pass' if a.smoke else 'training_completed',
            iterations=a.iterations,success_verified=False),indent=2))
    finally:
        env.close()
    print(f'Run saved: {run}',flush=True)

if __name__=='__main__': main()
