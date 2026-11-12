"""View native HoST-port .pt checkpoints and hotload atomic new saves."""
from pathlib import Path
import sys,argparse,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir',type=Path);p.add_argument('--port',type=int,default=8080)
    p.add_argument('--device',default='cuda:0');p.add_argument('--assist',action='store_true')
    a=p.parse_args()
    import torch,viser
    import warp
    warp.config.kernel_cache_dir=str(ROOT/'logs/warp_cache')
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import ViserPlayViewer
    from src.tasks.HoST_getUp.config import recovery_cfg
    from src.tasks.HoST_getUp.ppo import load
    from src.tasks.HoST_getUp.mdp import state
    def latest():
        files=[p for p in a.run_dir.glob('model_*.pt') if p.stem[6:].isdigit()]
        return max(files,key=lambda p:int(p.stem[6:])) if files else None
    path=latest()
    if path is None: raise FileNotFoundError('No atomic model_*.pt checkpoint available')
    policy,ckpt=load(path,a.device)
    action_mode=ckpt['metadata'].get('action_mode','absolute')
    cfg=recovery_cfg(ckpt['robot'],play=True,action_mode=action_mode);cfg.scene.num_envs=1
    env=ManagerBasedRlEnv(cfg=cfg,device=a.device)
    current=path;last_poll=0.
    def infer(obs):
        nonlocal policy,current,last_poll
        if time.monotonic()-last_poll>2:
            candidate=latest();last_poll=time.monotonic()
            if candidate and candidate!=current:
                new,data=load(candidate,a.device)
                if data['metadata'].get('action_mode','absolute')!=action_mode:
                    raise ValueError('Cannot hotload a different action convention into this environment')
                policy=new;current=candidate
                state(env)['scale'].fill_(data['metadata'].get('action_scale',1.))
                with (a.run_dir/'hotload.jsonl').open('a') as out:
                    out.write(__import__('json').dumps(dict(checkpoint=candidate.name,time=time.time()))+'\n')
                print('Hotloaded '+candidate.name,flush=True)
        if not a.assist: state(env)['assist'].zero_()
        if current==path: state(env)['scale'].fill_(ckpt['metadata'].get('action_scale',1.))
        with torch.no_grad(): return policy.inference(obs['actor'],state(env)['scale'])
    try:
        server=viser.ViserServer(host='127.0.0.1',port=a.port)
        ViserPlayViewer(env,infer,viser_server=server).run()
    finally: env.close()

if __name__=='__main__': main()
