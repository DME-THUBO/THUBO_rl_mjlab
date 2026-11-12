"""Check grouped timeout GAE and checkpoint round trips without training."""
from pathlib import Path
import sys,tempfile,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.tasks.HoST_getUp.ppo import Policy,advantages,save,load

def main():
    rewards=torch.ones(2,1,4);values=torch.zeros_like(rewards);nxt=torch.ones_like(rewards)*2
    term=torch.tensor([[False],[True]]);done=torch.ones(2,1,dtype=torch.bool)
    adv,ret=advantages(rewards,values,nxt,term,done)
    torch.testing.assert_close(adv[0],torch.ones(1,4)*2.98)
    torch.testing.assert_close(adv[1],torch.ones(1,4))
    policy=Policy(hidden=64);optimizer=torch.optim.Adam(policy.parameters())
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=logs,prefix='host_check_') as tmp:
        assert Path(tmp).resolve().is_relative_to(logs.resolve())
        p=Path(tmp)/'model_0.pt'
        save(p,policy,optimizer,0,'thubo',{})
        restored,meta=load(p)
        x=torch.randn(3,258)
        torch.testing.assert_close(policy.actor(x),restored.actor(x))
        assert meta['iteration']==0
    report=dict(grouped_timeout_bootstrap=True,terminal_bootstrap_disabled=True,
        atomic_checkpoint_roundtrip=True,critic_outputs=4,actor_observations=258,actor_actions=12)
    (logs/'host_unit_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
if __name__=='__main__':main()
