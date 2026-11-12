"""Four-value PPO with separately normalized group advantages.

MJLab supplies simulation/observations; no IsaacGym imports or old RSL-RL
storage are needed. Checkpoints are native to this runner, not old HoST weights.
"""
from pathlib import Path
import os
import torch
from torch import nn
from torch.distributions import Normal
from .mdp import GROUP_WEIGHTS

class Policy(nn.Module):
    def __init__(self,observations=258,actions=12,hidden=256):
        super().__init__()
        def net(outputs):
            return nn.Sequential(nn.Linear(observations,hidden),nn.ELU(),
                nn.Linear(hidden,hidden),nn.ELU(),nn.Linear(hidden,outputs))
        self.actor=net(actions);self.critic=net(4)
        self.log_std=nn.Parameter(torch.full((actions,),-.7))
        self.arch=dict(observations=observations,actions=actions,hidden=hidden)
    def distribution(self,x):
        return Normal(self.actor(x),self.log_std.clamp(-5,1).exp())
    def act(self,x):
        d=self.distribution(x);raw=d.sample()
        return raw,d.log_prob(raw).sum(-1),self.critic(x)
    def inference(self,x,scale):
        return torch.tanh(self.actor(x))*scale[:,None]

def advantages(rewards,values,next_values,terminated,done,gamma=.99,lam=.95):
    out=torch.zeros_like(rewards);carry=torch.zeros_like(rewards[0])
    for i in reversed(range(len(rewards))):
        delta=rewards[i]+gamma*next_values[i]*(~terminated[i])[:,None]-values[i]
        carry=delta+gamma*lam*(~done[i])[:,None]*carry
        out[i]=carry
    return out,out+values

def update(policy,optimizer,batch,epochs=5,minibatches=4):
    obs,raw,oldlog,adv,returns=batch
    adv=(adv-adv.mean((0,1),keepdim=True))/(adv.std((0,1),keepdim=True)+1e-8)
    weighted=(adv*torch.tensor(GROUP_WEIGHTS,device=adv.device)).sum(-1)
    obs=obs.flatten(0,1);raw=raw.flatten(0,1);oldlog=oldlog.flatten()
    weighted=weighted.flatten();returns=returns.flatten(0,1)
    stats=[]
    for _ in range(epochs):
        for idx in torch.randperm(len(obs),device=obs.device).chunk(minibatches):
            d=policy.distribution(obs[idx]);lp=d.log_prob(raw[idx]).sum(-1)
            ratio=(lp-oldlog[idx]).exp()
            actor=-torch.minimum(ratio*weighted[idx],ratio.clamp(.8,1.2)*weighted[idx]).mean()
            value=(policy.critic(obs[idx])-returns[idx]).square().mean()
            loss=actor+.5*value-.005*d.entropy().sum(-1).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),1.)
            optimizer.step();stats.append(float(loss.detach()))
    return sum(stats)/len(stats)

def save(path,policy,optimizer,iteration,robot,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pt.tmp')
    torch.save(dict(format='THUBO-HoST-mjlab-v1',policy=policy.state_dict(),optimizer=optimizer.state_dict(),
        iteration=iteration,robot=robot,architecture=policy.arch,metadata=metadata),temporary)
    os.replace(temporary,path)

def load(path,device='cpu'):
    obj=torch.load(path,map_location=device,weights_only=True)
    if obj['format']!='THUBO-HoST-mjlab-v1': raise ValueError('Wrong checkpoint format')
    p=Policy(**obj['architecture']).to(device);p.load_state_dict(obj['policy'],strict=True);p.eval()
    return p,obj
