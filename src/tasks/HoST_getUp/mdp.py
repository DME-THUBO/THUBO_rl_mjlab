"""HoST recovery mechanisms re-expressed using MJLab batched entities.

Reference: InternRobotics/HoST commit 70bb5809, Pi ground implementation.
Root/sole height substitutes for the head keyframe on the headless THUBO.
"""
import math
import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz

GROUP_WEIGHTS = (2.5,.1,1.,1.)

def tolerance(x, lower, margin, value_at_margin=.1):
    distance = (lower-x).clamp_min(0)/margin
    return torch.exp(math.log(value_at_margin)*distance.square())

def state(env):
    if not hasattr(env,"_host"):
        env._host = dict(assist=torch.full((env.num_envs,),.6,device=env.device),
            scale=torch.ones(env.num_envs,device=env.device),
            best=torch.zeros(env.num_envs,device=env.device),
            last_height=torch.zeros(env.num_envs,device=env.device),
            last_upright=torch.zeros(env.num_envs,device=env.device),
            groups=torch.zeros(env.num_envs,4,device=env.device))
    return env._host

def reset_recovery(env,env_ids, height=.39, reset_height=.12, poses=("supine",),curriculum_mode='terminal'):
    ids = torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    s=state(env)
    # Update only completed episodes; initial state cannot advance the curriculum.
    complete=env.episode_length_buf[ids]>0
    if curriculum_mode=='terminal':
        passed=(s['last_height'][ids]>.95*height)&(s['last_upright'][ids]>.9)&complete
    else:
        passed=(s["best"][ids]>.95*height)&complete
    s["assist"][ids]=(s["assist"][ids]-.05*passed).clamp_min(0)
    s["scale"][ids]=(s["scale"][ids]-.02*passed).clamp_min(.25)
    s["best"][ids]=0
    s['last_height'][ids]=0;s['last_upright'][ids]=0
    robot=env.scene["robot"]
    n=len(ids);choice=torch.randint(len(poses),(n,),device=env.device)
    roll=torch.zeros(n,device=env.device);pitch=roll.clone()
    for i,pose in enumerate(poses):
        mask=choice==i
        if pose=="supine": pitch[mask]=-math.pi/2
        elif pose=="prone": pitch[mask]=math.pi/2
        elif pose=="left": roll[mask]=math.pi/2
        elif pose=="right": roll[mask]=-math.pi/2
        else: raise ValueError(pose)
    yaw=(torch.rand(n,device=env.device)*2-1)*math.pi
    quat=quat_from_euler_xyz(roll,pitch,yaw)
    root=robot.data.default_root_state[ids].clone()
    pos=root[:,:3]+env.scene.env_origins[ids]
    pos[:,2]=reset_height+env.scene.env_origins[ids,2]
    robot.write_root_link_pose_to_sim(torch.cat((pos,quat),-1),env_ids=ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(n,6,device=env.device),env_ids=ids)
    q=robot.data.default_joint_pos[ids].clone()
    lim=robot.data.joint_pos_limits[ids]
    q=(q+(torch.rand_like(q)*2-1)*.1).clamp(lim[...,0],lim[...,1])
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)

def proprioception(env):
    r=env.scene["robot"];s=state(env)
    # One contiguous 43-vector permits frame-major history (upstream semantics).
    obs=torch.cat((r.data.root_link_ang_vel_b,r.data.projected_gravity_b,
        r.data.joint_pos,r.data.joint_vel,env.action_manager.action,s["scale"][:,None]),-1)
    return obs.clamp(-100,100)

def grouped_reward(env,height=.39,feet=("left_ankle_roll","right_ankle_roll")):
    r=env.scene["robot"];s=state(env)
    z=r.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
    upright=-r.data.projected_gravity_b[:,2]
    s["best"]=torch.maximum(s["best"],z)
    s['last_height']=z.clone();s['last_upright']=upright.clone()
    fids,_=r.find_bodies(feet)
    foot=r.data.body_link_pos_w[:,fids]
    clearance=z-foot[:,:,2].mean(-1)+env.scene.env_origins[:,2]
    # Multiplicative task objective, four separately valued reward groups.
    task=tolerance(upright,.99,1.,.05)*tolerance(clearance,height,height,.1)
    action=env.action_manager.action
    prev=env.action_manager.prev_action
    tau=r.data.qfrc_actuator
    vel=r.data.joint_vel
    limits=r.data.joint_pos_limits
    violation=(limits[...,0]-r.data.joint_pos).clamp_min(0)+(r.data.joint_pos-limits[...,1]).clamp_min(0)
    regu=-.01*(action-prev).square().sum(-1)-.001*vel.square().sum(-1)
    regu-=2.5e-6*tau.square().sum(-1)+2.5e-5*(tau*vel).abs().sum(-1)+100*violation.sum(-1)
    standing=z>.95*height
    style=-.25*(r.data.joint_pos-r.data.default_joint_pos).square().sum(-1)
    separation=(foot[:,0,1]-foot[:,1,1]).abs()
    style-=10*(separation-.12).abs()*standing
    target=(10*torch.exp(-2*r.data.root_link_ang_vel_b[:,:2].square().sum(-1))+
        10*torch.exp(-5*r.data.root_link_lin_vel_b[:,:2].square().sum(-1))+
        10*torch.exp(-5*r.data.projected_gravity_b[:,:2].square().sum(-1))+
        10*torch.exp(-20*(z-height).abs()))*standing
    s["groups"]=torch.stack((task,regu,style,target),-1)
    return (s["groups"]*torch.tensor(GROUP_WEIGHTS,device=env.device)).sum(-1)

def apply_assistance(env,env_ids=None,base="pelvis",mass=2.946841787):
    r=env.scene["robot"];s=state(env)
    ids,_=r.find_bodies((base,))
    force=torch.zeros(env.num_envs,1,3,device=env.device)
    gate=(env.episode_length_buf>30)&(-r.data.projected_gravity_b[:,2]>.8)
    force[:,0,2]=s["assist"]*mass*9.81*gate
    r.write_external_wrench_to_sim(force,torch.zeros_like(force),body_ids=ids)

def nonfinite(env):
    r=env.scene["robot"]
    return ~torch.isfinite(r.data.joint_pos).all(-1)
