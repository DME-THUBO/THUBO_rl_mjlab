"""Standing-task terms kept small and explicit for the first PPO milestone."""

import torch


def terrain_curriculum(env, env_ids):
    """Advance successful terrain episodes and retry failed ones."""
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return 0.0
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    active = env.episode_length_buf[env_ids] > 0
    terminated = env.termination_manager.terminated[env_ids]
    levels = terrain.terrain_levels[env_ids] + active.to(torch.long) * (
        (~terminated).to(torch.long) - terminated.to(torch.long)
    )
    levels = levels.clamp(0, terrain.terrain_origins.shape[0] - 1)
    terrain.terrain_levels[env_ids] = levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        levels, terrain.terrain_types[env_ids]
    ]
    return levels.float().mean()


def feel_slide(env, asset_cfg, contact_height: float = 0.035):
    """Penalize horizontal foot motion while a foot is close to the ground."""
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    foot_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    near_ground = (foot_pos[..., 2] < contact_height).to(foot_vel.dtype)
    return torch.sum(near_ground * torch.sum(torch.square(foot_vel[..., :2]), dim=-1), dim=1)


def root_height_reward(env, target: float = 0.39, std: float = 0.04):
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    return torch.exp(-torch.square(z - target) / (2.0 * std * std))


def upright_reward(env):
    gravity = env.scene["robot"].data.projected_gravity_b
    return torch.clamp(-gravity[:, 2], min=0.0, max=1.0)


def joint_position_penalty(env):
    robot = env.scene["robot"]
    default = robot.data.default_joint_pos
    assert default is not None
    return torch.sum(torch.square(robot.data.joint_pos - default), dim=1)
