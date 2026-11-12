"""Standing-task terms kept small and explicit for the first PPO milestone."""

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz
from mjlab.utils.lab_api.string import resolve_matching_names_values


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


def _gait_phase(env, period: float) -> torch.Tensor:
    env_index = torch.arange(env.num_envs, device=env.device, dtype=torch.float32)
    env_offset = torch.remainder(env_index * 0.61803398875, 1.0)
    elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
    return torch.remainder(elapsed / period + env_offset, 1.0)


def gait_phase_observation(env, period: float = 0.8) -> torch.Tensor:
    """Encode the left-foot gait phase; the right foot is half a cycle later."""
    phase = _gait_phase(env, period) * (2.0 * torch.pi)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def feet_slide(
    env,
    asset_cfg,
    default_offsets: tuple[float, float] = (0.066965, -0.055035),
    width_tolerance: float = 0.05,
    command_name: str = "twist",
    command_threshold: float = 0.01,
):
    """Penalize outward spread for vx, and inward collapse during lateral motion."""
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    root_pos = asset.data.root_link_pos_w
    num_feet = foot_pos.shape[1]
    foot_pos_b = quat_apply_inverse(
        asset.data.root_link_quat_w[:, None, :].expand(-1, num_feet, -1),
        foot_pos - root_pos[:, None, :],
    )
    lateral_offset = foot_pos_b[..., 1]
    default_offsets_tensor = torch.tensor(
        default_offsets,
        device=foot_pos.device,
        dtype=foot_pos.dtype,
    )
    signed_offset = (lateral_offset - default_offsets_tensor) * torch.sign(default_offsets_tensor)
    command = env.command_manager.get_command(command_name)
    lateral_motion = torch.abs(command[:, 1]) > command_threshold
    outward_excess = torch.clamp(signed_offset - width_tolerance, min=0.0)
    inward_excess = torch.clamp(-signed_offset, min=0.0)
    excess = torch.where(lateral_motion[:, None], inward_excess, outward_excess)
    return torch.sum(torch.square(excess / width_tolerance), dim=1)


def feet_air_time(env, asset_cfg, contact_height: float = 0.065, max_air_time: float = 0.5):
    """Reward each foot when it lands after spending time above the ground."""
    asset = env.scene[asset_cfg.name]
    foot_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    in_air = foot_height > contact_height
    if not hasattr(env, "_thubo_feet_air_time"):
        env._thubo_feet_air_time = torch.zeros_like(foot_height)
    reset = env.episode_length_buf == 0
    env._thubo_feet_air_time[reset] = 0.0
    was_air = env._thubo_feet_air_time > 0.0
    air_duration = env._thubo_feet_air_time
    env._thubo_feet_air_time = torch.where(
        in_air,
        env._thubo_feet_air_time + env.step_dt,
        torch.zeros_like(env._thubo_feet_air_time),
    )
    landed = was_air & ~in_air
    return torch.sum(torch.where(landed, torch.clamp(air_duration, max=max_air_time), torch.zeros_like(foot_height)), dim=1)


def feet_clearance_swing_reward(
    env,
    sensor_name: str,
    asset_cfg,
    command_name: str = "twist",
    period: float = 0.8,
    swing_ratio: float = 0.6,
    target_height: float = 0.10,
    clearance_tolerance: float = 0.02,
    command_threshold: float = 0.05,
    sole_half_thickness: float = 0.001,
    min_swing_height: float = 0.04,
    contact_penalty: float = 1.0,
):
    """Track a sinusoidal sole-clearance target during each foot's swing phase.

    The sinusoid dips toward zero at the swing boundaries, so the target is
    floored at ``min_swing_height``: a sole hovering a few millimetres above the
    ground can no longer harvest the near-zero ends of the arc. Dragging the
    sole through a nominal swing window used to be masked to zero reward; it is
    now charged ``contact_penalty`` per foot so shuffling along the ground is
    strictly worse than stepping.
    """
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    if found is None:
        raise RuntimeError(f"Sensor '{sensor_name}' must expose contact findings")
    asset = env.scene[asset_cfg.name]
    foot_clearance = torch.clamp(
        asset.data.geom_pos_w[:, asset_cfg.geom_ids, 2] - sole_half_thickness,
        min=0.0,
    )
    phase = _gait_phase(env, period)
    foot_phase = torch.stack((phase, torch.remainder(phase + 0.5, 1.0)), dim=1)
    in_swing = foot_phase < swing_ratio
    swing_progress = torch.clamp(foot_phase / swing_ratio, min=0.0, max=1.0)
    target_clearance = torch.clamp(
        target_height * torch.sin(torch.pi * swing_progress), min=min_swing_height
    )
    in_contact = found.view(found.shape[0], 2, -1).any(dim=-1)
    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :3], dim=1) > command_threshold
    active_swing = in_swing & ~in_contact & moving[:, None]
    error = torch.square((foot_clearance - target_clearance) / clearance_tolerance)
    per_foot_reward = torch.exp(-error) * active_swing.to(foot_clearance.dtype)
    shuffling = in_swing & in_contact & moving[:, None]
    per_foot_penalty = shuffling.to(foot_clearance.dtype) * contact_penalty
    return per_foot_reward.sum(dim=1) - per_foot_penalty.sum(dim=1)


def feet_flat_orientation_reward(
    env,
    sensor_name: str,
    asset_cfg,
    tilt_tolerance: float = 0.15,
):
    """Reward contacting soles whose local normal points vertically upward."""
    found = env.scene[sensor_name].data.found
    if found is None:
        raise RuntimeError(f"Sensor '{sensor_name}' must expose contact findings")
    asset = env.scene[asset_cfg.name]
    sole_quat = asset.data.geom_quat_w[:, asset_cfg.geom_ids]
    w, x, y, z = sole_quat.unbind(dim=-1)
    normal_x = 2.0 * (x * z + w * y)
    normal_y = 2.0 * (y * z - w * x)
    horizontal_normal_sq = torch.square(normal_x) + torch.square(normal_y)
    per_foot_reward = torch.exp(-horizontal_normal_sq / (tilt_tolerance * tilt_tolerance))
    in_contact = found.view(found.shape[0], 2, -1).any(dim=-1)
    contact_weight = in_contact.to(per_foot_reward.dtype)
    return torch.sum(per_foot_reward * contact_weight, dim=1) / torch.clamp(
        torch.sum(contact_weight, dim=1), min=1.0
    )


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


class VariablePosture:
    """Reward default pose with per-joint tolerances selected by command speed.

    Follows Unitree's variable_posture: exp(-mean((q - q_default)^2 / std^2)).
    Tolerances are reward scales in radians, not hard joint limits.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        asset_cfg = cfg.params["asset_cfg"]
        robot = env.scene[asset_cfg.name]
        _, names = robot.find_joints(asset_cfg.joint_names)
        self.stds = []
        for key in ("std_standing", "std_walking", "std_running"):
            _, _, values = resolve_matching_names_values(cfg.params[key], names)
            std = torch.tensor(values, device=env.device, dtype=torch.float32)
            if len(values) != len(names) or not torch.all(std > 0):
                raise ValueError(f"{key} must provide a positive tolerance for every joint")
            self.stds.append(std)

    def __call__(self, env, asset_cfg, command_name, std_standing, std_walking,
                 std_running, walking_threshold=0.1, running_threshold=1.5):
        del std_standing, std_walking, std_running
        command = env.command_manager.get_command(command_name)
        speed = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
        std = torch.where(
            (speed < walking_threshold)[:, None], self.stds[0],
            torch.where((speed < running_threshold)[:, None], self.stds[1], self.stds[2]),
        )
        robot = env.scene[asset_cfg.name]
        error = (robot.data.joint_pos[:, asset_cfg.joint_ids]
                 - robot.data.default_joint_pos[:, asset_cfg.joint_ids])
        return torch.exp(-torch.mean(torch.square(error / std), dim=1))


def stand_still(env, asset_cfg, command_name="twist", command_threshold=0.1):
    """Return squared default-pose error only for near-zero velocity commands."""
    robot = env.scene[asset_cfg.name]
    error = (robot.data.joint_pos[:, asset_cfg.joint_ids]
             - robot.data.default_joint_pos[:, asset_cfg.joint_ids])
    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
    return torch.square(error).sum(dim=1) * (speed <= command_threshold)


def reset_fallen_pose(
    env,
    env_ids,
    ratio: float = 0.4,
    fallen_env_ids: tuple[int, ...] | None = None,
    root_height: float = 0.12,
    root_height_jitter: float = 0.02,
    joint_jitter: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Drop a random subset of resetting envs into a supine lying pose.

    The pose is face-up (pitch=-pi/2) with random yaw and small roll/pitch
    jitter, zero velocities, and default joints offset by a bounded jitter.
    Register after ``reset_scene_to_default`` so it only overrides the picked
    subset of envs.

    When ``fallen_env_ids`` is given it pins exactly those env indices to the
    fallen pose on every reset (deterministic split for viewers); otherwise a
    Bernoulli draw with ``ratio`` picks the fallen subset.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if fallen_env_ids is not None:
        pinned = torch.as_tensor(fallen_env_ids, device=env.device)
        pick = torch.isin(env_ids, pinned)
    else:
        pick = torch.rand(env_ids.shape[0], device=env.device) < ratio
    if not pick.any():
        return
    fallen_ids = env_ids[pick]
    asset = env.scene[asset_cfg.name]
    num = fallen_ids.shape[0]

    roll = (torch.rand(num, device=env.device) * 2.0 - 1.0) * 0.15
    pitch = -0.5 * torch.pi + (torch.rand(num, device=env.device) * 2.0 - 1.0) * 0.15
    yaw = (torch.rand(num, device=env.device) * 2.0 - 1.0) * torch.pi
    quat = quat_from_euler_xyz(roll, pitch, yaw)

    root = asset.data.default_root_state[fallen_ids].clone()
    pos = root[:, 0:3] + env.scene.env_origins[fallen_ids]
    pos[:, 2] = (
        root_height + (torch.rand(num, device=env.device) * 2.0 - 1.0) * root_height_jitter
    )
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=fallen_ids)
    asset.write_root_link_velocity_to_sim(
        torch.zeros(num, 6, device=env.device), env_ids=fallen_ids
    )

    default_joint_pos = asset.data.default_joint_pos[fallen_ids]
    joint_pos = default_joint_pos + (
        torch.rand_like(default_joint_pos) * 2.0 - 1.0
    ) * joint_jitter
    limits = asset.data.joint_pos_limits[fallen_ids]
    joint_pos = torch.max(torch.min(joint_pos, limits[..., 1]), limits[..., 0])
    asset.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), env_ids=fallen_ids
    )


def getup_phase(env, z_low: float = 0.30, z_high: float = 0.36) -> torch.Tensor:
    """Per-env hysteresis latch: True while in the getup phase.

    Below ``z_low`` the env enters the getup phase, above ``z_high`` it returns
    to the velocity-tracking phase, and in between the previous phase is kept.
    """
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    if not hasattr(env, "_thubo_getup_down"):
        env._thubo_getup_down = z < z_low
    down = env._thubo_getup_down
    down = (down | (z < z_low)) & ~(z > z_high)
    env._thubo_getup_down = down
    return down


def gated_reward(func, *, down: bool):
    """Gate a reward term to the getup phase (down=True) or the velocity phase.

    Reward weights are global scalars, so per-env phase gating happens here as
    a 0/1 mask following the same idiom as ``feet_clearance_swing_reward``.
    """

    def gated(env, **params):
        is_down = getup_phase(env)
        mask = is_down if down else ~is_down
        if not mask.any():
            return torch.zeros(env.num_envs, device=env.device)
        return func(env, **params) * mask.to(torch.float32)

    return gated


def getup_root_height_reward(env, target: float = 0.39, std: float = 0.15):
    """Shaped root-height recovery with gradient from a lying pose upward.

    The L1 exponential kernel keeps a non-zero slope at low heights, unlike the
    squared ``root_height_reward`` kernel which is flat far from the target.
    """
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    return torch.exp(-torch.abs(z - target) / std)


def root_height_observation(env) -> torch.Tensor:
    return env.scene["robot"].data.root_link_pos_w[:, 2:3]
