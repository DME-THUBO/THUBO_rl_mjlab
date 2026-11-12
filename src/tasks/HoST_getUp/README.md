# HoST_getUp: 12-DoF MJLab recovery port

Reference: [InternRobotics/HoST](https://github.com/InternRobotics/HoST), immutable
commit `70bb580949a336a920833700e4b5dc3bf7fe87ce`, especially
`legged_gym/legged_gym/envs/pi/pi_host_ground.py` and `pi_config_ground.py`.
The upstream MIT license is preserved in `LICENSE_HoST`.

Two embodiments: `HoST-GetUp-THUBO` and `HoST-GetUp-PI`. The Pi URDF is converted
to MJCF with a free root and named fixed keyframes. The conversion report records
joint names, source hash, modeled mass and random-pose FK comparisons.

## Modules

- `robots.py`: separate model, body, initial-state and controller adapters.
- `mdp.py`: recovery resets, frame-major six-step proprioceptive history,
  multiplicative orientation/height task reward, regularization/style/target
  groups, success-dependent assistance/action-scale curricula.
- `config.py`: MJLab task registration, flat contacts, reset and step events.
- `ppo.py`: four-value critic, per-group GAE and normalized advantages,
  weighted clipped PPO, timeout bootstrapping and atomic native `.pt` saves.
- `scripts/host_train.py`: CUDA smoke/training/resume entry point.
- `scripts/host_viser.py`: inference viewer and latest-checkpoint hotload.

## Reproduce

```bash
conda env create -f environment-host.yml
conda activate THUBO_mjlab
python scripts/validate_model.py
python scripts/host_train.py --robot pi --smoke
python scripts/host_train.py --robot thubo --smoke
python scripts/host_train.py --robot thubo --num-envs 256 --iterations 2000 --run-dir logs/host_getup/thubo_recovery
python scripts/host_viser.py logs/host_getup/thubo_recovery --port 8080
```

Default reset is supine, matching the released Pi ground task. Explicitly select
other starts with `HOST_POSES=supine,prone,left,right`; convergence for additional
starts is a separate experiment, not an upstream guarantee. Viewer assistance
defaults to zero; `--assist` enables the training curriculum for diagnosis.
`hotload.jsonl` records actual checkpoint swaps. Saved `.pt` files contain model,
four-value critic, optimizer state, iteration and embodiment metadata. They are
not compatible with the previous IsaacGym or standard RSL-RL checkpoint format.

## Migration decisions and limits

This is a recovery task/algorithm port, not a claim of numerical equivalence to
IsaacGym trajectories or a reproduction of the paper's reported success rate.
Physics is MuJoCo Warp, not PhysX. Controller dt is 0.02 s with 0.002 s substeps.
The target repository's THUBO controller profile is inherited, while Pi uses
upstream stiffness/damping and its URDF 20 Nm limits. These are simulation
profiles, not measured THUBO continuous-duty ratings.

THUBO has no head or arms: root-to-foot height replaces HoST's head keyframe.
Robot-normalized stage thresholds and fractional body-weight assistance replace
Pi-specific dimensional constants. The action is bounded and applied around
the nominal pose, rather than the upstream current-joint-position target.
The six 43-element observation frames are otherwise kept in time-major order.
Current regularization/style rewards are explicit in `mdp.py`; not every
robot-specific Pi ankle keyframe penalty is reproduced. Domain randomization,
sensor noise parity and exhaustive upstream reward-by-reward parity remain
future validation items. Existing velocity tasks and old launchers are unchanged.

Do not treat smoke-test checkpoints, assisted lifting, or a training process
running on CUDA as proof of unassisted recovery. Evaluate supine/prone/side starts
separately and report assistance, hold time, torque saturation and success rates.
