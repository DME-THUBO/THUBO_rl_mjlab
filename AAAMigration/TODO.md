# 12-DoF Feetech STS3250: MJLab PPO + MuJoCo simulation

Architecture reference: [`ARCHITECTURE.md`](ARCHITECTURE.md), following the
ManagerBased organization of
[`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab).

This migration is intentionally limited to a simulation training milestone:

```text
MJCF + STL meshes + actuator metadata
                 |
                 v
        MJLab manager-based task
                 |
                 v
           PPO (RSL-RL)
                 |
                 v
       MuJoCo / MuJoCo Warp rollout
```

Hardware drivers, IMU integration, ONNX export, deployment, and sim-to-real
validation are out of scope for this milestone and must not block training.

## 1. Source assets and verified model properties

- [x] Create the project-owned `.venv` with MJLab, MuJoCo Warp, Torch CUDA, Warp, and RSL-RL.
- [x] Use [`mjcf/THUBO.xml`](../mjcf/THUBO.xml) as the source robot model.
- [x] Keep the 13 checked-in STL meshes under [`meshes/`](../meshes/).
- [x] Keep the STS3250 metadata in [`sts3250_official.json`](../sts3250_official.json).
- [x] Validate the model before environment construction:
  - 12 actuated revolute joints, `nq=19`, `nv=18`, `nu=12`.
  - Total modeled mass: `2.946841787012769 kg`.
  - Pelvis/root height in the source home pose: `0.39 m`.
  - 13 mesh assets resolve successfully.
  - Actuator metadata type is `feetech-sts3250`.
- [x] Provide a machine-readable validator: `python scripts/validate_model.py`.

The validator is a required pre-training gate. If a mesh, joint, actuator, root
height, or mass check fails, fix the model or its expected property explicitly
before running PPO.

## 2. MJLab environment

- [x] Add [`dme_thubo/robot.py`](../dme_thubo/robot.py): loads the MJCF,
  removes standalone XML motors, and adds MJLab position actuators using the
  candidate STS3250 armature, damping, friction, and effort values.
- [x] Add [`dme_thubo/env_cfg.py`](../dme_thubo/env_cfg.py) with a
  manager-based `ManagerBasedRlEnvCfg`.
- [x] Use 64 parallel environments for training and one for play/smoke tests.
- [x] Use MuJoCo timestep `0.002 s`, decimation `10`, and a `0.02 s` policy step.
- [x] Define position-target actions with 12 dimensions and `[-1, 1]` policy input.
  The action is applied around the model's default joint pose with scale `0.35 rad`.
- [x] Define the 42-dimensional actor observation:
  `joint_pos_rel (12) + joint_vel (12) + base_ang_vel (3) + projected_gravity (3) + last_action (12)`.
- [x] Define standing/balance rewards for alive state, upright posture, root height,
  default joint posture, joint velocity, and action rate.
- [x] Define termination for bad orientation, root height below `0.20 m`, and timeout.
- [ ] Add optional disturbances only after the deterministic standing task is stable.

Registered ManagerBased task IDs:

- [x] `THUBO-Flat-Stand` — standing acceptance task on a plane.
- [x] `THUBO-Rough-Stand` — standing robustness task on procedural rough terrain.
- [x] `THUBO-Flat-Velocity` — walking/velocity tracking configuration staged after standing.
- [x] `THUBO-Rough-Velocity` — rough-terrain velocity tracking configuration staged after standing.
- [x] Smoke-test all four task configurations with local `.venv`: flat/rough standing use 42 observations; flat/rough velocity use 45 observations.

## 3. PPO training

- [x] Add [`dme_thubo/rl_cfg.py`](../dme_thubo/rl_cfg.py) with RSL-RL PPO:
  ELU actor/critic MLPs `(256, 128)`, Adam learning rate `3e-4`, GAE
  `gamma=0.99`, `lambda=0.95`, 24 rollout steps per environment, and TensorBoard logging.
- [x] Register task ID `Feetech-Standing-12DoF` with MJLab's task registry.
- [x] Add [`scripts/train.py`](../scripts/train.py) to launch MJLab's PPO runner.
- [x] Run a one-iteration PPO smoke test on CUDA: 8 envs × 4 steps; PPO completed with finite losses.
- [x] Run the same local `.venv` PPO smoke entry point for `THUBO-Flat-Velocity`: 45-observation actor/critic, finite losses.
- [ ] Train P0 standing until the validator, reset, step, and reward metrics remain finite.
- [ ] Pass flat standing acceptance criteria before velocity training.
- [ ] Pass rough standing acceptance criteria before rough velocity training.
- [ ] Train walking with `THUBO-Flat-Velocity` after flat standing passes.
- [ ] Train walking with `THUBO-Rough-Velocity` after rough standing passes.

## 4. Required commands

From the repository root, using an environment that provides `mjlab`, `mujoco`,
`mujoco-warp`, `torch`, and `rsl_rl`:

```bash
.venv/bin/python scripts/validate_model.py
.venv/bin/python scripts/train.py THUBO-Flat-Stand
```

For a short smoke run through the same entry point:

```bash
DME_THUBO_NUM_ENVS=8 DME_THUBO_NUM_STEPS=4 DME_THUBO_MAX_ITERATIONS=1 \
  DME_THUBO_LOG_ROOT=/tmp/dme_thubo_smoke .venv/bin/python scripts/train.py THUBO-Flat-Stand
```

The first command must pass before training. The second command is the only
training entry point for this migration.

## 5. Deferred work

The following are deliberately deferred and should not be reintroduced into this
TODO until the simulation PPO task is verified:

- Serial bus, servo discovery, hardware safety, Orange Pi runtime, and BNO085 IMU.
- ONNX export and policy deployment.
- Sim-to-real actuator identification and real-robot experiments.
- Hardware/software observation-consistency checks.
