# DME-THUBO

This repository contains the simulation-only first milestone for the 12-DoF
THUBO robot with STS3250 actuators. The task uses the checked-in `mjcf/THUBO.xml` and STL meshes,
MJLab's ManagerBased environment, MuJoCo Warp simulation, and RSL-RL PPO.

The package layout follows the upstream
[`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab)
organization. See [`AAAMigration/ARCHITECTURE.md`](AAAMigration/ARCHITECTURE.md).

Create and populate the project-owned environment with `uv`:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

Use the Python environment that provides `mjlab`, `mujoco`, `mujoco-warp`,
PyTorch, and `rsl_rl`:

```bash
.venv/bin/python scripts/validate_model.py
.venv/bin/python scripts/train.py THUBO-Rough-Velocity
```

The default training profile uses 2048 parallel environments and 4000 PPO
iterations. The entry point also accepts `DME_THUBO_NUM_ENVS`,
`DME_THUBO_NUM_STEPS`, `DME_THUBO_MAX_ITERATIONS` (maximum 4000), and
`DME_THUBO_LOG_ROOT` for a short smoke run.

Run rough-terrain velocity tracking with a Viser viewer that automatically
hotloads each new 100-iteration checkpoint:

```bash
./.venv/bin/python scripts/train_with_viser.py THUBO-Rough-Velocity \
  --num-envs 2048 --iterations 4000 --save-interval 100 --viser-port 8080
```

Open `http://localhost:8080` to view the policy. Training runs on CUDA while
the single-environment Viser simulation runs on CPU.

TensorBoard for the completed run is available at `http://localhost:6006`:

```bash
./.venv/bin/tensorboard --logdir logs/rsl_rl/dme_thubo_rough_velocity \
  --host 127.0.0.1 --port 6006
```

Replay any local checkpoint in the native MuJoCo viewer:

```bash
./.venv/bin/python scripts/play.py THUBO-Rough-Velocity \
  logs/rsl_rl/dme_thubo_rough_velocity/<run>/model_3999.pt
```

Use `--viewer viser` to replay it in a browser-based viewer.

`validate_model.py` checks the 12 joints and actuators, mesh count, source root
height, total mass, and actuator metadata before the environment is built.
Registered tasks are `THUBO-Flat-Stand`, `THUBO-Rough-Stand`,
`THUBO-Flat-Velocity`, and `THUBO-Rough-Velocity`. The demonstration training
profile uses `THUBO-Rough-Velocity` on procedural rough terrain.

The velocity task has 12 position-target actions and 45 actor observations.
Hardware drivers, deployment, and sim-to-real work are intentionally outside
this milestone; see
[`AAAMigration/TODO.md`](AAAMigration/TODO.md) for the active scope.
