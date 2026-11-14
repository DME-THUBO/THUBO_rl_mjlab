# DME-THUBO

Simulation training for the 12-DoF THUBO robot using MJLab, MuJoCo and RSL-RL PPO.
The checked-in robot model is [`mjcf/THUBO.xml`](mjcf/THUBO.xml), with STL geometry in
[`meshes/`](meshes/) and actuator metadata in [`sts3250_official.json`](sts3250_official.json).

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python scripts/validate_model.py
```

The validator checks the MJCF, meshes, joints, actuators and model metadata before
simulation.

## Tasks

- `THUBO-Flat-Stand`: standing and balance on a plane.
- `THUBO-Flat-Velocity`: commanded velocity tracking on a plane.
- `HoST-GetUp-THUBO`: HoST get-up task for the THUBO robot.

Train a task with:

```bash
.venv/bin/python scripts/train.py THUBO-Flat-Stand
```

For a short smoke run:

```bash
DME_THUBO_NUM_ENVS=8 DME_THUBO_NUM_STEPS=4 DME_THUBO_MAX_ITERATIONS=1 \
  DME_THUBO_LOG_ROOT=/tmp/dme_thubo_smoke \
  .venv/bin/python scripts/train.py THUBO-Flat-Stand
```

Replay a checkpoint in the native viewer:

```bash
.venv/bin/python scripts/play.py THUBO-Flat-Stand path/to/model.pt
```

Use the Viser training entry point when a browser viewer is needed:

```bash
.venv/bin/python scripts/train_with_viser.py THUBO-Flat-Stand \
  --num-envs 2048 --iterations 4000 --viser-port 8080
```

Open `http://localhost:8080` after the viewer starts. Training runs in simulation;
hardware drivers, deployment and sim-to-real validation are outside this milestone.
See [`AGENTS.md`](AGENTS.md) and [`AAAMigration/TODO.md`](AAAMigration/TODO.md) for
repository constraints and deferred work.
