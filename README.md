# THUBO ManagerBased RL

12-DoF THUBO simulation with MJLab, MuJoCo Warp, and STL geometry from `mjcf/THUBO.xml`.
Hardware deployment is outside this repository's current scope.

## Shared configuration and MDP

- `dme_thubo/mdp.py`: every custom standing, velocity, combined recovery, and HoST term and per-environment runtime state. Upstream MJLab terms remain upstream references.
- `dme_thubo/env_cfg.py`: the only environment construction entry, `dme_thubo_env_cfg(task=..., robot=..., terrain="flat", play=False, action_mode=None)`. Each call builds independent managers, scene, actor/critic terms and configuration instances.
- `dme_thubo/embodiments.py`: side-effect-free THUBO/Pi adapters; `robot.py` defines THUBO.
- `dme_thubo/rl_cfg.py`: standard RSL-RL PPO settings.
- `src/tasks/*/config.py`: thin registration and compatibility wrappers; no standing-baseline overrides.
- `src/tasks/HoST_getUp/ppo.py`: dedicated four-value HoST PPO and native checkpoint format.

Velocity organization and task design reference [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab).
Recovery references [InternRobotics/HoST](https://github.com/InternRobotics/HoST), commit
`70bb580949a336a920833700e4b5dc3bf7fe87ce`; see [license and adaptation notes](src/tasks/HoST_getUp/README.md).
THUBO has no arms or head, so recovery uses pelvis/foot geometry and normalized thresholds.
The HoST runner retains grouped GAE, separate group advantage normalization and a four-output critic.

## Enabled tasks

| Task ID | Actor/critic observations | Actions and reset | Runner |
| --- | --- | --- | --- |
| `THUBO-Flat-Stand` | 42 | 12 default-offset position targets, scale 0.35; standing reset | RSL-RL |
| `THUBO-Flat-Velocity` | 47 | same actions; twist command and gait phase | RSL-RL |
| `HoST-GetUp-THUBO` | 43 per frame × 6 = 258 | 12 relative position actions; fallen reset and curricula | HoST PPO |

Default `scripts/train.py` task is `THUBO-Flat-Velocity`, 2048 environments and 4000 updates.
`THUBO-Rough-Velocity` remains disabled. `THUBO-Rough-Stand` is not registered.
Rough terrain height/obstacle issues require separate work and acceptance.
Simulation timestep is 0.002 s, decimation 10 (50 Hz control).
Standard tasks terminate on falling, low root, or timeout; HoST only on nonfinite state or timeout.
The combined task retains its distinct mixed reset and hysteresis-gated reward semantics.

## Install and run

```bash
conda env create -f environment-host.yml
conda activate THUBO_mjlab
python -m pip install -e .
python scripts/validate_model.py
python scripts/train.py THUBO-Flat-Stand
python scripts/train.py THUBO-Flat-Velocity
python scripts/train.py THUBO-Flat-Velocity-Getup
python scripts/host/train.py --robot thubo --smoke
python scripts/host/train.py --robot thubo --num-envs 256 --iterations 2000 --run-dir logs/host_getup/thubo_recovery
```

Standard training overrides: `DME_THUBO_NUM_ENVS`, `DME_THUBO_NUM_STEPS`,
`DME_THUBO_MAX_ITERATIONS` (at most 4000), `DME_THUBO_SAVE_INTERVAL`, `DME_THUBO_RUN_NAME`, `DME_THUBO_LOG_ROOT`.
HoST's entry point parses `HOST_ACTION_MODE` (default `relative`) and `HOST_POSES` (default `supine`) at runtime and passes them explicitly to the common factory.
The factory and registration modules do not read environment variables.
New standard training runs use the experiment directory `dme_thubo_flat_velocity`.
Existing checkpoints under `dme_thubo_rough_velocity` can still be replayed by passing their original paths.

```bash
python scripts/play.py THUBO-Flat-Velocity logs/rsl_rl/dme_thubo_flat_velocity/<run>/model_3999.pt --viewer viser
python scripts/play.py HoST-GetUp-THUBO logs/host_getup/thubo_recovery/model_1999.pt --viewer viser
python scripts/train_with_viser.py HoST-GetUp-THUBO --num-envs 256 --iterations 2000 --viewer-num-envs 4 --save-interval 25
python scripts/train_with_viser.py --viewer-only logs/host_getup/thubo_recovery --device cpu --viser-port 8080
python scripts/host/eval.py logs/host_getup/thubo_recovery/model_1999.pt
```

Viser defaults to localhost:8080. Standard replay uses MJLab/RSL-RL loading;
HoST replay uses its dedicated loader. Old HoST checkpoints without `action_mode`
retain the `absolute` compatibility rule. Different action modes cannot be hotloaded into the same viewer.
Short training proves runnable execution, not successful unassisted recovery.
Acceptance requires separate zero-assistance, sustained-standing evaluation per pose.

## Optional Pi assets

`src/assets/robots/pi/pi.xml` and `pi_12dof_release_v1.urdf` are locally generated,
Git-ignored optional assets. Generate from a licensed upstream checkout:

```bash
python scripts/host/convert_pi.py /path/to/HoST/legged_gym/resources/robots/pi_12dof
python scripts/host/train.py --robot pi --smoke
```

The converter, provenance metadata and license remain tracked. Missing local Pi
assets skip Pi registration without affecting THUBO; explicitly requesting Pi
reports the missing files and generation command. Existing history still contains
older Pi assets; no history rewrite is required.

## Refactor validation

```bash
python scripts/refactor_checks.py
python scripts/refactor_asset_checks.py
python scripts/refactor_runtime_checks.py
python scripts/host/checks.py
python scripts/host/train.py --robot thubo --smoke --run-dir logs/refactor_host_smoke
python scripts/host/eval.py logs/refactor_host_smoke/model_1.pt --poses supine --num-envs 4 --seconds 0.16 --output logs/refactor_host_replay.json
```

The frozen configuration fixture compares against `e6331e03432f1f6dbed137b9e3d0080ebb1b3d55`,
including dictionary/observation order and task parameters. Pi snapshots are checked when local Pi assets exist.
Runtime checks create/reset enabled tasks, check combined-task phase transitions,
and train/reload/replay standing and velocity checkpoints. Output stays under ignored `logs/`.
See [architecture](AAAMigration/ARCHITECTURE.md) and [HoST training flow](src/tasks/HoST_getUp/TRAINING_FLOW.md).

Recorded validation results: [REFACTOR_VALIDATION.md](AAAMigration/REFACTOR_VALIDATION.md).
