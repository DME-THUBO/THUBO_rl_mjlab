# Repository Guidelines

## Project Structure & Module Organization

`dme_thubo/` contains the task package: robot and environment definitions, RL configuration, and model validation. `src/tasks/` registers standing and velocity tasks; `src/assets/robots/thubo/` holds robot constants. The MuJoCo model is `mjcf/THUBO.xml`, with geometry in `meshes/`. Runnable entry points live in `scripts/` (`train.py`, `play.py`, and `validate_model.py`). Architecture notes and active scope are in `AAAMigration/`. Training output is written under `logs/`; do not commit generated runs or local environment files.

## Build, Test, and Development Commands

Install the editable package in the project environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

Run `.venv/bin/python scripts/validate_model.py` to check the MJCF, joints, actuators, meshes, and model metadata before simulation. Start training with `.venv/bin/python scripts/train.py THUBO-Flat-Stand`; use `DME_THUBO_NUM_ENVS`, `DME_THUBO_NUM_STEPS`, and `DME_THUBO_MAX_ITERATIONS` for a short smoke run. Replay a checkpoint with `.venv/bin/python scripts/play.py THUBO-Flat-Stand <checkpoint.pt>`. See `README.md` for Viser and TensorBoard commands.

## Coding Style & Naming Conventions

Use four spaces for Python indentation. Follow existing Python conventions: `snake_case` for functions, variables, and modules; `PascalCase` for classes; and descriptive task IDs such as `THUBO-Flat-Stand`. Put every custom MDP term and runtime state in `dme_thubo/mdp.py`. Build all environments through `dme_thubo/env_cfg.py:dme_thubo_env_cfg`; task directories must not copy common environment configuration or construct a standing baseline and override it. The common factory must not import registration modules. No formatter or linter is configured in `pyproject.toml`; keep imports tidy and match nearby code.

## Testing Guidelines

There is no configured pytest suite or coverage threshold. Run `scripts/validate_model.py` after changing the MJCF, meshes, robot constants, or actuator metadata. For task or training changes, run a brief training smoke test with reduced environment count and iterations, then replay a produced checkpoint when applicable. Keep generated checkpoints and logs out of commits.

## Commit & Pull Request Guidelines

The repository has no commit history from which to infer a message convention. Use concise imperative commit subjects, for example `Add rough-terrain standing task`. Pull requests should describe behavior and validation performed, link related issues when available, and include viewer screenshots for visible simulation changes. Call out changes to model assets, task registration, or configuration defaults.

## Configuration & Scope

This milestone targets simulation training for the 12-DoF THUBO robot. Hardware drivers and deployment are outside the current scope. Keep credentials and machine-specific paths out of tracked files, and review `AAAMigration/TODO.md` before broadening project scope.

## Shared tasks and HoST validation

`src/tasks/standing`, `velocity`, `velocity_getup` and `HoST_getUp` contain thin registration/compatibility wrappers. HoST additionally owns its dedicated four-value `ppo.py`, license and a `robots.py` compatibility re-export. Pure embodiment adapters belong in `dme_thubo/embodiments.py`.

Run model validation, `scripts/refactor_checks.py`, `scripts/refactor_asset_checks.py`, `scripts/refactor_runtime_checks.py`, `scripts/host/checks.py`, then HoST smoke training and checkpoint replay as documented in README. Preserve six-frame observation order, relative control, independent curricula, four-group rewards/GAE and old checkpoint missing-action-mode absolute behavior. Verify train/play and actor/critic mutable isolation. Short training is not proof of unassisted recovery; use held-standing evaluation separately.

Pi XML and URDF are ignored local optional assets: do not force-add them. Preserve the converter, source metadata and licenses. Missing Pi must not break THUBO registration; explicit selection must explain generation. Rough velocity remains disabled; rough stand has no registration. Attribute velocity to unitree_rl_mjlab and recovery to InternRobotics/HoST with its immutable source commit and LICENSE_HoST. Never invent measured servo parameters or successful recovery results.
