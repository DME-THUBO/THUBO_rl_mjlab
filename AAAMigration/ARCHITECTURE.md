# THUBO configuration ownership

```text
scripts/train.py or scripts/host/train.py
    -> src/tasks/*/config.py (registration only) / dme_thubo_env_cfg
    -> dme_thubo/env_cfg.py (fresh scene + simulation + task manager preset)
        -> dme_thubo/embodiments.py (no task registration imports)
            -> dme_thubo/robot.py -> mjcf/THUBO.xml + meshes/
        -> dme_thubo/mdp.py (all custom MDP terms and environment-owned state)
        -> upstream MJLab MDP terms
    -> ManagerBasedRlEnv -> MuJoCo Warp
    -> RSL-RL PPO OR src/tasks/HoST_getUp/ppo.py
```

`dme_thubo_env_cfg` validates task/robot/terrain/action combinations, creates common
simulation and scene configuration, installs the selected task's managers, applies
train/play settings and returns a new instance. It never imports `src.tasks`.
Compatibility functions only delegate to this factory. Actor and critic term
objects and separate calls must not share mutable configuration.
Environment variables are parsed in executable entry points, not during registration.

Presets: standing, velocity, velocity_getup, host_getup. Only flat terrain is enabled.
Pi is an optional local HoST embodiment; THUBO works without its generated models.
`embodiments.py` owns adapter logic; task `robots.py` is a compatibility re-export.
The factory never constructs a standing baseline and then overrides it for velocity or HoST.

Standard actions retain default-offset targets (0.35); HoST defaults to relative
targets, six frame-major observation frames, fallen resets and independent curricula.
The combined task keeps its different fallen sampling and phase-gated rewards.
Do not merge these reset functions just because both place the robot on the floor.
HoST state is attached to each environment and provided through `host_state`.

The dedicated HoST runner keeps four critic values, per-group GAE and normalized
advantages. Its checkpoint is not a standard RSL-RL file. Missing `action_mode`
metadata means the legacy absolute controller. No checkpoint conversion is implied.

Sources: [Unitree RL MJLab](https://github.com/unitreerobotics/unitree_rl_mjlab)
for velocity organization; [HoST](https://github.com/InternRobotics/HoST),
`70bb580949a336a920833700e4b5dc3bf7fe87ce`, for recovery. Preserve
`src/tasks/HoST_getUp/LICENSE_HoST` and Pi source metadata.

Validation against e6331e0: frozen task configuration semantics, model checks,
mutable-state isolation, missing-Pi import checks, reset/step and combined phase
transitions, short training and checkpoint replay, grouped GAE and native saves.
See README commands. Logs and checkpoints are not committed. Short runs are not
recovery success measurements; use zero-assistance sustained-standing evaluation.
Rough terrain height/obstacle semantics remain a separate issue.
