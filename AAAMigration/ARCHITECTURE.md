# DME-THUBO architecture

This repository follows the ManagerBased RL organization used by the upstream
[Unitree RL MJLab project](https://github.com/unitreerobotics/unitree_rl_mjlab).
The upstream repository is the layout reference; its Unitree robot assets and
deployment code are not copied into this project.

The active scope is simulation training with MJLab, MuJoCo Warp, and PPO/RSL-RL.
The source model remains the checked-in THUBO MJCF and STL set.

## Runtime flow

```text
scripts/train.py
    |
    +--> src.tasks                 # import task registrations
    +--> mjlab.tasks.registry      # task id -> env/agent config
    +--> ManagerBasedRlEnv         # scene + managers
             |
             +--> src.assets.robots.thubo
             |       +--> MJCF and STL meshes
             |       +--> STS3250 position actuator parameters
             |
             +--> observation manager: joint state, IMU-like base state, command
             +--> action manager: 12 normalized joint position targets
             +--> reward manager: posture/upright/height or velocity tracking
             +--> termination manager: fall, low root, timeout
             +--> event manager: deterministic reset and future randomization
             |
             +--> MuJoCo / MuJoCo Warp
             +--> RSL-RL PPO
```

## Repository layout

```text
DME-THUBO/
├── mjcf/                         # source MuJoCo XML models
├── meshes/                       # source STL meshes referenced by MJCF
├── sts3250_official.json         # actuator metadata
├── src/
│   ├── assets/robots/thubo/      # ManagerBased robot entity adapter
│   └── tasks/
│       ├── standing/             # THUBO-Flat-Stand, THUBO-Rough-Stand
│       └── velocity/             # THUBO-Flat-Velocity, THUBO-Rough-Velocity
├── dme_thubo/                # shared model validation and task primitives
├── scripts/train.py              # PPO entry point
├── scripts/validate_model.py     # model-property gate
└── AAAMigration/
    ├── ARCHITECTURE.md
    └── TODO.md
```

`dme_thubo` owns reusable implementation details. `src/assets` and
`src/tasks` provide the upstream-compatible package boundary and task registry
layout. A task configuration should compose the shared standing baseline and
change only its terrain, commands, observations, rewards, or curriculum.

## Task progression

### Standing first

`THUBO-Flat-Stand` is the deterministic acceptance task. It validates the
12-joint position action path, reset behavior, observation shapes, root height,
upright reward, and fall termination on a plane.

`THUBO-Rough-Stand` uses MJLab's procedural rough terrain generator with the
same action and standing objective. It is a robustness task after flat standing
is stable.

### Walking after standing

`THUBO-Flat-Velocity` and `THUBO-Rough-Velocity` add a
`UniformVelocityCommandCfg` named `twist`, a 3-element command observation, and
linear/angular velocity tracking rewards. They are registered now so their
configuration is reviewable, but training should begin only after the standing
acceptance criteria in `TODO.md` pass.

## Configuration ownership

- Robot XML, meshes, inertial properties, and joint limits belong under `mjcf/`
  and `meshes/`.
- Actuator construction belongs in the THUBO robot constants adapter.
- Scene, observation, action, reward, termination, and event managers belong in
  task environment configs.
- PPO network and algorithm settings belong in the task RL config.
- Model-property validation runs before the robot entity is built and is also
  available as a standalone command.

Hardware drivers, deployment, ONNX export, and sim-to-real are outside this
architecture milestone.
