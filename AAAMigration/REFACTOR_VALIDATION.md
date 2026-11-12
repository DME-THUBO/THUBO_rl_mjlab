# Shared-factory refactor validation

Baseline: `e6331e03432f1f6dbed137b9e3d0080ebb1b3d55`.
Validated on 2026-11-12 in `THUBO_mjlab`, Windows, RTX 5070 Ti Laptop GPU (12 GB).
Runtime: MJLab 1.6.0, MuJoCo 3.11.0, PyTorch 2.11.0+cu128.

| Check | Result |
| --- | --- |
| `scripts/validate_model.py` | Pass: 12 joints/actuators, 13 meshes, source mass and actuator metadata checks |
| `scripts/refactor_checks.py` | Pass: all 14 baseline configurations, preserving ordered terms and parameters; fresh calls, train/play and actor/critic isolation; invalid combinations rejected |
| `scripts/refactor_asset_checks.py` | Pass: common factory imports without task registration; missing Pi skips only Pi registration; explicit Pi request explains generation; both files ignored/untracked and local copies restored |
| `scripts/refactor_runtime_checks.py` | Pass: CUDA create/reset and eight finite steps for all five locally available tasks; observation dimensions 42/47/48/258/258 |
| Combined velocity/getup | Pass: low/high height transitions and hysteresis in an actual environment; partial reset |
| Standing and velocity training | Pass: each ran two PPO updates, four environments, eight rollout steps; native RSL-RL checkpoint reloaded and replayed for eight steps |
| `scripts/host/checks.py` | Pass: per-group timeout GAE/terminal handling, native checkpoint roundtrip, four critic outputs; missing-action-mode absolute fallback and wrong-format rejection |
| HoST THUBO smoke | Pass: two updates, four environments, eight rollout steps; native `model_1.pt` saved |
| HoST THUBO replay | Pass: saved native checkpoint loaded and replayed for eight steps with zero assistance; all observations finite |

HoST smoke command:

```bash
python scripts/host/train.py --robot thubo --smoke --run-dir logs/refactor_host_smoke
python scripts/host/eval.py logs/refactor_host_smoke/model_1.pt --poses supine --num-envs 4 --seconds 0.16 --output logs/refactor_host_replay.json
```

The 0.16-second replay checks executable compatibility only. It is shorter than
the evaluator's 0.5-second hold criterion and cannot establish recovery success.
This smoke policy achieved no held recoveries; no trained-performance claim is made.
Full acceptance still requires zero-assistance sustained standing, separately for
each pose and action scale. Rough terrain and hardware validation are outside this refactor.

Runtime reports, console logs and checkpoints remain under ignored `logs/` and
are not committed. The baseline fixture is a semantic configuration contract,
not a training output. Pi source metadata, conversion script and license stay tracked;
the XML and URDF are retained locally but removed from the latest Git tree.
