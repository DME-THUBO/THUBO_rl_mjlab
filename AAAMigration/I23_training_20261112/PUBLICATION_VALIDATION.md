# Publication validation — 2026-11-13

Validated in the isolated publication checkout with the existing THUBO_mjlab CUDA environment. This checks execution and checkpoint compatibility, not walking quality or hardware readiness.

- `python scripts/validate_model.py`: base model passes; the I23 model is additionally validated by `check_i23.py`.
- `python scripts/refactor_checks.py`: 10 available baseline configurations, mutable isolation, registry and invalid combinations pass.
- `python scripts/refactor_asset_checks.py`: pure factory import, optional Pi absence, explicit-selection errors and ignore contracts pass.
- `python scripts/host/checks.py`: four-group PPO timeout/bootstrap and atomic checkpoint contracts pass.
- `python scripts/refactor_runtime_checks.py`: CUDA reset/finite steps for enabled legacy tasks, short standing/velocity PPO and native checkpoint replay pass.
- `python scripts/check_i23.py --checkpoint <model_3999.pt>`: I23 mass/geometry/contact masks, 100 finite CUDA rollout steps, two short PPO updates, then 16 finite replay steps from the uploaded final checkpoint pass. Actor observations 47, actions 12.
- `python scripts/host/train.py --robot thubo --smoke ...` and the documented `scripts/host/eval.py` short replay pass. This is not sustained unassisted recovery acceptance.
- Both viewer command-line interfaces accept `--help`.
- A separate CUDA Viser process loads the exact `model_3999.pt` release asset and serves HTTP 200. Its hotload receipt is written inside its own run directory. The user's existing port 8082 viewer is not restarted or changed.

The final checkpoint has native iteration 3999, 52 stored tensors, and all tensors are finite. SHA-256 is recorded in [training_status.json](training_status.json). The I23 MJCF checksum matches the settings recorded for the original run.

[Machine-readable results](validation.json)

The source snapshot and both historical run directories are preserved in Release assets, with generated logs/checkpoints kept out of this change's Git commit. Personal computer paths in the historical text records are sanitized. The original local training directory is unchanged.
