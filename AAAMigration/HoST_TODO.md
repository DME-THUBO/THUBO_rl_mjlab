# HoST → MJLab execution checklist

Branch: `HoST/THUBO-mjlab-RL-recovery`.

- [x] Inspect target baseline `20008ee2e742397fea3563eed1cc39f80a5daa9f`.
- [x] Inspect HoST source `70bb580949a336a920833700e4b5dc3bf7fe87ce`.
- [x] Create isolated branch; preserve existing velocity/getup tasks.
- [x] Install Miniforge and create Conda environment `THUBO_mjlab`.
- [x] Install CUDA PyTorch 2.11.0+cu128 / Torchvision 0.26.0+cu128, MJLab 1.6.0, MuJoCo/Warp 3.11.0.
- [x] Convert `pi_12dof_release_v1.urdf` into floating-base MJCF; 32 random-pose FK comparisons match before XML rounding.
- [x] Implement `src/tasks/HoST_getUp`: robot adapters, fallen resets, six-frame observation history, grouped rewards, curriculum.
- [x] Implement four-value PPO and atomic `.pt` saves; document differences from the upstream implementation.
- [x] Both robots pass CUDA rollout/PPO smoke tests: 4 worlds, 2 updates, 64 transitions each, checkpoints saved.
- [x] Repeat both CUDA smoke tests after correcting the controller to upstream current-position-relative semantics.
- [x] Grouped timeout GAE and checkpoint round-trip checks pass.
- [x] Start THUBO recovery training: 256 worlds, 2,000 planned updates on RTX 5070 Ti Laptop GPU (12GB).
- [x] Start a separate corrected relative-action run; action convention is recorded in checkpoint metadata.
- [x] Viser HTTP 200 at localhost:8080; actual hotloads of model_450.pt, model_460.pt, model_470.pt observed.
- [x] Evaluate the initial absolute-action model_1200: 0/24 held supine recoveries without assistance. Record this failure.
- [ ] Evaluate corrected relative-action checkpoints; prone/side success remains unverified.
- [x] Commit source on the requested branch with the supplied author email.
- [ ] Push branch/open PR: requires GitHub device-login approval; no stored credentials were present.

## Follow-up validation, not claimed complete

- [ ] Finish the planned training run; a running process is not convergence evidence.
- [ ] Compare every upstream optional style term, domain randomization and action-target convention.
- [ ] Quantify held, unassisted recovery success separately from curriculum-assisted progress.

Never label an initial `.pt` or finite rollout as a successful recovery policy.
Do not copy Pi torque ratings, scale-dependent forces or body names onto THUBO.
