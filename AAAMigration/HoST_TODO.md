# HoST → MJLab execution checklist

Branch: `HoST/THUBO-mjlab-RL-recovery`.

- [x] Inspect target baseline `20008ee2e742397fea3563eed1cc39f80a5daa9f`.
- [x] Inspect HoST source `70bb580949a336a920833700e4b5dc3bf7fe87ce`.
- [x] Create isolated branch; preserve existing velocity/getup tasks.
- [x] Install Miniforge and create Conda environment `THUBO_mjlab`.
- [ ] Install CUDA-enabled PyTorch, MJLab, MuJoCo Warp and viewer dependencies.
- [ ] Convert upstream `pi_12dof_release_v1.urdf` into floating-base MJCF and verify FK/inertia/joint limits.
- [ ] Implement `src/tasks/HoST_getUp`: robot adapters, reset poses, observation history, grouped rewards, curriculum.
- [ ] Implement grouped-value PPO and atomic `.pt` saves; map upstream mechanisms and document intentional differences.
- [ ] Validate both robots' models and run finite/reset/step/PPO smoke tests on CUDA.
- [ ] Start THUBO recovery training on the detected RTX 5070 GPU.
- [ ] Demonstrate Viser hotload of two successive checkpoints.
- [ ] Evaluate unassisted recovery from supine/prone/side starts; report success separately from smoke completion.
- [ ] Commit reviewed source; push branch and open PR if authenticated access allows it.

Never label an initial `.pt` or finite rollout as a successful recovery policy.
Do not copy Pi torque ratings, scale-dependent forces or body names onto THUBO.
