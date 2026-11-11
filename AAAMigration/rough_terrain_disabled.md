# THUBO-Rough-Velocity 停用说明：当前粗糙地形设置不合理

## 结论先行

当前 rough 地形设置**不合理，暂不用于训练**。`THUBO-Rough-Velocity` 的任务注册已在
[`src/tasks/velocity/config.py`](../src/tasks/velocity/config.py) 中注释停用，`THUBO-Flat-Velocity`
不受影响。核心问题：

1. **地形峰值 0.30 m 过高**，而策略没有任何外感受观测（盲走），无法预判障碍，0.30 m 箱体接近随机绊摔。
2. **多个奖励/终止项使用绝对世界高度**，在箱顶等抬升地面上语义错误（净空奖励失效甚至反向引导、摔倒不触发终止）。
3. **没有"被绊倒"相关惩罚**（非足端碰撞、摆动相接触力），拖脚蹭箱只会零奖励而无代价。

恢复注册前请按文末清单逐项修复并验证。本文记录判断依据与修复方向。

## 1. 当前地形配置事实

配置位置：[`dme_thubo/env_cfg.py`](../dme_thubo/env_cfg.py)（`terrain="rough"` 分支）。
子地形为 `box_random_grid` 与 `random_spread_boxes`，课程 `num_rows=3`、
`difficulty_range=(0.0, 1.0)`，行号到难度映射 `t = row / (num_rows - 1)`，即
L0=0.0、L1=0.5、L2=1.0；`max_init_terrain_level=0`，从最易行起步。

高度规则来自 `mjlab/terrains/config.py` 预设与 `mjlab/terrains/primitive_terrains.py`：

| 子地形 | 高度规则 | L0 (d=0.0) | L1 (d=0.5) | L2 (d=1.0) |
| --- | --- | --- | --- | --- |
| `box_random_grid` | 格子高采样自 `[-h, +h]`，`h = 0.3 * d` | 0（全平） | ±0.15 m | **+0.30 m / −0.30 m** |
| `random_spread_boxes` | 箱高 `[0.05, 0.3] * (0.2 + 0.8d)` | 0.01–0.06 m | 0.03–0.18 m | **0.05–0.30 m** |

即满难度行的箱顶/格子凸起峰值为 **+0.30 m**（`box_random_grid` 另有等深 0.30 m 凹坑，
总起伏 0.60 m），约等于 THUBO 站立腿长尺度，对无感知策略过大。

## 2. 不合理的具体原因

### 2.1 无感知（盲走）与障碍尺寸不匹配

actor/critic 观测只有本体感觉与命令：`joint_pos`、`joint_vel`、`base_ang_vel`、
`projected_gravity`、`last_action`、`gait_phase`、`command`
（[`dme_thubo/env_cfg.py`](../dme_thubo/env_cfg.py) 与
[`src/tasks/velocity/config.py`](../src/tasks/velocity/config.py)）。
没有 height scan / raycast / camera，策略无法预判前方 0.30 m 箱体，
只能学到被动救摔，学不出"抬腿跨过"。无感知时障碍高度应控制在 5–10 cm 量级；
若要保留 0.30 m，需增加局部高度扫描观测。

### 2.2 绝对世界高度项在抬升地面上失效

| 项 | 位置 | 问题 |
| --- | --- | --- |
| `feet_clearance_swing` | [`dme_thubo/mdp.py`](../dme_thubo/mdp.py)（`foot_clearance` 取 `geom_pos_w[..., 2]`） | 摆动净空目标 0.04–0.10 m 是绝对 z；站上 0.25 m 箱顶后目标低于脚下地面，奖励恒为 0，甚至引导脚向下踩进箱子 |
| `root_height_reward` | 同上（目标 0.39 m） | 在箱顶自然根高为 0.39+箱高，奖励趋 0 并把 root 往下压 |
| `low_root` 终止 | [`dme_thubo/env_cfg.py`](../dme_thubo/env_cfg.py)（阈值 0.20 m） | 在箱顶摔倒 root 仍高于 0.20，不触发终止，倒地后继续无效回合 |
| `feet_air_time` | [`dme_thubo/mdp.py`](../dme_thubo/mdp.py)（`contact_height=0.065`） | 离地判定用绝对高度，在抬升地面附近失真（次要） |

这些项在 flat 地形语义正确，是 rough 地形复用 flat 奖励时未做地形相对化的遗留问题。

### 2.3 缺少防绊倒的惩罚项

现有 `feet_slide`、摆动相 `contact_penalty`（二值 `found`）不足以约束蹭箱、踢箱边。
非足端（小腿/膝盖）与地形的碰撞完全没有惩罚。关于"穿模体积惩罚
`volume_penetration()`"：**不建议实现**——MuJoCo 为软接触求解，穿模是瞬态且被求解器
允许的，没有原生穿模体积计算，mesh-terrain 求交在 2048 env GPU 仿真中不现实。
`ContactSensor` 的 `force` / `dist`（穿透深度）字段是充分且便宜的代理。

## 3. 修复方向（恢复注册前的建议）

1. **摆动净空改为地形相对**：从足端向下 raycast（mjlab 提供 `RayCastSensor`）取足下
   地形高 `z_g`，净空 = `foot_z - z_g`，目标 = `z_g + 0.10 * sin(摆动进度)`
   （下限 `z_g + min_swing_height`）。奖励侧使用地形高度属特权信息，不影响观测端盲走设定。
2. **非足端碰撞惩罚**：`ContactSensor` primary 取小腿/膝盖等非足底 geom，secondary 为
   terrain body，按 `force` 求和惩罚。
3. **摆动相接触惩罚改计量**：`shuffling` 惩罚从二值 `found` 升级为 `force`/`dist` 加权。
4. **高度项相对化**：`root_height_reward` 与 `low_root` 阈值减去 root 足下地形高。
5. **地形限高或加感知**：无感知路线把满难度峰值压到 5–10 cm；保留 0.30 m 则加局部
   height scan 观测。二选一后再启用课程升级。

已完成的相关修复（重新启用时保留）：接触传感器 `secondary=ContactMatch` 从
`mode="geom", pattern="terrain"` 改为 `mode="body", pattern="terrain"`——generator
地形只有 body 名为 `terrain`，geom 名为 `terrain_0..N`，按 geom 字面名匹配会在
`spec.compile()` 报 `unrecognized name 'terrain' of object`。

## 4. 恢复 THUBO-Rough-Velocity 注册的条件

- [ ] 峰值高度与感知方案定案（盲走限高，或加 height scan）
- [ ] 摆动净空、root 高度、`low_root` 终止完成地形相对化
- [ ] 增加非足端碰撞惩罚与摆动相接触力惩罚
- [ ] 短程 smoke run + checkpoint 回放验证（低难度行起步，课程可正常升级）
- [ ] 恢复 `src/tasks/velocity/config.py` 底部注释掉的 `register_mjlab_task` 块

## 参考文件

- 地形配置：[`dme_thubo/env_cfg.py`](../dme_thubo/env_cfg.py)
- 奖励实现：[`dme_thubo/mdp.py`](../dme_thubo/mdp.py)
- 任务注册（已注释）：[`src/tasks/velocity/config.py`](../src/tasks/velocity/config.py)
- 速度命令可行性核算：[`AAAMigration/velocity_limits.md`](velocity_limits.md)
