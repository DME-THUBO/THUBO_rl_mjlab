# train.py / play.py 的 IMU 与动作接口记录

## 1. 训练和回放中的 IMU 相关观测

`scripts/train.py` 本身只负责加载任务配置并启动 PPO，不直接访问 IMU。`scripts/play.py` 也只是加载 checkpoint、创建回放环境并调用相同的任务观测配置。

当前任务配置位于 [`dme_thubo/env_cfg.py`](../dme_thubo/env_cfg.py)：

- `base_ang_vel`：3 维基座坐标系角速度，读取 MJLab 的 `root_link_ang_vel_b`。
- `projected_gravity`：3 维重力方向在基座坐标系的投影，读取 `projected_gravity_b`，用于表达机身姿态。

两项同时提供给 actor 和 critic。完整 actor 观测为：

```text
joint_pos_rel (12)
+ joint_vel_rel (12)
+ base_ang_vel (3)
+ projected_gravity (3)
+ last_action (12)
= 42 dimensions
```

MJCF 中虽然声明了 `imu_site_quat`、`imu_gyro` 和 `imu_acc` 传感器，但当前观测函数没有直接读取这些 sensor。仿真中的观测来自机器人状态缓存；因此这里的 IMU 相关观测是“IMU 等价量”，不是硬件消息接口。

## 2. 真机部署时的数据来源

真机部署需要在策略推理前增加观测适配层：

- `base_ang_vel` 通常由 IMU 陀螺仪提供的三轴角速度得到。
- `projected_gravity` 通常由 IMU 的陀螺仪和加速度计经过姿态估计或传感器融合得到机身姿态，再计算重力方向在机身坐标系的投影。它不是直接把加速度计原始值作为该观测。
- `joint_pos_rel` 和 `joint_vel_rel` 由关节舵机反馈提供，并按训练时的关节顺序、单位和默认姿态计算相对量。
- `last_action` 是上一策略周期的 12 维动作，部署端需要自行保存。

部署端必须保持与训练一致的坐标系、符号、单位、关节顺序、观测顺序和归一化方式。当前仓库没有实现 IMU 驱动、姿态融合、真机通信或策略部署接口；这些属于后续硬件部署工作。

## 3. play.py 的动作接口

任务配置使用：

```python
JointPositionActionCfg(
    scale=0.35,
    use_default_offset=True,
)
```

策略输出的归一化动作先乘以 `0.35 rad`，再以默认关节角为 offset，形成 12 个关节位置目标。机器人配置使用 MJLab 的 `BuiltinPositionActuatorCfg`，包括 stiffness、damping 和 effort limit。因此 `play.py` 回放的是位置目标闭环，不是策略直接输出力矩。

这与 STS3250 的位置闭环控制思路一致，但当前 `play.py` 只控制 MuJoCo 仿真，不向真实 STS3250 发送指令。MJCF 中原有的 standalone `<motor>` 定义在 [`dme_thubo/robot.py`](../dme_thubo/robot.py) 加载时会被删除，再由 MJLab 添加位置执行器。
