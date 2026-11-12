# HoST 从倒地到起身的训练迭代

本图对应 THUBO 默认的 `relative` 动作模式：2,048 个训练环境、4,000 次 PPO 迭代，每次迭代每个环境采集 32 个控制步。单个回合最长 10 秒；回合重置与 PPO 更新是两个独立循环，回合可以跨越多次 PPO 迭代。

```mermaid
flowchart TD
    START["初始化策略、四输出价值网络和 2,048 个训练环境"]
    INIT["每个环境独立初始化课程参数<br/>辅助比例 a = 0.60；动作比例 s = 1.00"]
    RESET["倒地重置：默认仰躺<br/>骨盆高度 0.12 m；随机朝向和关节扰动"]
    OBS["读取最近 6 帧本体观测<br/>角速度、重力方向、关节位置和速度、动作、动作比例"]
    ACT["策略采样动作<br/>控制量 = tanh(raw) × s<br/>相对关节位置控制"]
    GATE{"已超过约 0.6 秒<br/>且身体直立度大于 0.8？"}
    HELP["向骨盆施加向上力 F = a × m × g<br/>初始约抵消 60% 重力；质量与惯性不变"]
    NOHELP["本步向上辅助力为零"]
    SIM["推进物理仿真，得到新姿态"]
    REWARD["计算四组奖励<br/>task：转正并撑高<br/>regu：限制抖动、高速、用力和越限<br/>style：接近默认站姿<br/>target：达到高度后保持稳定"]
    SAVE["保存观测、动作、四组奖励和价值估计<br/>终止状态与下一状态价值用于 GAE"]
    DONE{"该环境回合结束？<br/>10 秒超时或关节位置非有限"}
    PASS{"回合末骨盆高度大于 0.3705 m<br/>且直立度大于 0.9？"}
    REDUCE["该环境下回合降低辅助与动作幅度<br/>a = max(0, a - 0.05)<br/>s = max(0.25, s - 0.02)"]
    KEEP["保留该环境的 a 和 s"]
    NEXTRESET["仅重置已结束的环境为倒地姿态"]
    ENOUGH{"本次迭代已采集 32 步？"}
    GAE["四组分别计算 GAE 与回报<br/>分别归一化优势<br/>按权重 2.5、0.1、1、1 合并优势"]
    PPO["PPO 更新策略和四输出价值网络<br/>默认 5 个训练轮次、4 个 minibatch"]
    LOG["记录 loss、四组奖励、平均 a 和 s<br/>到保存间隔或最后一次迭代时原子保存 checkpoint"]
    FINISH{"已完成 4,000 次迭代？"}
    END["结束训练；查看器继续运行"]

    START --> INIT --> RESET --> OBS --> ACT --> GATE
    GATE -->|是| HELP --> SIM
    GATE -->|否| NOHELP --> SIM
    SIM --> REWARD --> SAVE --> DONE
    DONE -->|否| ENOUGH
    DONE -->|是| PASS
    PASS -->|是| REDUCE --> NEXTRESET
    PASS -->|否| KEEP --> NEXTRESET
    NEXTRESET --> ENOUGH
    ENOUGH -->|否，继续收集| OBS
    ENOUGH -->|是| GAE --> PPO --> LOG --> FINISH
    FINISH -->|否，下一次迭代| OBS
    FINISH -->|是| END

    LOG -.->|新 checkpoint| WATCH
    subgraph VIEW["独立 Viser 进程：4 个展示环境"]
        WATCH["约每 2 秒检查并热加载最新 checkpoint"]
        VRESET["默认仰躺重置；使用 checkpoint 中的动作比例"]
        INFER["策略推理与物理仿真<br/>默认无辅助力；显式传 --assist 才开启辅助"]
        DISPLAY["显示起身和站稳过程<br/>不更新策略参数"]
        WATCH --> VRESET --> INFER --> DISPLAY
        DISPLAY -->|持续推理并检查热加载；回合结束时重置| INFER
    end
```

## 奖励怎样引导起身

四组奖励的标量返回值为 `2.5 × task + 0.1 × regu + style + target`。实际 HoST 训练器读取四组原始奖励，分别计算、归一化优势后再加权更新策略。

| 奖励组 | 当前实现与引导作用 |
| --- | --- |
| `task` | 身体直立评分乘以骨盆相对双踝参考点的高度评分。连续鼓励身体转正、撑高；THUBO 目标高度为 0.39 m。 |
| `regu` | 惩罚动作变化、关节速度、力矩、机械功率和关节越限，限制粗暴动作。 |
| `style` | 惩罚偏离默认关节姿态；骨盆高度超过 0.3705 m 后，还约束双脚横向间距接近 0.12 m。 |
| `target` | 骨盆高度超过 0.3705 m 后，奖励低横滚/俯仰角速度、低水平速度、身体直立和接近 0.39 m 高度。 |

策略没有预设“翻身—收腿—撑起”的动作序列，需通过探索和奖励反馈学习。向上辅助力只在身体接近直立后开启，主要降低后半段撑起的难度；每个训练环境按自身回合末表现独立减少辅助。

## 如何判断已经学会

课程通过条件只看回合末的高度和直立度，没有要求持续站稳。四环境查看器默认无辅助，能够展示无辅助起身，但它不能替代批量成功率评估。

正式验证使用 `scripts/host/eval.py`，分别检查仰躺、俯卧和左右侧躺；默认评估要求零辅助，在高度、直立度及速度均达标时持续站稳至少 0.5 秒。

## 实现位置

- [config.py](config.py)：奖励注册、10 秒回合和事件配置。
- [mdp.py](mdp.py)：倒地重置、四组奖励、辅助力及课程递减。
- [ppo.py](ppo.py)：四组 GAE、优势归一化与 PPO 更新。
- [训练入口](../../../scripts/host/train.py)：采样、手动回合重置、迭代与 checkpoint 保存。
- [训练与查看器入口](../../../scripts/train_with_viser.py)：四环境展示和 checkpoint 热加载。
- [恢复评估入口](../../../scripts/host/eval.py)：无辅助持续站稳评估。
