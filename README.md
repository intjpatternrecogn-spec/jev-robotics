# JEV Architecture for Robotics

> A clean-room, executable **JEV-inspired decision layer for robotics**: shared world state is encoded once, a task question produces a decision embedding, and a dynamic set of independently encoded robot actions is scored in parallel.

[中文说明](#中文说明) · [Architecture](docs/ARCHITECTURE.md) · [Safety](SECURITY.md)

![JEV-inspired robotics architecture](docs/assets/jev-robotics-architecture.svg)

## Overview

Robots often have a motion planner and a controller but still need a fast layer that selects the next bounded action. This repository implements that layer using a testable candidate-embedding design:

```text
world state ──► shared state memory Hx
task question + Hx ──► decision embedding hqx
candidate skills ──► independent embeddings o1 ... oM
zi = hqxᵀoi + cost_prior_i
constraint mask + softmax ──► typed action probabilities
```

This is an independent architectural hypothesis inspired by JEV's public typed-decision interface. TypeSafe AI has not published JEV's internal layers, weights or parameter count; this project is not official JEV code and does not claim otherwise.

The diagram separates the reusable JEV-style candidate-ranking core from our robotics contribution. Shared world state and the task question produce one decision embedding; the skill list remains dynamic. Physical constraints and transparent task/cost priors operate before the bounded output, while a confidence-and-entropy gate can request an operator.

## Robotics-specific improvements

1. **Constraint-first planning** — reachability, payload, human-separation and collision checks mask invalid skills before probability normalization.
2. **Cost-aware ranking** — motion time and energy estimates modify the semantic logits without hiding the reason.
3. **Dynamic skill sets** — tools and capabilities can be added at runtime; there is no fixed classifier output dimension.
4. **Uncertainty-aware fallback** — high entropy or insufficient top probability yields `request_operator`, never an invented action.
5. **Decision trace** — every candidate includes feasibility reasons, probability, estimated time and energy.

The scorer proposes among bounded choices; deterministic safety systems retain authority.

The included offline model has not been empirically calibrated. Its softmax values demonstrate the decision interface and uncertainty gates; they are not validated real-world probabilities.

## Quick start

Python 3.10+ is sufficient; no package download is required.

```bash
python -m jev_robotics.cli examples/bin_picking.json
python -m unittest discover -s tests -v
```

## Python API

```python
from jev_robotics import RobotDecisionEngine, RobotState, Skill

state = RobotState(
    end_effector_xyz=(0.2, 0.0, 0.4),
    target_xyz=(0.45, 0.1, 0.25),
    reach_radius_m=0.9,
    payload_kg=0.4,
    payload_limit_kg=2.0,
    human_distance_m=1.8,
    collision_detected=False,
    gripper_has_object=False,
)

skills = [
    Skill("move_to_target", "move the gripper toward the target", 1.4, 8.0),
    Skill("grasp", "close the gripper around the target", 0.4, 2.0),
    Skill("hold", "hold position and wait", 0.1, 0.2),
]

result = RobotDecisionEngine().decide(
    state,
    "Which bounded skill should run next to pick the object safely?",
    skills,
)
print(result.choice, result.probabilities)
```

## Clean-room boundary

All implementation code and synthetic examples were created specifically for this public repository. It contains no employer code, robot logs, production configuration, credentials, private datasets or proprietary model weights.

## 中文说明

这是一个从零实现、可直接运行的 **JEV 风格机器人决策层**。世界状态只编码一次；任务问题与状态交互生成决策向量；任意数量的机器人技能独立编码后并行打分。机器人版本增加了可达性、载荷、人机距离与碰撞硬约束，并使用时间/能耗先验重排；置信度不足时请求操作员介入。

图中蓝色模块表示共享世界状态，紫色模块把任务问题与状态记忆融合为决策向量，橙色模块独立编码动态技能集合。红色物理可行性门控和黄色任务/代价先验是机器人方向的改进：不可达、超载、碰撞风险或人机距离不足的技能会被直接屏蔽，剩余技能才进行并行打分；分布过平或最高概率不足时返回 `request_operator`。

JEV 的底层网络结构尚未公开，因此本项目是独立架构提案，不是 TypeSafe AI 的官方实现，也不包含任何公司代码或内部资料。

## Responsible use

This software is for simulation and research only. It is not a certified safety component and must not directly command physical hardware. See [SECURITY.md](SECURITY.md).

## References

- [TypeSafe AI — System One Models and JEV](https://typesafe.ai/)
- [Introducing System One Models & JEV](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

These sources describe JEV's public behavior and training goals, not the undisclosed internal layer design implemented as a hypothesis here.

## License

MIT. See [LICENSE](LICENSE).
