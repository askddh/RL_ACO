# ONoC Optimization Core

面向光网络片上网络（Optical Network-on-Chip，ONoC）的优化核心模块，提供蚁群优化（ACO）、PPO 强化学习、微环谐振器建模、光网络仿真和结果绘图功能。算法以信号透射率、链路损耗、弯曲损耗和串扰为基础，并使用 OSNR（Optical Signal-to-Noise Ratio）评价候选方案。

> 本目录是项目的可复用核心包，不包含完整实验入口、训练数据和结果文件。通常应从上级 `ONoC_Optimization` 目录运行项目脚本，以保证模型等相对路径正确。

## 功能

- **ACO 映射优化**：搜索核（core）到光网络节点（ONI）的映射，支持信息素/启发式概率选择、传统与精英信息素更新，以及可选的 XGBoost OSNR 预测重试。
- **波长分配**：支持目的节点、源节点、通信顺序、逆通信顺序，以及基于路径重叠和信息素的分配策略。
- **光学建模**：计算微环谐振器 through/drop 端透射率、路径损耗、弯曲损耗、信号功率、串扰和 OSNR。
- **PPO 强化学习**：提供 Gymnasium 环境、PyTorch Actor-Critic 网络、PPO 智能体和动作掩码。
- **可视化**：绘制映射结果、收敛曲线和概率变化。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `__init__.py` | 导出核心类和函数 |
| `mr.py` | 微环谐振器模型 `MR` |
| `opticalnetwork.py` | ONoC 光网络环境 `OpticalNetworkEnv` |
| `ant_colony_optimize.py` | ACO 映射、波长分配、OSNR 评估和信息素更新 |
| `onoc_ppo.py` | `RLOpticalNetworkEnv`、`ActorCritic`、`PPOAgent` 和动作掩码 |
| `plot.py` | 实验数据容器及绘图工具 |

## 安装依赖

建议使用 Python 3.9+：

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
pip install numpy matplotlib joblib gymnasium torch
```

使用 `apply_mapping_ml_predict` 时，还需要安装与保存模型匹配的 XGBoost 版本。代码默认尝试加载：

```text
RL_ACO_compare/ONoC_Optimization/ml/model_xgb_mapping.joblib
```

## 基本用法

从项目根目录导入核心接口：

```python
from ONoC_Optimization.core import MR, AntColonySystem, OpticalNetworkEnv

env = OpticalNetworkEnv(C, communication)
aco = AntColonySystem(
    com_wavelengths=wavelengths,
    ONI=oni_nodes,
    num_ants=15,
    iterations=2000,
    alpha=2.0,
    beta=2.0,
    rho=0.5,
)

mapping = aco.apply_mapping(C, W, dis_matrix, pheromone)
M = aco.get_m(mapping, C)
A = aco.get_a(M, W)
```

`C`、`W`、`dis_matrix`、`pheromone`、波长列表以及 `MR` 参数依赖具体实验拓扑，请参考上级目录中的实验脚本传入完整配置。

## ACO 流程

1. 构造拓扑、通信矩阵和微环模型。
2. 初始化映射信息素矩阵。
3. 调用 `apply_mapping` 或 `apply_mapping_ml_predict` 生成候选映射。
4. 使用 `get_m`、`get_a` 和波长分配函数生成通信方案。
5. 计算链路透射率、信号功率、串扰和平均 OSNR。
6. 调用信息素更新函数并迭代。
7. 使用 `plot.py` 绘制收敛曲线并保存结果。

## PPO 流程

```python
from ONoC_Optimization.core.onoc_ppo import (
    RLOpticalNetworkEnv, PPOAgent, construct_action_mask
)

env = RLOpticalNetworkEnv(C, communication)
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
)
state, info = env.reset()
action_mask = construct_action_mask(env)
# 根据训练脚本选择动作、执行 env.step(action)，并更新 agent
```

## 参数与注意事项

- `alpha` 控制信息素重要性，`beta` 控制启发式信息重要性，`rho` 控制挥发率。
- `num_ants` 和 `iterations` 会显著影响运行时间，建议先用较小值验证流程。
- `calc_type` 支持 `pip`、`mwd`、`mp3dec` 和 `mp3enc`，用于不同业务拓扑的弯曲次数计算。
- 建议从项目根目录运行，避免模型等相对路径失效。
- 为复现实验，请同时固定 Python、NumPy 和 PyTorch 随机种子。
- GitHub 发布时建议排除 `.venv`、`__pycache__`、模型缓存和大体积结果文件，并补充 `requirements.txt` 或 `pyproject.toml`。

## 许可证

当前目录未包含明确许可证。公开发布前，请在仓库根目录添加合适的 `LICENSE`，并补充论文、数据集或上游代码引用信息。
