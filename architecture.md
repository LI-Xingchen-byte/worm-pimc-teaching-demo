# Worm PIMC 架构与源码阅读指南

本文介绍当前实现的模块职责、一次更新的流程和主要扩展入口。
使用方法见 [README](README.md) 和 [Python 使用指南](docs/python_workflow.md)；
物理权重、周期图像及观测量定义见 [derivations.md](derivations.md)。

## 从一次更新开始

入口在 `Simulation.step()`，它将单次尝试交给 `Sampler.step()`。
`Sampler._step()` 集中呈现以下顺序：

```text
选择 move
    ↓
生成候选 ProposalPatch
    ↓
检查适用性与候选构型
    ↓
计算目标权重变化、正反提议概率和 move 选择概率
    ↓
抽取接受随机数，进行 Metropolis–Hastings 判断
    ↓
接受：state.apply(patch)；拒绝：保留原构型
    ↓
返回 StepResult，更新尝试与接受计数
```

`Simulation.advance()` 重复上述更新，管理热化、measurement sweeps 和量测时机。
`run()` 在此基础上增加运行目录、结果文件和 checkpoint。
因此单步演示与正式运行共用同一套更新算法。

建议先读 [simulation.py](src/wormpimc/simulation.py) 的 `step/advance`，再读
[sampler.py](src/wormpimc/sampler.py) 的 `_step`，最后沿具体 move 进入 `moves/`。

## 模块地图

| 模块 | 职责 | 阅读时关注 |
|---|---|---|
| `config.py` | 参数、校验、TOML 与解析后的配置 | `small_system()` 和分组参数 |
| `types.py` | patch、状态与接受率分解的数据类型 | `ProposalPatch`、`AcceptanceBreakdown` |
| `geometry.py` | 周期坐标、显式链接图像与绕数 | 坐标折回和真实路径位移的区别 |
| `propagator.py` | 自由传播子、bridge 和 free walk | 采样与对应提议密度 |
| `potentials.py` | 对势、外势与势能组合 | 周期模型和参数校验 |
| `configuration.py` | bead/link 图、Z/G 扇区与构型检查 | `preview()`、`apply()`、拓扑不变式 |
| `worm_measure.py` | Worm 测度与更新比的纯参考计算 | 独立于 live 图修改的公式核对 |
| `measure.py` | primitive 目标权重及其变化 | 动能、势能、化学势和扇区项 |
| `moves/` | 生成候选改动、计算正反提议概率 | 更新逆对、端点和图像的选择 |
| `sampler.py` | 组合提议与目标，作接受判断 | `_step()` 与 `StepResult` |
| `simulation.py` | 初始化、运行阶段、量测与持久化 | 单步与 sweep 的区别 |
| `estimators/` | scalar、Green、关联量与密度量测 | 量测扇区、分母及结果接口 |
| `statistics.py` | ratio blocking 与协方差误差 | 相关样本、平台状态和稀疏数据 |
| `checkpoint.py` / `output.py` | 保存续算状态与可读结果 | checkpoint 与结果文件的区别 |
| `cli.py` / `__main__.py` | 命令行入口 | 配置检查、运行、恢复与摘要 |

核心依赖 NumPy；Matplotlib 只用于示例和验证绘图。
`examples/` 调用库并展示结果，不保存另一份采样算法。
`validation/` 提供独立参考与复现实验，不属于安装后包的公共 API。

## 构型、提议与目标权重

### Configuration

构型按 bead 储存位置、虚时间 slice、前后连接及 outgoing link 的显式周期图像。
Z 扇区由闭合有向环组成；G 扇区还包含具有 head/tail 的开放链。
巨正则模拟的粒子数可变化，初始化粒子数不是正则约束。

坐标处于基本周期盒内，但一条 link 的位移还取决于其 image。
计算作用量和画跨界路径时都保留该 image，不能用 minimum image 代替。
配置与构型的一致性、端点结构、链接互反和 slice 占据由构型校验检查。

### ProposalPatch

move 提出 `ProposalPatch`，表达 bead 位置、链接、增删或端点的候选修改。
`preview(patch)` 产生候选构型，不改动 live state；`apply(patch)` 在接受后提交。
revision 用来拒绝基于旧构型生成的 patch。

这一设计让目标权重能够比较更新前与候选状态，也让拒绝更新自然保留原构型。
拓扑修改集中到同一入口，避免某个 move 绕过共同的图一致性检查。

### Move、TargetMeasure 和 Sampler

move 决定如何选择路径与端点，并给出正反 proposal 密度。
`PrimitiveTargetMeasure` 计算目标权重，包括动能 link、primitive 势能作用量、
化学势以及 Worm 扇区/测度贡献；具体势公式集中在 `potentials.py`。
Sampler 将这些量与 move 选择概率组合成完整的 log acceptance ratio。

固定拓扑更新可使用局部 target delta；拓扑变化保留完整重算路径。
`verify_local_delta=True` 用完整重算核对局部差值，适合调试和小系统验证。
逆对为 Open/Close、Insert/Remove、Advance/Recede；另有 Swap、Wiggle 和 Displace。
各更新的公式与适用扇区见物理推导。

## 运行与追踪

| 调用 | 行为 |
|---|---|
| `step()` | 尝试一次更新，不推进热化/量测 sweep，也不自动量测 |
| `advance()` | 按配置推进运行阶段和量测，默认只在内存中运行 |
| `run()` | 推进运行，并保存正式运行材料 |
| `results()` | 返回核心结果的独立副本 |
| `checkpoint(path)` | 保存主模拟的续算状态 |
| `from_checkpoint(path)` | 恢复主模拟状态 |

`step(trace=True)` 保留更新前后的只读快照。`StepResult.candidate()` 使用记录的
同一 patch 重建候选，不重新抽样；被拒绝的有效 proposal 也有候选。
不适用或结构无效的尝试没有有效候选。

`step(move="open", trace=True)` 跳过 move 抽签，但仍使用正常的提议和接受规则，
包括配置中的正反选择权重。人为指定的序列用于理解机制；它的事件频次不代表
配置中的随机链分布。

`after_step` 用于观察每次尝试，`after_sweep` 用于观察 sweep 边界或附加量测。
回调读取模拟状态；修改构型或消耗 sampler RNG 会改变随后运行的轨迹。
追踪和绘图本身不消耗采样随机数，默认也不积累完整追踪历史。

## 量测与统计

`EstimatorManager` 管理核心 scalar 与 Green 结果。scalar 只在 Z 扇区取物理量；
统计同时记录 Z/G 驻留和全部计划量测机会。拒绝后保留的重复构型也是链的一部分。
Green 计数通过扇区驻留及 bin 定义归一化，不能直接把端点计数当作 g1。

`statistics.py` 对分子、分母做 dyadic ratio blocking。
短链、稀疏 bin 或没有合格平台时，`standard_error` 为 `None`，并附带误差状态。
blocking 平台是诊断信息，仍需检查热化、seed 间差异和慢相关性。

附加量测器通过回调接入现有 Simulation：

- `EqualTimeCorrelations`：g2 与非零 S(k)，保存相关量的完整样本历史和协方差。
- `DensityProfile`：Fourier 外势下的密度剖面，提供 `after_sweep` 便捷回调。

它们各自提供 `results()`、`snapshot()` 和 `restore()`，不自动加入主模拟的
`results()` 或 checkpoint。其量测时机和归一化详见
[关联量](docs/equal_time_correlations.md) 与 [密度剖面](docs/periodic_external.md)。

## 配置、输出与续算

`small_system()` 是简短入口；`from_mapping()` 和 `from_toml()` 提供完整配置。
`input_dict()` 和 `resolved_dict()` 用于检查输入及默认值、衍生参数。
便捷入口、直接构造与 dataclass 替换均经过配置校验。

配置分组描述系统、虚时间离散、势、更新权重和运行预算。
解析后的配置 hash 进入运行及恢复记录，用于核对参数一致性。
Fourier 外势由常数项和连续正整数谐波的 cosine/sine 系数表示。
其稳定性校验与配置示例见外势指南。

主 checkpoint 保存配置、构型、RNG、阶段、sweep 进度、move 计数和核心统计状态，
带有 schema 和 checksum 检查。结果 CSV/JSON 是读取和分析材料，不替代 checkpoint。
同环境下的续算一致性由测试检查；跨 schema 或数值环境的兼容性需另外核实。

续算附加量测器时，在同一 sweep 边界保存主 checkpoint 和 collector snapshot，
恢复两者后再继续。只恢复主状态会丢失附加量测的前半段统计历史。
命令行和输出文件导航见 [运行与续算](docs/running.md)。

## 修改时从哪里入手

| 目标 | 入口 | 相关检查 |
|---|---|---|
| 新的周期势 | `potentials.py`、`config.py` | 周期性、作用量、配置与续算一致性 |
| 修改更新提议 | `moves/`、`sampler.py` | 图不变式、正反密度、选择概率、接受率 |
| 新增观测量 | `estimators/`、回调 | 系综、归一化、重复样本及误差传播 |
| 改善构型显示 | `examples/worldline_plot.py` | 周期 image、候选与决策状态、RNG 不变 |
| 改善使用入口 | `simulation.py`、示例与使用指南 | 参数校验、量测时机和副作用 |

现有势通过配置和 factory 构造；任意用户 callable、通用插件系统、多维展示与
加速后端尚未实现。若扩展这些能力，应结合具体需求讨论接口和复现方式。
参与开发的方法见 [CONTRIBUTING.md](CONTRIBUTING.md)，已完成的版本变化见
[CHANGELOG.md](CHANGELOG.md)。
