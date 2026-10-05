# 用 Python 阅读与运行 Worm 更新

本教程介绍直接 Python 调用、单步追踪和结果读取。安装后可从任意工作目录调用库，不需要
TOML、Notebook、绘图库或输出目录。小规模物理计算仍需后续平衡化、统计和
离散误差验证；本教程的默认参数用于理解流程。

`small_system(external="fourier", ...)` 可配置外势并接入可选密度量测。
外势系数、化学势界限和辅助量测的续算方法见
[周期外势与密度剖面](periodic_external.md)。

## 1. 最短的带量测运行

```python
from wormpimc import Simulation, SimulationConfig

config = SimulationConfig.small_system(n_slices=8, measurement_sweeps=100)
simulation = Simulation(config)
simulation.advance()
number = simulation.results()["scalars"]["particle_number"]
print(number["mean"], number["standard_error"], number["uncertainty_status"])
```

`advance()` 先完成配置中的 warmup，再完成 measurement sweeps；重复调用在达到
目标后不再增加采样。`advance(max_sweeps=10)` 限制本次推进量，随后可以继续。
它不建立文件；需要持久化记录时使用既有的 `run()`、`checkpoint()` 和
`from_checkpoint()`。`run()` 返回的 `SimulationResult` 是运行记录；物理结果
统一从 `simulation.results()` 读取。

`small_system()` 默认是一维周期盒中的理想 Bose 气体，参数为
`beta=1`、`box_length=4`、`lambda_kin=0.5`、`chemical_potential=-2`、
`n_slices=16`、初始 `particle_count=2`、`seed=97531`、20 个 warmup sweeps、
100 个 measurement sweeps、每 sweep 20 次原子尝试和 8 个空间 bins。
所有九类 moves 都启用，Wiggle 权重为 4，其余为 1；最大更新段长度为
`min(6, n_slices-1)`，位移尺度为 0.5，Worm sector weight 为 1。
每个 measurement sweep 量测一次。这些是可检查的默认值，不是收敛保证。

相互作用需明确提供强度与范围：

```python
config = SimulationConfig.small_system(
    pair="periodized_gaussian", epsilon=1.0, sigma=0.5,
)
print(config.input_dict())       # 所有输入值
print(config.resolved_dict())    # 还包含 tau、归一化 move 权重等
```

Gaussian image tolerance 默认为 `1e-12`。更多配置可以通过相同的 dataclass
替换方式调整；完整 `SimulationConfig` 构造时会重新校验所有分组及交叉约束：

```python
from dataclasses import replace

config = replace(config, run=replace(config.run, measurement_stride=2))
config = replace(config, moves=replace(config.moves, worm_sector_weight=2.0))
```

分组 dataclass 是参数容器；校验发生在完整 `SimulationConfig` 的构造/替换边界。
例如将 beta 改为负数后重新组装完整配置会抛出 `ConfigError`。注入已有
`Configuration` 时，ndim、盒长、切片数也必须匹配。`particle_count` 仅控制
初始化，后续完整 Worm 链在巨正则扩展系综中变化，不固定为两个粒子。

## 2. 观察一次真实更新

```python
simulation = Simulation(SimulationConfig.small_system(n_slices=8))
event = simulation.step(trace=True)
print(event.move_name, event.status.value, event.accepted)
print(event.before["sector"], event.after["sector"])
if event.breakdown is not None:
    print(event.breakdown.log_ratio, event.breakdown.log_uniform)
    candidate = event.candidate()
else:
    print(event.patch.invalid_reason if event.patch else "no proposal")
```

完整可执行示例见 [observe_step.py](../examples/observe_step.py)。它调用的
唯一更新核心是 [sampler.py 中的 Sampler._step](../src/wormpimc/sampler.py)。
阅读该函数时按以下顺序追踪变量即可：

| 核心操作 | 对象或变量 | 含义 |
|---|---|---|
| 选择 | `rng.choice(..., p=...)` | 从固定 all_moves 权重选择，不按当前 sector 偷换归一化 |
| 提议 | `move.propose(state, rng)` | 建立 patch，记录修改和正反提议密度，不改变 live state |
| 构型权重 | `measure.delta_local(state, patch)` | kinetic、potential、chemical、sector/measure 的变化 |
| 提议概率 | `move.proposal_ratio(patch)` | 反向/正向密度及 move 选择概率之比 |
| 接受判断 | `log_uniform < min(0, log_ratio)` | 比较同一步的随机数与总对数接受比 |
| 提交 | `state.apply(patch)` | 仅在接受时改变 live state |

`step()` 在这段核心外按需记录快照；`advance()`、`run()` 与手动 `step()` 共用
这段实现。图像展示也将复用它，示例不重复写一份接受判断或拓扑修改。

`event.breakdown` 保存五类对数贡献、`log_uniform` 和 `accepted`。
`event.proposal_ratio` 进一步区分正反密度与正反 move 选择概率。
`MoveStatus.PROPOSED` 表示 proposal 已生成，既可能接受，也可能拒绝；判断接受
必须用 `event.accepted`。不适用和结构无效事件保留原因，其 breakdown 为 `None`。

追踪的 `before/after` 是独立的只读 mapping，内部数组也不可写。它们不会随
后续构型改变。`candidate()` 根据本次 `before` 和 patch 重建独立构型，不重新
采样，因而拒绝的候选也可以查看；不适用/结构无效时返回 `None`。
未开启 trace 时 before/after 为 `None`，调用 candidate 会明确报错。
对接受后的 live state 再次直接 apply 同一个 patch 会触发 stale revision 检查，
应使用 candidate() 查看历史提议。

### 指定一种 move

```python
simulation = Simulation(SimulationConfig.small_system(n_slices=8))
opened = simulation.step(move="open", trace=True)
closed = simulation.step(move="close", trace=True)
print(opened.accepted, closed.accepted)
```

支持 `wiggle`、`displace`、`open`、`close`、`insert`、`remove`、`advance`、
`recede`、`swap`。一次调用只尝试指定的类型；不会自动完成前置更新，也不会
强制接受。比如初始 Z 构型直接指定 Close 会返回 `not_applicable`。配置中权重为
零的 move 或错误名称会提前报错，不消耗 RNG 或增加计数。

`move=None`（默认）是正常随机选择。指定模式只省略选择 move 的随机数，
proposal 生成、target、正反密度、Metropolis 判断和提交仍用同一实现；接受率
中的正反 move 权重继续来自配置。事件通过 `selection_mode="specified"`
标明这一点。手工安排的更新序列用于解释机制，其频次不代表正常随机链的
平衡统计；做物理量计算仍使用正常 `advance()`，建议与演示使用独立 simulation。

可视化用法见 [更新可视化](visualize_updates.md)。

## 3. 构型与量测的语义

- **Z 扇区**：世界线闭合，可以读取粒子数、交换环和 winding，scalar 在这里量测。
- **G 扇区**：存在一条开放世界线；head 没有 outgoing link，tail 没有 incoming
  link。端点位置和虚时间差用于 Green 统计。此时不能简单把 beads 数除以切片数
  解释为固定粒子数。
- **step**：一次 move 选择与至多一次 Metropolis 判断，不自动量测，也不增加
  warmup/measurement sweep 计数。拒绝及不适用都算一次尝试。
- **sweep**：固定 `steps_per_sweep` 次尝试。`advance` 在完成 sweep 后推进阶段，
  只在 measurement 阶段按 stride 量测。
- **measurement opportunity**：一次计划量测，无论此时是 Z 还是 G 都保留记录；
  它与仅计 Z 扇区的 scalar 样本数不同。

混用手动 step 与 advance 是允许的：手动 step 改变构型、RNG 和 sampler 计数，
但不是已完成的 warmup/measurement sweeps。正式比较时应保持调用序列一致。

## 4. 在带量测的运行中观察每一步

```python
def observe(event):
    print(event.move_name, event.accepted, event.after["sector"])

simulation = Simulation(SimulationConfig.small_system())
simulation.advance(max_sweeps=2, trace=True, after_step=observe)
```

回调收到 warmup 与 measurement 中的每次尝试，包括拒绝和不适用；回调发生在
本步决策后、所在 sweep 的量测之前。只读取传入的事件，不从闭包修改 simulation
或使用它的 RNG。需要整个 sweep 完成后的量测状态时，使用 `after_sweep`。

默认不开启 trace，也不在 simulation 中保存历史。可以用 `after_step=events.append`
主动保留历史，但大量快照会占用内存。仅需要名称与接受率时，无须开启 trace。
内置追踪及 candidate 重建不会改变随机数序列或采样结果。

## 5. 读取有适用范围的结果

`results()` 返回与 live accumulator 分离的普通字典，可直接 JSON 序列化：

| 键 | 内容 |
|---|---|
| `summary` | 阶段、sweeps、atomic_steps、Z/G visits、量测机会数、Z 量测数与 config hash |
| `scalars` | 以 observable 名称索引，包含 mean、standard_error、n_measurements、n_effective、blocking 诊断及归一化信息 |
| `green_function` | 每个时空 bin 的边界、raw count、G、beta-minus g1、各自误差状态及归一化状态 |

不可用或不适用的值为 `None`，不以零补齐。`standard_error=None` 时先看
`uncertainty_status`，例如短链的 `insufficient_blocking_levels`。同时保留
`estimator_variant`、`normalization_status`、`units` 与 `warning`；未产生量测时
scalar mean 也为 `None`。`uncorrected_standard_error` 是诊断量，不替代相关链误差。
Green bins 计数不足时不会发布渐近误差；beta-minus g1 仍为 finite-tau 估计。

`results()` 不重新量测，不增加样本，也不读写文件。CLI 和 CSV 保持原有 schema；
Python 结果将 CSV 中的空值和非有限数字规范为 `None`。当前验证范围及相互作用、时间步验证的待完善部分见 [验证说明](validation.md)。

## 6. 导入边界

常用入口 `Simulation`、`SimulationConfig`、`Configuration`、`StepResult`、
`Sector`、`MoveStatus` 可从 `wormpimc` 导入。进阶对象使用明确子模块路径：

```python
from wormpimc.sampler import Sampler
from wormpimc.measure import PrimitiveTargetMeasure
from wormpimc.configuration import initialize_configuration
from wormpimc.geometry import winding_from_images
```

`Sampler` 同样检查 state 与 config 的几何一致性。`Move` 构造、内部数组布局和
patch 序列化属于进阶实现细节；普通调用不需要操作这些内容。
