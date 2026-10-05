# 周期外势与密度剖面

当前支持一维周期盒中的静态、对所有粒子相同的单粒子 Fourier
外势。它可独立使用，也可与既有 periodized Gaussian 对势叠加。Worm 更新仍
使用同一套 primitive target；本版不提供非周期势、时间相关势或任意 callable。

## Python 入口

```python
from wormpimc import Simulation, SimulationConfig
from wormpimc.estimators import DensityProfile
from wormpimc.potentials import external_from_config

config = SimulationConfig.small_system(
    n_slices=8,
    chemical_potential=-0.7,
    external="fourier",
    external_offset=0.5,
    external_cosine=[-0.5],
    external_sine=[],
)
simulation = Simulation(config)
density = DensityProfile(config, spatial_bins=16)
simulation.advance(after_sweep=density.after_sweep)
result = density.results()
field = external_from_config(config)
print(field.energy([0.0, 1.0, 2.0]))  # [0, 0.5, 1]
print(result["integrated_density"])  # <N>，不是 1
```

此例使用 `V(x)=0.5*(1-cos(2*pi*x/L))`。`external_offset` 是常数项；
`external_cosine[n-1]` 与 `external_sine[n-1]` 是第 n 次整数谐波的系数。
两个序列不等长时缺项为零，空序列表示常数势。所有系数均为有限实数，单位
与对势的 epsilon 相同。序列在配置内转为不可变 tuple；修改原始 list 不会
改变模型。周期性相对于 `system.box_length` 保证；模型没有另一个不相容的周期。

加入对势只需同时指定 `pair="periodized_gaussian", epsilon=1.0, sigma=0.5`。
接受率中的 potential 项与输出的 potential_energy 均包括外势和对势，能量不
先减去势的最小值。常数项也不被丢弃：势增加 c 等价于 target 中 mu 减少 c，
而物理总能量增加 c*N。

## TOML 与稳定性

```toml
[potential]
pair = "none"
external = "fourier"
external_offset = 0.5
external_cosine = [-0.5]
external_sine = []
```

完整可运行配置见 `configs/periodic_external.toml`。解析后参数、规范化 TOML、
config hash、metadata 和主 checkpoint 都包含外势参数。无外势配置的既有
schema-4 resolved 表与 hash 保持不变；当前版本可恢复旧 checkpoint。

无对势时，第一版要求

`mu < external_offset - sum(hypot(a_n, b_n))`。

这是势的下界给出的充分稳定性条件，可能拒绝真实基态能量仍高于 mu 的输入；
本版不在配置阶段求基态，也不以“有周期势”作为放开化学势限制的理由。
对势为现有严格正 Gaussian 排斥时沿用既有化学势规则。

## 密度量测与恢复

`DensityProfile` 是可选 collector，不自动进入 `Simulation.results()`、正式 run
的 CSV 或主 checkpoint。`after_sweep` 只在与主量测一致的 measurement/stride
位置量测，不使用 RNG。与其他 observer 共用时，可在一个回调内依次调用它们。

空间 bins 是 [0,L) 上等宽半开区间。在 Z 构型内先将 M 个时间片的粒子数平均，
再在 Monte Carlo 机会之间求平均。G 对分子/分母都贡献零，空 Z 分母贡献一，
拒绝后的重复状态保留。因此密度积分等于主量测的 <N>，不按逐构型 N 归一化。

每个 bin 以完整机会序列进行 numerator/Z-residence ratio blocking，包含二者
协方差。时间片不当作独立样本。只有满足既有平台判据且至少 32 个机会对该 bin
有非零贡献时才发布 SE；其余为 None。即使误差状态给出平台，短链也不保证
热化或收敛。内存为 O(bins*log K)，不保留全部 bead 历史。

辅助历史需要与主状态在同一时刻保存：

```python
import json
from pathlib import Path

simulation = Simulation(config)
density = DensityProfile(config, spatial_bins=16)
simulation.advance(max_sweeps=50, after_sweep=density.after_sweep)
simulation.checkpoint("simulation.npz")
Path("density_snapshot.json").write_text(
    json.dumps(density.snapshot(), allow_nan=False), encoding="utf-8"
)

restored = Simulation.from_checkpoint("simulation.npz")
resumed_density = DensityProfile(restored.config, spatial_bins=16)
resumed_density.restore(json.loads(Path("density_snapshot.json").read_text(encoding="utf-8")))
restored.advance(after_sweep=resumed_density.after_sweep)
```

上段保存尚未完成的 simulation。仅恢复主 checkpoint 无法恢复辅助历史；
不同参数或 bin 数的辅助 snapshot 会被拒绝。`results()` 是结果，`snapshot()`
才是含所有 blocking pending 状态的续算材料。

## 图与教学脚本

```powershell
python examples/periodic_external.py
python examples/periodic_external.py --no-show --output tmp/external_demo
python examples/periodic_external.py --no-plot --sweeps 32
python examples/visualize_updates.py --config configs/periodic_external.toml --moves open close
```

`periodic_external.py` 使用同一个 Simulation，画两行共用 x 轴的 V_ext(x) 与 rho(x)，仅对
可发布的 SE 画误差棒。更新图在三个 worldline 面板下分别加势能曲线，不把势能
画到虚时间轴上。原有无外势更新图保持三个面板。绘图 helper 在 examples 中，
core 不依赖 Matplotlib；`--no-plot` 可在只安装 NumPy 的环境运行。

使用 `--output` 时保存 input.toml、simulation.npz、density_snapshot.json、
density.json，以及启用绘图时的 density.png/update.png。更新展示发生在保存
之后，其单步事件不混入已保存的量测历史。

## 验证范围与后续

测试覆盖周期性、多谐波和相位、对势组合、空 slice、常数外势/化学势平移，
九类更新的独立 link-endpoint 外势作用量核对、local/full/reverse 核对、完整
权重的 beta 导数、外势输出、主与辅助统计的精确续算，以及密度归一化和
重复样本的相关误差。

`validation/external_reference.py` 另提供不导入 wormpimc 的周期 heat-kernel
空间求积参考，输出相同 M 的理想气体密度/能量及单粒子正则密度/能量。
必须分别检查空间网格与 M；有限 M 参考不是 continuum 认证；系统验证范围见 [验证说明](validation.md)。

现有 Green、g1、g2 只保留相对位移，在非均匀外势下解释为空间平均，不能
解释成某个固定位置的局域值。S(k) 包含平均密度调制；局域两点关联与 connected
量的新增接口按本轮范围留到后续。
