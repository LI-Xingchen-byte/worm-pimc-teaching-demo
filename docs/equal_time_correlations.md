# 小系统中的 g2 和 S(k)

`EqualTimeCorrelations` 是一个可独立导入的量测器，用显式回调附加到已有 Simulation。
这样教学代码能看清何时量测、量测了什么，无须修改采样更新。

```python
from wormpimc import Simulation, SimulationConfig
from wormpimc.estimators import EqualTimeCorrelations

config = SimulationConfig.small_system(
    n_slices=8, chemical_potential=-0.7,
    warmup_sweeps=128, measurement_sweeps=1024,
)
simulation = Simulation(config)
correlations = EqualTimeCorrelations(config, spatial_bins=8, modes=(1, 2, 3))

def measure(sim):
    if sim.phase == "measurement" and sim.measurement_completed % config.run.measurement_stride == 0:
        correlations.measure(sim.state)

simulation.advance(after_sweep=measure)
result = correlations.results()
print(result["g2"])
print(result["structure_factor"])
```

完整短示例为 `examples/correlations.py`。`measure(state)` 不检查
warmup/stride，因为单独的 Configuration 不携带这些生命周期信息；回调必须
如上显式筛选。每个机会只调用一次。若需同时监视单步，不要在两个回调里重复量测。

1024-opportunity 示例是快速的 API 演示；本次实测中它的 g2 均值仍明显偏离
参考，即使自动 blocking 检查出现平台。报告中的物理趋势使用每 seed 16384
个机会及更长 warmup，不应把短示例的输出当作同等质量的结果。

## 定义与读数

- `g2`：8 个均匀的 [0,L) 位移 bins，输出左右边界、均值、SE、误差状态、
  原始 slice 平均对计数之和及全部 blocking 层。排除自配对，保留两个有序方向。
- `structure_factor`：只接受互不重复的正整数 modes，实际 k=2*pi*mode/L。
  输出均值、SE、波矢、密度 Fourier 强度之和与 blocking 层；包含自项 N。
- 两者都在所有时间片平均之后作为一个 Monte Carlo 观察值。G 机会存零，
  空 Z 存 Z=1、N=0；被拒绝更新后的重复构型仍须保留。
- `standard_error=None` 表示非零机会不足、分母为零或尚未形成平台，不是零误差。
  `blocking_plateau` 也不保证已排除所有慢相关性。

实际归一化是

\[
\bar g_{2,b}=\frac{L}{\Delta r}\frac{\langle C_b\rangle_Z}{\langle N\rangle_Z^2},
\qquad S(k)=\frac{\langle|\rho_k|^2\rangle_Z}{\langle N\rangle_Z}.
\]

**g2 的分母是平均粒子数的平方**。`g2_pair` 是另一种 factorial 分母归一化的
数学定义，本模块不输出它；两者的空间平均不同。不要逐构型除 N²，也不要把 g2 的空间平均强制拉到 1。
定义详见 [物理推导](../derivations.md) 的对关联一节。

`results()["g2_spatial_average_sum_rule"]` 应等于所有均宽 g2 bins 的均值：
`<N(N-1)>/<N>²`。这是计数与归一化的内部恒等式；物理检验另用独立参考，
不能只靠这条恒等式宣称采样正确。

误差使用 `(I_Z, I_Z*N, I_Z*N(N-1), I_Z*C_b, I_Z*|rho|², ...)` 的完整
block 协方差，经 delta method 传播，包含 g2 的随机归一化分母。
最低 32 blocks、最高三个合格层的平台以及 32 个非零贡献机会，沿用现有门槛。
这是渐近误差估计；短链的非线性比值偏差、seed 间散布都需单独留意。

## 原始数据、恢复和范围

`correlations.samples` 返回独立副本，每行为一个机会，列顺序为
`Z, N, N(N-1), 8 个对计数, 3 个 Fourier 强度`。
不同 bins/modes 下列数随之改变。每个量已乘 Z 指示量；对计数和强度先对 M 平均。
例如 `np.save("samples.npy", correlations.samples)` 可保存审计数据。

量测器使用 O(K*(bins+modes)) 内存保存完整历史；16384 个机会、8 bins、
3 modes 的连续 float64 数组约 1.75 MiB，运行中的 Python 容器另有开销。
它面向小系统和 pilot，暂不适合无界长运行。

`snapshot()` 返回可 JSON 保存的历史；新建同配置、同网格的 collector 后，
用 `restore(snapshot)` 恢复。配置哈希和 bin/mode 必须匹配。
**主 Simulation checkpoint 不包含该 collector**，两者必须在同一个 sweep
边界一起保存。仅恢复主 checkpoint 会缺失关联量的前半段历史。
`Simulation.results()` 和既有 CSV 也没有自动新增这些量，读取此 collector
自己的 `results()`。本轮没有修改 config/checkpoint/output schema。

模块可读取现有一维周期 Z 构型，包括 Gaussian 相互作用，但本次实测验证
仅覆盖理想气体。没有完成外势关联量、接触相互作用或高精度 continuum 验证。

## 独立参考与复现

理想气体参考由独立 mode 占据构造。Wick 分解给出等时二阶关联；相干性与
密度关联的定义可参见 [Naraschewski 与 Glauber, PRA 59, 4595](https://doi.org/10.1103/PhysRevA.59.4595)。
本仓库使用有限周期盒和巨正则几何占据，并在每个 bin 内解析积分 Fourier 项，
不使用该文的局域密度近似或固定粒子数凝聚态修正。

```powershell
python -m validation.run_correlations --output runs/validation/correlations/my_run --seed 2718
```

默认 2048 warmup + 16384 measurement sweeps、每 sweep 8 次更新，最多
180 秒。输出目录必须不存在；超时会保存已收集数据并标记 TIMEOUT。
三 seed 矩阵保存后可用 `validation.plot_correlations` 重绘；已做检查及其范围见
[验证说明](validation.md)。这些命令从仓库根目录运行，validation 参考
与绘图工具不属于安装后的公共库；collector 和量测函数属于库。
