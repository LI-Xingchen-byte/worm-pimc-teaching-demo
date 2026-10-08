# Worm PIMC

一个可直接调用、逐步观察更新过程的 Python 路径积分 Monte Carlo 教学库。
核心采用连续空间 Worm 算法，用于一维周期盒中的有限温度玻色子小系统。
支持理想气体、周期 Gaussian 排斥对势，以及静态 Fourier 外势。

## 安装

需要 Python 3.11 或更新版本。先下载仓库：

```shell
git clone https://github.com/LI-Xingchen-byte/worm-pimc-teaching.git
cd worm-pimc-teaching
```

然后在仓库根目录执行：

```shell
python -m pip install -e .
```

核心依赖只有 NumPy。需要绘图时安装可选依赖：

```shell
python -m pip install -e ".[plot]"
```

## 第一个计算

```python
from wormpimc import Simulation, SimulationConfig

config = SimulationConfig.small_system(
    n_slices=8, warmup_sweeps=5, measurement_sweeps=100,
)
simulation = Simulation(config)
simulation.advance()
number = simulation.results()["scalars"]["particle_number"]
print(number["mean"], number["standard_error"], number["uncertainty_status"])
```

这段代码先热化，再量测平均粒子数，不需要输入文件，也不生成运行目录。
默认模型是理想气体；加入 Gaussian 对势时，在 `small_system()` 中传入
`pair="periodized_gaussian", epsilon=1.0, sigma=0.5`。

这是短链演示。`standard_error=None` 表示当前数据不足以给出误差估计，
不是零误差；延长链并检查不同 seed 和时间步，才适合讨论物理结果。

## 示例

也可以打开 [Worm 更新 notebook](examples/worm_walkthrough.ipynb)，在同一页面中阅读
最短计算、构型数据与九类更新的内嵌图。首次使用在仓库根目录执行：

```shell
python -m pip install -e ".[notebook]"
python -m jupyter lab examples/worm_walkthrough.ipynb
```

选择相应 Python kernel 后从头运行。每组更新独立初始化，支持按组重跑；
这份静态版本保留接受、拒绝和不适用事件，不需要逐个关闭图窗。

下面的脚本从仓库根目录运行。前三个依次展示计算、单步观察和构型绘图。

| 示例 | 运行命令 | 内容 |
|---|---|---|
| 最短计算 | `python examples/simulate.py` | 配置、运行、读取结果 |
| 单步观察 | `python examples/observe_step.py` | 提议、接受率分项和决策 |
| 更新可视化 | `python examples/visualize_updates.py` | 更新前、候选、决策后三栏图 |
| 关联量 | `python examples/correlations.py` | 等时 g2 与 S(k) 的附加量测 |
| 外势与密度 | `python examples/periodic_external.py` | Fourier 外势和密度剖面 |

绘图示例需要 `.[plot]`。外势示例可加 `--no-plot` 只运行计算。
更新图默认尝试 Open、Close，关闭当前图窗后继续；其他选项见
[可视化指南](docs/visualize_updates.md)。

## 怎样阅读这个库

- `SimulationConfig` 描述系统、势、更新权重和运行参数。
- `Simulation` 管理构型与运行：`step()` 尝试一次更新，`advance()` 推进热化和量测，
  `run()` 还会保存结果与 checkpoint。
- `Sampler` 将 `moves/` 的提议与 `measure.py` 的目标权重组合成接受判断。
- `estimators/` 和 `statistics.py` 负责量测、归一化与相关样本的误差估计。

只想写一个程序，从上面的用法和 [Python 使用指南](docs/python_workflow.md) 开始。
想读更新实现，接着读 [架构说明](architecture.md)；公式与物理约定见
[物理推导](derivations.md)。

## 适用范围

目前实现限于一维周期系统，势能采用 primitive 离散作用量。已有理想气体对照、
更新正反检查和续算测试；相互作用体系的系统验证与时间步外推仍需完善。
具体证据范围和复现方法见 [验证说明](docs/validation.md)。

指定 `step(move="open")` 等更新序列用于机制演示；物理量计算使用正常随机选择的
`advance()`。外势下的现有关联量表示空间平均，不能直接当作局域两点关联。

## 进一步阅读与贡献

- [周期外势与密度剖面](docs/periodic_external.md)
- [等时关联量](docs/equal_time_correlations.md)
- [命令行运行与续算](docs/running.md)
- [参与开发](CONTRIBUTING.md)
- [版本更新](CHANGELOG.md)

本项目采用 [MIT 许可](LICENSE)。
