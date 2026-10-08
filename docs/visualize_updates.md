# 逐步查看 Worm 更新

希望把图和代码放在同一页面时，打开
[Worm 更新 notebook](../examples/worm_walkthrough.ipynb)。它包含最短计算、
构型表、九类更新及接受判断的静态内嵌图。安装与启动方法见 [README](../README.md#示例)。
各组使用独立 seed 和 simulation，保存的候选包括被拒绝的更新。

先安装可选绘图依赖。也可用 `--config configs/periodic_external.toml` 加载周期外势，
在每个世界线面板下显示同空间坐标的势能曲线。密度剖面与新示例见
[周期外势与密度剖面](periodic_external.md)。

```powershell
python -m pip install -e ".[plot]"
```

核心采样库仍只依赖 NumPy。绘图辅助在
[examples/worldline_plot.py](../examples/worldline_plot.py)，不被求解器导入。

## 从两次更新开始

```powershell
python examples/visualize_updates.py
```

默认依次尝试 Open、Close。每步有三个面板：**更新前、候选、决策后**。
关闭当前图窗后继续下一步。默认 seed=97531、M=8 的两步结果为 Open 接受、
Close 拒绝；候选图仍显示被拒绝的闭合桥段，决策后构型保留开放状态。

主脚本见 [visualize_updates.py](../examples/visualize_updates.py)。
其中采样与显示的关键调用为：

```python
from wormpimc import Simulation, SimulationConfig
from worldline_plot import plot_step  # 在 examples 目录内；项目根可用 examples.worldline_plot
import matplotlib.pyplot as plt

config = SimulationConfig.small_system(n_slices=8)
simulation = Simulation(config)
for move in ("open", "close"):
    event = simulation.step(move=move, trace=True)
    figure = plot_step(event, tau=config.tau)
    plt.show()
    plt.close(figure)
```

这段调用复用 [Sampler._step](../src/wormpimc/sampler.py) 的选择/提议、target、
proposal ratio、接受判断和提交顺序。新增的 `move` 只让选择步骤可由调用者指定；
没有第二份更新算法。脚本的其他代码仅用于命令行选项、PNG 保存与窗口关闭。

## 选择更新与保存图像

```powershell
# 单独尝试一种 move；不会自动寻找可用状态
python examples/visualize_updates.py --moves open

# Open 后观察 Swap，再尝试 Close；显示 bead ID 便于追踪连接
python examples/visualize_updates.py --moves open swap close --seed 1 --ids

# 一组可重复的开放链生长与缩短演示
python examples/visualize_updates.py --moves insert advance recede remove --seed 5

# 重复单一类型，或循环指定序列
python examples/visualize_updates.py --moves wiggle --steps 5
python examples/visualize_updates.py --moves open close --steps 6

# 正常随机选择，只观察，不自动量测
python examples/visualize_updates.py --moves random --steps 20

# 无图窗地保存每次尝试，包括拒绝与不适用
python examples/visualize_updates.py --moves open swap close --seed 1 --no-show --output tmp/my-demo
```

每个 PNG 的名称含步骤编号与 move 名称。`--steps` 省略时每个 `--moves` 项尝试
一次；指定时循环列表，仍包含失败尝试。API 的随机选择使用 `move=None`；
字符串 `random` 只作为命令行的便利选项。

需要 G 扇区的 Close、Advance、Recede、Remove、Swap，应在适当的 Open/Insert
之后调用。前置更新也可能被拒绝，因此指定 move 不保证产生图中期望的变化。
程序如实显示不适用原因，不隐藏重试或强制接受。

## 图像怎么读

- 灰线表示本步未改变的连接；橙线是修改前的连接，蓝线是候选的新连接或新几何。
- head 用绿色圆点，tail 用紫色菱形；标注 bead ID。`--ids` 额外标注全部 beads。
- 横轴是位置，0 与 L 为同一空间边界；纵轴是虚时间，0 与 beta 为同一时间边界。
- link 使用其显式 `image_to_next` 绘制，在穿过每个空间边界时拆分；不会用
  minimum image 替换实际路径。最后一个时间切片的 outgoing link 画到 beta，
  而不是斜着连回纵轴底部。顶部空心 bead 是 slice 0 的周期副本，不是新粒子。
- 下方展示 log R 的五类贡献和本次 log u，读者可以直接核对是否满足
  `log u < min(0, log R)`。
- 被拒绝时中间面板保留候选，右面板与左面板的构型相同；不适用时中间面板
  明确写出无候选，下方说明原因。

绘图只读取 `StepResult`，`candidate()` 重建不抽取随机数。`plot_step` 返回
Figure，由调用者决定显示、保存或关闭；保存路径也由调用者控制。绘图关闭
后采样代码不变。自定义 notebook 可从项目根导入 `examples.worldline_plot`，
使用同一函数，不需要复制绘图逻辑。

## 机制演示与物理采样

指定类型的事件标记 `selection_mode="specified"`，图中标明指定演示。
接受率仍使用配置的正反 move 选择权重，表示正常随机 sampler 在选中此
move 后的接受规则。手工调度序列本身不是配置中的随机选择过程，不应把
演示事件频次或其 residence 当作平衡分布结果。

这些调用仍通过普通 Metropolis 规则改变当前构型、RNG 和 move 计数，
`step` 不自动量测。进行带量测的物理计算时使用独立的 simulation 和正常
`advance()`；已有 `after_step` 回调也可复用 `plot_step` 观察随机运行。

## 已检查与尚未覆盖的范围

这一版检查了拒绝候选、不适用状态、Swap 的真实连接变化、Advance 跨空间
边界、虚时间接缝和多重绕行，并测试有无绘图时的状态、RNG 和计数一致性。
支持一维 snapshot；二维/三维展示、播放控件和视频导出不在本切片范围。
自动测试使用无窗口 Agg 后端，真实桌面图窗的外观及交互体验仍可继续调试。
