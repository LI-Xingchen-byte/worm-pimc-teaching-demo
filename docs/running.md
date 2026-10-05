# 命令行运行与续算

从仓库根目录运行。Python 内存调用与单步观察见 [使用指南](python_workflow.md)。

## 配置与运行

```shell
python -m wormpimc check-config configs/worm_sampler.toml
python -m wormpimc run configs/worm_sampler.toml
```

`worm_sampler.toml` 启用完整巨正则 Worm 更新。
`closed_sampler.toml` 是固定粒子数、置换连通性和绕数的回归参考，
不能将其结果当作完整拓扑系综的采样。
`periodic_external.toml` 展示 Fourier 外势。

检查局部 target delta 时，可增加完整重算：

```shell
python -m wormpimc run configs/worm_sampler.toml --verify-local-delta
```

## 输出

默认运行目录位于 `runs/`。每次正式运行保存输入与解析后配置、环境与状态信息、
move 统计、Z/G 驻留、scalar 结果、Green histogram、blocking 诊断和 checkpoints。
CSV 中的空误差对应统计不足状态，不能当作零误差。
命令的全部选项可通过 `python -m wormpimc --help` 和各子命令的 `--help` 查看。

## 恢复与摘要

将下列占位路径替换为实际运行目录与 checkpoint：

```shell
python -m wormpimc resume runs/<run-id>/checkpoints/<checkpoint>.npz
python -m wormpimc summarize runs/<run-id>
```

主 checkpoint 保存继续采样所需的构型、随机数和核心量测状态。
附加 `EqualTimeCorrelations`、`DensityProfile` 的历史不自动包含在其中，
需要在同一个 sweep 边界保存并恢复各自 snapshot。
具体代码见 [关联量](equal_time_correlations.md) 和 [密度剖面](periodic_external.md)。
