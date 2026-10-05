# 验证范围与复现入口

这里区分算法和实现检查、已有物理对照，以及仍需完成的收敛研究。
短示例用于展示调用方式，不是已经收敛的物理结果。

## 已有检查

| 检查 | 工具或测试 | 能说明什么 |
|---|---|---|
| 周期几何、自由传播和 Brownian bridge | geometry/propagator 测试、`tests/reference_examples/` | 自由路径与显式 image 的基本一致性 |
| 拓扑与正反更新 | topology/measure/selected-move 测试 | 候选图、逆对、权重和选择概率的内部一致性 |
| 独立 proposal 重建 | `validation/proposal_audit.py` | 从实际前后图重算提议密度，而不是复用保存的 q 值 |
| 理想 Bose 气体整链 | `validation/ideal_reference.py`、`run_p1.py` | 有限盒工作点上的粒子数、能量和交换统计对照 |
| 等时关联 | `validation/correlation_reference.py`、`run_correlations.py` | 理想气体有限 bin 的 g2 与 S(k) 参考 |
| 周期外势 | `validation/external_reference.py`、外势/密度测试 | 相同 M 的独立 heat-kernel 求积与作用量、密度检查 |
| 误差与续算 | statistics/checkpoint/simulation 测试 | 相关样本统计状态和同环境恢复的一致性 |

## 已完成的整链对照

2026-09-16，使用当时的 0.9.0 完成一个有限周期盒理想气体工作点的对照：

- L=4、beta=1、lambda=0.5、mu=-0.7、M=8、无对势，最大更新段长为 3。
- seed 为 2718、3141、5772；C0=0.5、1、2，另有 C0=1 的不对称更新权重组。
- 每条链 2048 warmup + 16384 measurement sweeps，每 sweep 8 次更新，共 12 条链。
- N、N²、E、交换环中的粒子数和 P(N) 的 120 项比较通过运行前设定的 4 SE 筛查。

独立动量参考为平均 N=1.331737611、E=0.452679978、交换粒子数=0.538770452。
筛查支持这个工作点的结果与参考相容，不是对全部参数或无偏性的证明。
不对称权重组的 seed 间散布更大，blocking 平台也不能排除全部慢相关性。
上述数字是历史实验摘要，本次文档整理没有重新运行这套长链矩阵。
原始历史数据保留在本地；公开仓库提供参考和重新生成材料的脚本。

等时关联已有三个 seed 的理想气体探索性运行，使用 8 个空间 bins 和前三个
非零 Fourier modes；观察到相应量级与趋势，尚不代表高精度收敛。
外势量测还需区分有限 M、空间求积网格和 continuum 极限。

## 本地复现

从仓库根目录运行；输出目录需要尚不存在。

```shell
python -m pytest
python -m validation.run_p1 --output runs/validation/ideal/pilot --pilot --seed 1701
python -m validation.run_correlations --output runs/validation/correlations/seed2718 --seed 2718
```

pilot 检查流程和预算，不复现完整的 12 链证据。完整参数、矩阵和 proposal 审核见
[整链验证方法](full_chain_validation.md)，关联量接口见
[等时关联说明](equal_time_correlations.md)。`run_p1` 等文件名沿用既有验证工具名称。

## 尚需完善

- 更广参数范围、相互作用体系与匹配系综的两粒子独立参考。
- 多个时间切片数 M 的研究，以及 primitive 势能作用量的时间步外推。
- 关联量的热化、长链、多 seed 精度检查；外势下局域/connected 关联的额外定义。

初始化 N=2 不构成正则约束；比较正则两粒子参考时使用匹配系综或条件量测。
理想气体的精确自由传播检验，不能代替相互作用的时间步收敛研究。
