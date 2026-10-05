# 怎样检验完整采样链

整链验证运行正常的随机 Worm 链，指定 move 的教学演示不作为采样数据。
运行前确定参数、seed、预算与比较判据；已有检查的范围见 [验证说明](validation.md)。
验证工具位于仓库的 `validation/`，不加入库的公共入口，也不复制更新算法。

## 独立参考

有限一维周期盒的单粒子能级为

\[
\epsilon_j=\lambda(2\pi j/L)^2,\quad
q_j=e^{\beta(\mu-\epsilon_j)},\quad
\bar n_j=\frac{q_j}{1-q_j},\quad \mu<0.
\]

各动量 mode 的占据独立且服从几何分布。因此

\[
\langle N\rangle=\sum_j\bar n_j,\qquad
\langle E\rangle=\sum_j\epsilon_j\bar n_j,\qquad
\operatorname{Var}N=\sum_j\bar n_j(1+\bar n_j).
\]

`ideal_reference.py` 直接卷积各 mode 的几何分布得到 P(N)，不调用库中的
propagator、action 或 sampler。截断后的 P(N) 不重新归一化，尾部另存。
动量截断另给出遗漏的平均粒子数与能量的上界。
几何分布依据见 [MIT 8.333 的 Bose 占据统计习题与解答](https://ocw.mit.edu/courses/8-333-statistical-mechanics-i-statistical-mechanics-of-particles-fall-2013/9e4f82085311e3bee72e9244f449cbe8_MIT8_333F13_ExamRevFinlSol.pdf)。

交换量不是“发生了几次 Swap”。这里按闭合图统计：含 lM 个 beads 的环
包含 l 个粒子，累加 l>1 的环中的粒子数。由
`log Xi = sum_l z^l Z_1(l beta)/l` 展开，各 l 环的平均数量为
`z^l Z_1(l beta)/l`，所以交换粒子数的参考为
`<N> - z Z_1(beta)`。这检验了实际 Bose 交换权重，而不只是 move 计数。

## 复现命令

在仓库根目录、已安装 NumPy 的 Python 环境运行：

```powershell
python -m pytest tests/test_p1_validation.py -q
python -m validation.run_p1 --output runs/validation/ideal/my_pilot --pilot --seed 1701
python -m validation.run_p1 --output runs/validation/ideal/my_run --seed 2718 --c0 0.5
```

输出目录必须尚不存在，防止覆盖旧实验。runner 使用当前仓库的源码。
`--warmup`、`--sweeps`、`--timeout` 可调整预算；超时标为 TIMEOUT，保存已完成
时间序列与 checkpoint，不作为已完成确认链。统计辅助时间序列尚不支持
自动断点续跑；`final.npz` 可以恢复核心 Simulation，不能据此宣称完整重建
未保存的辅助统计历史。

每个目录有配置、每次量测机会的数组、最终 checkpoint 和结果 JSON。
数组第一列为 Z 指示量，其后为 N、N²、E、交换粒子数、P0…P4、P(N>=5)
的分子；G 行全部为零。保留拒绝后的重复构型，不用 accepted moves 子集
估计物理分布。数组和源码保存 SHA256；结果可独立重新分析。

正式矩阵使用 `c0.5_s2718` 等目录名，可运行：

```powershell
python -m validation.report_p1 --root runs/validation/ideal
python -m validation.check_replay runs/validation/ideal/c1_s2718 runs/validation/ideal/replay_s2718 --output runs/validation/ideal/reproducibility.json
```

汇总始终检查预定的 12 条链，缺失或未完成者显式列出。每个物理量保存
全部 blocking 层、误差平台状态、与参考之差、前后半链均值及有效样本量。
`CONSISTENT` 表示通过预定精度下的偏差筛查，不等于证明无偏或完成所有物理验证。

## 正反 proposal 的额外检查

`proposal_audit.py` 只读取 before/candidate 的原始图数据。独立高斯公式
计算每条 image-resolved 链接；周期端点传播密度用另一份 image 求和计算。
再根据图找回 Open/Close、Insert/Remove、Advance/Recede 的被删或新增路径。
Swap 在正反两个图中枚举目标 slice 的所有 beads，逐条前向核对桥路径、
反向沿 predecessor 检查候选合法性，重新累加候选归一化。

参考计算不读取 `patch.metadata` 的 q 字段，也不把 `patch.reversed()` 保存的
交换值当作反向概率证据。测试还核对不对称 move 权重的选择概率项和独立
理想气体 target log weight。仍需完整链的独立物理参考来检验宏观分布。

## 延伸到关联量与时间步验证

g2 和 S(k) 的现有量测、归一化与误差接口见 [关联量指南](equal_time_correlations.md)。
运行短 pilot 检查各 bin/mode 的误差，再为固定参数确定多个 seed 的生产预算。
参数或量测成本变化后重新试跑，不用更新吞吐量代替有效样本数。

相互作用两粒子参考需使用匹配系综或条件量测；初始化 N=2 不等于正则约束。
时间步研究比较多个 M，并另行检查参考的空间求积精度。
理想气体的精确自由传播不包含 primitive 势能离散误差，不能替代相互作用收敛验证。
