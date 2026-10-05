# 版本更新

这里记录对使用者有影响的功能与行为变化；详细使用方法见 README 和专题指南。
版本号与现有库保持一致。本次文档与目录整理没有改变采样算法或提升库版本。

## 未发布

- 采用 MIT 许可，整理公开仓库的文件范围。
- 精简 README，按当前实现重写架构与源码阅读指南。
- 教学示例使用描述性文件名；参考采样脚本移到 `tests/reference_examples/`。
- 增加贡献、运行和验证说明，将历史开发记录保留在本地。

## 0.11.0

- 增加静态周期 Fourier 外势，可与 Gaussian 对势组合。
- 增加可选 `DensityProfile` 及其统计 snapshot/restore、势能曲线和密度示例。
- Fourier 参数进入配置和恢复记录；附加密度历史需与主 checkpoint 同步保存。

## 0.10.0

- 增加可选 `EqualTimeCorrelations`，量测等时 g2 与非零 S(k)。
- 使用包含随机归一化分母的协方差误差；附加结果与历史独立保存。

## 0.9.0

- 支持指定 move 的单步机制演示。
- 增加更新前、候选、决策后的三栏构型可视化。

## 0.8.0

- 增加小系统便捷配置、内存结果读取、只读追踪和候选重建。
- 统一 Python 公共入口的配置与注入构型校验。

## 0.7.0

- 增加 scalar、Green 与 g1 的相关样本 ratio blocking 和误差诊断。
- 主 checkpoint 保存 blocking 状态；稀疏数据或平台不足时保留误差不可用状态。

## 0.6.0

- 增加 primitive thermodynamic 动能/总能量和归一化 finite-bin Green 量测。
- 提供 beta-minus finite-tau g1 及相关输出。

## 0.5.0

- 实现完整 Z/G 构型及 Open/Close、Insert/Remove、Advance/Recede、Swap 更新。
- 保存扇区驻留和完整 Worm 状态，支持续算。

## 0.4.0

- 增加 Worm 测度、周期 bridge 和更新比的纯参考公式核对。

## 0.3.0

- 增加 Gaussian 对势、固定拓扑 Wiggle/Displace、模拟生命周期和运行/续算 CLI。

## 0.2.0

- 增加周期几何、自由传播子、Brownian bridge 和闭合路径参考采样。

## 0.1.0

- 建立可安装的包、配置校验与配置检查 CLI。
