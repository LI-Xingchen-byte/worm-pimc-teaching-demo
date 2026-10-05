# 连续空间 Worm PIMC：符号与约定

本文给出当前实现的单位、周期路径、Worm 测度、更新概率和观测量定义。
简短用法见 [README](README.md)，代码组织见 [架构说明](architecture.md)。

## 1. 范围与阅读顺序

当前实现为一维周期盒中的有限温度玻色子，采用对称 primitive 势能作用量。
支持理想气体、光滑排斥 Gaussian 对势和静态周期 Fourier 外势。
接触或硬核相互作用需要另外验证的短时作用量，当前未提供。

阅读周期路径先看第 4–8 节；阅读 Worm 更新看第 9–12 节；
观测量与相关样本统计见第 13–14 节。公式的 `eq-*` 锚点供源码和测试引用。
实现与推导应使用一致定义；更改约定时同步说明、代码和验证。
系统验证及尚未完成的时间步外推见 [验证说明](docs/validation.md)。

## 2. 单位与基本物理符号

除非明确在测试某个物理单位换算层，否则代码使用

$$
\hbar=k_{\mathrm B}=1.
$$

因此，逆温度具有能量倒数的单位，虚时间遍历 $[0,\beta)$。

| 符号 | 代码面向名称 | 含义 |
|---|---|---|
| $d$ | **ndim** | 空间维度；第一个实现中 $d=1$ |
| $N$ | **n_particles** | 闭合、对角构型中的物理粒子数 |
| $m$ | **mass** | 玻色子质量；通常通过 $\lambda$ 表示，而非单独存储 |
| $\lambda$ | **lambda_kin** | 动能系数， $\lambda=\hbar^2/(2m)$ ； $\lambda\tau$  具有长度平方的单位 |
| $T$ | **temperature** | 温度 |
| $\beta$ | **beta** | 逆温度，代码单位下 $\beta=1/T$ |
| $\mu$ | **chemical_potential** | 化学势；其符号以 $+\mu\int N(\tau')\,d\tau'$ 进入权重 |
| $L$ | **box_length** | 一维环的周长 |
| $\Lambda$ | **box** | 周期空间域， $[0,L)^d$ |
| $M$ | **n_slices** | 虚时间切片数 |
| $\tau$ | **tau** | 时间步长， $\tau=\beta/M$ |
| $j$ | **slice_id** | 切片索引， $j\in\{0,\ldots,M-1\}$ |
| $\mathbf R_j$ | **positions_on_slice(j)** | 切片 $j$ 上的全部坐标 |
| $V(\mathbf R)$ | **potential_energy** | 总的外势加对势能量 |
| $\rho_0$ | **free_density_matrix** | 自由粒子短时密度矩阵 |

保留符号约定：

- $m$ 在物理公式中始终表示质量。
- 以时间切片计量的段长用 $s$ 或 **n_links** 表示，绝不用 $m$。
- $M$ 始终表示切片总数。
- **bead_id** 是存储标识符，不是物理粒子标签。

本项目与主要 Worm 参考文献的符号略有差异：Boninsegni--Prokof'ev--Svistunov 用 $P$ 表示切片总数、$\varepsilon$ 表示时间步长、$M$ 表示提议段长。在本项目中这些符号分别映射为 $M$、$\tau$ 与 $s$。我们保留 $\Xi$ 表示物理巨配分函数；“$Z$ 扇区”仍是对角构型扇区的惯用名称，并不引入第二个配分函数符号。

所有浮点物理计算在可用时使用 IEEE binary64。Python 使用 **numpy.float64**；Fortran 实现将定义项目精度 kind，而不是依赖编译器默认的实型 kind。

## 3. 哈密顿量与默认模型

<a id="eq-hamiltonian"></a>
对于 $N$ 个全同无自旋玻色子，被模拟的哈密顿量为

$$
H_N=-\lambda\sum_{i=1}^{N}\nabla_i^2
    +\sum_{i=1}^{N}V_{\mathrm{ext}}(\mathbf r_i)
    +\sum_{1\le i\lt k\le N}U_L(\mathbf r_i-\mathbf r_k).
$$

构型 $\mathbf R=(\mathbf r_1,\ldots,\mathbf r_N)$ 的总势能为

$$
V(\mathbf R)
=\sum_i V_{\mathrm{ext}}(\mathbf r_i)
+\sum_{i\lt k}U_L(\mathbf r_i-\mathbf r_k).
$$

<a id="eq-periodized-gaussian"></a>
第一个示例使用

$$
d=1,\qquad V_{\mathrm{ext}}(x)=0,
$$

以及一种 periodized Gaussian 排斥

$$
U_L(\Delta x)
=\epsilon\sum_{n\in\mathbb Z}
\exp\!\left[-\frac{(\Delta x+nL)^2}{2\sigma^2}\right],
\qquad \epsilon>0.
$$

在数值工作中，图像求和只有在被省略的尾部低于某个已记录的容差之后才被截断。最近邻近似只允许作为显式的替代模型出现，或者在测试表明被忽略的图像低于所要求的数值容差之后使用。

势接口必须接受卷绕坐标并返回周期势。任何 Monte Carlo 更新都不得为这个 Gaussian 示例包含模型特异的代数。

2026-10-01 外势扩展使用静态 Fourier 模型（系数单位为能量）：

$$
V_{\mathrm{ext}}(x)=c_0+\sum_{n=1}^{K}
\left[a_n\cos(2\pi nx/L)+b_n\sin(2\pi nx/L)\right].
$$

两个系数序列从 n=1 开始，不等长时缺项为零。空序列给出常数外势。
周期性由整数谐波保证，且外势与对势相加；头尾的单粒子外势也遵循第 10.2
节的半权重。在无对势的巨正则输入中采用充分条件
`mu < c0 - sum(hypot(a_n,b_n))`，该下界可能比真实基态能量保守。
常数外势 c 等价于 target 中的 `mu -> mu-c`，能量估计则增加 c*N。

可选密度剖面将 [0,L) 划成宽度 dx 的半开 bins。令 n_a(C) 是 Z 构型
各时间片的 bin 粒子数平均，则 `rho_a = <I_Z*n_a> / (<I_Z>*dx)`，
满足 `sum(rho_a*dx)=<N>_Z`。G 提供零分子和零分母，空 Z 的分母为一。
同一构型的时间片和拒绝后的重复状态不作为额外独立样本；误差沿用完整机会
序列上的 ratio blocking。外势下仅按相对位移计数的关联量是空间平均量，
局域两点函数及其归一化留到后续实现。

## 4. 周期几何

### 4.1 卷绕坐标

存储的 bead 坐标满足

$$
\mathbf r_b\in[0,L)^d.
$$

<a id="eq-wrap"></a>
对于标量坐标，

$$
\mathrm{wrap}_L(x)
=x-L\left\lfloor\frac{x}{L}\right\rfloor
\in[0,L).
$$

<a id="eq-centered-displacement"></a>
中心化位移为

$$
\mathrm{disp}_L(x,y)
=(x-y)-L\left\lfloor\frac{x-y}{L}+\frac12\right\rfloor
\in[-L/2,L/2).
$$

因此半盒平局归属于 $-L/2$。这一平局约定必须在 Python、Fortran 与测试中完全一致。

### 4.2 动能链接与周期图像

仅靠卷绕坐标不足以保留路径绕环的拓扑。每条有向动能链接 $b\to b'$ 因此携带一个整数图像向量

$$
\mathbf n_{b\to b'}\in\mathbb Z^d.
$$

<a id="eq-image-resolved-displacement"></a>
它的未卷绕位移为

$$
\Delta\widetilde{\mathbf r}_{b\to b'}
=\mathbf r_{b'}-\mathbf r_b+L\mathbf n_{b\to b'}.
$$

进入自由粒子链接作用量的是这个位移，而不是最小图像位移。对所有整数图像求和即可在周期盒上恢复精确的自由密度矩阵。

规划的存储名称是 **image_to_next(bead_id, ndim)**。若某个 bead 没有出射链接，其图像值被忽略，并应在 debug 构建中置零。

对于一个闭合有向环 $\gamma$，卷绕坐标会互相消去，且

$$
\sum_{(b\to b')\in\gamma}
\Delta\widetilde{\mathbf r}_{b\to b'}
=L\mathbf W_\gamma,
\qquad
\mathbf W_\gamma
=\sum_{(b\to b')\in\gamma}\mathbf n_{b\to b'}
\in\mathbb Z^d.
$$

总绕数向量是对所有闭合环求和得到。在一维中它简单地写成 $W\in\mathbb Z$。

## 5. 虚时间离散化

虚时间沿正方向取向

$$
0\longrightarrow\tau\longrightarrow2\tau\longrightarrow\cdots
\longrightarrow(M-1)\tau\longrightarrow0.
$$

每条被占据的动能链接恰好前进一个切片：

$$
\mathrm{slice}(\mathrm{next}(b))
=\mathrm{slice}(b)+1\pmod M.
$$

被接受的构型不包含跳过切片的“长链接”。多切片更新会显式插入或移除中间 bead。

对于两个切片索引，正向模分离为

$$
\Delta j(a\to b)
=(j_b-j_a)\bmod M
\in\{0,\ldots,M-1\},
$$

对应的分离为 $\Delta\tau=\tau\Delta j$。当一条开放路径穿过超过一个时间周期时，它实际的有向链接数（而不只是模分离）记录了这一历史。

## 6. 巨正则配分函数

物理对角的系综为

$$
\Xi(\beta,\mu,L)
=\mathrm{Tr}\exp[-\beta(H-\mu\hat N)].
$$

以标记坐标为中间表示，并显式恢复玻色对称性，

$$
\Xi_M
=\sum_{N=0}^{\infty}\frac{e^{\beta\mu N}}{N!}
\sum_{P\in S_N}
\int_{\Lambda^{NM}}
\prod_{j=0}^{M-1}d\mathbf R_j
\prod_{j=0}^{M-1}
\rho_\tau(\mathbf R_j,\mathbf R_{j+1}),
$$

其中时间边界条件为

$$
\mathbf R_M=P\mathbf R_0.
$$

链接 bead 图通过其连通性存储置换；它不存储单独的置换数组。存储 ID 只是用于枚举图的标签，因此提议选择概率与 $1/N!$ 测度因子必须在每个改变拓扑的推导中一致地处理。

展示出的 $1/N!$ 属于这个数学迹的标记坐标形式。实现遵循标准的 Worm 图约定，其中对全同 bead 的重新标记被吸收进图测度。因此它不把 $-\log(N!)$ 作为单独的构型能量项来求值。这是一种测度约定，不是可以在局部随意加减的消去。

## 7. 短时密度矩阵与 primitive 作用量

<a id="eq-symmetric-primitive"></a>
记 $H=K+V$。版本 1 采用对称 primitive 因式分解

$$
e^{-\tau(K+V)}
=e^{-\tau V/2}e^{-\tau K}e^{-\tau V/2}
+O(\tau^3).
$$

局部因式分解误差为 $O(\tau^3)$，在通常的正则性假设下，对光滑可观测量给出 $O(\tau^2)$ 的主导全局时间步偏差。这个标度是需要数值检验的假设，不能代替时间步外推。

<a id="eq-free-density"></a>
对于无穷空间中一个自由粒子，

$$
\rho_0(\mathbf r,\mathbf r';\tau)
=(4\pi\lambda\tau)^{-d/2}
\exp\!\left[-\frac{|\mathbf r'-\mathbf r|^2}{4\lambda\tau}\right].
$$

<a id="eq-image-free-density"></a>
每条周期链接带显式图像时，其贡献为

$$
\rho_0^{(\mathbf n)}(\mathbf r,\mathbf r';\tau)
=(4\pi\lambda\tau)^{-d/2}
\exp\!\left[
-\frac{|\mathbf r'-\mathbf r+L\mathbf n|^2}{4\lambda\tau}
\right].
$$

对这个表达式在所有 $\mathbf n\in\mathbb Z^d$ 上求和即得周期自由粒子密度矩阵。

<a id="eq-z-log-weight"></a>
对于一个在 $Z$ 扇区、每个时间间隔有 $N$ 条被占据链接的闭合图，在采用的 Worm 图测度下，primitive 目标对数权重为

$$
\begin{aligned}
\log W_Z(C)={}&
-\frac{d}{2}NM\log(4\pi\lambda\tau)
-\sum_{\ell\in\mathcal L(C)}
\frac{|\Delta\widetilde{\mathbf r}_\ell|^2}{4\lambda\tau}\\
&-\tau\sum_{j=0}^{M-1}V(\mathbf R_j)
+\beta\mu N.
\end{aligned}
$$

这里 $\mathcal L(C)$ 是有向动能链接的集合。玻色对称因子已属于第 6 节定义的图测度的一部分。常数可以在某个特定的接受率中被代数地消去，但在展示这种消去之前，它们必须保留在书写的推导中。

任何代码路径都不得把 primitive 作用量静默地替换成 pair-product 作用量。所选的作用量类型必须出现在输入元数据与输出文件中。

## 8. 自由 Brownian-bridge 约定

<a id="eq-bridge-recursion"></a>
类 Wiggle 提议使用未卷绕的自由 Brownian bridge。假设当前未卷绕点 $\widetilde{\mathbf r}_0$ 距离固定端点 $\widetilde{\mathbf r}_s$ 有 $s$ 条链接。第一个新点采样为

$$
\widetilde{\mathbf r}_1\sim
\mathcal N\!\left(
\widetilde{\mathbf r}_0
+\frac{\widetilde{\mathbf r}_s-\widetilde{\mathbf r}_0}{s},
\;2\lambda\tau\frac{s-1}{s}\,I_d
\right).
$$

<a id="eq-bridge-moments"></a>
递归应用此式直到到达端点。等价地，在链接 $a\in\{1,\ldots,s-1\}$ 处的边际分布为

$$
\mathbb E[\widetilde{\mathbf r}_a]
=\widetilde{\mathbf r}_0
+\frac{a}{s}
(\widetilde{\mathbf r}_s-\widetilde{\mathbf r}_0),
$$

$$
\mathrm{Cov}(\widetilde{\mathbf r}_a)
=2\lambda\tau\frac{a(s-a)}{s}I_d.
$$

对于周期端点，提议必须首先选择或继承一个确定的端点图像，从而固定 $\widetilde{\mathbf r}_s$。采样最小图像 bridge 并丢弃所选图像并不等价于周期自由传播子。

### 8.1 周期 bridge 混合

对于改变拓扑的更新，端点图像并不总是从既有段继承。定义精确的周期自由密度

<a id="eq-periodic-free-density"></a>

$$
\rho_{0,L}(\mathbf r,\mathbf r';s\tau)
=\sum_{\mathbf k\in\mathbb Z^d}
\rho_0^{(\mathbf k)}(\mathbf r,\mathbf r';s\tau).
$$

一个归一化的周期 bridge 分两个显式阶段采样。首先按

$$
P(\mathbf k\mid\mathbf r,\mathbf r')
=\frac{\rho_0^{(\mathbf k)}(\mathbf r,\mathbf r';s\tau)}
{\rho_{0,L}(\mathbf r,\mathbf r';s\tau)},
$$

选择总端点图像 $\mathbf k$，然后以该端点图像采样一条未卷绕 Brownian bridge。若
$K_s=\sum_{a=0}^{s-1}\log\rho_0^{(\mathbf n_a)}
(\mathbf r_a,\mathbf r_{a+1};\tau)$ 是所得图像分辨链接的对数权重，则所选取图像、中间卷绕位置与链接图像的联合密度为

<a id="eq-periodic-bridge-mixture"></a>

$$
Q_L
=\frac{\exp(K_s)}
{\rho_{0,L}(\mathbf r_0,\mathbf r_s;s\tau)}.
$$

对所有 $\mathbf k$、链接图像与中间位置求和，结果为 1。这一混合（而不是最小图像 bridge）才是权威的 Close 提议。它同时也允许后续 Close 移动生成一个与先前 Open 移动移除的不同绕数扇区。

## 9. 构型扇区与图拓扑

### 9.1 通用存储约定

构型使用 bead 记录池，采用 structure-of-arrays 存储：

    position(bead_id, ndim)       wrapped coordinate in [0, L)
    slice_of(bead_id)             integer in [0, M-1]
    next_of(bead_id)              outgoing neighbour or NONE
    prev_of(bead_id)              incoming neighbour or NONE
    image_to_next(bead_id, ndim)  periodic image of the outgoing link
    active(bead_id)               allocation flag

在面向 Python 与面向 Fortran 的规格中，哨兵值都是 **NONE = -1**。Fortran 数组索引内部可以是从 1 起，但持久化 ID 与跨语言测试夹具使用从 0 起。外部 ID 与数组下标之间的转换必须发生在唯一一个已记录的辅助例程中。

没有持久的物理 **particle_id**。粒子是不可区分的，置换更新可能改变哪些 bead 属于同一有向环。

### 9.2 对角扇区 $Z$

合法的 $Z$ 扇区构型满足以下全部条件：

1. 没有 Worm 端点。
2. 每个活跃 bead 恰好有一个前驱和一个后继。
3. 每个后继都前进一个时间切片模 $M$。
4. 跨越每个区间 $j\to j+1$ 的被占链接数是同一个整数 $N$。
5. 每个连通分量都是一个有向环，其链接数是 $M$ 的整数倍。
6. 每个环周围的图像和是一个整数绕数向量。

包含 $qM$ 条链接的环表示一个长度为 $q$ 的玻色置换环。

### 9.3 非对角扇区 $G$

第一个实现恰好允许一个有向开放分量，并使用端点名

- 尾部 $\mathcal M$（**worm_tail**）：产生端， $\mathrm{prev}(\mathcal M)=\mathrm{NONE}$；
- 头部 $\mathcal I$（**worm_head**）：湮灭端， $\mathrm{next}(\mathcal I)=\mathrm{NONE}$。

沿正虚时间方向，开放分量从 $\mathcal M$ 遍历到 $\mathcal I$。所有内部 bead 都有一个前驱和一个后继。其他连通分量（如果存在）是闭合环。

玻色单体 Green 函数约定为

$$
G(\mathbf r,\tau)
=\left\langle
\mathcal T_\tau
\hat\psi(\mathbf r,\tau)
\hat\psi^\dagger(\mathbf 0,0)
\right\rangle,
$$

不带整体负号。端点直方图使用

$$
\Delta\mathbf r_{\mathcal M\to\mathcal I}
=\mathrm{disp}_L(\mathbf r_{\mathcal I},\mathbf r_{\mathcal M}),
$$

以及从尾部到头部的正向有向虚时间分离。任何需要未卷绕端点间隔的估计量都必须沿开放链使用图像，而不是中心化位移。

定义

$$
N_j(C)
=\text{跨越 }j\to j+1\text{ 的被占链接数}.
$$

那么任一扇区的化学势贡献为

$$
W_\mu(C)
=\exp\!\left[
\mu\tau\sum_{j=0}^{M-1}N_j(C)
\right].
$$

在 $Z$ 扇区，$N_j=N$，上式化为 $e^{\beta\mu N}$。在 $G$ 扇区，$N_j$ 是分段常数，只在端点切片处改变。这个定义是权威的；bead 计数绝不能被用作物理粒子数或虚时间占据的替代品。

零长度的开放分量不属于版本 1 的状态空间。若一个转移会消除最后一条开放链接，就必须使用适当的 Close 或 Remove 更新。

## 10. 扩展 Worm 系综

### 10.1 来源映射与端点测度

Worm 端点测度约定遵循 Boninsegni--Prokof'ev--Svistunov，
*Phys. Rev. E* **74**, 036701 (2006)，第 II A--B 节，尤其式
(2.6)--(2.24)，但保留第 4 节与第 8 节的图像分辨周期几何。

令 $g_M(\mathbf r_{\mathcal I},j_{\mathcal I};
\mathbf r_{\mathcal M},j_{\mathcal M})$ 为玻色 Green 函数未归一化的离散分子，使得

$$
G_M(\mathbf r_{\mathcal I},j_{\mathcal I};
\mathbf r_{\mathcal M},j_{\mathcal M})
=\frac{g_M(\mathbf r_{\mathcal I},j_{\mathcal I};
\mathbf r_{\mathcal M},j_{\mathcal M})}{\Xi_M}.
$$

端点测度固定为对两个端点切片求和、对两个端点位置做 Lebesgue 积分的未归一化形式。因此

<a id="eq-worm-extended-measure"></a>

$$
\mathcal Z_{\mathrm W}
=\Xi_M+C_G
\sum_{j_{\mathcal I}=0}^{M-1}
\sum_{j_{\mathcal M}=0}^{M-1}
\int_\Lambda d\mathbf r_{\mathcal I}
\int_\Lambda d\mathbf r_{\mathcal M}\,
g_M(\mathbf r_{\mathcal I},j_{\mathcal I};
\mathbf r_{\mathcal M},j_{\mathcal M}).
$$

场算符具有量纲 $L^{-d/2}$，因此 $g_M$ 与 $G_M$ 具有量纲 $L^{-d}$。于是

$$
[C_G]=L^{-d}=V^{-1},\qquad V=L^d.
$$

$C_G$ 是一个算法系数，不影响归一化的 $Z$ 扇区可观测量。面向用户的调节参数是无量纲正数 $C_0$，名为 **worm_sector_weight**。对于第 12.2 节的均匀提议族，项目定义

<a id="eq-worm-sector-coefficient"></a>

$$
C_G=\frac{C_0}{VM s_{\max}}.
$$

这一选择不是一个物理近似：它只是吸收掉原本会出现在驻留比中的体积、切片数与段选择标度。一旦启用 Worm 更新，resolved configuration 与 checkpoint 元数据必须同时存储 $C_0$ 与推导出的 $C_G$。

### 10.2 Worm 端点处的 primitive 作用量

闭合扇区恒等式 $S_V=\tau\sum_jV(\mathbf R_j)$ 在 Worm 端点处是有歧义的，因为入射与出射粒子集合不同。定义

$$
\mathbf R_j^-
=\{\mathbf r_b:\mathrm{slice}(b)=j,
\mathrm{prev}(b)\ne\mathrm{NONE}\},
$$

$$
\mathbf R_j^+
=\{\mathbf r_b:\mathrm{slice}(b)=j,
\mathrm{next}(b)\ne\mathrm{NONE}\}.
$$

$\mathbf R_j^-$ 是从区间 $j-1\to j$ 到达的构型， $\mathbf R_j^+$ 是跨越 $j\to j+1$ 离开的构型。任一扇区的对称 primitive 作用量为

<a id="eq-worm-primitive-action"></a>

$$
S_V(C)=\frac{\tau}{2}\sum_{j=0}^{M-1}
\left[V(\mathbf R_j^-)+V(\mathbf R_j^+)\right].
$$

在普通 bead 处，这两个集合一致。在 $\mathcal M$ 处，只有
$\mathbf R_{j_{\mathcal M}}^+$ 包含尾部；在 $\mathcal I$ 处，只有
$\mathbf R_{j_{\mathcal I}}^-$ 包含头部。在 $Z$ 扇区，这两个集合相同，上式精确退化为第 7 节。

相对于下文固定的图测度，任一扇区的绝对对数密度为

<a id="eq-worm-log-weight"></a>

$$
\log W(C)=
\sum_{\ell\in\mathcal L(C)}\log\rho_0^{(\mathbf n_\ell)}
-S_V(C)
+\mu\tau\sum_{j=0}^{M-1}N_j(C)
+\mathbf 1_G(C)\log C_G.
$$

这是 $G$ 扇区 `TargetMeasure` 未来的全量重算 oracle。端点半权重、归一化的动能前置因子以及扇区系数都不得被移入提议代码。

### 10.3 图重数（graph multiplicity）

图测度对每个有向 worldline 图只计数一次（模去存储 ID 的重新标记）。闭合分量是不可区分的；开放分量通过标记的端点 $\mathcal M$ 与 $\mathcal I$ 加以区分。因此 bead 分配 ID 不是积分标签，在使用确定性分配器时不引入任何提议概率。

这一约定与标记坐标的迹一致。使用归一化玻色位置态，

$$
|\mathbf R_N\rangle
=\frac{1}{\sqrt{N!}}
\hat\psi^\dagger(\mathbf r_1)\cdots
\hat\psi^\dagger(\mathbf r_N)|0\rangle,
$$

一次场插入给出通常的 $\sqrt{N+1}$ 矩阵元因子。在标记展开中，两个端点因子的乘积提供 $N+1$，而中间 $(N+1)$-体对称化提供 $1/(N+1)!$：

$$
\frac{N+1}{(N+1)!}=\frac1{N!}.
$$

因此，相对于闭合背景，标记的开放线具有单位图重数。在上面扩展对数权重中，没有额外的 $N!$、$N+1$、端点矩阵元或
`particle_id` 项。像 Open 中对角 bead 计数 $B_Z$ 这样的因子只来自提议选择。

### 10.4 扇区驻留与 Green 函数归一化

对于一个平移不变系统，定义端点间隔密度

$$
h_s(\mathbf r)=\left\langle
\mathbf 1_G\,
\mathbf 1_{\Delta j(\mathcal M\to\mathcal I)=s}\,
\delta_L^{(d)}\!\left(
\mathbf r-\mathrm{disp}_L(
\mathbf r_{\mathcal I},\mathbf r_{\mathcal M})
\right)
\right\rangle_{\mathcal Z_{\mathrm W}}.
$$

尾部切片有 $M$ 种选择，对绝对尾部位置积分得到 $V$。除以对角驻留
$P_Z=\langle\mathbf1_Z\rangle_{\mathcal Z_{\mathrm W}}$ 因此给出

<a id="eq-green-residence-normalization"></a>

$$
G_M(\mathbf r,s\tau)
=\frac{h_s(\mathbf r)}{P_Z C_G M V}.
$$

对于直方图 bin，$h_s$ 是访问次数除以总测量次数，再除以空间 bin 体积。$\mathbf r=0$ 处的等时接触约定与最终 $g_1$ 直方图 API 需按下文 Green/g1 估计量定义处理；任何原始端点计数都不得标记为归一化 $g_1$。

## 11. 更新名称与逆对

本项目保留以下规范名称：

| 更新 | 扇区转移 | 逆类 | 意图角色 |
|---|---|---|---|
| **Open** | $Z\to G$ | **Close** | 从既有闭合路径移除一段 |
| **Close** | $G\to Z$ | **Open** | 重建缺失的段并闭合路径 |
| **Insert** | $Z\to G$ | **Remove** | 创建一个开放路径分量 |
| **Remove** | $G\to Z$ | **Insert** | 删除对应的可移除开放分量 |
| **Advance** | $G\to G$ | **Recede** | 沿虚时间向前延伸一个 Worm 端点 |
| **Recede** | $G\to G$ | **Advance** | 缩短对应的开放段 |
| **Swap** | $G\to G$ | **Swap** 类 | 将头部与另一条路径重连并采样置换 |
| **Wiggle** | 同扇区 | **Wiggle** 类 | 从 free bridge 重采样中间 bead |
| **Displace** | 同扇区 | **Displace** 类 | 平移一个选定的连通对象 |

更新名称描述的是一个提议类，而不是接受保证。若引入端点特异的变体，其实现名称必须包含 **head** 或 **tail**，并记录其逆映射。

## 12. 细致平衡约定

令 $p_a(C)$ 为在构型 $C$ 中选择更新类 $a$ 的概率，并令 $q_a(C'|C)$ 包含之后每一个离散选择概率与连续提议密度。完整的提议概率为

$$
T_a(C\to C')=p_a(C)q_a(C'|C).
$$

对于目标密度 $\pi(C)$，Metropolis--Hastings 比为

$$
R_a(C\to C')
=\frac{\pi(C')T_{\bar a}(C'\to C)}
{\pi(C)T_a(C\to C')},
$$

且

$$
A_a(C\to C')=\min(1,R_a).
$$

<a id="eq-metropolis-log-ratio"></a>
实现总是计算

$$
\log R_a
=\Delta\log W_{\mathrm{kin}}
-\Delta S_V
+\Delta\log W_\mu
+\Delta\log W_{\mathrm{sector/comb}}
+\log T_{\bar a}
-\log T_a.
$$

接受在 log 空间判断：

    accept if log(u) < min(0, log_ratio),  u ~ Uniform(0,1)
    （若 log(u) < min(0, log_ratio) 则接受，u ~ Uniform(0,1)）

任何更新都不得仅仅因为某个提议因子对某个特定输入文件是常数就省略它。消去必须符号化地展示。

每次提议都会产生一个包含所有已更改 bead、链接、图像、端点与选择元数据的可逆 patch。拒绝时丢弃 patch 而不改变活动构型；接受时原子地提交它。

对于存储的正向提议与构造出的反向提议，主要的单元测试为

$$
\log R_a(C\to C')
+\log R_{\bar a}(C'\to C)=0
$$

在已记录的浮点容差内成立。

### 12.1 固定拓扑闭合扇区提议族

Wiggle 与 Displace 保持拓扑与粒子数固定。移动选择使用通过归一化所有已配置移动权重而得到的、与状态无关的概率。不支持的移动名称可以具有零权重；不执行与状态相关的再归一化。

对于 **Wiggle**，令 $B$ 为活跃 bead 数并令
$s_{\max}<M$。提议依次选择

1. 一个起始 bead，概率 $1/B$，均匀选取；
2. 一个段长，从 $s\in\{2,\ldots,s_{\max}\}$ 中均匀选取；
3. 通过沿这 $s$ 条链接遍历继承的、既有有向端点图像；
4. 第 8 节自由 Brownian bridge 的 $s-1$ 个中间未卷绕坐标。

将 $Q_0[\widetilde{\mathbf r}_{1:s-1}\mid
\widetilde{\mathbf r}_0,\widetilde{\mathbf r}_s]$ 记作归一化的 bridge
密度，则条件提议密度为

$$
q_{\rm W}(C'|C)
=\frac{1}{B(s_{\max}-1)}
Q_0[\widetilde{\mathbf r}'_{1:s-1}\mid
\widetilde{\mathbf r}_0,\widetilde{\mathbf r}_s].
$$

反向提议使用相同的起始 bead、长度与端点图像，并采样旧的中间坐标。因此

$$
\log\frac{q_{\rm W}(C|C')}{q_{\rm W}(C'|C)}
=\log Q_0[\widetilde{\mathbf r}_{1:s-1}]
-\log Q_0[\widetilde{\mathbf r}'_{1:s-1}].
$$

由于 bridge 密度是 $s$ 个自由链接传播子的乘积除以固定端点传播子，它的比会消去所修改段上的完整动能目标之比。粒子数、扇区、移动选择概率与图测度都保持不变。只有在逐分量记录完这种消去之后，才可使用

<a id="eq-wiggle-log-ratio"></a>

$$
\log R_{\rm W}
=-\tau\sum_{j\in J_{\rm W}}
\left[V(\mathbf R'_j)-V(\mathbf R_j)\right],
$$

其中 $J_{\rm W}$ 是包含已更改中间 bead 的切片集合。

对于 **Displace**，从 $n_{\rm cyc}$ 个分量中均匀选取一个有向闭合分量，并从固定超立方体 $[-\Delta,\Delta]^d$ 中均匀抽取向量
$\boldsymbol\delta$。该分量的每个 bead 都平移相同的未卷绕位移；重新计算卷绕位置与链接图像，使每条未卷绕链接位移都被保持。逆过程使用
$-\boldsymbol\delta$，具有相同的密度 $(2\Delta)^{-d}$ 与相同的
$1/n_{\rm cyc}$ 分量概率。因此提议比为 1，动能比为 1，且

<a id="eq-displace-log-ratio"></a>

$$
\log R_{\rm D}
=-\tau\sum_{j\in J_{\rm D}}
\left[V(\mathbf R'_j)-V(\mathbf R_j)\right].
$$

对于这两个移动，实现仍必须计算并暴露完整的目标与提议分解。测试必须验证 Wiggle 的动能--提议消去、Displace 的零动能变化、局部/完整目标的吻合，以及正向/反向对数比的消去。

这两个移动都不改变有向连通性或总环绕数。仅使用这两种更新时对固定初始粒子数、置换连通性与绕数扇区下的目标进行采样。它是一个闭合扇区参考采样器，还不是完整正则或巨正则玻色系综的遍历采样器。

### 12.2 Open/Close 与 Insert/Remove 族

以下定义 Open/Close 与 Insert/Remove 的目标比和提议比。令

$$
\Delta S_V=S_V(C')-S_V(C),
\qquad
K_s=\sum_{a=0}^{s-1}\log\rho_0^{(\mathbf n_a)}
(\mathbf r_a,\mathbf r_{a+1};\tau),
$$

并令 $p_a$ 表示从完整已配置移动集中选择移动类 $a$ 的、与状态无关的概率。四个移动都使用
$s\sim\mathrm{Uniform}\{1,\ldots,s_{\max}\}$ 且
$s_{\max}<M$。一旦启用这一族，每一逆对的两个成员都必须具有正的选择概率。在错误扇区中被选中的移动记为不适用；概率不按扇区重新归一化。

#### Open：（ $Z\to G$ ）

Open 从当前对角图的 $B_Z$ 个 bead 中均匀选取一个 bead $\mathcal I$，并均匀选取 $s$。令
$\mathcal M=\mathrm{next}^s(\mathcal I)$。$\mathcal I\to\mathcal M$ 上的 $s-1$ 个中间 bead 与全部 $s$ 条链接被移除；保留的 bead 成为头部 $\mathcal I$ 与尾部 $\mathcal M$。由于 $s_{\max}<M$，这不可能删除整个闭合分量。

正向条件概率为

$$
q_{\rm op}(C'|C)=\frac1{B_Zs_{\max}}.
$$

对于精确的逆路径，Close 具有周期 bridge 密度

$$
q_{\rm cl}(C|C')
=\frac{\exp(K_s)}
{\rho_{0,L}(\mathbf r_{\mathcal I},
\mathbf r_{\mathcal M};s\tau)}.
$$

目标动能变化为 $-K_s$，而提议贡献包含
$+K_s$；这两者只在两者都被记录后才消去。Open 移除 $s$ 条被占链接，因此 $\Delta\log W_\mu=-\mu\tau s$。完整的简化比为

<a id="eq-open-log-ratio"></a>

$$
\boxed{
\log R_{\rm op}
=\log\!\left[
\frac{C_G B_Zs_{\max}}
{\rho_{0,L}(\mathbf r_{\mathcal I},
\mathbf r_{\mathcal M};s\tau)}
\right]
-\Delta S_V-\mu\tau s
+\log\frac{p_{\rm cl}}{p_{\rm op}}
}.
$$

没有任何端点距离截断属于目标或提议。如果将来把它作为效率过滤器引入，它必须作为两个方向上的支撑限制出现，并接受单独的推导。

#### Close：（ $G\to Z$ ）

Close 计算从头部 $\mathcal I$ 到尾部 $\mathcal M$ 的正向模间隔。只有当这个间隔是唯一的 $s\in\{1,\ldots,s_{\max}\}$ 时才适用。随后它从第 8.1 节的周期 bridge 混合中采样总端点图像与 $s-1$ 个中间 bead，并插入全部 $s$ 条图像分辨链接。令 $B_Z'$ 为所得对角图的 bead 数。新链接对数权重为 $K_s$。

正向 bridge 密度为 $\exp(K_s)/\rho_{0,L}$，反向 Open 选择为 $1/(B_Z's_{\max})$。Close 增加 $s$ 条被占链接。因此

<a id="eq-close-log-ratio"></a>

$$
\boxed{
\log R_{\rm cl}
=\log\!\left[
\frac{\rho_{0,L}(\mathbf r_{\mathcal I},
\mathbf r_{\mathcal M};s\tau)}
{C_G B_Z's_{\max}}
\right]
-\Delta S_V+\mu\tau s
+\log\frac{p_{\rm op}}{p_{\rm cl}}
}.
$$

对于一个精确的正向 Open patch 与其反向 Close patch，
$B_Z'=B_Z$，两者的作用量变化符号相反，且每个目标与提议分量都分别消去。

#### Insert：（ $Z\to G$ ）

Insert 创建一个具有 $s$ 条链接与 $s+1$ 个 bead 的新开放分量。它依次选择

1. $j_{\mathcal M}$，从 $M$ 个切片中均匀选取；
2. $\mathbf r_{\mathcal M}$，在盒中均匀选取，密度 $1/V$；
3. $s$，从 $1,\ldots,s_{\max}$ 中均匀选取；
4. 每个后续的未卷绕自由位移，来自归一化 Gaussian，卷绕端点并在每条链接上存储对应图像。

最终 bead 是在切片 $j_{\mathcal M}+s\pmod M$ 上的头部 $\mathcal I$。以 $K_s$ 为新链接对数权重，

$$
q_{\rm ins}(C'|C)
=\frac{\exp(K_s)}{VMs_{\max}},
\qquad q_{\rm rm}(C|C')=1.
$$

动能目标与自由随机游走提议因子消去。Insert 增加 $s$ 条被占链接，给出

<a id="eq-insert-log-ratio"></a>

$$
\boxed{
\log R_{\rm ins}
=\log(C_GVMs_{\max})
-\Delta S_V+\mu\tau s
+\log\frac{p_{\rm rm}}{p_{\rm ins}}
}.
$$

采用第 10.1 节 $C_G=C_0/(VMs_{\max})$ 的约定，第一项就是 $\log C_0$。

#### Remove：（ $G\to Z$ ）

只有当唯一的开放分量实际链接数 $s\in\{1,\ldots,s_{\max}\}$ 时才适用 Remove。使用的是实际有向链接数，而不仅是端点的模切片间隔。开放分量的全部 $s+1$ 个 bead 与其 $s$ 条链接被删除。反向 Insert 概率密度在被删除的尾部位置、切片、长度、链接图像与 bead 位置上求值；确定性 bead 分配器不贡献概率。

<a id="eq-remove-log-ratio"></a>

$$
\boxed{
\log R_{\rm rm}
=-\log(C_GVMs_{\max})
-\Delta S_V-\mu\tau s
+\log\frac{p_{\rm ins}}{p_{\rm rm}}
}.
$$

#### AcceptanceBreakdown 映射

在上述解析消去之前，提供给采样器的分量为：

| 移动 | kinetic | potential | chemical | sector measure | 反向减正向提议 |
|---|---:|---:|---:|---:|---:|
| Open | $-K_s$ | $-\Delta S_V$ | $-\mu\tau s$ | $+\log C_G$ | $K_s-\log\rho_{0,L}+\log(B_Zs_{\max})+\log(p_{\rm cl}/p_{\rm op})$ |
| Close | $+K_s$ | $-\Delta S_V$ | $+\mu\tau s$ | $-\log C_G$ | $-K_s+\log\rho_{0,L}-\log(B_Z's_{\max})+\log(p_{\rm op}/p_{\rm cl})$ |
| Insert | $+K_s$ | $-\Delta S_V$ | $+\mu\tau s$ | $+\log C_G$ | $-K_s+\log(VMs_{\max})+\log(p_{\rm rm}/p_{\rm ins})$ |
| Remove | $-K_s$ | $-\Delta S_V$ | $-\mu\tau s$ | $-\log C_G$ | $+K_s-\log(VMs_{\max})+\log(p_{\rm ins}/p_{\rm rm})$ |

`worm_measure.py` 中的纯参考函数编码了这个表，而不执行图突变。它们用于完整/局部目标检查与精确正向/逆向测试。

### 12.3 Advance/Recede 与 Swap 族

#### Advance：（ $G\to G$ ）

Advance 从 $1,\ldots,s_{\max}$ 中均匀选取 $s$，并从当前头部 $\mathcal I$ 向前生长开放分量。它采样 $s$ 个图像分辨自由步，创建 $s$ 个新 bead 与链接，并让最终 bead 成为新头部。以 $K_s$ 为新增动能对数权重，

$$
q_{\rm ad}(C'|C)=\frac{\exp(K_s)}{s_{\max}},
\qquad
q_{\rm re}(C|C')=\frac1{s_{\max}}.
$$

动能目标与自由游走密度消去，同时增加 $s$ 条被占链接：

<a id="eq-advance-log-ratio"></a>

$$
\boxed{
\log R_{\rm ad}
=-\Delta S_V+\mu\tau s
+\log\frac{p_{\rm re}}{p_{\rm ad}}
}.
$$

#### Recede：（ $G\to G$ ）

Recede 均匀选取 $s$，并删除头部紧后方的最后 $s$ 条链接及它们的末端 bead。只有当开放分量严格多于 $s$ 条链接时才适用；消除整个开放分量属于 Remove。精确的反向 Advance 密度在被删除的位置与图像上求值。因此

<a id="eq-recede-log-ratio"></a>

$$
\boxed{
\log R_{\rm re}
=-\Delta S_V-\mu\tau s
+\log\frac{p_{\rm ad}}{p_{\rm re}}
}.
$$

实际有向链接数对适用性具有权威性。模切片差无法区分一个短分量与一个缠绕虚时间圆柱的分量。

#### Swap：（ $G\to G$ ）

Swap 使用固定 bridge 长度 $s_{\rm sw}=s_{\max}$。令当前头部为切片 $j$ 上的 $h$。合法候选集包含切片 $j+s_{\rm sw}\pmod M$ 上的活跃 bead $\alpha$，其前驱链
$\alpha,\mathrm{prev}(\alpha),\ldots,
\mathrm{prev}^{s_{\rm sw}}(\alpha)$ 有定义且不包含尾部 $\mathcal M$。定义

$$
\Sigma_h(C)=\sum_{\alpha\in\mathcal C_h(C)}
\rho_{0,L}(\mathbf r_h,\mathbf r_\alpha;s_{\rm sw}\tau).
$$

候选 $\alpha$ 以其项除以 $\Sigma_h$ 的概率被选取。令
$\zeta=\mathrm{prev}^{s_{\rm sw}}(\alpha)$。旧段
$\zeta\to\alpha$ 具有对数动能权重 $K_{\rm old}$。从 $h$ 到 $\alpha$ 采样一条周期 bridge，它具有新链接对数权重 $K_{\rm new}$。旧段的中间 bead ID 被复用，$h$ 与第一个 bridge bead 相连，$\zeta$ 成为新头部。合并的正向密度为

$$
q_{\rm sw}(C'|C)=\frac{\exp(K_{\rm new})}{\Sigma_h(C)}.
$$

在提议状态中，相同的构造具有归一化子 $\Sigma_\zeta(C')$，并以密度
$\exp(K_{\rm old})/\Sigma_\zeta(C')$ 重建旧段。动能与 bridge 因子消去，链接占据不变，Swap 是其自身的逆类。因此

<a id="eq-swap-log-ratio"></a>

$$
\boxed{
\log R_{\rm sw}
=-\Delta S_V
+\log\Sigma_h(C)-\log\Sigma_\zeta(C')
}.
$$

候选归一化子是提议密度的一部分，必须在两种状态中重算。邻接表实现只有在保持相同声明支撑与归一化的前提下才可以限制候选集；Python 参考使用目标切片上的所有合法 bead。

## 13. 可观测量与 Fourier 约定

物理对角可观测量只在 $Z$ 扇区累积，除非某个估计量推导显式说明了其他情况。

一维 Fourier 约定为

$$
f_k=\int_0^L dx\,e^{-ikx}f(x),
\qquad
f(x)=\frac1L\sum_k e^{ikx}f_k,
\qquad
k=\frac{2\pi n}{L},\quad n\in\mathbb Z.
$$

周期 delta 分布为

$$
\delta_L(x)=\sum_{n\in\mathbb Z}\delta(x+nL).
$$

保留的可观测量定义如下：

### 13.1 密度与粒子数

$$
\rho=\frac{\langle N\rangle_Z}{L}
$$

在一维中。下标 $Z$ 表示对对角扇区访问次数的条件平均，其条件分布即物理对角系综。

### 13.2 对关联

当前量测的 $g_2$ 采用密度平方归一化。
对于平移不变的一维周期系统，定义

<a id="eq-equal-time-g2"></a>

$$
g_2(r)
=\frac{L}{\langle N\rangle_Z^2}
\left\langle
\sum_{i\ne k}\delta_L\!\left(r-(x_i-x_k)\right)
\right\rangle_Z.
$$

这里排除自配对，累积两个有序方向。旧草案使用的分母
$\langle N(N-1)\rangle_Z$ 对应另一数量，今后只称为
$g_2^{\rm pair}(r)=g_2(r)\langle N\rangle_Z^2/\langle N(N-1)\rangle_Z$。
它的空间平均恒为 1，不能与本次密度平方归一化的输出混用。
这是尚未发布过 pair estimator 的显式约定修订，不改变 Worm 目标测度或更新。

令 $C_b=M^{-1}\sum_j\sum_{a\ne c}1[(x_{c,j}-x_{a,j})\bmod L\in b]$，
bin 宽为 $\Delta r_b$，则有限 bin 输出为
$\bar g_{2,b}=L\langle C_b\rangle_Z/(\Delta r_b\langle N\rangle_Z^2)$。
符号差异不改变有序对直方图，两个方向都被计入。全 bins 的和规则为

$$
\frac1L\sum_b\Delta r_b\bar g_{2,b}
=\frac{\langle N(N-1)\rangle_Z}{\langle N\rangle_Z^2}.
$$

有限固定 N 的独立均匀粒子在此定义下为 $1-1/N$，Poisson 巨正则气体为 1；
不能把有限体系的 $g_2$ 强制归一化到远处等于 1。

### 13.3 静态结构因子

定义

$$
\rho_k=\sum_{i=1}^{N}e^{-ikx_i},
\qquad
S(k)=\frac{\langle\rho_k\rho_{-k}\rangle_Z}
{\langle N\rangle_Z}.
$$

$k=0$ 值包含巨正则数涨落，不应把它当作普通非零波矢结构因子来绘制。

实际累计 $D_l=M^{-1}\sum_j|\rho_{k_l,j}|^2$，其中包含自项 N，
在系综平均后才除以 $\langle N\rangle_Z$。不能先对每个构型除以 N，
也不删去 N=0 的 Z 样本。实现只开放正整数 l，平移不变体系中负模与正模相同。

等时关联的 all-opportunity 向量为
$X=(I_Z,I_ZN,I_ZN(N-1),I_ZC_b, I_ZD_l,\ldots)$。
若其样本均值记为 $(z,n,f,c_b,d_l,\ldots)$，则
$g_{2,b}= (L/\Delta r_b)c_b z/n^2$， $S_l=d_l/n$。
每个 block 先平均完整向量，按
$\mathrm{SE}(F)^2=\nabla F^T\mathrm{Cov}(\bar X_{\rm block})\nabla F/B$
传播所有分子与分母的协方差，B 为完整 blocks 数。
时间片只用于一个构型内的平均，不作为独立 Monte Carlo 样本。
沿用至少 32 blocks 的三个相邻合格层平台门槛；少于 32 个非零贡献机会不发布 SE。
这是渐近 delta method，短链仍可能受比值偏差和未解析慢相关影响。

本次独立理想参考在有限盒中取
$n_p=[e^{\beta(\lambda(2\pi p/L)^2-\mu)}-1]^{-1}$、 $\mu<0$，给出
$g_2(r)=1+|\sum_p n_p e^{i2\pi p r/L}|^2/\langle N\rangle^2$ 和
$S_l=1+\sum_p n_p n_{p+l}/\langle N\rangle$。
对 g2 的每个 Fourier 项进行 bin 积分，核对的是有限-bin 平均而非中心点值。
这里明确使用巨正则独立几何 mode 占据，不将其用于固定 N 的凝聚态参考。

### 13.4 绕数估计量

<a id="eq-winding-response"></a>
对于一个 $d$ 维周期盒，保留的有限尺寸超流响应估计量为

$$
f_s
=\frac{L^2\langle|\mathbf W|^2\rangle_Z}
{2d\lambda\beta\langle N\rangle_Z}.
$$

对于一维环，这是一个有限尺寸响应诊断量。在有限 $L$ 与有限 $T$ 下非零，其本身并不构成热力学极限下有限温度超流的断言。

### 13.5 能量

在巨正则系综中，

$$
\langle H\rangle
=-\left.\frac{\partial\log\Xi}{\partial\beta}\right|_{\mu,L}
+\mu\langle N\rangle.
$$

对于一个对角构型，定义图像分辨的平方链接和

$$
Q_2(C)=\sum_{\ell\in\mathcal L(C)}
\left|\Delta\mathbf r_\ell^{(\mathbf n_\ell)}\right|^2.
$$

在固定 $M$、 $\mu$ 与 $L$ 下，用 $\tau=\beta/M$ 对完整的归一化 primitive 权重求导，得

<a id="eq-thermodynamic-energy-estimator"></a>

$$
K_{\mathrm{th}}(C)
=\frac{dN(C)}{2\tau}
-\frac{Q_2(C)}{4\lambda M\tau^2},
$$

$$
V_{\mathrm{prim}}(C)
=\frac1M\sum_{j=0}^{M-1}V(\mathbf R_j),
\qquad
E_{\mathrm{th}}(C)=K_{\mathrm{th}}(C)+V_{\mathrm{prim}}(C).
$$

$\langle H\rangle$ 定义中的 $+\mu\langle N\rangle$ 项逐构型地消去了 $e^{\beta\mu N}$ 的导数。 $Q_2$ 中的所有位移都使用存储的链接图像；把它们替换成中心化最小图像位移会给出错误的绕数贡献。这些估计量只在扇区 $Z$ 中累积，并使用元数据变体
**primitive_thermodynamic_kinetic** 和 **primitive_thermodynamic_total**。热力学估计量的方差随 $M$ 增大；未来的 centroid-virial 估计量可能改善统计，但不得静默替换此参考公式。

### 13.6 单体密度矩阵

无量纲归一化的单体密度矩阵预留为

$$
g_1(r)
=\frac{\langle\hat\psi^\dagger(r)\hat\psi(0)\rangle}{\rho}.
$$

对于第一个一维实现，把中心化周期区间 $[-L/2,L/2)$ 分成 $B_x$ 个半开 bin

$$
I_a=[x_a,x_{a+1}),\qquad
x_a=-\frac L2+a\Delta x,\qquad
\Delta x=\frac{L}{B_x}.
$$

在每个计划测量机会（包括两个扇区），递增 $K$。令 $K_Z$ 为这些机会中处于扇区 $Z$ 的次数，并令 $H_{s,a}$ 统计满足模正向端点时间分离与中心化端点位移的扇区-$G$ 构型数：

$$
s=(j_{\mathcal I}-j_{\mathcal M})\bmod M,
\qquad
\mathrm{disp}_L(x_{\mathcal I},x_{\mathcal M})\in I_a.
$$

那么 $\widehat h_{s,a}=H_{s,a}/(K\Delta x)$ 且
$\widehat P_Z=K_Z/K$。应用第 10.4 节给出 bin 平均估计量

<a id="eq-green-histogram-estimator"></a>

$$
\widehat G_{M,a}(s\tau)
=\frac{H_{s,a}}
{K_Z\,\Delta x\,C_G M L}.
$$

开放分量中实际链接数被保留为拓扑元数据，但 Matsubara 时间 bin 是模分离 $s$。令

$$
\widehat\rho
=\frac{\sum_{k\in Z}N(C_k)}{K_ZL},
$$

则无量纲单体密度矩阵的有限时间近似为

<a id="eq-g1-beta-minus-estimator"></a>

$$
\widehat g_{1,M,a}^{(\beta^-)}
=\frac{\widehat G_{M,a}((M-1)\tau)}{\widehat\rho}.
$$

这使用玻色时序断开点的 $\beta^-$ 侧。$s=0$ 直方图包含由
$\hat\psi(x)\hat\psi^\dagger(0)$ 隐含的周期接触项，绝不能被重新标记为 $g_1$。
因此输出对所有 $s$ 报告归一化 $G_M$，但对 $g_1$ 只在 $s=M-1$ 时报告，归一化状态为
**beta_minus_finite_tau**。计数与 bin 边界被精确地 checkpoint。在 blocking 实现之前，直方图标准误显式保持不可得，而不是用相关访问伪造出来。

## 14. Monte Carlo 时间、随机数与统计

一个原子 Monte Carlo **步骤（step）** 是一次更新类抽取，紧跟着一次提议尝试，包括在作量计算之前就被拒绝的无效提议。

一个 **sweep** 是用户配置的固定原子步数 **steps_per_sweep**。它不以当前 bead 数来定义，因为在巨正则与 Worm 扇区中 bead 数是变化的。

每次运行记录：

- 随机数生成器算法；
- 输入种子或种子序列；
- warm-up sweeps；
- measurement sweeps；
- steps per sweep；
- measurement stride；
- 每个更新被尝试、结构无效、作量已求值以及被接受的计数；
- $Z$- 与 $G$- 扇区驻留计数；
- 用于报告不确定度的 blocking 或 autocorrelation 设置。

使用相同语言、RNG 实现、种子、编译器/解释器配置与输入的重复运行预期复现相同的流。Python 与 Fortran 要求统计上一致，而不是逐位一致，除非它们有意共享并测试同一 RNG 实现。

### 14.1 条件估计量与二进制 blocking

标量估计量不能删除 $G$ 扇区量测机会后再把剩余值误当作独立样本。令
$D_k=\mathbf 1(C_k\in Z)$，并对 $Z$ 扇区可观测量 $O$ 定义
$X_k^{(O)}=D_k O(C_k)$，则实际输出的条件均值是

$$
\widehat O_Z=\frac{\overline X^{(O)}}{\overline D}.
$$

同理，令 $Y_{k,s,a}$ 表示第 $k$ 次计划量测是否落入 $G$ 扇区的
$(s,a)$ bin，则 Green 与 $g_1$ 都是相关样本均值之比：

$$
\widehat G_{s,a}
=\frac{1}{\Delta x C_GML}\frac{\overline Y_{s,a}}{\overline D},
\qquad
\widehat g_{1,M,a}^{(\beta^-)}
=\frac{1}{\Delta x C_GM}
\frac{\overline Y_{M-1,a}}{\overline{DN}}.
$$

统计实现保留每次计划量测机会，并使用长度
$B=1,2,4,\ldots$ 的连续、非重叠 block。对任一 numerator--denominator
对，在某一 blocking level 上令 block averages 为 $(x_b,d_b)$、
$r_B=\bar x_B/\bar d_B$，并定义线性化残差 $q_b=x_b-r_Bd_b$。该层的
ratio standard error 为

<a id="eq-blocking-ratio-standard-error"></a>

$$
\mathrm{SE}_B(c r_B)
=\frac{|c|}{|\bar d_B|}
\sqrt{\frac{s_q^2}{n_B}},
\qquad
s_q^2=\frac{1}{n_B-1}\sum_{b=1}^{n_B}q_b^2.
$$

每一层至少保留 32 个完整 blocks。只有最粗的三个合格层之方差区间
$\widehat v_B[1\pm2\sqrt{2/(n_B-1)}]$ 有共同交集时，才把其中最大的
standard error 作为相关性修正结果；否则正式误差留空并报告
`insufficient_blocking_levels` 或 `blocking_plateau_not_reached`。这个平台判据是
保守的自动发布 gate，不是 blocking fixed point 的数学证明。
这里采用 Flyvbjerg--Petersen 的 blocking 思路 [4]；Wolff 的
autocorrelation $\Gamma$ method [5] 保留为独立交叉检查，而不是当前发布值的
隐藏替代实现。

对 histogram ratio 还要求 numerator 至少出现 32 次。更稀疏的 bin 仍输出
count、ratio 与全部 blocking moments，但状态为
`insufficient_numerator_events`，不得由零或个位数事件产生渐近误差棒。所有
blocking moments、未配对的尾 block 与发布状态都进入 checkpoint；从旧
checkpoint 续算时，若 measurement 已经开始而历史 moments 不完整，则整条续算的
相关性修正保持不可得。

## 15. 强制 debug 不变式

在 debug 模式中，每个被接受的改变拓扑的更新至少必须检查：

1. 所有活跃 ID 唯一，且所有被引用的 ID 都活跃；
2. $\mathrm{next}(b)=c$ 蕴含 $\mathrm{prev}(c)=b$，反之亦然；
3. 每条被占链接都前进一个切片模 $M$；
4. 每条被占链接都有一个有定义的整数图像；
5. 卷绕坐标位于 $[0,L)$ 内；
6. 端点计数与端点度数匹配当前扇区；
7. 所有活跃 bead 恰好属于一个闭合环或唯一的开放分量；
8. $N_j$ 在 $Z$ 中为常数，在 $G$ 中只在端点切片处改变；
9. 从图像计算的绕数在每个闭合环上都是整数；
10. 缓存中的切片列表与对能量（若存在）与全量重算一致。

不变式失败是程序错误。它必须用一个可复现的构型 dump 停止 debug 运行，而不是被当作普通的被拒绝移动。

## 16. 主要参考文献

1. D. M. Ceperley, “Path integrals in the theory of condensed helium,” *Rev. Mod. Phys.* **67**, 279 (1995). <https://doi.org/10.1103/RevModPhys.67.279>
2. M. Boninsegni, N. Prokof'ev, and B. Svistunov, “Worm Algorithm for Continuous-space Path Integral Monte Carlo Simulations,” *Phys. Rev. Lett.* **96**, 070601 (2006). <https://doi.org/10.1103/PhysRevLett.96.070601>
3. M. Boninsegni, N. V. Prokof'ev, and B. V. Svistunov, “Worm algorithm and diagrammatic Monte Carlo: A new approach to continuous-space path integral Monte Carlo simulations,” *Phys. Rev. E* **74**, 036701 (2006). <https://doi.org/10.1103/PhysRevE.74.036701>
4. H. Flyvbjerg and H. G. Petersen, “Error estimates on averages of correlated data,” *J. Chem. Phys.* **91**, 461--466 (1989). <https://doi.org/10.1063/1.457480>
5. U. Wolff, “Monte Carlo errors with less errors,” *Comput. Phys. Commun.* **156**, 143--153 (2004). <https://doi.org/10.1016/S0010-4655(03)00467-3>
