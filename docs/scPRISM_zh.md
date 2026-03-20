# scPRISM：面向单细胞分子的概率独立信号提取

---

## 1. 符号与生成模型

### 1.1 符号表

| 符号 | 名称 | 定义 | 备注 |
|---|---|---|---|
| $g$ | 基因索引 | $g = 1, \ldots, G$ | |
| $c$ | 细胞索引 | $c = 1, \ldots, C$ | |
| $\mu_{gc} > 0$ | 潜在表达水平 | 细胞 $c$ 中基因 $g$ 的去噪潜在丰度 | 有效信号变量；理论上对应程序性异质性，但工程上会吸收部分不可观测扰动 |
| $\epsilon_c > 0$ | 细胞效率因子 | 细胞 $c$ 的全局乘性偏置 | 对所有基因共享；$\log \epsilon_c \sim \mathcal{N}(\mu_\epsilon, \sigma_\epsilon^2)$ |
| $\epsilon_{gc} > 0$ | 基因-细胞独立扰动 | $(g,c)$ 对上的独立乘性扰动 | $\mathbb{E}[\epsilon_{gc}] = 1$；各 $(g,c)$ 对独立 |
| $r \in (0, 1]$ | 捕获效率 | 分子被实际捕获为 UMI 的全局比例 | 超参数；所有基因和细胞共享 |
| $\Lambda_c$ | 有效采样池 | $\Lambda_c = \epsilon_c \cdot \sum_g \epsilon_{gc} \cdot \mu_{gc}$ | 细胞 $c$ 中经过所有扰动后的总有效分子数 |
| $X_{gc}$ | 观测 UMI 计数 | 最终观测值 | |
| $N_c$ | 总 UMI 计数 | $N_c = \sum_g X_{gc}$ | |
| $F_g$ | 基因先验 | $\mu_{gc} \sim F_g$ | 基因 $g$ 潜在表达的群体级有效先验 |
| $S$ | 采样池标尺 | 全局参照常数，通过 §2 从数据中估计 | 作用是将跨基因耦合的分母替换为已知常数，使各基因解耦 |

### 1.2 跨批次扩展（可选）

当需要进行跨批次或跨物种比较时，可以将 $\mu_{gc}$ 进一步分解为：

| 符号 | 名称 | 定义 |
|---|---|---|
| $p_{gc} \in (0, 1)$ | 相对基因活性 | 由细胞类型 / 状态决定的无量纲活性 |
| $L_g > 0$ | 表达域长度 | 基因 $g$ 表达范围的有效上界 |
| $\tau \in (0, 1)$ | 覆盖阈值 | 控制 $L_g$ 截断点的超参数，例如 $\tau = 0.99$ |

令 $\Phi_g$ 表示 $F_g$ 的 CDF。则 $L_g$ 定义为 $\tau$ 分位数：

$$L_g = \inf\{x : \Phi_g(x) \geq \tau\}$$

从而有 $\mu_{gc} = L_g \cdot p_{gc}$。

**单批次分析不需要这一步分解。**

### 1.3 生成模型

数据生成过程由四层构成：

**生物层（信号来源）：**

$$\mu_{gc} \sim F_g$$

每个基因 $g$ 有独立先验 $F_g$，可以是单峰、多峰、重尾或任意混合分布。理论上，$F_g$ 只描述程序性异质性；但在实际估计中，由于部分非程序性扰动不可分辨，$F_g$ 应理解为对去噪潜在丰度的有效先验（effective prior over denoised latent abundance）。跨基因依赖被留给下游模型处理。

**基因-细胞独立扰动层：**

$$\epsilon_{gc} > 0, \qquad \mathbb{E}[\epsilon_{gc}] = 1, \qquad \text{各 } (g,c) \text{ 对独立}$$

每个 $(g,c)$ 对独立采样一个乘性扰动。$\epsilon_{gc}$ 刻画的是在群体级先验 $F_g$ 和细胞级效率 $\epsilon_c$ 之外，每个基因在每个细胞中各自经历的、具有一定范围的未知扰动。这包括单分子捕获的随机波动、局部反应微环境差异、转录脉冲噪声等非程序性来源。

这里将 $\epsilon_{gc}$ 显式写出，主要是为了严谨分析分母

$$\sum_{g'} \epsilon_{g'c} \mu_{g'c}$$

在大数条件下为何可以稳定到 $S$，从而说明这类局部扰动对全局标尺估计的影响是次阶的。换言之，$\epsilon_{gc}$ 对 **$S$ 的角色** 是“需要被显式分析以证明其近似可消去”；但对 **$F_g$ 的角色** 则不同：由于 $\epsilon_{gc}$ 不可直接观测、影响相对较小，且本框架本身就是工作近似，在实际工程实现中更合理的做法是让其效应吸收到 $\hat{F}_g$ 的 effective prior 中，而不是试图单独恢复一个可辨识的 $\epsilon_{gc}$ 层。

**细胞效率层：**

$$\log \epsilon_c \sim \mathcal{N}(\mu_\epsilon, \sigma_\epsilon^2)$$

$\epsilon_c$ 是细胞层面的全局乘性因子，来源于多个独立乘性过程的乘积（固定效率、透化效率、液滴反应产率等）。它对所有基因共享。

**观测层（捕获 + 组成型抽样）：**

经过所有扰动后，细胞 $c$ 的有效采样池为：

$$\Lambda_c = \epsilon_c \cdot \sum_g \epsilon_{gc} \cdot \mu_{gc}$$

测序过程从 $\Lambda_c$ 中以比例 $r$ 进行捕获，引入 Poisson 采样：

$$N_c \sim \mathrm{Poisson}(r \cdot \Lambda_c)$$

在给定 $N_c$ 的条件下，这 $N_c$ 个 UMI 按各基因有效丰度的相对比例分配。单基因的条件边缘为：

$$X_{gc} \mid N_c \sim \mathrm{Binomial}\!\left(N_c,\; \frac{\epsilon_{gc} \cdot \mu_{gc}}{\sum_{g'} \epsilon_{g'c} \cdot \mu_{g'c}}\right)$$

这是一个刻意简化的工作观测模型（deliberately simplified observation model）：真实 UMI 数据中可能存在额外的 overdispersion、零膨胀、gene-specific capture bias 等效应。本文的立场是，$\epsilon_{gc}$ 的显式引入主要服务于对 $S$ 的理论分析；而在实际推断中，这些局部复杂性并不需要被逐项恢复，它们最终会体现在 effective prior $F_g$ 的形状以及后验不确定性中。完成 nuisance 消去后，Binomial 是用于构建 per-gene 推断接口的主近似，而不是对实验物理过程的完整复刻。

### 1.4 三重消去

观测方程中包含三个 nuisance 量：$r$、$\epsilon_c$、$\epsilon_{gc}$。它们通过不同机制被消除：

**$r$ 的消去（精确）。** 捕获效率 $r$ 只出现在 $N_c$ 的生成过程中。条件于 $N_c$ 后，$r$ 完全不出现在 Binomial 的参数里。

**$\epsilon_c$ 的消去（精确）。** $\epsilon_c$ 在 Binomial 比例的分子分母中相互抵消。$\epsilon_c$ 的全部影响已被编码进 $N_c$ 的大小中。

**$\epsilon_{gc}$ 的消去（大数近似）。** 分母 $\sum_{g'} \epsilon_{g'c} \cdot \mu_{g'c}$ 是大量独立扰动的加权和。由大数定律：

$$\sum_{g'} \epsilon_{g'c} \cdot \mu_{g'c} \approx \sum_{g'} \mu_{g'c} \approx S$$

第一个近似消去了 $\epsilon_{gc}$，第二个近似是大量独立基因求和后各细胞总转录组大小趋于稳定。两步联合精度为 $O(G_{\mathrm{eff}}^{-1/2})$（§1.5）。

分子中该基因自己的 $\epsilon_{gc}$ 在理论上仍然残留——我们实际观测的是带有局部乘性扰动的信号经过采样后的结果。对方法实现而言，这并不意味着要单独恢复一个 $\epsilon_{gc}$；更实际的理解是：这类未显式建模的局部扰动最终会表现为 effective noise，并体现在 $\hat{F}_g$ 的形状和后验分布 $p(\mu_{gc} \mid X_{gc}, N_c, S, F_g)$ 的宽度中。

**消去后的简化方程：**

$$X_{gc} \mid N_c \;\approx\; \mathrm{Binomial}\!\left(N_c,\; \frac{\mu_{gc}}{S}\right), \qquad \mu_{gc} \sim F_g$$

| 因子 | 性质 | 消去机制 | 精度 |
|---|---|---|---|
| $r$ | 全局常数 | 条件于 $N_c$ 后不出现 | 精确 |
| $\epsilon_c$ | 跨基因共享 | 分子分母抵消 | 精确 |
| $\epsilon_{gc}$（分母） | 每个 $(g,c)$ 独立 | 大数定律 | $O(G_{\mathrm{eff}}^{-1/2})$ |
| $\epsilon_{gc}$（分子） | 同上 | 不消去；残留为不可约噪声 | — |

### 1.5 采样池稳定性：大数定律分析

§1.4 的核心近似是 $\sum_{g'} \epsilon_{g'c} \cdot \mu_{g'c} \approx S$。本节量化精度。

由于 $\epsilon_{gc}$（均值 1、独立）和 $\mu_{gc} \sim F_g$（各基因独立采样）是两层独立的波动来源，分母的 CV 为二者的平方和开根号。引入表达占比 $\phi_g = \mu_{gc} / \sum_{g'} \mu_{g'c}$ 和有效基因数 $G_{\mathrm{eff}} = 1 / \sum_g \phi_g^2$，设 $\epsilon_{gc}$ 的对数方差为 $\sigma_\epsilon^2$（简化为全局常数），$\mu_{gc}$ 围绕基因均值的对数方差为 $\sigma_g^2$，则：

$$\mathrm{CV}^2\!\left(\sum_{g'} \epsilon_{g'c} \cdot \mu_{g'c}\right) \;\approx\; \frac{\langle e^{\sigma_\epsilon^2} - 1 \rangle_\phi + \langle e^{\sigma_g^2} - 1 \rangle_\phi}{G_{\mathrm{eff}}}$$

在典型参数下（$\sigma_\epsilon \lesssim 0.5$，$\sigma_g \lesssim 1$，$G_{\mathrm{eff}} \geq 500$），联合 CV 不超过 $\sim 10\%$，相比 $\epsilon_c$ 的 50%–300% 变异可忽略。

这一估计基于跨基因独立的假设。共享转录程序等会引入正相关，使 CV 偏乐观。即使高出数倍，相对于 $\epsilon_c$ 仍然很小。

**退化条件。** 当少数基因主导总表达时（如线粒体基因 $> 30\%$），$G_{\mathrm{eff}}$ 急剧下降，近似变差。

---

## 2. 采样池标尺估计

### 2.1 目标

由 §1.4 的简化方程：

$$X_{gc} \mid N_c \approx \mathrm{Binomial}\!\left(N_c,\; \frac{\mu_{gc}}{S}\right), \qquad \mu_{gc} \sim F_g$$

$S$ 是所有基因和细胞共享的全局常数。一旦 $S$ 被确定，各基因完全解耦。本步骤的目标：从 $\{N_c\}_{c=1}^C$ 和超参 $r$ 估计 $S$。

### 2.2 模型

由 §1.3 的观测层：

$$N_c \sim \mathrm{Poisson}(r \cdot \Lambda_c) \approx \mathrm{Poisson}(r \cdot S \cdot \epsilon_c)$$

定义 $\eta_c = rS \cdot \epsilon_c$，则：

$$N_c \sim \mathrm{Poisson}(\eta_c), \qquad \log \eta_c \sim \mathcal{N}(\mu, \sigma^2)$$

其中 $\mu = \log(rS) + \mu_\epsilon$，$\sigma = \sigma_\epsilon$。这是标准 **Poisson-LogNormal** 混合模型，未知量只有 $(\mu, \sigma)$ 两个标量。

### 2.3 EM 算法

将 $\eta_c$ 视为潜变量，通过 EM 估计 $(\mu, \sigma)$。

**E 步。** 对每个细胞 $c$，计算一维后验：

$$p(\eta_c \mid N_c, \mu, \sigma) \propto \mathrm{Poisson}(N_c \mid \eta_c) \cdot \mathrm{LogNormal}(\eta_c \mid \mu, \sigma^2)$$

所需的矩 $\mathbb{E}[\log \eta_c \mid N_c]$ 与 $\mathbb{E}[(\log \eta_c)^2 \mid N_c]$ 通过 Gauss-Hermite 求积高效计算。

**M 步。** 更新参数：

$$\mu \leftarrow \frac{1}{C} \sum_c \mathbb{E}[\log \eta_c \mid N_c]$$

$$\sigma^2 \leftarrow \frac{1}{C} \sum_c \mathrm{Var}[\log \eta_c \mid N_c] + \mathrm{Var}_c\!\left(\mathbb{E}[\log \eta_c \mid N_c]\right)$$

迭代直至收敛。参数空间只有二维，收敛表现稳定。

### 2.4 提取 $S$

EM 收敛后得到 $(\hat{\mu}, \hat{\sigma})$。当前代码默认取：

$$\widehat{rS} = \exp(\hat{\mu}), \qquad \hat{S} = \frac{\widehat{rS}}{r}$$

选择 $\exp(\hat{\mu})$（LogNormal 中位数）对应 $\mathrm{median}(\epsilon_c) = 1$ 的约定。其他选择（均值、众数）的差异在 $\sim 20\%$ 以内，被 $r$ 的调节吸收。当前 `prism.model.fit_pool_scale` 还提供一个**可选消融**：先聚合群体 posterior，再用 softargmax 产生代表性 `point_mu`；但默认实现仍使用 $\hat{\mu}$ 本身，因此 $\hat{S}$ 默认就是标量 $\exp(\hat{\mu})/r$。

### 2.4.1 可识别性与标度约定

需要强调：从 $\mu = \log(rS) + \mu_\epsilon$ 出发，数据直接识别的是 $rS$ 与 $\epsilon_c$ 标度约定的组合，而不是某个脱离约定的“真实物理 $S$”。因此，$S$ 只能在一个 gauge fixing 约定下被定义。

本文采用的约定是：

$$\mathrm{median}(\epsilon_c) = 1 \quad \Longleftrightarrow \quad \mu_\epsilon = 0$$

在此约定下，$\exp(\hat{\mu})$ 被解释为 $\widehat{rS}$，再由 $r$ 提取 $\hat{S}$。若改用 $\mathbb{E}[\epsilon_c] = 1$ 或其他等价标度约定，得到的 $S$ 数值会整体缩放，但这种缩放会被 $r$ 吸收，最终的 per-gene Binomial 方程和后验接口语义不变。因此，本文中的 $S$ 应理解为规范化后的 effective pool scale，而不是由数据唯一识别出的原始物理量。

### 2.5 解耦后的 per-gene 方程

$$X_{gc} \mid N_c \sim \mathrm{Binomial}\!\left(N_c,\; \frac{\mu_{gc}}{\hat{S}}\right), \qquad \mu_{gc} \sim F_g$$

- $N_c$：已知观测值（每个细胞不同）
- $\hat{S}$：全局常数（所有基因所有细胞共享）
- $\mu_{gc}$, $F_g$：待估量

**所有基因完全解耦。** $r$ 通过 $\hat{S} = \widehat{rS}/r$ 控制分母大小，决定了先验和似然在后验中的相对权重。

**网格范围。** $N_{gc}^{\mathrm{eff}} = (X_{gc}/N_c) \cdot \hat{S}$，$S_{\mathrm{eff}} = \max_c N_{gc}^{\mathrm{eff}}$。由于 $X_{gc} \leq N_c$，有 $N_{gc}^{\mathrm{eff}} \leq \hat{S}$，不存在网格越界问题。

### 2.6 适用条件与失效模式

分母稳定近似

$$\sum_{g'} \epsilon_{g'c} \cdot \mu_{g'c} \approx S$$

是整个解耦链条的核心适用条件。它依赖于足够大的有效基因数 $G_{\mathrm{eff}}$、有限的跨基因相关性，以及不存在少数基因长期主导总表达。以下情形会显著削弱该近似：

- 高线粒体占比或核糖体基因占比，使少数基因主导总表达
- 强细胞周期、stress response、interferon response 等共享程序，使跨基因正相关升高
- 极端细胞类型混合或稀有亚群占比很高，使“稳定总池”假设变差
- 少数超高表达 marker 或 burst-like 基因长期占据大部分分子预算
- 跨物种、跨平台或强 batch 场景中，$\epsilon_c$ 与组成型偏差可能不再被单个全局标尺充分吸收

这些场景下，$S$ 更应被解释为近似有效标尺，Signal / Confidence / Surprisal 可能出现系统偏差。实践上应结合 QC 指标（如 mt%）、$N_c$ 分布、以及模拟或留出验证来判断该近似是否可接受。

---

## 3. 基因先验估计

### 3.1 目标与范围

对每个基因 $g$，从观测 $\{(X_{gc}, N_c)\}_{c=1}^C$ 和全局常数 $\hat{S}$ 估计群体级分布 $F_g$。这里的 $F_g$ 应理解为潜在表达量的有效群体先验，而不是可与所有实验扰动完全分离的“纯生物真值分布”。

本框架是**逐基因的边际信号模型**，不建模跨基因依赖。其目的是为下游模型提供信号接口层。

### 3.2 非参数离散表示

$F_g$ 表示在 $M$ 个等距网格点上（当前代码默认 $M = 512$）：

$$0 \leq v_1 < v_2 < \cdots < v_M \leq S_{\mathrm{eff}}$$

其中 $S_{\mathrm{eff}} = \max_c (X_{gc}/N_c) \cdot \hat{S}$。由于 $X_{gc} \leq N_c$，有 $S_{\mathrm{eff}} \leq \hat{S}$，Binomial 参数 $v_j/\hat{S} \leq 1$ 自动满足，不存在网格越界问题。

**参数化。** $M$ 个无约束 logit $\{l_j\}_{j=1}^M$ 经过两步映射得到 $F_g$：

**Stage A：Softmax 归一化。**

$$w_j = \frac{\exp(l_j)}{\sum_k \exp(l_k)}$$

**Stage B：高斯卷积平滑。**

$$\tilde{w}_j = \sum_{k=1}^{M} w_k \; G_\sigma(v_j - v_k), \qquad G_\sigma(x) = \exp\!\left(-\frac{x^2}{2\sigma^2}\right)$$

边缘处理使用 replicate padding（边缘外的值等于边缘值），这是图像处理中的标准做法。由于高斯卷积保持均值不变，且边缘位置的概率质量通常接近零，卷积后无需重归一化。若确实需要严格归一化，可使用环状卷积（circular padding）代替重归一化，因为重归一化会引入梯度不稳定。

$\tilde{w}$ 就是 $\hat{F}_g$——不存在任何"逆变换"步骤，卷积后的分布就是最终的先验估计。

**超参 $\sigma$。** 以网格间距 $\Delta v$ 为单位，$\sigma_{\mathrm{bins}}$ 是高斯核的标准差。当前代码默认值是 $\sigma_{\mathrm{bins}} = 1$。这对应较弱的构造性平滑：锯齿会被抑制，但窄峰不会被非常激进地抹平。更大的平滑核（例如 $\sigma_{\mathrm{bins}} = 5$）仍然是合理的**可选消融**，对应更强的低通约束；但它不再是当前实现的默认设置。

**为什么用卷积而不是惩罚项？** 二阶差分惩罚 $\sum_j(w_{j-1} - 2w_j + w_{j+1})^2$ 是软约束，优化器可以牺牲平滑性换取似然。卷积是构造性保证——无论 logit 如何变化，$\tilde{w}$ 都不会出现锯齿。代价是偏差-方差权衡：$\sigma$ 过大会抹掉真实窄峰。

### 3.3 似然计算与分块训练

对每个细胞 $c$ 和每个网格点 $v_j$，Binomial 似然为：

$$\mathrm{lik}_{cj} = \mathrm{Binomial}(X_{gc} \mid N_c, \; v_j / \hat{S})$$

在理论上，由于 $\hat{S}$ 是全局常数，$X_{gc}$ 和 $N_c$ 在训练过程中不变，整个似然矩阵 $\mathrm{lik} \in \mathbb{R}^{C \times M}$ 可以在训练开始前一次性计算并缓存。

但当前 `PriorEngine` 的最终实现默认采用**按细胞分块（cell chunking）即时计算**：每次迭代只对一个 cell slice 计算对应的 `log_lik`，以降低大数据集上的峰值内存占用，并支持多基因批量训练。也就是说，预计算缓存更适合作为参考实现或可选工程变体，而不是当前主代码路径的默认策略。

实际存储的是对数似然 $\log \mathrm{lik}_{cj}$，所有后续计算在 log 空间中进行以保证数值稳定。

每个细胞的似然向量形状类似一个钟形：在 $v_j \approx (X_{gc}/N_c) \cdot \hat{S}$ 附近最高，往两边衰减。$N_c$ 大的细胞钟形窄（信息多），$N_c$ 小的细胞钟形宽（信息少）。但所有细胞的钟形都定义在同一个网格上、指向同一个信号空间——这就是为什么群体级先验 $F_g$ 能起到去噪作用。

### 3.4 损失函数

总损失由两项组成：

$$\mathcal{L} = \lambda_{\mathrm{nll}} \cdot \mathcal{L}_{\mathrm{NLL}} + \lambda_{\mathrm{align}} \cdot \mathcal{L}_{\mathrm{align}}$$

无密度加权，无空间变换。$\tilde{w}$ 就是拟合目标，也是最终输出。

#### 项一：负对数似然（NLL）

对每个细胞 $c$，边际似然为先验和似然的加权求和：

$$d_c = \sum_{j=1}^{M} \tilde{w}_j \cdot \mathrm{lik}_{cj}$$

NLL 为所有细胞的均值：

$$\mathcal{L}_{\mathrm{NLL}} = -\frac{1}{C} \sum_{c=1}^{C} \log d_c$$

这是标准的经验贝叶斯 MLE 目标。每个观测 $(X_{gc}, N_c)$ 对 NLL 等权贡献。在模型设定充分接近真实数据、且样本量足够大时，MLE 提供一致的恢复基础；但在有限样本、模型失配或强噪声条件下，稀有结构是否能被恢复仍取决于信噪比、网格分辨率和正则化强度，不能仅由渐近性质保证。

#### 项二：后验-先验对齐（JSD 正则化）

**动机。** NLL 在高噪声或稀疏数据下可能对 $F_g$ 约束不足。JSD 对齐提供额外的自洽性正则：如果 $\tilde{w}$ 是合理的先验，那用它做后验推断再汇总，得到的群体决策分布不应与 $\tilde{w}$ 相差过大。这个机制有潜力保护受到少量细胞支持的稀有状态，但它不是无条件成立的 theorem；在低覆盖或多峰场景下，若初始化或当前迭代中出现窄小伪峰，JSD 也可能通过正反馈强化这些结构。因此它应被视为可调正则化项，并通过 $\sigma_{\mathrm{bins}}$、$\lambda_{\mathrm{align}}$ 与消融实验共同校准。

**构造四步：**

**第一步：后验计算。** 对每个细胞 $c$，用当前 $\tilde{w}$ 作为先验，通过 Bayes 公式计算后验：

$$p(v_j \mid X_{gc}) = \frac{\tilde{w}_j \cdot \mathrm{lik}_{cj}}{\sum_k \tilde{w}_k \cdot \mathrm{lik}_{ck}}$$

实现上在 log 空间完成：$\log p_j = \log \tilde{w}_j + \log \mathrm{lik}_{cj} - \mathrm{logsumexp}$。在当前主实现里，这一步通常和 cell chunking 结合：对每个 chunk 做广播加法和归一化，再把结果沿细胞维聚合。

$F_g$ 在这里的作用是**调制似然**。一个测到 $X_{gc} = 0$ 的细胞，如果先验在 $\mu = 50$ 处有峰，后验不会完全塌到 $\mu = 0$，而是在 $\mu = 0$ 和 $\mu = 50$ 之间形成折衷——先验利用群体信息把噪声观测拉向更合理的位置。

**第二步：默认汇总方式。** 当前代码默认不先做每细胞硬决策，而是直接将完整后验在细胞维求平均，得到后验对齐分布：

$$\hat{Q}_g(v_j) = \frac{1}{C} \sum_{c=1}^{C} p(v_j \mid X_{gc})$$

这对应 `posterior average`。它直接聚合完整后验分布，避免了硬决策的信息丢失；代价是会显式保留单细胞后验宽度，因此通常比 MAP-histogram 更平滑，也更容易把稀疏或窄峰摊宽。

**第三步：可选点决策 / MAP histogram 消融。** 文档早期版本中的默认做法，是先对每个细胞取后验 MAP：

$$\hat{\mu}_{gc} = v_{j^*}, \qquad j^* = \arg\max_j \; p(v_j \mid X_{gc})$$

若采用这一路径，默认使用 $T = 0$（MAP，最硬的决策）。所有决策值精确落在网格点上。

然后再汇总为后验对齐分布 $\hat{Q}_g$：

MAP-histogram 消融是将所有细胞的 MAP 决策汇总成网格上的直方图。由于 $T = 0$ 时决策值就是网格点，直接统计频率即可：

$$\hat{Q}_g(v_j) = \frac{1}{C} \sum_{c=1}^{C} \mathbf{1}[\hat{\mu}_{gc} = v_j]$$

MAP histogram 不再是当前主实现的默认设置，而应视为**可选消融**。它的优点是语义更接近“群体中各离散状态的占比”，通常更利于保留稀疏峰；代价是更硬、更依赖局部决策稳定性，也更容易受早期伪峰影响。

若使用 soft-argmax（$T > 0$），决策值可能不在网格点上，此时用线性插值法：对每个 $\hat{\mu}_{gc}$ 落在 $v_j \leq \hat{\mu}_{gc} < v_{j+1}$ 之间的，按距离比例将质量分配到左右两个网格点：

$$\hat{Q}_g[j] \mathrel{+}= \frac{v_{j+1} - \hat{\mu}_{gc}}{\Delta v}, \qquad \hat{Q}_g[j+1] \mathrel{+}= \frac{\hat{\mu}_{gc} - v_j}{\Delta v}$$

最后归一化。线性插值比直接 round 到最近网格点精度高一倍。

**第四步：Jensen-Shannon 散度。**

$$\mathcal{L}_{\mathrm{align}} = \mathrm{JSD}(\hat{Q}_g \| \tilde{w}) = \frac{1}{2}\mathrm{KL}(\hat{Q}_g \| \bar{M}) + \frac{1}{2}\mathrm{KL}(\tilde{w} \| \bar{M}), \qquad \bar{M} = \frac{1}{2}(\hat{Q}_g + \tilde{w})$$

$\hat{Q}_g$ 在每步迭代开始时重新计算（用当前 $\tilde{w}$ 做后验），不参与梯度。梯度只通过 $\tilde{w}$ 回传到 logit。这形成了类似 EM 的交替结构：固定 $\hat{Q}_g$（E-step），优化 $\tilde{w}$（M-step）。当前主实现里，`posterior average` 与这一交替结构配合构成默认训练路径；MAP histogram 与 soft-decision 都应视为对齐项构造的可选消融。

**JSD 对稀有信号的作用。** 稀有态在数据中只有少量细胞支持。NLL 中它们的梯度贡献通常较小，而 JSD 可能提供额外的群体级稳定作用：当某个稀有峰确实得到局部观测支持时，后验对齐分布会在该区域汇聚，从而减弱优化过程中该峰被完全抹平的趋势。需要同时强调的是，这种机制并不自动区分“真实稀有峰”和“初始化或噪声诱发的小峰”；在低覆盖和多峰先验下，JSD 也可能放大伪结构。因此，关于“保护稀有信号”的说法应理解为经验假设，需通过 ablation 和 simulation 验证，而不是先验保证。

### 3.5 总损失与输出

$$\mathcal{L} = \lambda_{\mathrm{nll}} \cdot \mathcal{L}_{\mathrm{NLL}} + \lambda_{\mathrm{align}} \cdot \mathcal{L}_{\mathrm{align}}$$

$\tilde{w}$ 就是最终的 $\hat{F}_g$。不存在逆变换。

其中，$\mathcal{L}_{\mathrm{NLL}}$ 是主数据项，$\mathcal{L}_{\mathrm{align}}$ 是可选的自洽性正则。实践上建议将 NLL-only 作为基线，并用消融实验评估 JSD 项对稀有峰恢复、过平滑和伪峰强化的影响。

### 3.6 优化

- **优化器：** AdamW + 余弦学习率调度
- **批模式：** 默认按基因批量训练，并在细胞维使用 chunked full-batch；每次优化遍历全部细胞，但计算按 `cell_chunk_size` 分块完成
- **初始化：** 对未拟合基因，先用均匀先验做一轮 posterior-average 聚合，得到 `posterior-generated distribution`，再将其反解析为 logits 作为 warm start；`init_temperature` 默认为 1.0
- **似然计算：** 默认按 chunk 即时计算 `log_lik`，不预缓存完整矩阵

| 超参 | 符号 | 作用 | 默认值 |
|---|---|---|---|
| 网格大小 | $M$ | $F_g$ 分辨率 | 512 |
| 卷积标准差 | $\sigma_{\mathrm{bins}}$ | 构造性平滑强度（网格间距单位） | 1.0 |
| 对齐分布 | — | JSD 中 $\hat{Q}_g$ 的默认构造 | posterior average |
| 决策温度 | $T$ | MAP / soft-decision 消融中的后验点决策锐度 | 默认主实现未使用 |
| NLL 权重 | $\lambda_{\mathrm{nll}}$ | 数据保真项权重 | 1.0 |
| 对齐权重 | $\lambda_{\mathrm{align}}$ | JSD 自洽性权重 | 1.0 |
| 学习率 | `lr` | AdamW 初始学习率 | 0.05 |
| 迭代次数 | `n_iter` | 优化预算 | 100 |
| 学习率下界比例 | `lr_min_ratio` | 余弦调度终点 / 初始 lr | 0.1 |
| 初始化温度 | `init_temperature` | 冷启动 posterior 聚合温度 | 1.0 |
| 细胞分块大小 | `cell_chunk_size` | 每次 likelihood 计算的 cell chunk 大小 | 512 |
| 捕获效率 | $r$ | 通过 $\hat{S} = \widehat{rS}/r$ 控制 Binomial 似然宽度 | 0.02–0.1 |

---

## 4. 概率信号接口

### 4.1 后验推断

给定估计先验 $\hat{F}_g$（即 §3 输出的 $\tilde{w}$）和全局常数 $\hat{S}$，对每个细胞-基因对 $(g, c)$ 计算离散网格上的完整后验分布：

$$p(v_j \mid X_{gc}, N_c, \hat{S}, \hat{F}_g) = \frac{\tilde{w}_j \cdot \mathrm{Binomial}(X_{gc} \mid N_c, v_j/\hat{S})}{\sum_k \tilde{w}_k \cdot \mathrm{Binomial}(X_{gc} \mid N_c, v_k/\hat{S})}$$

由于当前实现通常采用 cell chunking，此步在算法上仍是同一个 Bayes 更新，但工程上会按 cell slice 分块执行后验计算与聚合。

从后验中提取三个核心指标和一个可选指标：

- **观测 → 信号映射：** 观测能否被稳定地映射到一个信号值（Signal, Confidence）
- **信号 → 群体语义：** 该信号在基因表达分布中的位置与结构（Surprisal, Sharpness）

### 4.2 Signal

$$s_{gc} = v_{j^*}, \qquad j^* = \arg\max_j \; p(v_j \mid X_{gc})$$

后验 MAP 估计——后验概率最高的那个网格点。这是去噪后的表达量，交付给下游模型的主信号。

Signal 回答：**在当前先验和观测下，这个细胞-基因对最合理的潜在表达量是什么？**

Signal 的去噪机制来自先验 $\tilde{w}$ 对似然的调制。$N_c$ 大的细胞似然窄，后验接近似然峰值（Signal ≈ $N_{\mathrm{eff}}$）；$N_c$ 小的细胞似然宽，后验被先验主导，Signal 被拉向群体中的高概率状态。这就是经验贝叶斯收缩——信息弱的观测借用群体信息去噪。

*约定：* 当前 `Posterior` / `SignalExtractor` 主接口默认并直接输出 MAP Signal。soft-argmax（$T > 0$）可作为参考实现或消融版本，但不是当前包默认暴露的信号定义。

### 4.3 Confidence

$$\mathrm{Conf}_{gc} = 1 - \frac{H\!\left[p(v_j \mid X_{gc})\right]}{\log M}$$

其中 $H[\cdot]$ 是后验分布的 Shannon 熵，$\log M$ 是最大可能熵（均匀分布）。$\mathrm{Conf} \in [0, 1]$：

- $\mathrm{Conf} = 1$：后验是 delta 函数，观测完全确定了信号
- $\mathrm{Conf} = 0$：后验是均匀分布，观测没有提供超出先验的任何信息

Confidence 回答：**观测 $(X_{gc}, N_c)$ 在当前先验下，为确定 $\mu_{gc}$ 提供了多少信息？**

Confidence 主要受 $N_c$ 控制：深测序细胞的 Binomial 似然窄，后验集中，Confidence 高；浅测序细胞似然宽，后验分散，Confidence 低。先验形状也有影响——多峰先验可能产生多峰后验，即使在中等深度下也会降低 Confidence。

$\log M$ 归一化使 Confidence 在不同网格范围 $S_{\mathrm{eff}}$ 的基因之间可比。

### 4.4 Surprisal

$$\mathrm{Surprisal}_{gc} = -\log \tilde{w}_{j^*}$$

其中 $j^*$ 是 Signal 对应的网格点（MAP 位置）。这是信号值在先验 $\hat{F}_g$ 下的负对数概率——先验在该位置分配的概率越低，Surprisal 越高。

Surprisal 回答：**这个信号状态在该基因的群体分布中有多罕见？**

Surprisal 和 Confidence 正交：

| | 高 Confidence | 低 Confidence |
|---|---|---|
| **高 Surprisal** | 可信的稀有状态（如激活亚群 marker） | 观测模糊，稀有性不确定 |
| **低 Surprisal** | 常见状态，判定清晰 | 常见区域，但观测信息不足 |

高 Confidence + 高 Surprisal 的组合是下游稀有态发现和异常检测中最有价值的信号。

**可选归一化。** 为跨基因可比，可归一化到 $[0, 1]$：

$$\mathrm{Surprisal}_{gc}^{\mathrm{norm}} = \frac{-\log \tilde{w}_{j^*}}{\max_j(-\log \tilde{w}_j)}$$

### 4.5 Sharpness（可选）

$$\mathrm{Sharpness}_{gc} = -\bigl(\log(\tilde{w}_{j^*-1}+\varepsilon) - 2\log(\tilde{w}_{j^*}+\varepsilon) + \log(\tilde{w}_{j^*+1}+\varepsilon)\bigr)$$

离散二阶差分近似对数密度的局部曲率 $-d^2/ds^2 \log f_g(s)$。高斯卷积保证 $\tilde{w}$ 光滑，$\varepsilon$ 用于避免尾部极小概率导致的数值不稳定。边缘网格点（$j^* = 1$ 或 $j^* = M$）取邻近内部值。

Sharpness 回答：**这个信号位于一个尖锐的窄峰上，还是一个宽泛的平坦区域？**

Sharpness 与 Surprisal 的区分：

- 常见状态可以位于尖锐峰上（管家基因的紧密调控——低 Surprisal，高 Sharpness）
- 稀有状态可以位于宽泛尾部（高 Surprisal，低 Sharpness）

Sharpness 标记为可选，因为对多数单峰基因它与 Surprisal 高度相关。主要价值在多峰基因中，用于区分离散激活态（窄峰）和连续分化梯度（宽分布）。

### 4.6 下游使用

| 指标 | 状态 | 范围 | 用途 |
|---|---|---|---|
| **Signal** | 核心 | $[0, S_{\mathrm{eff}}]$ | 下游模型主输入（去噪表达矩阵） |
| **Confidence** | 核心 | $[0, 1]$ | 样本筛选、损失加权、不确定性感知训练 |
| **Surprisal** | 核心 | $[0, +\infty)$ 或 $[0, 1]$ | 异常检测、稀有亚群发现 |
| **Sharpness** | 可选 | $[0, +\infty)$ | 离散态 vs 连续态区分 |

**典型使用模式：**

- **去噪：** 直接用 Signal 矩阵替代 raw UMI 矩阵
- **不确定性感知自监督学习：** 用 Confidence 加权训练损失，高置信样本贡献更大
- **稀有态发现：** 按 Surprisal 排序（或 Surprisal × Confidence 做联合筛选），阈值化识别候选
- **离散 vs 连续状态标注：** 用 Sharpness 区分尖峰离散态和宽泛连续态，辅助 GRN 推断中的因果锚点选择
