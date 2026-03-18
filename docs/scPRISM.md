# scPRISM: Probabilistic Independent Signal Extraction for Single-Cell Molecules

## 1. 符号定义与生成模型

### 1.1 符号总表

| 符号 | 名称 | 定义 | 备注 |
|---|---|---|---|
| $g$ | 基因索引 | $g = 1, \ldots, G$ | |
| $c$ | 细胞索引 | $c = 1, \ldots, C$ | |
| $\mu_{gc} > 0$ | 真实基因表达量 | 细胞 $c$ 中基因 $g$ 的真实mRNA分子数 | 核心信号，不可直接观测 |
| $\epsilon_c > 0$ | 细胞效率因子 | 细胞 $c$ 的全局乘性偏差（捕获/裂解/文库效率） | 所有基因共享，$\log \epsilon_c \sim \mathcal{N}(\mu_\epsilon, \sigma_\epsilon^2)$ |
| $\tilde{\mu}_{gc}$ | 扰动后表达量 | $\tilde{\mu}_{gc} = \epsilon_c \cdot \mu_{gc}$ | 实际暴露给采样过程的分子数 |
| $\Lambda_c$ | 有效采样池 (effective sampling pool) | $\Lambda_c = \sum_g \tilde{\mu}_{gc} = \epsilon_c \sum_g \mu_{gc}$ | 细胞 $c$ 中所有基因扰动后分子总数 |
| $X_{gc}$ | 观测UMI count | 最终观测值 | |
| $N_c$ | 总UMI数 | $N_c = \sum_g X_{gc}$ | |
| $F_g$ | 基因底层分布 (gene prior) | $\mu_{gc} \sim F_g$ | 基因 $g$ 的表达量在细胞群体中的分布 |

### 1.2 跨批次扩展符号（可选）

当需要跨批次或跨物种比较时，可将 $\mu_{gc}$ 进一步分解：

| 符号 | 名称 | 定义 |
|---|---|---|
| $p_{gc} \in (0, 1)$ | 相对基因活性 (relative gene activity) | 归一化活性，由细胞类型/状态等程序性因素决定 |
| $L_g > 0$ | 表达域长 (expression domain length) | 基因 $g$ 在该体系中可达的最大真实分子数 |

此时 $\mu_{gc} = L_g \cdot p_{gc}$。$L_g$ 吸收了基因长度、启动子强度、mRNA稳定性等基因特异性标度，$p_{gc}$ 成为跨基因可比的无量纲活性。

**单批次分析时不需要这一分解**，直接使用 $\mu_{gc}$ 和 $F_g$ 即可。$p_{gc}$ / $L_g$ 的引入仅在需要跨批次对齐或跨物种比较时才有意义——它将批次间共享的"相对调控程序"与批次特异的"绝对标度"显式分离。

### 1.3 生成模型

完整的数据生成过程分为三层：

**生物层（信号源）：**

$$\mu_{gc} \sim F_g$$

每个基因 $g$ 拥有独立的底层分布 $F_g$，可为单峰、双峰、重尾或任意混合形态。不同基因的 $F_g$ 之间独立——基因间调控关系由下游模型（如GRN推断）处理。

**噪声层（细胞效率）：**

$$\log \epsilon_c \sim \mathcal{N}(\mu_\epsilon, \sigma_\epsilon^2)$$

$\epsilon_c$ 是细胞级别的全局乘性因子，来源于固定效率、透化程度、液滴内反应效率等多个独立乘性过程的乘积，对数空间近似正态。

**观测层（伯努利采样）：**

$$X_{gc} \mid N_c \sim \text{Binomial}\left(N_c,\ \frac{\tilde{\mu}_{gc}}{\Lambda_c}\right) = \text{Binomial}\left(N_c,\ \frac{\mu_{gc}}{\sum_{g'} \mu_{g'c}}\right)$$

注意 $\epsilon_c$ 在分子分母中约去——条件于 $N_c$ 时，观测比例只取决于各基因的相对表达量。

### 1.4 基因特异性固定因子（不纳入模型）

Random hexamer偏好性、rRNA去除附带损伤、基因长度效应、甲醛交联偏好性等因素，对每个基因施加一个跨细胞恒定的乘性常数 $\alpha_g$。由于 $\alpha_g$ 在所有细胞中取值相同，在细胞间比较时自动消除，无需建模或校正。

---

## 2. 采样池估计 (Pool Estimation)

### 2.1 动机

Per-gene采样方程的分母 $\sum_{g'} \mu_{g'c}$ 是未知的，且耦合了所有基因。如果不先处理这个耦合项，每个基因的 $F_g$ 无法独立估计。

我们引入**有效采样池大小** $\Lambda_c$ 作为关键中间量。通过total count $N_c$ 预先估计 $\Lambda_c$，从而将基因间解耦。

### 2.2 模型

定义**群体采样池均值**：

$$S = \sum_{g'} \overline{\mu}_{g'} \quad \text{其中} \quad \overline{\mu}_{g'} = \mathbb{E}_c[\mu_{g'c}]$$

$S$ 是所有基因群体平均表达量之和，反映该细胞群体的"平均转录组大小"。

**核心近似：** 由于 $\Lambda_c = \epsilon_c \cdot \sum_g \mu_{gc}$，而 $\sum_g \mu_{gc}$ 是大量基因的加和，在细胞间的相对变异远小于 $\epsilon_c$。因此近似：

$$\sum_g \mu_{gc} \approx S \quad \Longrightarrow \quad \Lambda_c \approx S \cdot \epsilon_c$$

此时total count的生成过程为：

$$N_c \sim \text{Poisson}(\Lambda_c) \approx \text{Poisson}(S \cdot \epsilon_c)$$

### 2.3 参数化

令 $\eta_c = S \cdot \epsilon_c$，合并全局常数和细胞效率的分布参数：

$$\log \eta_c = (\log S + \mu_\epsilon) + \sigma_\epsilon \cdot z, \quad z \sim \mathcal{N}(0,1)$$

记 $\mu = \log S + \mu_\epsilon$，$\sigma = \sigma_\epsilon$，则：

$$N_c \sim \text{Poisson}(\eta_c), \quad \log \eta_c \sim \mathcal{N}(\mu, \sigma^2)$$

这是一个标准的**Poisson-LogNormal**混合模型。待估参数仅 $(\mu, \sigma)$ 两个标量。

> **注意：** $S$ 和 $\epsilon_c$ 不可单独辨识——我们只估计它们的乘积 $\eta_c$。这不影响下游分析，因为per-gene方程只需要 $\eta_c$ 的值。

### 2.4 EM求解

将 $\eta_c$ 视为隐变量，对 $(\mu, \sigma)$ 执行EM算法：

**E-step：** 对每个细胞 $c$，计算隐变量的后验分布

$$p(\eta_c \mid N_c, \mu, \sigma) \propto \text{Poisson}(N_c \mid \eta_c) \cdot \text{LogNormal}(\eta_c \mid \mu, \sigma^2)$$

这是一维后验，通过数值积分（如Gauss-Hermite求积）高效计算后验矩 $\mathbb{E}[\log \eta_c \mid N_c]$ 和 $\mathbb{E}[(\log \eta_c)^2 \mid N_c]$。

**M-step：** 更新参数

$$\mu \leftarrow \frac{1}{C} \sum_c \mathbb{E}[\log \eta_c \mid N_c], \qquad \sigma^2 \leftarrow \frac{1}{C} \sum_c \text{Var}[\log \eta_c \mid N_c] + \text{Var}_c\left(\mathbb{E}[\log \eta_c \mid N_c]\right)$$

迭代至收敛。由于参数空间仅二维、后验单峰光滑，收敛快且无局部最优问题。

### 2.5 点估计

收敛后，对每个细胞取后验众数（MAP估计）：

$$\hat{\eta}_c = \arg\max_{\eta} \; p(\eta \mid N_c, \hat{\mu}, \hat{\sigma})$$

一维优化，可在对数空间网格上取argmax。

### 2.6 解耦后的per-gene方程

用 $\hat{\eta}_c$ 替代采样方程中的分母，得到每个基因的独立方程：

$$X_{gc} \mid N_c \sim \text{Binomial}\left(N_c,\ \frac{\mu_{gc}}{\hat{\eta}_c}\right), \quad \mu_{gc} \sim F_g$$

至此，**每个基因完全独立**，可进入下一阶段分别估计 $F_g$。

---

## 3. 基因先验估计 (Gene Prior Estimation)

### 3.1 目标

对每个基因 $g$，利用所有细胞的观测 $\{(X_{gc}, N_c, \hat{\eta}_c)\}_{c=1}^C$ 估计其底层分布 $F_g$。

$F_g$ 的参数化形式可以选择：

- **混合分布法：** $F_g$ 参数化为 $K$ 个分量的混合分布（如混合LogNormal、混合Beta等），参数包括混合权重 $\{\pi_k\}$ 和各分量参数 $\{\theta_k\}$。
- **离散法：** $F_g$ 参数化为有限网格上的离散概率分布 $\{(v_j, w_j)\}$，其中 $v_j$ 是网格点，$w_j$ 是对应概率（通过softmax归一化）。

### 3.2 优化方法

使用**梯度下降**（或其变体如Adam）直接优化 $F_g$ 的参数。总损失函数由以下三项组成：

#### 损失一：负对数似然 (NLL)

这是核心驱动项。对每个细胞 $c$，观测 $X_{gc}$ 在模型下的边际似然为：

$$p(X_{gc} \mid N_c, \hat{\eta}_c, F_g) = \int \text{Binomial}\left(X_{gc} \;\middle|\; N_c,\; \frac{\mu}{\hat{\eta}_c}\right) \; dF_g(\mu)$$

NLL损失为：

$$\mathcal{L}_{\text{NLL}} = -\frac{1}{C} \sum_{c=1}^{C} \log p(X_{gc} \mid N_c, \hat{\eta}_c, F_g)$$

对于混合分布，积分化为对分量的加权求和；对于离散法，化为网格点上的加权求和。

#### 损失二：后验-先验对齐 (Posterior-Prior Alignment)

**动机：** NLL只约束 $F_g$ 拟合观测数据，但高噪声环境下（特别是低表达基因），NLL的梯度信号很弱，$F_g$ 可能退化为过拟合特定观测的尖锐分布。我们需要一个额外的自洽性约束：**如果 $F_g$ 是正确的先验，那么对所有细胞做贝叶斯后验推断再汇总，得到的"后验决策分布"应当与 $F_g$ 本身一致。**

**具体实现：**

1. **后验决策分布构造：** 对每个细胞 $c$，用当前 $F_g$ 和观测 $X_{gc}$ 计算后验分布 $p(\mu \mid X_{gc}, N_c, \hat{\eta}_c, F_g)$，从中取点估计 $\hat{\mu}_{gc}$（如后验均值或MAP）。将所有细胞的 $\hat{\mu}_{gc}$ 汇总，得到经验后验决策分布 $\hat{Q}_g$。

2. **对齐损失：** 度量 $\hat{Q}_g$ 与 $F_g$ 之间的差异：

$$\mathcal{L}_{\text{align}} = D(\hat{Q}_g \;\|\; F_g)$$

其中 $D$ 可选Jensen-Shannon散度（JSD）、Wasserstein距离或KL散度等。工程上，$\hat{Q}_g$ 通过"从后验采样 → 构建直方图/KDE"得到，因此这一项本质是**采样 + 对齐loss**。

**直觉：** 这一项迫使 $F_g$ 达到一种"不动点"——先验生成数据、数据更新后验、后验汇总回先验，三者自洽。

#### 损失三：多峰鼓励 (Mode Separation Encouragement)

> **仅在使用混合分布参数化时需要。离散法无需此项。**

**动机：** 混合分布的分量在优化过程中容易坍缩——多个分量收敛到相同的位置，退化为实质上的单峰分布，浪费模型容量。

**实现：** 对混合分布的各分量中心 $\{c_k\}$ 施加排斥力：

$$\mathcal{L}_{\text{sep}} = -\sum_{k \neq k'} \log |c_k - c_{k'}|$$

或等价地，最小化分量间距离的负对数。这鼓励分量在参数空间中保持分离，使模型能真正捕获多峰结构（如bet-hedging基因的双模态表达）。

### 3.3 总损失

$$\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda_1 \cdot \mathcal{L}_{\text{align}} + \lambda_2 \cdot \mathcal{L}_{\text{sep}}$$

其中 $\lambda_1, \lambda_2$ 为超参数。$\lambda_2 = 0$ 当使用离散法时。

### 3.4 输出

估计完成后，$\hat{F}_g$ 即为基因 $g$ 的底层表达分布。结合采样池估计 $\hat{\eta}_c$，可以对每个细胞-基因对计算：

- **后验分布** $p(\mu_{gc} \mid X_{gc}, N_c, \hat{\eta}_c, \hat{F}_g)$：该基因在该细胞中真实表达量的完整不确定性刻画。
- **后验点估计** $\hat{\mu}_{gc}$：去噪后的表达量估计。
- **后验可信度**：后验分布的集中程度，反映该观测值的信息含量。

---

## 4. 流程总结

```
输入：UMI count矩阵 X (G × C)，total count向量 N

第一阶段 — 采样池估计 (Pool Estimation)
│  模型：N_c ~ Poisson(η_c), log η_c ~ N(μ, σ²)
│  方法：EM算法
│  输出：全局参数 (μ̂, σ̂)；每细胞采样池估计 η̂_c
│  作用：将基因间解耦
│
第二阶段 — 基因先验估计 (Gene Prior Estimation)
│  对每个基因 g 独立执行：
│  模型：X_gc | N_c ~ Binom(N_c, μ_gc / η̂_c), μ_gc ~ F_g
│  方法：梯度下降优化 F_g 的参数
│  损失：L_NLL + λ₁·L_align + λ₂·L_sep
│  输出：基因底层分布 F̂_g
│
最终输出：
│  每个 (细胞, 基因) 对的后验分布 p(μ_gc | X_gc, N_c, η̂_c, F̂_g)
│  → 去噪点估计、可信度、异常检测等下游任务
```