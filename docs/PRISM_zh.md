# PRISM：面向单细胞分子的概率独立信号提取

---

## 1. 符号体系与生成模型

### 1.1 符号表

| 符号 | 名称 | 定义 | 备注 |
|---|---|---|---|
| $\mathcal{G}^{\mathrm{ref}}$ | 参考基因集合 | 用于构造细胞参考总计数的基因集合 | 应尽可能大且结构均衡；应排除对总量有压倒性贡献的模块（如高占比线粒体基因、核糖体基因等） |
| $\mathcal{G}^{\mathrm{fit}}$ | 拟合基因集合 | 需要进行单基因建模的目标基因集合 | 不要求与 $\mathcal{G}^{\mathrm{ref}}$ 重合；未参与 $N_c$ 构造的基因仍可被拟合 |
| $G^{\mathrm{ref}}$ | 参考集合基数 | $G^{\mathrm{ref}} = \lvert\mathcal{G}^{\mathrm{ref}}\rvert$ | 决定 $N_c$ 的聚合稳定性 |
| $g$ | 拟合基因索引 | $g \in \mathcal{G}^{\mathrm{fit}}$ | 不要求 $g \in \mathcal{G}^{\mathrm{ref}}$ |
| $h$ | 参考基因索引 | $h \in \mathcal{G}^{\mathrm{ref}}$ | 仅用于定义 $N_c$ |
| $c$ | 细胞索引 | $c = 1, \ldots, C$ | |
| $X_{gc}$ | 观测 UMI 计数 | 基因 $g$ 在细胞 $c$ 中的观测计数 | 直接可观测量 |
| $N_c$ | 参考总计数 | $N_c = \sum_{h \in \mathcal{G}^{\mathrm{ref}}} X_{hc}$ | 细胞级 exposure 代理；是对潜在窗口 $T_c$ 的带噪估计，而非精确真值 |
| $T_c$ | 潜在总窗口 | 细胞 $c$ 中不可直接完整观测的有效分子总体 | 概念性参照量；$N_c$ 是其有噪代理 |
| $p_{gc} \in [0,1]$ | 相对表达比率 | 基因 $g$ 在细胞 $c$ 中相对于参考窗口的表达占比 | 框架的核心潜变量；语义由 $\mathcal{G}^{\mathrm{ref}}$ 诱导 |
| $F_g$ | 基因级有效先验 | $p_{gc} \sim F_g$ | 定义在相对表达比率空间的群体级先验，吸收所有无法显式分解的局部扰动 |
| $S > 0$ | 缩放标尺 | 任意选定的正实数常数 | 纯坐标变换量纲；仅在需要将 $p_{gc}$ 映射至绝对数值网格时引入 |
| $\mu_{gc}$ | 缩放表达坐标 | $\mu_{gc} = S \cdot p_{gc}$ | $p_{gc}$ 的等价重参数化；便于可视化与旧接口兼容 |
| $\epsilon_c > 0$ | 细胞级整体扰动 | 细胞 $c$ 共享的整体窗口缩放因子 | 主要效应由 $N_c$ 的细胞间差异吸收，不作为独立潜变量推断 |
| $\epsilon_{gc} > 0$ | 基因-细胞局部扰动 | 基因 $g$ 在细胞 $c$ 中的局部组成波动 | 包括局部捕获偏差、微环境差异、转录 burst 等；其效应吸收入有效先验 $F_g$ |
| $\bar N$ | 参考总计数均值 | $\bar N = \frac{1}{C}\sum_{c=1}^C N_c$ | 用于将不同细胞的 exposure 归一到共同尺度 |
| $N_c^{\mathrm{eff}}$ | 有效参考计数 | $N_c^{\mathrm{eff}} = \frac{N_c}{\bar N} S$ | 进入工作 likelihood 的有效 exposure；同时编码细胞相对窗口大小与全局尺度 $S$ |

**集合分工约定。** $\mathcal{G}^{\mathrm{ref}}$ 的职责是稳定估计细胞级参考 exposure，因此要求集合规模大、组成均衡、无强主导模块；$\mathcal{G}^{\mathrm{fit}}$ 的职责是定义待建模目标基因的范围，二者不必重合。值得强调的是，$N_c$ 不是一个可任意重标的纯数值量——改变其构造方式既改变数值尺度，也改变其方差结构，并最终影响 $F_g$ 的拟合不确定性。

---

### 1.2 框架立场与建模目标

本框架不从"绝对分子数的真实物理生成过程"出发，而直接以**相对表达比率** $p_{gc}$ 为核心建模对象。对每个目标基因 $g$，观测计数 $X_{gc}$ 被理解为：在有限 exposure 条件下，对细胞内真实表达占比 $p_{gc}$ 的带噪测量。

建模目标因此聚焦于以下两类随机性的分离：

- **采样随机性**：从有限分子总体中抽取有限计数所产生的不可避免波动，由显式 likelihood 处理；
- **组成异质性**：采样前已经存在的真实生物变异与不可辨识局部扰动，吸收入基因级有效先验 $F_g$。

scPRISM 不试图还原实验物理链条的每一个中间步骤，而是在一个可操作的统计框架内将上述两类随机性系统地分离。

---

### 1.3 从有限总体抽样到工作模型

**严格模型。** 设细胞 $c$ 中存在一个有效分子总体 $T_c$，其中基因 $g$ 的分子数为 $M_{gc} = p_{gc} T_c$。若从该总体中无放回地抽取 $n_c$ 个分子，则在有限总体抽样框架下，单基因计数服从超几何分布：

$$X_{gc} \mid T_c,\, M_{gc},\, n_c \;\sim\; \mathrm{Hypergeometric}(T_c,\, M_{gc},\, n_c)$$

其条件一阶、二阶矩为：

$$\mathbb{E}[X_{gc} \mid T_c, M_{gc}, n_c] = n_c p_{gc}$$

$$\mathrm{Var}(X_{gc} \mid T_c, M_{gc}, n_c) = n_c p_{gc}(1-p_{gc}) \cdot \underbrace{\frac{T_c - n_c}{T_c - 1}}_{\text{有限总体修正}}$$

**Binomial 近似的适用条件。** 当以下条件同时成立时，超几何分布可由 Binomial 稳定近似：（i）总体规模 $T_c$ 足够大；（ii）采样比例 $n_c/T_c$ 较小，使有限总体修正因子接近 1；（iii）单基因占比 $p_{gc}$ 通常较小；（iv）关注单基因边缘分布而非多基因联合分配。此时一阶矩严格一致，二阶矩之差由有限总体修正控制，从而有：

$$\mathrm{Hypergeometric}(T_c,\, M_{gc},\, n_c) \;\approx\; \mathrm{Binomial}(n_c,\, p_{gc})$$

**工作模型。** 由于实际抽样次数 $n_c$ 不可直接观测，以有效参考计数 $N_c^{\mathrm{eff}}$ 作为其统计代理，得到 scPRISM 的单基因工作公式：

$$\boxed{X_{gc} \mid N_c^{\mathrm{eff}},\, p_{gc} \;\approx\; \mathrm{Binomial}(N_c^{\mathrm{eff}},\, p_{gc}), \qquad p_{gc} \sim F_g}$$

其中

$$N_c^{\mathrm{eff}} = \frac{N_c}{\bar N} S, \qquad \bar N = \frac{1}{C}\sum_{c=1}^C N_c, \qquad \mu_{gc} = Sp_{gc}$$

该公式是后续所有先验拟合、后验推断与信号提取的起点。需要强调：$N_c^{\mathrm{eff}}$ 并非"真实抽样次数"的还原，而是一个匹配一阶矩并近似刻画观测精度的有效量；当 $\frac{N_c}{\bar N}S$ 不是整数时，应将其理解为连续意义下的 effective sample size。上式应理解为**比例观测模型**（proportion observation model），而非对实验物理过程的逐步复刻。

---

### 1.4 参考总计数 $N_c$ 的统计语义

参考总计数定义为：

$$N_c = \sum_{h \in \mathcal{G}^{\mathrm{ref}}} X_{hc}$$

**$N_c$ 是带噪 exposure 代理，而非精确真值。** 从现象学角度，可将其近似写作：

$$N_c \approx T_c \cdot \epsilon_c \cdot \frac{G^{\mathrm{ref}}}{G_t}$$

其中 $G_t$ 为更大背景窗口中的有效基因规模。这一关系式不是严格恒等式，而是强调：$N_c$ 的数值与参考集合的选择直接耦合，其方差结构同时受 $G^{\mathrm{ref}}$ 的规模与内部组成影响。当 $\mathcal{G}^{\mathrm{ref}}$ 足够大且无强主导模块时，聚合过程通过大数平均效应压低内部随机波动，使 $N_c$ 主要反映细胞级整体 exposure 的变化，而非少数基因局部起伏的直接投影。相反，若参考集合中存在整体性强干扰模块，则 $N_c$ 会偏离“共享 exposure 代理”的语义，转而混入显著的组成型偏差。

**$N_c$ 可用于拟合 $\mathcal{G}^{\mathrm{fit}}$ 中任意基因。** 由于 $N_c$ 的语义是**细胞级共享参考窗口**，而非"由目标基因自身定义的窗口"，因此它可以为任何 $g \in \mathcal{G}^{\mathrm{fit}}$ 提供 exposure 基准，无论该基因是否属于 $\mathcal{G}^{\mathrm{ref}}$。将潜在干扰组分（如高占比线粒体基因）排除出 $\mathcal{G}^{\mathrm{ref}}$ 以提升 $N_c$ 的稳定性，与在同一 $N_c$ 上拟合这些基因的相对表达比率 $p_{gc}$，这两件事并不矛盾。

**$N_c$ 不可任意重标。** 在本框架中，真正进入 likelihood 的并不是原始 $N_c$，而是归一化后的有效参考计数

$$N_c^{\mathrm{eff}} = \frac{N_c}{\bar N} S$$

因此，$N_c$ 只负责提供细胞间的相对 exposure 缩放，而全局观测精度由固定标尺 $S$ 统一控制。表面上，将 $N_c^{\mathrm{eff}}$ 乘以常数 $\lambda$、同时将 $p_{gc}$ 除以 $\lambda$，可以保持期望 $\mathbb{E}[X_{gc} \mid N_c^{\mathrm{eff}}, p_{gc}] = N_c^{\mathrm{eff}} p_{gc}$ 不变。但 Binomial likelihood 不仅依赖期望，还依赖观测精度，因为

$$\mathrm{Var}(X_{gc} \mid N_c^{\mathrm{eff}}, p_{gc}) = N_c^{\mathrm{eff}} p_{gc}(1-p_{gc})$$

且在小 $p_{gc}$ 区域，对 $p_{gc}$ 的 Fisher information 近似正比于 $N_c^{\mathrm{eff}}$。因此，决定拟合质量的不是原始 $N_c$ 的绝对量级，而是归一化后 exposure 代理的**有效信噪比**。使用 $\frac{N_c}{\bar N}S$ 的好处在于：不同细胞之间保留相对窗口差异，而不同数据集之间只要采用相同的 $S$，则整体缩放语义保持一致，不会因为某个数据集的平均 $N_c$ 偏大或偏小而系统性改变后验精度。

---

### 1.5 噪声来源的分类与处理策略

在 scPRISM 的建模视角下，观测 UMI 计数的随机性可概念性地归结为四类来源，各自采取不同的处理方式。

**第一类：采样随机性（显式建模）。** 这是从有限 exposure 中得到离散计数时不可避免的波动，对应工作公式中的 Binomial likelihood。框架显式处理并"消化"的核心正是这一部分：给定 $N_c^{\mathrm{eff}}$ 和 $p_{gc}$，观测值 $X_{gc}$ 的随机性被归因于采样过程，并通过后验推断加以分离。该层噪声决定了在固定 $p_{gc}$ 下单细胞观测会有多分散，也是本框架唯一被显式参数化的随机层。

**第二类：细胞级整体扰动 $\epsilon_c$（由 $N_c$ 吸收）。** $\epsilon_c$ 代表某一细胞整体窗口的系统性缩放，例如总捕获水平或整体反应效率的差异。它不作为独立潜变量被推断，而是通过 $N_c$ 的细胞间差异进入 likelihood 精度，从而隐式地被处理。也就是说，$\epsilon_c$ 的主要作用不是改变某个特定基因的相对位置，而是改变该细胞整体观测的有效分辨率。

**第三类：基因-细胞局部扰动 $\epsilon_{gc}$（吸收入有效先验 $F_g$）。** 这类扰动包括局部捕获偏差、微环境差异、转录 burst 等无法逐项辨识的 gene-cell 特异性效应。参考集合聚合只能使 $N_c$ 成为较稳定的 exposure 代理，不能消除目标基因自身的局部组成波动。这些采样前已经存在、且来源无法进一步区分的波动，最终表现为 $p_{gc}$ 分布的有效变异性，并直接耦合进 $F_g$ 的形状与宽度。换言之，$F_g$ 是"包含不可辨识局部扰动后的有效先验"，而非"去除一切技术项后的纯净生物分布"。因此，scPRISM 所恢复的是适用于后验推断的 effective ratio prior，而不是严格意义上的机制分解结果。

**第四类：测序残余误差（默认忽略）。** 碱基识别错误、比对错误、UMI 冲突与纠错残差等采样后的技术误差，对于现代高质量 UMI 流程而言，经过 UMI 聚合、比对过滤与条形码纠错后，其对基因级计数的净影响通常远低于前三类效应，在当前主模型中视为高阶小量忽略不计。经验上，这类 read-level 或 base-level 错误常已被压低到约 $10^{-2}$ 或更低量级，而其传递到 gene-level UMI count 的净扰动通常更小。若特定平台、低质量样本或异常流程中此类误差不可忽略，可将其作为附加观测噪声层引入扩展模型。

---

### 1.6 固定标尺 $S$ 与归一化 exposure

在新的表述中，不再单独引入观测精度超参 $r$。相反，我们将全局尺度与观测精度统一吸收到固定标尺 $S$ 中，并定义

$$N_c^{\mathrm{eff}} = \frac{N_c}{\bar N} S, \qquad \bar N = \frac{1}{C}\sum_{c=1}^C N_c$$

于是，$N_c/\bar N$ 仅编码细胞间的相对 exposure 差异，而 $S$ 统一给出全局的参考尺度。这样做有三个直接好处：

- 保留了不同细胞之间由 $N_c$ 反映的相对观测深度差异；
- 避免了不同数据集因平均 $N_c$ 不同而导致整体 likelihood 宽度系统性漂移；
- 使跨数据集比较时，只要使用相同的 $S$，就能够保持一致的 $\mu$ 轴与观测精度语义。

因此，$S$ 在本框架中兼具两重角色：一方面，它定义缩放坐标 $\mu = Sp$；另一方面，它也控制整体 likelihood 的集中程度。$S$ 越大，$N_c^{\mathrm{eff}}$ 整体越大，观测约束越强；$S$ 越小，后验越依赖群体先验 $F_g$。

从实践上看，$S$ 不是需要从数据中识别的物理量，而是一个统一的全局参考标尺。对于单一数据集，一个自然的默认选择是令

$$S = \bar N$$

此时有 $N_c^{\mathrm{eff}} = N_c$。而当需要在多个数据集、多个平台或多个参考集合定义之间保持一致尺度时，可显式固定同一个 $S$，从而获得可比的 $\mu$ 轴与 likelihood 精度解释。

---

### 1.7 平台差异与 $\mu$ 轴缩放

跨测序平台、跨建库流程或跨引物设计比较时，需要额外注意：即便在相同生物体系下，观测到的 $\mu$ 轴也可能发生系统性缩放。造成这一现象的主要来源并非细胞内真实比例结构本身改变，而是平台相关的技术映射差异，例如引物体系、逆转录效率、扩增偏好、read 结构、比对策略与 UMI 处理流程等。这些因素会改变“相同真实比例”在观测坐标上的投影尺度。

可将这种现象学差异写为

$$\mu_{gc}^{(k)} = a_g^{(k)} \cdot \mu_{gc}^{\star}$$

其中 $k$ 表示平台或实验流程，$\mu_{gc}^{\star}$ 表示某个共同潜在坐标下的表达量，$a_g^{(k)} > 0$ 表示平台 $k$ 对基因 $g$ 的有效缩放系数。若平台差异是纯全局缩放，则有 $a_g^{(k)} \equiv a^{(k)}$；而在更现实的情形下，$a_g^{(k)}$ 往往是**gene-specific** 的，即不同基因会有不同程度的偏移。

这意味着：跨平台时，$\mu$ 轴不应被直接解释为绝对可比的物理尺度。某些平台会整体压缩或拉伸表达轴，而另一些差异则体现在特定基因的额外偏移上。因此，若不做额外校正，不同平台下的 $\mu$ 数值本身通常只具有平台内语义，而不必然具有平台间的一一对应关系。

然而，实验上经常可以观察到：即使 $\mu$ 轴发生缩放，部分基因所对应的 $F_g$ 形状仍然在不同平台之间表现出高度一致，尤其是在单峰/双峰结构、尾部厚度以及主要模态位置的相对关系上。这说明平台差异的主效应往往首先表现为坐标缩放或轻度基因特异性扭曲，而不是彻底改写该基因的群体分布结构。换言之，平台差异更像是对观测坐标的重参数化，而不是对生物分布形状的完全重写。

因此，在 scPRISM 中，跨平台比较应区分两个层面：

- **轴尺度差异：** 主要表现为 $\mu$ 轴的整体缩放或 gene-specific 偏移，需要通过额外归一化、锚定基因或校准模型处理；
- **形状稳定性：** 若某些基因的 $F_g$ 形状在不同平台间仍保持一致，则这些基因可视为较稳健的结构锚点，说明框架所恢复的 effective prior 具有一定平台不变性。

这也进一步说明：$F_g$ 的可迁移性并不要求 $\mu$ 轴数值逐点一致。更现实、也更重要的目标，是在允许平台相关缩放与偏移存在的前提下，识别哪些基因保留了稳定的分布形状，以及哪些基因受平台效应影响更强。前者支持跨平台结构比较，后者则提示需要额外的 gene-specific 校正。

---

## 2. 基因先验估计

### 2.1 目标与范围

对每个基因 $g$，从观测 $\{(X_{gc}, N_c)\}_{c=1}^C$、有效参考计数 $N_c^{\mathrm{eff}} = \frac{N_c}{\bar N}S$ 以及固定缩放标尺 $S$ 估计群体级有效先验 $F_g$。这里的 $F_g$ 定义在相对表达比率 $p_{gc} \in [0,1]$ 上；若使用缩放坐标 $\mu_{gc} = Sp_{gc}$，则也可等价地表示为 $\mu$ 轴上的离散分布。

需要强调，$F_g$ 是 effective prior：它描述的是在当前观测框架下，基因 $g$ 的有效群体分布，而不是已经与所有局部扰动、平台偏差和采样前噪声完全分离的“纯生物真值分布”。本框架是逐基因的边际经验贝叶斯模型，不显式建模跨基因依赖；其目标是为后续后验推断与信号提取提供稳定的单基因先验层。

### 2.2 离散表示与缩放坐标

对每个基因 $g$，在 $\mu$ 轴上引入 $M$ 个等距网格点：

$$0 \le v_1 < v_2 < \cdots < v_M \le \mu_g^{\max}$$

其中对应的比例坐标为

$$p_j = \frac{v_j}{S} \in [0,1]$$

于是 $F_g$ 被表示为这些网格点上的离散分布。设无约束参数为 logits $\{l_j\}_{j=1}^M$，先经 softmax 得到概率权重：

$$w_j = \frac{\exp(l_j)}{\sum_{k=1}^M \exp(l_k)}$$

随后对 $w$ 施加高斯平滑，得到最终分布：

$$\tilde{w}_j = \sum_{k=1}^M w_k\,G_\sigma(v_j-v_k), \qquad G_\sigma(x)=\exp\left(-\frac{x^2}{2\sigma^2}\right)$$

卷积后的 $\tilde{w}$ 即为估计先验 $\hat F_g$。这里不存在额外的“逆变换”：平滑后的分布本身就是最终输出。

在新的框架下，$S$ 不再是待估参数，而是固定缩放标尺。一个自然且稳定的默认选择是

$$S = \frac{1}{C}\sum_{c=1}^C N_c$$

即取参考总计数的细胞均值。这样做的作用不是赋予 $S$ 物理可识别性，而是给 $\mu = Sp$ 提供一个稳定的公共尺度；特别地，当改变 $\mathcal{G}^{\mathrm{ref}}$ 的构造方式时，使用 $\mathrm{mean}(N_c)$ 作为标尺通常能使 $\mu$ 轴保持更稳定的量级语义。

对于具体基因 $g$，网格上界建议取为

$$\mu_g^{\max} = \max_c X_{gc} \cdot \frac{S}{\bar{N}}, \qquad \bar{N} = \frac{1}{C}\sum_{c=1}^C N_c$$

亦即用该基因在全体细胞中的最大观测计数乘以 $S/N_c$ 的平均尺度。这样做的目的，是在网格范围与有效分辨率之间取得平衡：若上界取得过大，则大量网格点会落在几乎没有后验质量支撑的区域，导致有效分辨率下降；若上界取得过小，则会出现尾部质量被截断、后验概率在边界附近严重堆积的质量逸散问题。上述经验选择通常能够在不引入过大冗余范围的前提下，为该基因保留足够的动态空间。

### 2.3 单细胞似然

在第 1 节的工作模型下，单细胞观测满足

$$X_{gc}\mid N_c^{\mathrm{eff}},p_{gc} \approx \mathrm{Binomial}(N_c^{\mathrm{eff}},p_{gc})$$

因此，对离散网格点 $v_j$，单细胞似然写为

$$\mathrm{lik}_{cj} = \mathrm{Binomial}\!\left(X_{gc}\mid N_c^{\mathrm{eff}}, \frac{v_j}{S}\right)$$

其中 $N_c^{\mathrm{eff}} = \frac{N_c}{\bar N}S$，$v_j/S$ 是对应的比例参数。于是每个细胞都在同一条 $\mu$ 轴上诱导出一个单峰或近单峰的似然轮廓；其峰值大致位于

$$v_j \approx \frac{X_{gc}}{N_c^{\mathrm{eff}}}\,S = X_{gc}\frac{\bar N}{N_c}$$

附近。$N_c^{\mathrm{eff}}$ 越大，似然越窄，说明该细胞对 $p_{gc}$ 提供的约束越强；$N_c^{\mathrm{eff}}$ 越小，似然越宽，说明后验会更依赖群体先验。

### 2.4 经验贝叶斯目标

对每个细胞 $c$，边际似然为先验与单细胞似然的加权和：

$$d_c = \sum_{j=1}^M \tilde{w}_j\,\mathrm{lik}_{cj}$$

对应的负对数似然为

$$\mathcal{L}_{\mathrm{NLL}} = -\frac{1}{C}\sum_{c=1}^C \log d_c$$

这是经验贝叶斯意义下的主数据项：它要求找到一个群体级分布 $\tilde w$，使得所有单细胞观测在该分布下具有尽可能高的边际解释度。

在样本量充足且模型近似合理时，NLL 提供了恢复 $F_g$ 的基本统计依据；但在有限样本、低覆盖或强噪声条件下，仅依赖 NLL 仍可能对稀有峰或弱结构约束不足。

### 2.5 后验生成分布与自洽正则

给定当前先验 $\tilde w$，对每个细胞可由 Bayes 公式得到离散后验：

$$p(v_j\mid X_{gc}) = \frac{\tilde{w}_j\,\mathrm{lik}_{cj}}{\sum_{k=1}^M \tilde{w}_k\,\mathrm{lik}_{ck}}$$

这里默认采用完整后验的平均来构造群体级“后验生成分布”：

$$\hat Q_g(v_j) = \frac{1}{C}\sum_{c=1}^C p(v_j\mid X_{gc})$$

$\hat Q_g$ 可理解为：在当前先验下，所有细胞经后验更新后重新“生成”的群体分布。它保留了单细胞后验的全部概率质量，而不是先做 MAP 硬决策再聚合，因此比 MAP histogram 更连续，也更符合后验一致性的原始动机。

基于这一分布，引入自洽正则项：

$$\mathcal{L}_{\mathrm{align}} = \mathrm{JSD}(\hat Q_g\|\tilde w)$$

其中

$$\mathrm{JSD}(\hat Q_g\|\tilde w)=\frac{1}{2}\mathrm{KL}(\hat Q_g\|\bar M)+\frac{1}{2}\mathrm{KL}(\tilde w\|\bar M), \qquad \bar M=\frac{1}{2}(\hat Q_g+\tilde w)$$

总目标函数写为

$$\mathcal{L} = \lambda_{\mathrm{nll}}\mathcal{L}_{\mathrm{NLL}}+\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}$$

这个正则项的含义是：若 $\tilde w$ 是一个合理的群体级先验，那么用它对单细胞观测做后验推断后，再将所有后验分布平均得到的群体分布，不应与原先验相差过大。

需要强调，JSD 对齐是经验上的自洽约束，而不是普适定理。它可能帮助稳定稀有结构，也可能在低覆盖和多峰场景中强化偶然伪峰。因此，它应被理解为可调正则项，而非先验保证。

### 2.6 平滑、梯度结构与偏差-方差权衡

高斯卷积在这里的作用不只是让输出分布 $\tilde w$ 更平滑，也会通过链式法则平滑优化过程中的梯度场。也就是说，卷积不仅抑制最终分布中的锯齿，也降低了相邻网格点之间梯度的高频震荡，使优化更倾向于产生连续结构，而不是网格级别的尖锐抖动。

即使不显式加入高斯模糊，仅使用 softmax-logit 参数化，在 NLL 与后验一致性目标的共同作用下，优化得到的分布也往往会趋向一定程度的平滑：因为相邻网格点对应的似然函数高度相关，梯度天然具有局部耦合性。因此，高斯模糊并不是唯一的平滑来源，而是一种更强的、显式的低通约束。

这也解释了其偏差-方差权衡：

- 较小的 $\sigma$ 保留更多局部结构，但对噪声更敏感；
- 较大的 $\sigma$ 提供更强的稳定化作用，但可能抹平真实窄峰。

因此，卷积平滑的作用应理解为：在本来就存在局部梯度相关性的优化问题上，进一步施加构造性的光滑先验。

### 2.7 输出与超参数

最终输出的 $\tilde w$ 就是估计得到的 $\hat F_g$。它定义了基因 $g$ 在比例空间或缩放坐标空间上的有效群体先验，并直接用于后续的单细胞后验推断。

该部分的核心超参数为：

- $M$：离散网格分辨率；
- $\sigma$：高斯平滑核宽度；
- $\lambda_{\mathrm{nll}}$：数据项权重；
- $\lambda_{\mathrm{align}}$：后验一致性项权重；
- $S$：固定缩放标尺，默认建议取 $\mathrm{mean}(N_c)$。

在方法层面，最关键的不是具体优化器或工程实现，而是以下三点：

- 用离散分布表示单基因先验；
- 用 Binomial 比例观测模型连接观测与潜变量；
- 用“边际似然 + 后验生成分布自洽”共同约束 $F_g$。

---

## 3. 后验提取与 `prism extract signals` 接口

本节不再沿用旧版文档中的 `Confidence / Surprisal / Sharpness` 术语体系，而是严格按照当前代码实现说明 `prism extract signals` 的实际输入、后验计算方式、导出通道和输出文件布局。这里讨论的是 `prism extract` 命令组中的 `signals` 子命令；同一命令组下的 `kbulk` 与 `kbulk-mean` 属于聚合接口，不在本节展开。

### 3.1 命令边界与数据来源

当前 CLI 入口为：

```bash
prism extract signals CHECKPOINT INPUT.h5ad --output OUTPUT.h5ad
```

其实现逻辑可概括为以下四步：

1. 读取 checkpoint，获得：
   - 已拟合基因集合 `checkpoint.gene_names`
   - 全局先验 `checkpoint.priors` 或标签先验 `checkpoint.label_priors`
   - 参考基因集合 `checkpoint.metadata["reference_gene_names"]`
2. 读取输入 `h5ad`，从 `X` 或 `--layer` 指定矩阵中取观测计数。
3. 用 checkpoint 中记录的参考基因集合在输入数据上重新计算每个细胞的参考总计数：

$$N_c = \sum_{h \in \mathcal{G}^{\mathrm{ref}} \cap \mathcal{G}^{\mathrm{data}}} X_{hc}$$

4. 对选定基因逐批执行后验提取，并将结果写回输出 `AnnData.layers`。

这里有两个实现细节需要明确：

- `prism extract signals` 的参考总计数并不是从“当前待提取基因”重新定义，而是始终由 checkpoint 中保存的 `reference_gene_names` 决定。
- 如果 checkpoint 的参考基因与输入数据没有交集，命令会直接报错；如果只是部分重叠，则仅使用重叠部分重新计算 $N_c$。

目标基因集合的选择规则同样是代码固定的：

- 若未提供 `--genes`，默认尝试提取 `checkpoint.gene_names` 中所有同时存在于输入数据的基因；
- 若提供 `--genes`，则在该文本文件、输入数据 `var_names`、以及 checkpoint 已拟合基因三者的交集中提取；
- 若交集为空，命令报错而不是输出空文件。

### 3.2 实际后验公式与有效 exposure

当前实现并不直接把原始 $N_c$ 代入 Binomial likelihood，而是先构造有效 exposure：

$$N_c^{\mathrm{eff}} = \frac{N_c}{\bar N} S, \qquad \bar N = \frac{1}{C}\sum_{c=1}^C N_c$$

其中 $S$ 不是在提取时重新估计的，而是直接取自 checkpoint 内保存的 `PriorGrid.S`。因此，对给定基因 $g$ 的离散网格点 $p_{gj}$，代码真正计算的是

$$\log \mathrm{lik}_{gcj} = \log \mathrm{Binomial}\!\left(X_{gc} \mid N_c^{\mathrm{eff}}, p_{gj}\right)$$

对应的离散后验为

$$p(p_{gj}\mid X_{gc}, N_c, \hat F_g) = \frac{\tilde w_{gj}\,\mathrm{Binomial}(X_{gc}\mid N_c^{\mathrm{eff}}, p_{gj})}{\sum_k \tilde w_{gk}\,\mathrm{Binomial}(X_{gc}\mid N_c^{\mathrm{eff}}, p_{gk})}$$

若记 $\mu_{gj} = S p_{gj}$，则这与在 $\mu$ 轴上的后验完全等价，只是代码内部始终以 `p_grid` 为主、以 `mu_grid = S \cdot p_grid` 作为派生坐标。

实现上还需要注意三件事：

- `log_binomial_likelihood_grid()` 使用 `lgamma` 形式计算对数 Binomial 系数，因此 $N_c^{\mathrm{eff}}$ 不要求是整数，可以被解释为连续意义下的 effective sample size。
- 旧版文档里提到“extract 阶段通常采用 cell chunking”，这已不符合当前实现。现在 `signals` 命令的外层分块单位是基因，参数名为 `--batch-size`，含义是“每批处理多少个基因”。
- 当 `--prior-source label` 时，代码会在每个基因批次内部再按 `adata.obs[label_key]` 的标签对子细胞集分别运行后验提取；也就是说，当前工程分块是“gene batching + label-wise cell slicing”，而不是统一的 cell chunking。

### 3.3 点估计与信息量

当前 CLI 默认导出的不是旧版的 `Signal + Confidence + Surprisal (+ Sharpness)`，而是 `CORE_CHANNELS` 中定义的四个通道：

- `signal`
- `posterior_entropy`
- `prior_entropy`
- `mutual_information`

此外，CLI 允许额外请求两个非默认通道：

- `map_p`
- `map_mu`

因此，`prism extract signals` 当前总共只支持 6 个可写出通道。`posterior`、`support`、`prior_weights` 虽然在 Python API 中可获得，但并不属于 CLI 可选 `--channel` 的范围。

下面按源码语义逐一说明。

#### 3.3.1 `signal` / `map_mu`

给定离散后验，代码首先取后验 MAP 索引：

$$j_{gc}^{*} = \arg\max_j \; p(p_{gj}\mid X_{gc})$$

然后定义

$$\mathrm{map\_p}_{gc} = p_{g j_{gc}^{*}}, \qquad \mathrm{map\_mu}_{gc} = S \cdot \mathrm{map\_p}_{gc}$$

其中 `signal` 在当前实现中只是 `map_mu` 的别名。也就是说：

$$\mathrm{signal}_{gc} \equiv \mathrm{map\_mu}_{gc}$$

这点在 `Posterior.extract()` 中是显式硬编码的，而不是近似关系。因此：

- 若只请求 `signal`，输出的是 $\mu$ 轴上的 MAP 点估计；
- 若同时请求 `signal` 和 `map_mu`，两层数据数值相同，只是名字不同；
- 若需要比例空间上的 MAP，则应显式请求 `map_p`。

`signal` 仍然是当前下游最直接的去噪主信号，但其精确定义是“$\mu$ 轴上的后验 MAP”，而不是更早文档中可能出现的 soft-argmax 或其他软决策形式。

#### 3.3.2 `posterior_entropy`

代码对每个细胞-基因对输出后验 Shannon 熵：

$$H_{gc}^{\mathrm{post}} = -\sum_{j=1}^{M} p(p_{gj}\mid X_{gc}) \log p(p_{gj}\mid X_{gc})$$

这对应实现中的 `posterior_entropy`。它的语义是：在看到当前观测之后，该细胞-基因对仍保留多少不确定性。

与旧版 `Confidence` 的差别很重要：

- `posterior_entropy` 越小，表示后验越集中、信息越确定；
- 它的量纲是熵（自然对数底下为 nats），不是归一化到 $[0,1]$ 的分数；
- 代码没有做 $\log M$ 归一化，因此不同网格大小、不同有效支持范围之间，数值解释应谨慎。

如果下游仍想构造类似旧版 `Confidence` 的归一化指标，可以在导出后自行定义

$$\mathrm{Conf}^{\mathrm{derived}}_{gc} = 1 - \frac{H_{gc}^{\mathrm{post}}}{\log M_g}$$

但这只是从当前输出推导出的二次指标，不是源码中的内置通道。

#### 3.3.3 `prior_entropy`

对每个基因，代码还计算先验熵：

$$H_g^{\mathrm{prior}} = -\sum_{j=1}^{M} \tilde w_{gj}\log \tilde w_{gj}$$

在实现中，该值先按基因计算，再在所有细胞上广播，因此输出矩阵中的 `prior_entropy[:, g]` 对同一个 prior 来源是常数列。

这里要区分两种情形：

- 当 `--prior-source global` 时，同一基因在所有细胞上共享同一个先验，因此 `prior_entropy` 对该基因是全体细胞相同的常数。
- 当 `--prior-source label` 时，不同标签使用不同的 label-specific prior，因此同一基因在不同标签细胞上可能呈现不同的 `prior_entropy` 常数块。

`prior_entropy` 不是“某个细胞观测后得到的不确定性”，而是该基因群体先验本身的分散程度。

#### 3.3.4 `mutual_information`

当前代码中的 `mutual_information` 定义为

$$\mathrm{MI}_{gc} = \max\!\left(H_g^{\mathrm{prior}} - H_{gc}^{\mathrm{post}},\; 0\right)$$

它表达的是：给定当前观测 $(X_{gc}, N_c)$ 后，相对于先验不确定性，后验熵下降了多少。

从语义上看，它更接近“单次观测带来的信息增益”或“熵减少量”：

- 值越大，说明该观测对锁定潜在状态越有帮助；
- 值越小，说明后验相对先验没有缩窄太多，当前观测提供的信息有限。

严格地说，经典互信息通常是对观测分布再取期望后的总体量；而这里的实现是对每个具体观测实例直接计算 `prior_entropy - posterior_entropy`，再截断到非负。因此它是一个 observation-specific information gain 指标，名称沿用源码中的 `mutual_information`。

#### 3.3.5 `map_p`

`map_p` 是比例空间上的 MAP：

$$\mathrm{map\_p}_{gc} = p_{g j_{gc}^{*}}$$

它与 `signal = map_mu` 只差一个固定比例因子 $S$：

$$\mathrm{signal}_{gc} = \mathrm{map\_mu}_{gc} = S \cdot \mathrm{map\_p}_{gc}$$

如果下游方法希望直接在相对表达比率空间工作，或者需要避免不同 checkpoint 的 $S$ 对数值尺度的影响，应优先读取 `map_p` 而不是 `signal`。

### 3.4 Python API 可访问但 CLI 不直接导出的载荷

虽然 `prism extract signals` CLI 只支持上述 6 个二维通道，但 Python 层的 `Posterior` / `SignalExtractor` 还可以返回更完整的后验载荷。

调用 `Posterior.extract(..., channels={...})` 时，若请求 `posterior`，当前实现会额外返回：

- `posterior`：形状为 `(n_cells, n_genes, M)` 的完整离散后验；
- `p_grid`：对应的比例网格；
- `mu_grid`：对应的 $\mu$ 网格；
- `support`：`mu_grid` 的别名；
- `prior_weights`：每个基因的离散先验权重。

这套载荷适合做以下事情：

- 单基因后验曲线可视化；
- 自定义置信度、稀有度或局部几何指标；
- 与旧版文档中的 `Surprisal`、`Sharpness` 一类派生量做离线对照。

但需要强调，CLI 当前故意不支持把这些三维或辅助数组直接写入 `h5ad` 层；`resolve_channels()` 允许的 `--channel` 只有 `signal`、`posterior_entropy`、`prior_entropy`、`mutual_information`、`map_p`、`map_mu` 六种。

### 3.5 输出文件布局与层命名

`prism extract signals` 的输出是一个新的 `AnnData` 文件，所有提取结果都写入 `output_adata.layers[channel_name]`。每个 layer 的数据类型由 `--dtype` 控制，目前仅支持：

- `float32`
- `float64`

输出矩阵布局由 `--output-mode` 控制：

#### 3.5.1 `fitted-only`

这是默认模式。输出对象只保留被成功提取的目标基因，因此：

- `output_adata.n_vars = n_selected_genes`
- 每个输出 layer 的形状都是 `(n_cells, n_selected_genes)`
- 列顺序与 `selected_genes` 一致，而不是原始输入矩阵的全部基因顺序

如果只关心已拟合基因的 PRISM 信号，这是更紧凑、更直接的模式。

#### 3.5.2 `full-matrix`

该模式复制原始 `adata` 的完整基因轴：

- `output_adata.n_vars = input_adata.n_vars`
- 每个输出 layer 的形状都是 `(n_cells, n_dataset_genes)`
- 只有被实际提取的基因位置会被填入数值
- 未选中或未拟合的基因位置保持为 `NaN`

这使得导出的 PRISM 层可以与原始矩阵在同一基因轴上逐列对齐，但代价是会产生大量 `NaN` 占位。

### 3.6 `global` 与 `label` 两种先验来源

当前命令通过 `--prior-source` 控制使用哪一类先验：

- `global`：使用 `checkpoint.priors`
- `label`：使用 `checkpoint.label_priors`

当选择 `label` 时，必须同时提供 `--label-key`，且该列必须存在于 `adata.obs` 中。实现上，代码会：

1. 读取每个细胞的标签值；
2. 对每个标签单独挑出对应细胞；
3. 用同名 label prior 对这部分细胞单独做后验提取；
4. 再把结果回填到输出矩阵的对应行。

因此，`label` 模式下的提取结果不是“先统一算完再按标签切片”，而是真正使用不同的先验对不同细胞子群分别推断。这一点会直接影响：

- `signal / map_mu / map_p` 的点估计；
- `posterior_entropy` 的集中程度；
- `prior_entropy` 的常数块取值；
- `mutual_information` 的信息增益幅度。