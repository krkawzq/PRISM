# `src/prism/model` 工程优化与算法提效分析

## 1. 目标与范围

本文面向 `src/prism/model` 模块，重点分析如下几类问题：

- 工程层面的吞吐、显存/内存、重复计算、重复拷贝、接口设计问题
- 拟合 (`fit`) / 推断 (`infer`) / kBulk 推断 (`kbulk`) 的热点路径
- 可以在不改变统计语义前提下优先落地的优化项
- 可以接受少量近似或启发式的算法提速方向

本分析基于两部分证据：

- 静态代码审阅：`fit.py`、`infer.py`、`types.py`、`posterior.py`、`kbulk.py`、`exposure.py`、`numeric.py`
- 合成基准：在仓库 `.venv` 环境、CPU 上执行小规模 benchmark，主要用于确认数量级趋势，而不是给出绝对性能上限

## 2. 模块现状概览

当前模块的核心执行链条如下：

1. `fit_gene_priors(...)`
   - 构建 support
   - 对所有基因并行跑离散 support 上的 EM
   - 输出 `PriorGrid`
2. `infer_posteriors(...)`
   - 用 `PriorGrid` 对 `ObservationBatch` 做 posterior 推断
   - 输出 `InferenceResult`
3. `infer_kbulk(...)`
   - 与 `infer_posteriors(...)` 基本同构，只是输入换成了 kBulk 聚合样本
4. `Posterior`
   - 在 `infer_posteriors(...)` 之上再次做包装和格式转换

整体实现比较干净，数值逻辑集中，代码可读性不错；但在大规模数据下，存在几个典型的工程性瓶颈：

- `fit` 中大量重复计算
- `infer`/`kbulk` 缺少与 `fit` 对等的 chunking 策略
- `numpy <-> torch`、`float32 <-> float64` 转换频繁
- 校验逻辑和结果对象在热点路径里做了全量扫描
- 默认 `torch.compile=True`，对短任务冷启动代价过高

## 3. 关键热点与问题定位

### 3.1 `fit` 的 EM 循环重复计算 likelihood

热点位置：`src/prism/model/fit.py` 中 `_run_em_pass(...)`

当前逻辑里，support、counts、effective exposure 在一次 EM pass 内是固定的，变化的只有 prior probabilities；但每一轮 EM 都会重新调用：

- `log_binomial_likelihood_support(...)`
- `log_negative_binomial_likelihood_support(...)`
- `log_poisson_likelihood_support(...)`

这意味着每一轮都在重复做同一批 `lgamma/log/log1p` 计算。对离散 support 的 EM 而言，这部分是常量项，本质上可以缓存。

影响：

- 计算复杂度的主项没有变，但每轮常数项非常大
- `max_em_iterations` 越大，这种重复越吃亏
- `negative_binomial` 和 `binomial` 的收益尤其大，因为 likelihood 本身更贵

结论：

- 这是当前 `fit` 路径里最值得优先处理的优化点
- 即使不改算法，只做 likelihood cache，也很可能是收益最大的单点工程优化

建议：

- 增加 `cache_likelihood: bool = True`
- 按 `cell_chunk` 预先计算并缓存 `log_likelihood_chunk`
- 每轮 EM 只做：
  - `log_posterior = log_likelihood_chunk + log_prior`
  - `logsumexp`
  - `posterior sum`
- 若显存不足，则支持：
  - GPU chunk cache
  - CPU pinned-memory cache
  - 自动 fallback 到“边算边用”

### 3.2 `infer` 没有 chunking，峰值内存按 `genes * cells * support` 线性爆炸

热点位置：

- `src/prism/model/infer.py` 中 `infer_posteriors(...)`
- `src/prism/model/kbulk.py` 中 `infer_kbulk_samples(...)`

当前 `infer` 与 `kbulk` 都是一次性把整个 `counts_t`、`support_t`、`prior_probabilities_t` 喂给 inferencer，然后直接产出 `posterior_probabilities_t`。

posterior/log-likelihood 的核心张量规模是：

`n_genes * n_cells * n_support_points * dtype_bytes`

以 `float32` 为例：

- `256 genes * 2048 cells * 512 support ≈ 1.0 GiB`
- `2000 genes * 20000 cells * 512 support ≈ 76.3 GiB`

这还是单个大张量，不包含额外中间张量、输出、副本和 autograd/allocator 开销。

影响：

- 大数据下 `infer_posteriors(...)` 很容易成为 OOM 源头
- `include_posterior=True` 时，问题会进一步放大
- `kbulk` 复用了几乎同样的策略，也有同样问题

结论：

- `fit` 已经有 `cell_chunk_size`，`infer` 却没有对应机制，这是当前接口层最明显的不对称点

建议：

- 给 `infer_posteriors(...)` / `infer_kbulk(...)` 增加：
  - `cell_chunk_size`
  - 可选的 `gene_chunk_size`
- 当 `include_posterior=False` 时，不保留完整 posterior，只在线聚合：
  - MAP
  - posterior entropy
  - prior entropy
  - mutual information
- 只有 `include_posterior=True` 时才保留完整三维张量

### 3.3 输出路径大量强制回到 `np.float64`，抵消了 `float32` 的收益

热点位置：

- `src/prism/model/constants.py`：`DTYPE_NP = np.float64`
- `src/prism/model/infer.py`：`InferenceResult` 返回前统一 `.astype(DTYPE_NP, copy=False)`
- `src/prism/model/posterior.py`：再次统一 `np.asarray(..., dtype=np.float64)`
- `src/prism/model/types.py`：`_as_matrix/_as_vector/_as_probabilities/_as_support` 全部按 `DTYPE_NP` 归一

这使得当前链路有两个问题：

1. 即使用户选择 `torch_dtype="float32"`，输出仍然会被提升到 `float64`
2. `float32` 在计算端节省的显存/内存，经常会在返回结果时被重新吃回去

小规模 CPU 合成基准结果：

| 配置 | `fit` | `infer(include_posterior=False)` | `infer(include_posterior=True)` |
| --- | ---: | ---: | ---: |
| `float32` | 8.32s | 0.91s | 0.94s |
| `float64` | 9.47s | 1.01s | 1.20s |

基准条件：

- `n_cells=256`
- `n_genes=32`
- `n_support_points=64`
- `max_em_iterations=4`
- CPU，`compile_model=False`

结论：

- 在这个实现上，`float32` 已经能稳定给出 10% 到 20% 的收益
- 如果输出继续强制回到 `float64`，收益会被明显稀释

建议：

- 将“内部计算 dtype”和“对外结果 dtype”拆开
- 增加 `result_dtype: Literal["float32", "float64"]`
- 对 inference/kbulk 默认优先输出 `float32`
- 对 checkpoint 和严肃分析报告再按需升到 `float64`

### 3.4 `torch.compile` 的冷启动代价很高，不适合作为通用默认值

热点位置：

- `src/prism/model/fit.py`：`compile_model=True`
- `src/prism/model/infer.py`：`compile_model=True`
- `src/prism/model/kbulk.py`：`compile_model=True`

小规模 CPU 合成基准结果：

| 配置 | 第一次调用 | 第二次调用 |
| --- | ---: | ---: |
| `compile_model=False` | 0.75s | 0.78s |
| `compile_model=True` | 48.69s | 0.40s |

基准条件：

- `infer_posteriors(...)`
- `n_cells=128`
- `n_genes=16`
- `n_support_points=48`
- CPU，`float32`

结论：

- `torch.compile` 在重复同形状调用下可能有收益
- 但对单次或少量调用任务，冷启动代价非常大
- 当前将其设为默认开启，风险偏高

建议：

- 默认改为 `compile_model=False`
- 或者仅在满足下面条件时自动开启：
  - 设备是 CUDA
  - shape 相对稳定
  - 预期会重复调用多次
- 或提供 `compile_policy = "never" | "auto" | "always"`

### 3.5 校验和对象重建在热点路径里做了整数据扫描

热点位置：

- `src/prism/model/types.py`：
  - `ObservationBatch.check_shape()`
  - `GeneBatch.__post_init__()`
  - `GeneBatch.to_observation_batch()`
  - `InferenceResult.__post_init__()`
  - `PriorFitResult.__post_init__()`

几个典型问题：

1. `ObservationBatch.check_shape()` 通过重新构造一个 `ObservationBatch` 来完成验证
2. `GeneBatch.__post_init__()` 先构造 `ObservationBatch`，再重新 `np.asarray(...)`
3. `InferenceResult.__post_init__()` 对 2D/3D 大数组再次做全量 `np.asarray + shape check + finite check`
4. `PriorGrid.support`、`PriorGrid.prior_probabilities`、`PriorGrid.scaled_support` 是 property，每次访问都会再次包装

这些操作单次看不大，但当：

- 批次很多
- `include_posterior=True`
- 推断结果很大

时，会形成明显的“结果整理成本”。

建议：

- 区分“外部输入校验对象”和“内部快路径对象”
- 给 `InferenceResult` / `PriorFitResult` 增加内部构造函数，例如：
  - `from_trusted_arrays(...)`
- 对已验证过的 `ObservationBatch/GeneBatch` 避免重复校验
- 对 `PriorGrid` 预缓存：
  - `support_np`
  - `prior_probabilities_np`
  - `scaled_support_np`

### 3.6 `support_max` 的构造会额外物化大矩阵

热点位置：`src/prism/model/fit.py` 中 `_default_probability_support_max(...)`

当前做法会构造：

`observed_probabilities = counts / effective_exposure[:, None]`

这会额外生成一个和 `counts` 同规模的 dense 浮点矩阵，然后再对其做：

- `max(axis=0)`，或
- `nanpercentile(..., axis=0)`

影响：

- 在大 `n_cells * n_genes` 下会出现一次额外的大内存峰值
- `quantile` 路径比 `observed_max` 更重

建议：

- `observed_max` 用 streaming/chunked max
- `quantile` 用近似分位数：
  - 随机采样 cell
  - t-digest / GK sketch
  - 分块 percentile 合并

### 3.7 `infer` 和 `kbulk` 的逻辑高度重复，维护成本高，也阻碍统一优化

热点位置：

- `src/prism/model/infer.py`
- `src/prism/model/kbulk.py`

重复内容包括：

- dtype 解析
- prior 选择
- support / prior tensor 构造
- binomial / negative_binomial / poisson 分发
- 输出 `InferenceResult` 的转换与搬运

影响：

- 同一个优化要做两遍
- 更容易出现默认值不一致
- 更容易在一个路径优化了，另一个路径遗漏

建议：

- 抽出统一的 `_infer_core(...)`
- 只让 `infer_posteriors(...)` 和 `infer_kbulk_samples(...)` 负责输入适配
- 所有 chunking / dtype / compile 策略统一放在 core 层

## 4. 优先级建议

### P0：优先马上做，收益最高

1. 给 `infer` / `kbulk` 增加 chunking
2. 把 `result_dtype` 从 `torch_dtype` 中解耦，默认允许输出 `float32`
3. 将 `compile_model` 默认值改成更保守的策略
4. 消除热点路径中的重复校验与重复 `np.asarray(...)`
5. 在 `fit` 里缓存固定 support 下的 likelihood

### P1：中期做，兼顾性能和结构

1. 统一 `infer` 与 `kbulk` 的核心逻辑
2. 支持 `gene_chunk_size`
3. 缓存 `PriorGrid` 的 lookup / scaled support
4. 支持 shape-stable 的 compile 策略
5. 将 `support_max` 改为流式统计

### P2：研究型优化

1. 多阶段 coarse-to-fine support
2. support pruning / active set EM
3. 近似分位数 support 上界
4. 基于基因统计特征的 warm start / support 共享

## 5. 算法层面的启发式提速方法

以下方法不一定保持与当前实现完全同轨，但在大规模任务里通常值得考虑。

### 5.1 多阶段 coarse-to-fine support

当前 `PriorFitConfig.n_support_points` 默认固定为 512，对所有基因统一处理。这个策略稳，但比较贵。

可改为三阶段：

1. 阶段 1：32 或 64 个 support 点粗拟合
2. 阶段 2：围绕高 posterior 质量区间重建 128 个点
3. 阶段 3：只对高熵/难收敛基因继续细化到 256 或 512 个点

优点：

- 对低表达、单峰、低不确定性基因，通常不需要 512 个点
- 可以显著降低平均 support 规模

建议：

- 把当前 `use_adaptive_support` 的单次 refinement 扩展成多阶段 refinement
- refinement 条件可由以下指标联合决定：
  - posterior entropy
  - objective 改善幅度
  - MAP 是否靠近 support 边界

### 5.2 Active-set EM：只更新还没收敛的基因

当前 `_run_em_pass(...)` 用的是全局 `max(abs(updated - probabilities_t))` 作为停止条件。这意味着只要有一个基因没收敛，所有基因都继续参与下一轮。

可以改为：

- 为每个基因维护 `delta_gene`
- 对已经收敛的基因冻结 `probabilities`
- 下一轮只对 active genes 继续计算 posterior

优点：

- 对“快收敛基因很多、慢收敛基因很少”的情况特别有效
- 可以和 chunking 同时使用

### 5.3 Support pruning：迭代中淘汰长期低质量 support 点

在 EM 的若干轮之后，很多 support 点会长期几乎没有 posterior mass。

启发式做法：

- burn-in 若干轮后
- 若某 support 点在连续 `m` 轮中平均 posterior 质量都低于阈值 `tau`
- 则从 support 中剔除
- 但保留 MAP 邻域和两端安全边界

优点：

- 后续轮次的 `support` 维度会越来越小
- 对高分辨率 support 非常有用

风险：

- 过早 pruning 可能剪掉次峰

因此建议：

- 只在 posterior entropy 已经明显下降后启用
- 保留每个峰附近固定宽度的 bins

### 5.4 基因级自适应 support 大小

当前所有基因都使用统一的 `n_support_points`。但不同基因的统计复杂度差异很大。

可以用以下特征给基因分配不同 support 大小：

- mean expression / scaled mean
- zero fraction
- dispersion / overdispersion proxy
- 上一阶段 posterior entropy

一个简单策略：

- 简单基因：32 或 64
- 中等基因：128
- 复杂基因：256 或 512

### 5.5 近似分位数与采样式 support 上界

`support_max_from="quantile"` 本质上是为了让 support 上界别被极端值主导。

但如果全量 percentile 太贵，可以改成：

- 随机采样一部分 cells
- 按 reference count 分层采样
- 用近似分位数 sketch

对 support 上界这种“粗定位”任务，近似统计通常足够。

### 5.6 基于先验特征的 warm start

当前 `_initialize_probabilities(...)` 默认均匀初始化。

可以考虑的 warm start：

- 用相邻批次/相邻 label 的 prior 作为初始化
- 按基因均值/方差聚类，对同簇共享初始化
- 同一基因在多次重复拟合时直接使用上一次 posterior mean

优点：

- 减少 EM 迭代数
- 对重复训练任务尤其有用

### 5.7 面向只读摘要任务的近似 posterior

很多下游只需要：

- MAP
- signal
- posterior entropy
- mutual information

并不需要完整 posterior 分布。

这时可以考虑：

- 仅在 top-k prior mass 或 MAP 邻域 support 上算 posterior
- 先粗网格定位，再细网格局部重算

这类启发式对交互式分析、批量导出尤其实用。

## 6. 建议的落地顺序

### 第一阶段：先把工程大头做掉

目标：先把 OOM 风险和明显重复计算压下去

建议顺序：

1. `infer` / `kbulk` 增加 chunking
2. 结果 dtype 改为可配置，默认允许 `float32`
3. `compile_model` 策略改为保守默认
4. 去掉热点路径中的重复校验

### 第二阶段：重构 `fit` 的 EM 实现

目标：降低每轮常数项

建议顺序：

1. 支持 likelihood cache
2. 加 active genes 机制
3. 如果需要，再做 support pruning

### 第三阶段：引入启发式算法

目标：降低平均 support 规模和平均迭代数

建议顺序：

1. coarse-to-fine support
2. 基因级 support 大小
3. warm start

## 7. 建议补充的评测指标

为了避免“优化后更快但结果漂了”，建议把下面几类指标固定下来：

### 性能指标

- wall-clock time
- peak CPU memory
- peak GPU memory
- compile cold-start time
- compile warm-run time

### 数值一致性指标

- `final_objective`
- posterior mean 的 L1 / KL / JSD
- MAP support 一致率
- posterior entropy 误差

### 数据规模维度

- `n_cells`
- `n_genes`
- `n_support_points`
- likelihood 类型
- `include_posterior` 是否开启
- `float32/float64`

## 8. 总结

如果只看“单位工作量的收益”，当前最值得优先做的不是零碎微优化，而是下面四件事：

1. `fit` 中缓存固定 support 下的 likelihood，减少 EM 的重复计算
2. `infer`/`kbulk` 引入 chunking，避免 `genes * cells * support` 级别的大张量峰值
3. 解耦结果 dtype，真正让 `float32` 在输出链路上生效
4. 把 `torch.compile` 改成按场景启用，而不是无条件默认开启

如果这四件事先做完，再叠加：

- active-set EM
- coarse-to-fine support
- support pruning

这个模块在大规模任务下的效率会有比较明显的提升，而且工程复杂度仍然可控。

## 9. 代码定位附录

下面给出与本文结论直接相关的代码位置，便于后续按点实施：

- `src/prism/model/fit.py`
  - `_default_probability_support_max(...)`：support 上界构造与大矩阵物化
  - `_adaptive_refine_support(...)`：当前单阶段 adaptive support
  - `_run_em_pass(...)`：EM 主循环与 likelihood 重复计算
  - `fit_gene_priors(...)`：`compile_model`、dtype、binomial validation、adaptive support 总入口
- `src/prism/model/infer.py`
  - `_BasePosteriorInferencer._summarize(...)`：MAP / entropy / MI 汇总
  - `infer_posteriors(...)`：当前一次性整批推断与结果搬运
- `src/prism/model/kbulk.py`
  - `infer_kbulk_samples(...)`：与 `infer_posteriors(...)` 基本重复的推断核心
- `src/prism/model/types.py`
  - `ObservationBatch.check_shape()`：通过重建对象重复校验
  - `GeneBatch.__post_init__()` / `to_observation_batch()`：重复 `np.asarray(...)`
  - `PriorGrid.support` / `prior_probabilities` / `scaled_support`：热点 property 包装
  - `InferenceResult.__post_init__()`：大结果对象的再次全量验证
- `src/prism/model/posterior.py`
  - `summarize(...)` / `summarize_batch(...)` / `extract(...)`：再次转换为 `np.float64`
- `src/prism/model/exposure.py`
  - `mean_reference_count(...)` / `effective_exposure(...)`：重复 reference 统计
- `src/prism/model/numeric.py`
  - `log_*_likelihood_support(...)`：likelihood 核心计算，适合成为缓存边界
  - `posterior_from_log_likelihood(...)`：E-step 的公共归一化逻辑
