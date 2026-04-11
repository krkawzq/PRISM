# PRISM GMM Hotpath Refactor Todo

## 范围

本轮重新审查 `src/prism/gmm` 全模块，覆盖：

- bug 与配置陷阱
- schema / report / search 的校验缺口
- `search / fit / optimize / numeric` 的热点瓶颈
- 不改变统计语义前提下可直接落地的工程优化

## 重新审查结论

### 1. 已确认并已修复的问题

- [x] `GMMTrainingConfig` 缺少 pruning 指标的早失败校验
  之前 `pruning_error_metric` / `pruning_significance_metric` 非法时，要到运行中更深层逻辑才报错；现在已在配置层直接校验。

- [x] `DistributionGMMSearch` 对 `support_mask` / padding 槽位约束不足
  之前允许非 prefix mask、masked-out probabilities 非零、active support / bin_edges 无严格单调保证；这些状态会让 search/fit 热路径在隐含前提下工作，风险较高。现在统一做结构校验。

- [x] `fit_distribution_gmm()` / `fit_prior_gmm()` 只校验 `support/probabilities`
  之前用户传入一个“support 一样但 bin_edges / bounds / mask 不一致”的 `search` 对象时，入口未必能拦住。现在已把 `support_mask/bin_edges/lower_bounds/upper_bounds` 一并对齐检查。

- [x] `config` 字段表面 frozen，内部实际可变
  `DistributionGMMSearch` / `DistributionGMMReport` / `PriorGMMReport` 的 `config` 原来是可变 `dict`，外部可以在构造后修改，导致对象状态漂移。现在统一冻结为只读 mapping。

- [x] `DistributionGMMReport` 校验路径通过临时构造 `DistributionGMMSearch` 复用逻辑
  这会制造额外 `repeat/zeros` 和不必要的临时对象，也让校验路径难维护。现在改为直接校验自身字段，不再重建 search 对象。

- [x] `DistributionGMMReport.to_mixture()` 通过临时构造 `DistributionGMMSearch` 间接实现
  现在改为直接从组件行构造 mixture，并补上 index range 检查。

- [x] `PriorGMMReport` 缺少结果一致性校验
  现在补上：
  `support_domain` 合法性
  `fitted_probabilities` 行和为 1
  masked-out fitted/residual 为 0
  `fitted + residual == probabilities`

### 2. 已完成的热点优化

- [x] 搜索候选打分矩阵化
  `_score_peak_candidates()` 不再逐候选做 Python 循环，改为整批计算 `alpha/improvement`。

- [x] 跨 peak 候选批量评估
  同一 gene/stage 内多个 peak 的候选参数先合并，再统一调用一次 `truncated_gaussian_bin_masses()` 做评分，减少 `ndtr` 调用批次。

- [x] 自定义 masked quantile 快路径
  用列排序 + 线性插值替换 `np.nanquantile(..., axis=0)` 的高开销路径。

- [x] `window_radii` 缓存
  为重复 `(n_points, candidate_window_count)` 组合增加缓存。

- [x] refit 初始化参数向量化
  `initialize_raw_parameters()` 去掉逐 gene / 逐 component 双层循环。

- [x] search 单行 dense kernel
  `_score_candidate_parameters()` 在 dense support 下直接走 `truncated_gaussian_bin_masses_dense_1d()`，不再把单 row 候选评分塞回通用 padded/masked batch kernel。

- [x] refit/evaluate workspace cache
  为 shape-stable 的非 CPU 路径增加 tensor workspace，复用 `target/bin_edges/mask/active_mask` 等输入张量；CPU 路径保留 `torch.as_tensor(...)`，避免 `copy_` 触发 PyTorch functionalization / dynamo 懒加载带来的冷启动回退。

- [x] refit no-op fast path
  `optimize_mixture_parameters()` 在 `max_iterations == 0` 或全部参数关闭优化时直接退化为 `evaluate_mixture_parameters()`，避免无意义地构造 module / optimizer / autograd 图。

- [x] compile 策略升级为 policy
  `GMMTrainingConfig` / `GMMSearchConfig` 新增 `compile_policy=never|auto|always`，同时兼容旧的布尔 `compile_model` / `search_refit_compile_model`。

- [x] RefitModule 静态张量预计算
  `span/mean_low/mean_span/min_window/bin_edges_left/bin_edges_right/bin_mask_expanded/active_component_count` 现在在模块初始化时缓存，减少每轮 forward 的重复张量整形。

- [x] 增加基准脚本
  新增 `scripts/experiments/benchmark_gmm_hotpath.py`，可固定 `n_genes/n_support/max_components/max_iterations` 复测 search / fit wall time，并支持 `cProfile`。

## 当前基准与热点

### 搜索阶段

- [x] `16 genes x 97 support` 合成样例，`max_components=4`、`search_refit_enabled=False`
  使用：
  `python scripts/experiments/benchmark_gmm_hotpath.py --genes 16 --support 97 --max-components 4 --fit-iterations 10 --warmup 1 --repeat 2 --torch-dtype float64 --device cpu`
  当前默认单峰合成样例（`true-components=1`）warm run 约 `3.65s ~ 3.74s`，mean `3.69s`。

- [x] `8 genes x 65 support` profile
  使用：
  `python scripts/experiments/benchmark_gmm_hotpath.py --genes 8 --support 65 --max-components 4 --fit-iterations 10 --warmup 1 --repeat 1 --profile both --profile-top 15`
  search 侧热点已经下沉到：
  `truncated_gaussian_bin_masses_dense_1d`
  `torch.special.ndtr`
  `_extract_stage_components_for_gene`
  说明单行 dense kernel 已经接入，剩余大头主要是 CDF 数值核本体。

### 拟合阶段

- [x] `16 genes x 97 support` 合成样例，`max_iterations=10`
  使用同一 benchmark 脚本 warm run 约 `7.95s ~ 7.98s`，mean `7.96s`。

- [x] `8 genes x 65 support` warm profile
  主要时间仍在：
  `run_backward`
  `torch.special.ndtr`
  `truncated_gaussian_bin_masses_from_edges`

- [x] fit 冷启动已单独识别
  当前环境里第一次 `optimize_mixture_parameters()` 会额外支付一次明显的 PyTorch 懒加载成本（`torch._dynamo` / `fsdp` / `sympy` 等导入链）；这不是 GMM 算法本体的 steady-state 开销，后续 wall time 结论统一以 warm run 为准。

- [x] report/schema 校验不是总运行时主热点
  `DistributionGMMReport` / `PriorGMMReport` 的重建校验平均约 `1ms` 量级，相对完整 fit 耗时占比很低；本轮更偏向修正确性与可维护性，而不是追求这一段的吞吐。

## 已发现，暂未继续下钻

- [ ] search / fit 仍共同受 `truncated_gaussian_bin_masses()` 主导
  当前最昂贵的底层核仍是截断高斯 bin mass 计算，尤其是 `torch.special.ndtr`。

- [ ] fit 的 autograd 开销已经接近数值核本体
  warm profile 里 `run_backward` 也是主项；进一步提速要么减少迭代次数，要么减少参数/图复杂度，要么专门优化单 row/小 K 情况。

- [ ] `build_bin_edges()` 在“提供 search 但仍要求严格入口校验”时会产生可见前置开销
  这是正确性和性能之间的权衡。目前保留严格校验，不做 risky fast path。

- [ ] `compile_policy=auto` 仍然是保守启发式
  目前只在较大 CUDA workload 下才会尝试编译；如果后续要继续打磨 GPU 吞吐，还可以补更细的 shape / iteration / device 级策略。

- [ ] workspace cache 目前只在非 CPU 路径启用
  CPU 上直接 `copy_` 到复用 tensor 会被当前 PyTorch 版本的 functionalization 路径放大冷启动，因此本轮保留“CPU 直接 as_tensor，非 CPU 复用 workspace”的折中方案。

## 后续更激进方向

- [ ] 为截断高斯 bin mass 设计更底层的 fused kernel / 近似 CDF
  这是 search 与 fit 共同的终极热点；不动这一层，很难再拿到数量级收益。

- [ ] 为低 `k` / fixed-bounds 做更激进的 refit 专用 fast path
  当前已有 no-op fast path，但 `k=1/2`、固定 bounds、固定部分参数仍有进一步裁剪 backward 图的空间。

- [ ] 继续区分 cold / warm / profile 三套口径
  现在 benchmark 脚本已经落地；如果要长期追踪，可以再补固定输出格式（例如 CSV / JSON）和历史结果归档。

## 验证记录

- [x] `python -m compileall src/prism/gmm src/tests/test_model_gmm.py scripts/experiments/benchmark_gmm_hotpath.py` 通过
- [x] `python -m pytest src/tests/test_model_gmm.py -q` 通过，当前 `24 passed`
- [x] 合成 benchmark 已复测
  包括新的 benchmark 脚本 wall time 以及 `--profile both` 的 search/fit cProfile；文档中的收益与热点均来自当前代码而不是旧结果。
