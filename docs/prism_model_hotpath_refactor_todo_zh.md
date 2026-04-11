# PRISM Model Hotpath Refactor Todo

## 范围

本轮只做 `src/prism/model` 的热点工程优化，不修改算法逻辑，不引入启发式近似，不改变结果语义。

## 已完成

- [x] EM `finalize` 融合
  将训练阶段按 chunk 的 `posterior_from_log_likelihood + 再算 logsumexp` 合并为单次 `log_joint` 归一化，去掉重复 `log_prior` / `logsumexp` 计算。

- [x] 训练 support 项预计算
  对 probability/rate support 的 `log(p)`、`log(1-p)`、`log(rate)` 做一次性预计算，避免 EM 循环中重复构造。

- [x] 训练 observation 常量缓存
  为 binomial / negative binomial / poisson 的 chunk 常量项增加缓存结构，避免每轮 EM 重复做 `lgamma` 和 shape/broadcast 准备。

- [x] chunk 级 likelihood cache 保留并接入新内核
  保持大显存场景下的 likelihood cache，同时让 cache miss 路径也走新的 observation/support 预计算逻辑。

- [x] 修复 likelihood cache 命中时的重复 observation 构造
  训练路径在 cache hit 时不再每轮重建 chunk observation terms，避免“cache 生效但前置常量仍重复计算”的隐性浪费。

- [x] infer/kbulk 改成统一分块执行路径
  为 posterior inference 和 kbulk inference 增加 `observation_chunk_size`，并加入自动 chunk 选择，降低大批量推断的峰值显存和中间张量体积。

- [x] 推断输出改为预分配写入
  `map_support` / `posterior_entropy` / `prior_entropy` / `mutual_information` / `posterior` 改为按 chunk 预分配和回填，避免整批结果多次 materialize。

- [x] posterior summarize 路径降额外展开
  `map_support` 改成直接按 `support` gather，不再显式展开到 `[genes, cells, support]`。

- [x] `ObservationBatch` / `GeneBatch` 轻量校验
  `check_shape()` 不再通过重建对象做校验；`GeneBatch.to_observation_batch()` 改为无重复归一化的快路径。

- [x] `PriorGrid.select_genes()` same-order 快路径
  请求顺序与现有 prior 完全一致时直接返回自身，减少 server/CLI 批量调用中的重复切片和重复 name lookup。

- [x] `Posterior` 静态数组缓存
  缓存输出 dtype 下的 `support` / `scaled_support` / `prior_probabilities`，避免重复 `np.asarray` 和重复缩放。

- [x] `adaptive_support` torch 化
  `support refine` 和 `support interpolation` 都已切到 torch 路径，避免逐 gene numpy 循环和 adaptive phase 的额外 CPU 往返。

- [x] Bugfix: `kbulk.py` 运行路径缺失 `DTYPE_NP` 导入
  这是本轮重构过程中发现的真实运行时 bug，已经并入修复。

- [x] Bugfix: `extract.common` 恢复导出 `resolve_prior_source`
  修复 `prism fit priors` 启动时被 `extract` 子命令导入链阻断的问题。

- [x] 测试补强
  新增 chunked infer / chunked kbulk 与单块结果一致性测试，以及 `PriorGrid.select_genes()` fast path 测试。

## 已发现，暂未继续下钻

- [ ] `adaptive_support` 仍走 CPU numpy 循环
  `fit.py::_adaptive_refine_support()` 仍有 GPU 到 CPU 的往返和逐 gene 的 numpy 循环；由于默认不是主热点，本轮未继续动。

- [ ] 小批量 infer 单块路径还有额外框架开销
  新的统一分块实现对大批量更稳，但非常小的推断任务可能比旧的一次性路径多一点预分配/拷贝开销。需要后续决定是否保留 single-chunk fast path。

- [ ] compile mode 未做专项 benchmark
  本轮保留 compile 相关接口和 cache key，但没有专门对编译态路径做新的收益评估。

- [ ] negative binomial 仍有可继续下沉的常量项
  目前已经缓存 support-independent 基项，但 `mu` 相关 log 项仍在每步计算。若后续继续压榨，可再做更细粒度 workspace。

## 验证记录

- [x] `.venv/bin/python -m py_compile` 通过
- [x] `src/tests/test_model_runtime_guards.py` 通过，当前 `20 passed`
- [x] CLI 导入 / dispatch 验证
  `from prism.cli.main import main` 通过，`uv run prism fit priors --help` 通过
- [x] CUDA microbenchmark
  训练样例从约 `1.52s` 降到约 `1.28s`，热点从“重复 likelihood + finalize”转移为“finalize + observation 常量构建”
- [x] 分块推断 smoke benchmark
  `8192 x 128 x 256` 规模下，`observation_chunk_size=1024` 可稳定完成，posterior 返回路径显存峰值受控
