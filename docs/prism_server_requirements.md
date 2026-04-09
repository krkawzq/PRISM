# 新版 `prism.server` 需求文档

## 1. 背景

旧版 server 建立在重构前的 PRISM API 之上，依赖了已经移除或不再稳定的接口，例如：

- 旧 checkpoint 字段结构
- 已删除的 `prism.baseline` 分析逻辑
- 旧版 `server` 中围绕 `global_eval` 的派生能力

当前仓库中的稳定模型接口已经收敛到 `src/prism/model`，核心能力包括：

- `ObservationBatch`
- `PriorGrid`
- `ModelCheckpoint`
- `fit_gene_priors`
- `infer_posteriors`
- `Posterior`
- `infer_kbulk`
- `KBulkBatch`

因此新 server 必须围绕这一套接口重新设计，而不是迁移旧版页面逻辑。

## 2. 产品目标

新版 server 的目标是成为一个本地交互式分析面板，用于：

1. 加载 `.h5ad` 数据集
2. 加载并检查 PRISM checkpoint
3. 浏览和搜索基因
4. 查看基因在数据集中的原始表达统计
5. 查看 checkpoint 中的 global prior / label prior
6. 对单基因执行 posterior 推断并查看关键 channel
7. 在浏览器里对单基因执行 on-demand fit
8. 对单基因做 kBulk 抽样分析

这个 server 面向单用户、本地分析、可解释调试，不是面向多租户生产环境。

## 3. 非目标

下面这些内容不属于本次新版 server 的目标范围：

- 用户系统、鉴权、权限控制
- 多用户隔离会话
- 文件上传
- 持久化任务队列
- 旧版 baseline metrics 体系
- 旧版 global evaluation 页面
- checkpoint 编辑、合并、写回
- 通用 REST 平台化接口

## 4. 设计原则

### 4.1 只依赖稳定的 `prism.model`

server 只允许依赖当前 `src/prism/model`、`src/prism/io`、`src/prism/plotting` 和稳定 CLI 共用辅助逻辑，不再引用旧版 `prism_v1` 代码。

### 4.2 以 checkpoint 原生语义为中心

server 展示和交互要围绕当前 checkpoint 真实支持的概念：

- global prior
- label prior
- support domain
- distribution
- scale metadata
- reference gene set

而不是把旧版概念硬映射过来。

### 4.3 页面能力优先，JSON 作为辅助

server 首先是本地分析 UI，因此以服务端渲染 HTML 页面为主。保留少量 JSON API 用于健康检查和搜索。

### 4.4 显式区分三类分析来源

单基因分析需要清晰区分：

- raw-only
- checkpoint posterior
- on-demand fit posterior

页面必须始终显示分析来源。

## 5. 用户故事

### 5.1 数据加载

作为用户，我希望输入 `.h5ad` 路径、可选 `layer`、可选 checkpoint 路径后，server 能：

- 读取数据集
- 给出细胞数、基因数、总 count 分布概览
- 识别可用 label 列
- 如果加载了 checkpoint，给出 checkpoint 摘要

### 5.2 checkpoint 检查

作为用户，我希望看到 checkpoint 的关键结构信息：

- 是否有 global prior
- 有多少 label priors
- 可用 label 列表预览
- distribution
- support domain
- scale
- mean reference count
- reference gene 数量和与数据集的 overlap

### 5.3 基因浏览

作为用户，我希望可以：

- 按基因名子串搜索
- 按总 count、检测率、索引排序
- 点击进入基因详情页

### 5.4 基因详情页

作为用户，我希望在一个页面中看到：

- 基因原始统计
- 当前分析模式
- prior 来源和配置
- checkpoint prior 曲线
- posterior signal / entropy / mutual information 诊断
- on-demand fit 入口
- kBulk 入口

### 5.5 checkpoint posterior 分析

作为用户，我希望使用 checkpoint 对单基因做 posterior 推断。

系统必须支持两种 prior source：

- `global`
- `label`

其中：

- `global` 表示对全部细胞使用 global prior
- `label` 表示用户选择某个 label，页面只对该 label 对应细胞子集分析，并使用该 label prior

### 5.6 on-demand fit

作为用户，我希望可以在当前页面上对单基因重新拟合 prior，并立即看到：

- 新 prior 曲线
- 与 checkpoint prior 的对比
- objective 历史曲线
- 用新 prior 重新做 posterior 后的 signal / entropy 图

### 5.7 kBulk 分析

作为用户，我希望在一个选定 gene 上，对某个 class key 的多个组进行 kBulk 抽样，并查看：

- 每组样本数
- 每组抽样次数
- 每组 MAP signal 分布
- 每组 posterior entropy 分布

## 6. 核心数据契约

### 6.1 数据集侧

server 从 `.h5ad` 中需要这些信息：

- `adata.X` 或指定 `layer`
- `adata.var_names`
- `adata.obs`

server 需要在加载时预计算：

- 每细胞总 count
- 每基因总 count
- 每基因检测细胞数
- 每细胞零比例
- 基因搜索索引
- 可用 label 列

### 6.2 checkpoint 侧

server 假定 checkpoint 遵循 `prism.model.ModelCheckpoint`：

- `prior`
- `label_priors`
- `scale_metadata`
- `label_scale_metadata`
- `fit_config`
- `metadata`

并且需要从 metadata 中消费：

- `reference_gene_names`
- `fit_distribution`
- `posterior_distribution`
- `support_domain`
- `label_key`（若存在）

### 6.3 reference genes

checkpoint-backed posterior 和 kBulk 分析都依赖 reference genes。

规则：

1. 默认从 checkpoint metadata 的 `reference_gene_names` 读取
2. 只保留和数据集重叠的 genes
3. overlap 为空时，checkpoint-backed 分析不可用

## 7. 页面与路由需求

### 7.1 页面路由

必须提供：

- `GET /`
  - 首页
- `GET /load`
  - 加载数据和 checkpoint
- `GET /gene`
  - 基因详情页
- `GET /api/health`
  - 健康检查
- `GET /api/search`
  - 基因搜索
- `GET /assets/*`
  - 静态资源

### 7.2 首页需求

首页必须展示：

- 加载表单
- 数据集摘要
- checkpoint 摘要
- 可用 label 列
- gene browser

### 7.3 基因页需求

基因页必须支持 query 参数：

- `q`
- `mode`
  - `checkpoint`
  - `fit`
- `prior_source`
  - `global`
  - `label`
- `label_key`
- `label`

fit 参数至少支持：

- `scale`
- `reference_source`
  - `checkpoint`
  - `dataset`
- `n_support_points`
- `max_em_iterations`
- `convergence_tolerance`
- `cell_chunk_size`
- `support_max_from`
- `support_spacing`
- `use_adaptive_support`
- `adaptive_support_fraction`
- `adaptive_support_quantile_hi`
- `likelihood`
- `nb_overdispersion`
- `torch_dtype`
- `compile_model`
- `device`

kBulk 参数至少支持：

- `class_key`
- `k`
- `n_samples`
- `sample_seed`
- `max_classes`
- `sample_batch_size`
- `kbulk_prior_source`

## 8. 分析能力需求

### 8.1 原始统计

对任意 gene，server 必须能给出：

- 总 count
- 平均 count
- 中位数
- P90
- P99
- 最大值
- 检测细胞比例
- 零比例
- 与总 count 的相关性

### 8.2 checkpoint prior 可视化

对于当前 gene，server 必须支持：

- 展示 global prior 曲线
- 若 checkpoint 有 label priors，展示选定 label 的 prior 曲线
- 在可行时对 global 和 selected label 做 overlay

### 8.3 checkpoint posterior 分析

对当前 gene，server 必须支持展示：

- signal
- map_probability
- map_support
- posterior_entropy
- prior_entropy
- mutual_information

同时提供以下图：

- raw count histogram
- raw count vs total count scatter
- signal vs raw proxy scatter
- posterior entropy histogram
- prior curve 图
- posterior gallery

### 8.4 on-demand fit 分析

server 必须能基于当前页面参数构造 `PriorFitConfig`，并执行：

- 选 reference genes
- 计算 reference counts
- 设定 scale
- `fit_gene_priors(...)`
- 基于拟合 prior 的 `Posterior.summarize(...)`

输出需要包含：

- 新 prior 曲线
- objective history
- 新 posterior 诊断
- 与 checkpoint prior 的对比

### 8.5 kBulk 分析

server 必须能对当前 gene 在所选 `class_key` 下执行：

- group 划分
- 每组随机抽样 k 个细胞
- 基于 prior 进行 `infer_kbulk(...)`
- 汇总每组 MAP signal / posterior entropy

对 binomial / negative_binomial 情况，effective exposure 必须使用：

- `effective_exposure(reference_counts, prior.scale)` 逐细胞计算后再按组合求和

而不是直接把 reference counts 总和当作 effective exposure。

## 9. 缓存需求

server 需要按当前数据上下文做缓存，至少包括：

- 数据集摘要
- 基因搜索结果
- 原始统计
- checkpoint posterior 分析结果
- on-demand fit 结果
- figure data URI

加载新的 `.h5ad` 或 checkpoint 后必须清空缓存。

## 10. 可视化需求

server 使用 Matplotlib，图像以 data URI 嵌入 HTML。

至少需要这些图：

- raw overview
- prior overlay
- objective trace
- signal interface
- posterior gallery
- kBulk group comparison

## 11. 运行约束

### 11.1 并发模型

使用单进程共享 `AppState`，允许多线程请求，但不做多用户隔离。

### 11.2 文件访问

server 只接收本地路径，不负责上传。

### 11.3 错误展示

加载失败、gene 不存在、checkpoint 缺少 global prior、label 不匹配等错误必须直接在 HTML 页面上可见。

## 12. 实现范围

本次实现必须交付：

1. 新版 `src/prism/server` 包
2. 新版 `prism cli serve`
3. 一套可工作的服务端渲染页面
4. 最小健康检查和搜索 API
5. 新模型下的单基因 checkpoint posterior、on-demand fit、kBulk 页面能力

## 13. 验收标准

满足以下条件视为完成：

1. `import prism.server` 成功
2. `python -m prism.cli.main serve --help` 可用
3. server 可以启动
4. 可加载 `.h5ad`
5. 可加载 checkpoint 并显示摘要
6. 可以浏览和搜索基因
7. 可以在 gene 页面查看 checkpoint posterior
8. 可以触发 on-demand fit
9. 可以触发 kBulk 分析

