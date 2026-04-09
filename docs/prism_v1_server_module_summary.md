# `src/prism_v1/server` 模块调研与功能总结

## 1. 调研范围

本文总结的是 `src/prism_v1/server` 当前代码中已经实现的能力，以及它依赖的直接调用链：

- `src/prism_v1/server/app.py`
- `src/prism_v1/server/router.py`
- `src/prism_v1/server/handlers.py`
- `src/prism_v1/server/state.py`
- `src/prism_v1/server/services/*`
- `src/prism_v1/server/views/*`
- `src/prism_v1/cli/serve/app.py`

同时结合了少量相邻模块的接口定义做交叉确认，例如：

- `src/prism/model/*`
- `src/prism_v1/model/*`
- `src/prism_v1/baseline/metrics.py`

目的不是解释 PRISM 全部算法，而是回答一个更具体的问题：

`src/prism_v1/server` 这个模块“设计上支持什么”，“当前仓库状态下实际上能不能直接跑起来”，以及“它对数据和 checkpoint 有什么要求”。

## 2. 一句话结论

从业务逻辑代码看，`src/prism_v1/server` 是一个轻量级、无外部 Web 框架的只读分析服务，目标是提供：

- 加载 `.h5ad` 数据集和可选 PRISM checkpoint
- 浏览、搜索和定位基因
- 查看单基因的原始表达统计
- 基于 checkpoint prior 做单基因后验分析
- 在浏览器流程里触发单基因 on-demand fit
- 做带标签分组的 kBulk 对比
- 做 checkpoint-backed 的全局表示评估
- 生成多类 PNG 图并以内嵌 data URI 的方式嵌入 HTML

但从当前仓库状态看，这个模块仍处在一次从 `prism.server` 向 `prism_v1.server` 的迁移中间态，存在多处旧包名引用和模型接口不匹配问题。结论上：

- 按代码意图，它支持的功能链路已经比较完整
- 按当前代码状态，它还不能稳定作为 `prism_v1` 的 server 直接启动

## 3. 模块整体架构

模块采用非常直接的分层：

### 3.1 启动层

- `app.py`
  - 创建 `ServerApp`
  - 持有 `ServerConfig`
  - 初始化全局 `AppState`
  - 注册 `Router`
  - 使用 `ThreadingHTTPServer` 提供服务

### 3.2 路由层

- `router.py`
  - 定义 `Request`
  - 定义 `Response`
  - 提供精确路径和前缀路径匹配
  - 只处理 GET 请求

### 3.3 处理器层

- `handlers.py`
  - 解析 query 参数
  - 调用各类 service
  - 渲染 HTML 或 JSON

### 3.4 状态层

- `state.py`
  - 加载 `.h5ad`
  - 加载 checkpoint
  - 预计算基因和细胞级统计
  - 提供多命名空间缓存

### 3.5 服务层

- `services/datasets.py`
  - 数据矩阵选择、切片、统计、基因定位
- `services/checkpoints.py`
  - checkpoint 读取与结构校验
- `services/analysis.py`
  - 数据集摘要、基因搜索/浏览、单基因分析、on-demand fit、kBulk
- `services/global_eval.py`
  - 全局表示评估
- `services/figures.py`
  - Matplotlib 绘图并转成 data URI

### 3.6 视图层

- `views/layout.py`
  - 页面壳、导航、加载表单、统计卡片
- `views/home.py`
  - 首页、数据集摘要、gene browser、global evaluation
- `views/gene.py`
  - 基因详情页、pending 页面、fit 表单、kBulk 表单

## 4. HTTP 入口与页面能力

### 4.1 当前注册的路由

`build_router()` 当前注册了 6 个入口：

| 路由 | 方法 | 作用 |
| --- | --- | --- |
| `/` | GET | 首页，展示数据集摘要、gene browser、global evaluation 入口 |
| `/load` | GET | 加载数据集和 checkpoint |
| `/gene` | GET | 基因详情页，支持 checkpoint 分析、on-demand fit、kBulk |
| `/api/health` | GET | 健康检查 JSON |
| `/api/search` | GET | 基因搜索 JSON |
| `/assets/*` | GET | 静态资源，目前主要是 CSS |

另外：

- `/favicon.ico` 返回 `204 No Content`
- 没有 POST 接口
- 没有文件上传接口
- 没有鉴权、权限控制、用户会话或持久化数据库

### 4.2 这是一个单实例共享状态服务

服务并不是“每个用户一个会话”。

当前实现里：

- 整个进程只有一个 `AppState`
- 任意一次 `/load` 都会替换当前已加载数据集
- 所有用户/请求共享同一份 loaded dataset 和 cache
- 新加载数据后会清空全部缓存

这意味着它更像一个本地分析面板或单用户工具，而不是一个真正多租户的 Web 服务。

## 5. 数据加载与前置要求

### 5.1 支持加载什么

当前加载流程支持：

- 一个必须的 `.h5ad` 文件路径
- 一个可选的 checkpoint 路径
- 一个可选的 `layer` 名称

入口是 `/load?h5ad=...&ckpt=...&layer=...`。

### 5.2 加载时会预计算什么

加载 `.h5ad` 后，`AppState` 会立即构建这些数据：

- `adata`
- 当前使用的表达矩阵 `matrix`
- 每个细胞总计数 `totals`
- 基因名数组 `gene_names`
- 小写化基因名 `gene_names_lower`
- 精确匹配字典 `gene_to_idx`
- 小写匹配字典 `gene_lower_to_idx`
- 每个基因总 UMI `gene_total_counts`
- 每个基因被检测到的细胞数 `gene_detected_counts`
- 每个细胞零值比例 `cell_zero_fraction`
- 按总表达量从高到低排序的基因索引 `ranked_gene_indices`

### 5.3 自动识别标签列

服务会从 `adata.obs` 中按顺序尝试寻找下面这些列：

- `treatment`
- `cell_type`
- `label`
- `group`

只要某列存在且唯一值数量不少于 2，就会被当成分组标签列。

这个标签列会被用于：

- gene-level baseline metrics 中的分组统计
- kBulk comparison
- global evaluation

如果这些列都不存在，或者只有单一类别，那么：

- 模块仍然可以做基本浏览和单基因分析
- 但所有依赖标签的能力会部分退化或不可用

### 5.4 checkpoint 的最小要求

按 `services/checkpoints.py` 的代码意图，server 希望 checkpoint 至少提供：

- 可用的全局 priors 或 label priors
- metadata 中的 `reference_gene_names`
- 与当前数据集有交集的 reference genes

进一步说，若要完整支持 checkpoint-backed 的单基因分析与 global evaluation，代码实际假定 checkpoint 还应具备：

- 全局 priors
- scale 元数据

否则会在后续分析阶段失败。

## 6. 当前支持的功能清单

下面的“支持”分两层理解：

- 第一层是“代码逻辑已经实现”
- 第二层是“当前仓库状态下可否直接执行”

本节先讲第一层，也就是功能设计本身。

### 6.1 首页与数据集概览

首页在加载成功后提供：

- 细胞数
- 基因数
- 已拟合基因数
- 当前 layer
- 模型来源
- `S`
- `S source`
- 平均 reference count
- reference gene 数量
- label key
- 当前 `.h5ad` 路径
- 当前 checkpoint 路径

因此它不仅是一个“载入成功页面”，也是一个简化的数据与模型快照面板。

### 6.2 Gene Browser

首页支持基因浏览，能力包括：

- 子串搜索
  - 使用基因名小写后的 substring 匹配
- 分页
  - 默认页大小来自 `browse_page_size`
- 排序
  - `total_umi`
  - `detected_cells`
  - `detected_fraction`
  - `gene_name`
  - `gene_index`
- 排序方向
  - `asc`
  - `desc`
- scope
  - `auto`
  - `fitted`
  - `all`

`scope=auto` 的逻辑是：

- 如果加载了 checkpoint，则默认只展示 fitted genes
- 如果没有 checkpoint，则展示全部基因

gene browser 列表项包含：

- gene name
- gene index
- total UMI
- detected cells
- detected fraction

### 6.3 JSON 搜索接口

`/api/search?q=...` 提供 JSON 形式的搜索结果。

返回字段为：

- `gene_name`
- `gene_index`
- `total_umi`
- `detected_cells`
- `detected_fraction`

特点：

- 未加载数据时返回空数组
- 默认最多返回 `top_gene_limit`
- 空 query 时会返回按总表达量排序的 top genes

### 6.4 基因定位方式

`/gene?q=...` 对 gene query 的解析支持：

- 精确基因名匹配
- 数字索引匹配
- 大小写不敏感的精确匹配

注意这里不是 substring 搜索。也就是说：

- `/api/search` 和 gene browser 支持模糊筛选
- `/gene?q=...` 本身要求 query 能唯一落到某个 gene 或 index

### 6.5 原始表达摘要

如果某个基因没有 checkpoint prior，且用户也没有显式触发 fit/kBulk，那么服务会进入 pending page，只展示原始层面的统计：

- 总计数
- 平均计数
- 中位数
- P90
- P99
- 最大值
- detected cells
- detected fraction
- zero fraction
- 与细胞 total count 的相关性

同时会提供一个基础图：

- 原始 count 直方图
- gene count vs total count 散点图
- `X_gc / total` 的 proxy 分布图

这部分能力不依赖 checkpoint。

### 6.6 checkpoint-backed 单基因分析

当满足以下条件时：

- 已加载 checkpoint
- 当前基因存在 checkpoint prior
- 当前请求没有显式要求 `fit=1`

服务会直接走 checkpoint-backed 的单基因分析链路。

这条链路会得到：

- 目标基因原始 counts
- 参考基因 reference counts
- 观测 proxy `counts * S / mean(reference_counts)`
- 每个细胞的 MAP `p`
- 每个细胞的 signal / MAP `mu`
- posterior entropy
- prior entropy
- mutual information
- posterior samples
- prior support grid
- prior weights

页面上会显示：

- Gene summary 卡片
- Baseline metrics 表格
- Gene overview 图
- Prior profile 图
- Signal interface 图
- Posterior gallery 图

### 6.7 On-demand single-gene fit

如果用户访问 `/gene?...&fit=1`，则服务会进行单基因即时拟合，而不是直接使用 checkpoint prior。

其核心逻辑是：

1. 选定目标基因
2. 选定 reference gene 集合
3. 计算每个细胞 reference counts
4. 设定 `S`
   - 用户传了就用用户值
   - 否则默认用 `mean(reference_counts)`
5. 调用 `fit_gene_priors(...)` 做单基因 prior 拟合
6. 用拟合得到的 prior 构造 `Posterior`
7. 计算各类单细胞后验摘要与诊断图

当前页面暴露的 fit 参数包括：

- `S`
- `reference_mode`
- `grid_size`
- `sigma_bins`
- `align_loss_weight`
- `lr`
- `n_iter`
- `lr_min_ratio`
- `grad_clip`
- `init_temperature`
- `cell_chunk_size`
- `optimizer`
- `scheduler`
- `torch_dtype`
- `device`

其中 `reference_mode` 支持：

- `checkpoint`
- `all`

含义为：

- `checkpoint`: 使用 checkpoint 中记录的 reference gene set，并排除目标基因
- `all`: 使用当前数据集除目标基因外的所有基因

如果没有 checkpoint，最终会自动退化为 `all`。

### 6.8 Baseline representation 对比

单基因分析页面不只展示 PRISM signal，也会并行构造几个 baseline 表示：

- `raw_count`
- `normalize_total`
- `log1p_normalize_total`
- `signal`

然后调用 `evaluate_representations(...)` 给出比较指标。

当前页面展示的指标包括：

- `mean`
- `median`
- `std`
- `var`
- `p95`
- `nonzero_frac`
- `depth_corr`
- `depth_mi`
- `sparsity_corr`
- `fisher_ratio`
- `kruskal_h`
- `kruskal_p`
- `auroc_ovr`
- `zero_consistency`
- `zero_rank_tau`
- `dropout_recovery`
- `treatment_cv`

其中部分指标依赖标签列。如果标签列不存在，则这些分组相关指标会显示为空。

### 6.9 kBulk comparison

页面支持通过 `/gene?...&kbulk=1` 触发 kBulk 分析。

它的逻辑不是单纯聚合原始值，而是：

1. 根据标签列把细胞分组
2. 选取样本量足够大的 group
3. 在每个 group 内重新拟合 class-specific `F_g`
4. 多次随机抽取 `k` 个细胞形成 kBulk
5. 对每次 kBulk 调用 `infer_kbulk(...)`
6. 比较不同 group 的 MAP `mu` 分布与 posterior entropy

当前页面暴露的 kBulk 参数包括：

- `kbulk_k`
- `kbulk_samples`
- `kbulk_groups`
- `kbulk_min_cells`
- `kbulk_seed`

最终页面会展示：

- 每个 group 的细胞数
- 平均 MAP `mu`
- MAP `mu` 标准差
- 平均 posterior entropy
- class-specific prior 曲线图
- kBulk MAP `mu` 分布图

### 6.10 Global evaluation

首页支持通过 `global_eval=1` 触发全局评估，但这个能力必须先加载 checkpoint。

它的逻辑是：

1. 选择标签列
2. 随机采样不超过 `max_cells` 个细胞
3. 从 fitted genes 中取表达量最高的前 `max_genes` 个基因
4. 构建四类表示矩阵
   - 原始矩阵 `X`
   - `NormalizeTotalX`
   - `Log1pNormalizeTotalX`
   - checkpoint-backed `signal`
5. 对每类表示做 PCA、KMeans、近邻一致性评估
6. 同时给出 prior entropy 最高的一批基因

页面暴露的参数包括：

- `ge_max_cells`
- `ge_max_genes`
- `ge_batch`
- `ge_seed`

输出指标包括：

- `silhouette`
- `ari`
- `nmi`
- `pca_var_ratio`
- `neighborhood_consistency`

同时会展示：

- 各表示方法在这些指标上的表格
- top prior-entropy genes 表格
- 一张全局 overview 柱状图

### 6.11 健康检查

`/api/health` 当前会返回：

- `status`
- `loaded`
- `n_cells`
- `n_genes`
- `fitted_genes`
- `has_checkpoint`

这对于外部脚本做最简单的服务存活检查是够用的。

### 6.12 静态资源与图像输出

模块当前没有单独的图片路由。

图像输出方案是：

- Matplotlib 在服务端生成 PNG
- 转成 base64
- 以 `data:image/png;base64,...` 的形式直接嵌入 HTML

优点：

- 无需管理图片文件生命周期
- 无需额外图片路由

代价：

- HTML 体积会增大
- 页面缓存粒度较粗

## 7. 查询参数与页面交互总表

### 7.1 `/`

首页支持的 query 参数：

- `q`
- `browse_q`
- `browse_sort`
- `browse_dir`
- `browse_scope`
- `browse_page`
- `global_eval`
- `ge_max_cells`
- `ge_max_genes`
- `ge_batch`
- `ge_seed`

### 7.2 `/load`

- `h5ad` 必填
- `ckpt` 可选
- `layer` 可选

### 7.3 `/gene`

基础参数：

- `q`
- `fit`
- `kbulk`

fit 参数：

- `S`
- `reference_mode`
- `grid_size`
- `sigma_bins`
- `align_loss_weight`
- `lr`
- `n_iter`
- `lr_min_ratio`
- `grad_clip`
- `init_temperature`
- `cell_chunk_size`
- `optimizer`
- `scheduler`
- `torch_dtype`
- `device`

kBulk 参数：

- `kbulk_k`
- `kbulk_samples`
- `kbulk_groups`
- `kbulk_min_cells`
- `kbulk_seed`

### 7.4 `/api/search`

- `q`

## 8. 缓存与性能特征

### 8.1 已实现的缓存命名空间

`AppState` 里当前定义了这些缓存：

- `summary`
- `search`
- `browse`
- `analysis`
- `figures`
- `html`
- `global_eval`

### 8.2 实际有使用的缓存

当前真正被业务逻辑用到的主要是：

- `summary`
- `search`
- `analysis`
- `figures`
- `global_eval`

当前几乎没有实际用到的有：

- `browse`
- `html`

### 8.3 缓存行为特点

- `summary`
  - 缓存数据集摘要
- `search`
  - 缓存 gene candidate 检索结果
- `analysis`
  - 只缓存“未显式 fit、未显式 kBulk”的 gene analysis
- `figures`
  - 缓存所有绘图结果
- `global_eval`
  - 缓存全局评估结果
- 静态资源字节还额外用了 `functools.lru_cache`

### 8.4 性能/使用侧的含义

这套设计说明模块面向的是：

- 中小规模交互式分析
- 重复查看同一批基因或同一套参数

而不是：

- 高并发生产服务
- 多用户同时加载不同数据集
- 需要长时间持有多个 project session 的环境

## 9. 当前实现中的重要限制与风险

这一节是本次调研最重要的部分之一。下面这些不是“未来可能优化”，而是当前代码里已经能看到的现实限制。

### 9.1 `prism_v1` 迁移尚未完成，启动链路存在阻塞

虽然代码位于 `src/prism_v1/server`，但当前仍有多处旧引用：

- `src/prism_v1/cli/serve/app.py` 仍然 `from prism.server import ServerConfig, run_server`
- `src/prism_v1/server/views/home.py` 仍然引用 `prism.server.services.*`
- `src/prism_v1/server/views/gene.py` 仍然引用 `prism.server.services.*`
- `src/prism_v1/server/handlers.py` 的静态资源读取仍然指向 `prism.server.assets`

这意味着按 `prism_v1` 路径启动时，会立刻碰到旧包名缺失问题。

### 9.2 服务层仍引用旧的 `prism.*` 接口，而不是 `prism_v1.*`

例如：

- `services/analysis.py` 依赖 `prism.baseline.metrics`
- `services/analysis.py` 和 `services/global_eval.py` 依赖 `prism.model`
- `services/datasets.py` 依赖 `prism.io`
- `services/checkpoints.py` 依赖 `prism.model.ModelCheckpoint`

但当前仓库里：

- `src/prism/baseline` 已不存在
- `src/prism.model` 与 `src/prism_v1.model` 的 API 已经发生了明显分叉

这不是简单的 import 路径小问题，而是接口契约已经不一致。

### 9.3 已验证的导入失败

本次调研中做了最小化导入验证，结果显示：

- `import prism_v1.cli.serve.app` 会因为 `No module named 'prism.server'` 失败
- `import prism_v1.server.views.home` 会在更早阶段因为 `prism.model` 中缺少 `OptimizerName` 导出而失败

因此“当前能否直接启动 server”这个问题的答案是：不能。

### 9.4 checkpoint 数据结构假设与当前 `prism.model` 不一致

`services/checkpoints.py` 当前假定 checkpoint 对象具备：

- `priors`
- `scale`
- `label_priors`

而当前 `src/prism/model/checkpoint.py` 中的 `ModelCheckpoint` 结构是另一套字段：

- `prior`
- `scale_metadata`
- `label_priors`
- `label_scale_metadata`

这说明即使先修掉 import，checkpoint 读取链路仍有高概率继续出错。

### 9.5 on-demand fit 的 figure cache key 过粗

`analyze_gene()` 在 `fit_params is not None` 时不会缓存 analysis 本体，这是对的；但 figure 缓存依然使用 `analysis.cache_key`，而这个 key 只区分：

- gene 名
- 是 `fit` 还是 `checkpoint`
- `kbulk` 的少量参数

它没有把全部 fit 参数编码进去。

结果是：

- 用户修改 `lr`、`n_iter`、`sigma_bins` 等参数重新 fit 时
- analysis 结果会重算
- 但图像缓存有机会复用旧图

这会导致图文不一致，是一个真实逻辑风险。

### 9.6 `plot_max_points` 目前未生效

`ServerConfig` 里定义了 `plot_max_points`，但当前 server 代码没有使用它。

因此：

- 这个参数现在只是“存在于配置中”
- 不是实际可调的性能控制开关

### 9.7 `browse` 和 `html` 缓存命名空间目前未被业务使用

这不影响正确性，但说明缓存设计和实际落地之间还有残留接口。

### 9.8 Global evaluation 对 checkpoint 形态有更强假设

虽然 checkpoint loader 允许“只存在 label priors”这类情况通过一部分校验，但 global evaluation 实际会：

- 从 `loaded.fitted_gene_names` 取基因
- 使用 checkpoint 全局 priors 提取 signal

所以想真正跑通 global evaluation，基本还是需要一个带全局 priors 的 checkpoint。

## 10. 能力矩阵

| 能力 | 设计上已实现 | 依赖 checkpoint | 依赖标签列 | 当前仓库状态可直接运行 |
| --- | --- | --- | --- | --- |
| 加载 `.h5ad` | 是 | 否 | 否 | 受迁移阻塞影响 |
| 加载 checkpoint | 是 | 是 | 否 | 受迁移阻塞影响 |
| 首页数据集摘要 | 是 | 否 | 否 | 受迁移阻塞影响 |
| Gene browser | 是 | 否 | 否 | 受迁移阻塞影响 |
| `/api/search` | 是 | 否 | 否 | 受迁移阻塞影响 |
| 原始单基因摘要 | 是 | 否 | 否 | 受迁移阻塞影响 |
| checkpoint-backed gene analysis | 是 | 是 | 否 | 受迁移阻塞影响 |
| on-demand single-gene fit | 是 | 否 | 否 | 受迁移阻塞影响 |
| baseline representation metrics | 是 | 否 | 部分是 | 受迁移阻塞影响 |
| kBulk comparison | 是 | 否 | 是 | 受迁移阻塞影响 |
| global evaluation | 是 | 是 | 是 | 受迁移阻塞影响 |
| `/api/health` | 是 | 否 | 否 | 受迁移阻塞影响 |

## 11. 对这个模块的准确定位

如果只看功能设计，`src/prism_v1/server` 已经不是一个“静态展示页”，而是一个比较完整的交互式分析面板，包含三类核心能力：

- 数据浏览
  - load dataset
  - 搜索/浏览 gene
  - 原始统计摘要
- 单基因诊断
  - checkpoint-backed posterior inspection
  - on-demand fit
  - baseline 对比
  - posterior 可视化
- 群体级诊断
  - kBulk group comparison
  - global representation evaluation

但如果看当前代码状态，它更准确的描述应该是：

`一个功能轮廓已经比较完整、但仍处于 prism -> prism_v1 迁移收尾阶段的本地分析服务模块。`

## 12. 后续建议

如果下一步要让这个模块真正可用，优先级建议如下：

1. 先统一 import 路径
   - 把 `prism.server.*`、`prism.model`、`prism.baseline.*`、`prism.io` 中所有应该迁到 `prism_v1.*` 的引用收口
2. 再对 checkpoint 接口做一次端到端对齐
   - 明确 server 期望的 `ModelCheckpoint` 字段结构到底以哪一版为准
3. 补最小启动验证
   - 至少覆盖 `import prism_v1.server`
   - `import prism_v1.cli.serve.app`
   - 用一个小 `.h5ad` 做 `/load` 到 `/gene` 的冒烟测试
4. 修 figure cache key
   - 把 fit 参数编码进 cache key，避免图文错配
5. 清理无效配置和未使用缓存
   - `plot_max_points`
   - `browse`
   - `html`

---

如果只回答“这个模块现在支持什么功能”，最准确的简版答案是：

它已经实现了一个以 GET 页面为主的 PRISM 分析 server，支持加载数据、搜索和浏览基因、查看单基因原始统计、用 checkpoint prior 做后验分析、做单基因 on-demand fit、做标签分组的 kBulk 对比、做 checkpoint-backed 的全局表示评估，并在页面中直接渲染多类诊断图；但当前 `prism_v1` 迁移尚未收口，启动链路和模型接口仍有明显阻塞，暂时不能把这些能力视为“开箱即用”。  
