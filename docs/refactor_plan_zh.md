# PRISM 重构探索总结与方案

更新日期：2026-04-01

## 1. 文档目的

本文档用于沉淀当前对 PRISM 项目结构、脚本组织、CLI 设计和模型实现的探索结果，并给出一份可执行的重构方案。目标不是立即改写所有代码，而是先统一边界、规范接口、明确阶段性目标，再按优先级逐步推进。

---

## 2. 本次探索范围

本次重点查看了以下部分：

- `docs/scPRISM_zh.md`
- `src/prism/model`
- `src/prism/cli`
- `scripts/`

同时对以下维度进行了交叉检查：

- 脚本和 CLI 的职责重叠情况
- gene list / label list / annotation list 等文件格式是否统一
- `AnnData` 读取、矩阵选择、参考计数计算等公共逻辑是否重复
- model 层的性能热点与参数控制缺口
- checkpoint / plotting / extract / rank 等命令之间的接口一致性

---

## 3. 现状结论

### 3.1 项目总体分层是合理的

当前项目已经具备比较清晰的主干：

- `src/prism/model`：单基因先验拟合、后验推断、kBulk 推断、checkpoint
- `src/prism/cli`：面向用户的命令入口
- `src/prism/server`：服务端和交互展示支持
- `scripts/`：混合了数据处理、分析工具、实验脚本、开发验证脚本

主问题不在于主干不存在，而在于：

- 公共逻辑还没有完全沉到共享模块
- CLI 与脚本之间存在功能重叠
- 文件格式和参数风格已经开始分叉
- model 层存在若干明显的工程优化点

### 3.2 `scripts/` 已完成第一轮整理，但仍有尾项

当前 `scripts/` 已按职责分层为：

- `scripts/data/`
- `scripts/analysis/`
- `scripts/experiments/`
- `scripts/dev/`
- `scripts/dist/`

其中：

- 已被正式 CLI 覆盖的旧 gene-list / data / plotting 脚本已经删除
- `scripts/analysis/calc_degs.py` 仍依赖外部 `hpdex`，暂不适合并入主 CLI
- `scripts/experiments/*` 仍属于研究/实验入口，不应直接并入主 CLI
- `scripts/dev/test_em.py` 仍属于开发验证脚本
- `scripts/dist/run_fit_distributed.sh` 仍属于调度包装脚本

剩余问题主要是：

- 部分 experiment/dev 脚本仍通过 `sys.path` 注入访问项目代码
- experiment 脚本之间仍存在跨脚本复用
- `analysis` 类脚本还未完全与新增的 `prism analyze` 边界对齐

---

## 4. 当前主要问题

### 4.1 公共逻辑已部分收敛，但 CLI 共享层还不完整

以下能力已经基本收敛到共享层：

- `read_gene_list`
- `select_matrix`
- `compute_reference_counts`

仍未完全收敛的能力包括：

- `resolve_dtype`
- `resolve_prior_source`
- mutually-exclusive 参数检查
- Rich summary / plan table 输出
- 某些 gene name 到 index 的映射与子集选择

这意味着 `src/prism/io` 已建立，但 `src/prism/cli/common` 这一层仍未真正落地。

### 4.2 gene list 已基本统一，但 string-list / 注释输入仍偏松散

当前 gene list 已具备统一的 `GeneListSpec` 读写兼容层，并兼容：

- 纯文本：每行一个 gene
- 旧 CLI JSON
- 旧训练脚本 JSON
- 新 schema JSON

剩余问题主要是：

- label list / annotation list 仍以“文本或 ad-hoc JSON”方式读取
- 某些命令仍保留 `--gene-list` 等旧别名，帮助文案还不够统一
- 旧字段兼容期尚未正式收口

### 4.3 CLI 参数风格已改善，但仍未完全统一

当前仍可见的风格分叉包括：

- `--gene` / `--genes` / `--gene-list` 仍存在兼容并存
- `--label` / `--labels` / `--label-key` 的语义层次还未完全收口
- 输出参数同时存在 `--output`、`--output-json`、`--output-csv`、`--output-dir`
- `plot` / `analyze` / `checkpoint` 的职责边界虽已改善，但帮助文案仍需继续统一

这会提高维护和使用成本。

### 4.4 剩余脚本仍未完全产品化

目前主要是 experiment/dev/analysis 脚本仍在通过手动修改 `sys.path` 使用项目代码，说明它们尚未纳入统一包接口和调用规范。

这类现象包括：

- `PROJECT_ROOT / SRC_ROOT / sys.path.insert(...)`
- 硬编码外部工程路径，例如 `scripts/analysis/calc_degs.py` 对 `hpdex` 的依赖

### 4.5 model 层存在明显性能优化空间

当前最值得优先关注的问题有：

1. `fit priors` 在 CLI 入口就把输入矩阵整体 densify。
   - 对大规模稀疏单细胞矩阵会明显增加内存压力。

2. `fit_gene_priors()` 每次优化 step 内会重复计算 likelihood。
   - 当前实现中，后验平均和 NLL 计算阶段都会调用 `log_binomial_likelihood_grid()`。

3. 单细胞推断默认走 `float64`。
   - 对许多推断任务来说，`float32` 可能已经足够，且可以降低显存和计算成本。

4. `extract signals` 的结果回填是逐基因列循环。
   - 对大批量基因输出会增加 Python 层开销。

5. scope 内参考计数重复求和。
   - 许多场景下可以利用预计算向量切片，而不必反复对参考基因矩阵求和。

### 4.6 参数控制仍不够完整

目前模型已经支持的参数主要包括：

- grid size
- sigma bins
- align loss weight
- lr / scheduler / optimizer
- `fit_method = gradient | em`
- `init_temperature`

但仍缺少一些很自然的控制项：

- 初始化策略本身，而不仅仅是初始化温度
- warm start / 从已有 prior 或 checkpoint 初始化
- grid 构造策略
- 提前停止
- 推断 dtype
- 对齐项计算频率
- 针对低表达基因的特殊网格策略

### 4.7 plotting 接口原先有一处值得修正

原先的 `checkpoint plot-fg --x-axis p` 使用的是 `mu / max(mu)` 的归一化横轴，而不是严格的 `p = mu / S`。  
这项问题已在后续重构中修正，当前 plotting 已改为独立的 `prism plot` 命令组，`checkpoint plot-fg` 仅作为兼容别名保留。

---

## 5. 重构方向结论

### 5.1 核心判断

本次重构最重要的目标不是“多写几个命令”，而是先完成下面三件事：

1. 建立稳定的公共 IO / schema / helper 层
2. 统一 CLI 接口和 list 文件规范
3. 再做 model 层性能优化和参数扩展

如果顺序反过来，后面仍然会因为 schema 和边界不统一而反复返工。

### 5.2 当前阶段不建议优先做的事情

- 不建议第一轮就把所有实验训练脚本并入主 CLI
- 已被正式 CLI 完全覆盖的旧脚本可以直接删除，但不建议把 remaining experiment/analysis/dev 脚本也强行并入 CLI
- 不建议在 schema 未统一前大面积改命令参数
- 不建议在没有回归测试前大改数值逻辑

---

## 6. 目标架构

建议将项目逐步收敛为下面的结构：

```text
src/prism/
  cli/
    analyze/
      overlap_de.py
    common/
      options.py
      validators.py
      output.py
    ...
  io/
    lists.py
    anndata.py
    paths.py
    checkpoint_io.py
  model/
    ...
  analysis/
    deg.py              # 如果未来正式纳入
  experiments/          # 可选，后续再决定是否建立

scripts/
  data/
    prepare_ebw4.py
  dist/
    run_fit_distributed.sh
  analysis/
    calc_degs.py        # 若仍保留外部依赖形式
  experiments/
    train_baseline.py
    train_gears.py
    train_gene_mae.py
    train_gene_jepa.py
    train_static_gene_net.py
  dev/
    test_em.py

docs/
  refactor_plan_zh.md
```

这个结构的原则是：

- 正式用户工具进 `src/prism/cli`
- 公共逻辑进 `src/prism/io` 或 `src/prism/cli/common`
- 一次性/研究性脚本从主工具链剥离

---

## 7. 统一规范建议

### 7.1 统一 list schema

建议新增统一的 `ListSpec` / `GeneListSpec` 规范，最低限度包含：

```json
{
  "schema_version": 1,
  "kind": "gene_list",
  "gene_names": ["GeneA", "GeneB", "GeneC"],
  "scores": [1.2, 0.8, 0.3],
  "source_path": "input.h5ad",
  "method": "hvg",
  "metadata": {
    "top_k": 1000,
    "label": null
  }
}
```

设计原则：

- `gene_names` 是唯一必备核心字段
- `scores` 可选，但若存在应与 `gene_names` 对齐
- 其他字段统一收敛进 `metadata`
- 不再让不同工具自行增加平级字段

兼容策略：

- 读取时兼容纯文本、旧 CLI JSON、旧训练脚本 JSON
- 写出时统一写新 schema
- 第一阶段保留旧格式兼容，不破坏现有流程

### 7.2 统一列表读取接口

建议建立：

- `read_list_text(path)`
- `read_gene_list(path)`
- `read_gene_list_spec(path)`
- `write_gene_list_text(path, gene_names)`
- `write_gene_list_spec(path, spec)`

并支持：

- `.txt`
- `.json`

如果后续有必要，再考虑 `.yaml`。

### 7.3 统一 CLI 参数命名

建议统一以下风格：

- 文件型基因集合：`--genes`
- 文件型标签集合：`--labels`
- 参考基因集合：`--reference-genes`
- 拟合基因集合：`--fit-genes`
- 观测矩阵层：`--layer`
- 输出主文件：`--output`
- 补充结构化输出：`--output-json` / `--output-csv`
- 通用控制：`--device`、`--dtype`、`--seed`、`--dry-run`

建议避免：

- 同类功能同时存在 `--gene-list`、`--genes-path`、`--restrict-genes`
- 同一命令中既接受 repeatable `--gene` 又接受 `--gene-list`，但没有统一封装其互斥检查

### 7.4 统一 AnnData 操作 helper

建议在 `src/prism/io/anndata.py` 中沉淀：

- `select_matrix(adata, layer)`
- `slice_gene_matrix(...)`
- `compute_reference_counts(...)`
- `resolve_gene_positions(...)`
- `ensure_dense_if_needed(...)`
- `write_h5ad_atomic(...)`

这样 CLI、server、scripts 就不会再各写一套。

---

## 8. CLI 重构方案

### 8.1 `genes` 命令组

当前已完成第一轮重构。

#### 现状

- 已有：`intersect`、`subset`、`rank`、`merge`、`filter`
- 原先的 `calc_gene_list.py`、`merge_gene_list.py`、`filter_gene_list.py` 已删除
- gene-list 输入已经统一兼容 text / 旧 JSON / 新 schema JSON

#### 建议目标

将 `genes` 命令组扩展为：

- `prism genes rank`
- `prism genes merge`
- `prism genes filter`
- `prism genes intersect`
- `prism genes subset`

#### 具体建议

1. 扩展 `prism genes rank`
   - 吸收 `calc_gene_list.py` 中 CLI 尚未具备的能力：
   - `signal-hvg`
   - `signal-variance`
   - `signal-dispersion`
   - `--max-cells`
   - `--restrict-genes`

2. 新增 `prism genes merge`
   - 替代 `merge_gene_list.py`
   - 输入统一为 gene-list JSON 或 text

3. 新增 `prism genes filter`
   - 替代 `filter_gene_list.py`
   - 将物种规则、正则规则、sidecar 输出保留

### 8.2 新增 `data` 命令组

当前 `data` 命令组已经建立，包含：

- `prism data subset-genes`
- `prism data downsample`

这些功能原先对应的旧脚本已经删除。  
如果未来还有 matrix transform、layer copy、obs split 等功能，也更容易继续扩展。

### 8.3 `checkpoint` / `plot` / `analyze` 边界

当前边界应明确为：

- `checkpoint`：inspect / merge / 兼容别名
- `plot`：所有以 figure 为主输出的可视化命令
- `analyze`：所有以表格、指标、排序结果为主输出的分析命令

目前已经完成的动作包括：

- plotting 已从 `checkpoint` 中拆出为独立 `plot` 命令组
- `checkpoint plot-fg` 仅保留为兼容别名
- `overlap-de` 的正式入口已迁移到 `prism analyze overlap-de`
- `checkpoint overlap-de` 仅保留为兼容别名

后续建议重点：

- 继续评估哪些“只导出 CSV/表格”的分析命令应进入 `analyze`
- 避免把 `checkpoint` 再次扩展成 analysis/plotting 杂糅入口

### 8.4 `extract` 命令组

建议重构重点：

- 把 `signals`、`kbulk`、`kbulk-mean` 共享的基因选择、参考基因选择、输出写入逻辑继续抽象
- 统一 `batch_size` / `sample_batch_size` 的命名和语义
- 明确 `S_source`、`N_avg_source` 的帮助文案和默认值解释

### 8.5 `fit` 命令组

建议重构重点：

- 参数分类更清晰：
  - 数据选择参数
  - 分组/标签参数
  - 拟合控制参数
  - 优化器参数
  - 初始化参数
  - 输出参数
- 将部分高级参数移到明确的高级分组文案中，降低默认使用复杂度

---

## 9. `scripts/` 整理方案

### 9.1 第一轮目录调整建议

#### 当前保留目录

- `scripts/data/prepare_ebw4.py`
- `scripts/dist/run_fit_distributed.sh`
- `scripts/analysis/calc_degs.py`
- `scripts/experiments/*`
- `scripts/dev/test_em.py`

#### 已删除并由 CLI 替代

- `scripts/calc_gene_list.py`
- `scripts/merge_gene_list.py`
- `scripts/filter_gene_list.py`
- `scripts/gene_subset_anndata.py`
- `scripts/down_sample_anndata.py`
- `scripts/plot_batch_perturbation_grid.py`

### 9.2 兼容策略

当前策略已更新为：

1. 先把正式实现迁入 `src/prism/cli`
2. 对已被正式 CLI 替代、且无额外独立价值的旧脚本直接删除
3. 保留 dataset-specific、analysis、experiments、dev、dist 类型脚本
4. 在 README 中明确新的正式入口与脚本目录边界
5. 对 remaining scripts，只做“归类、去耦、减少 `sys.path` 注入”，而不是硬塞进主 CLI

---

## 10. model 层优化方案

### 10.1 性能优化优先级

#### P0：避免入口整体 densify

当前 `fit priors` 在 CLI 入口就把矩阵 densify。建议改为：

- 保持 `adata.X` / layer 原始格式
- 在 gene batch 切片后，按需转 dense
- 参考计数计算优先使用 sparse sum

这是最直接的内存优化点。

#### P1：减少重复 likelihood 计算

`fit_gene_priors()` 中每步对同一批数据会多次调用 `log_binomial_likelihood_grid()`。  
建议评估以下方案：

- 将 step 内的 likelihood 计算结果在 cell chunk 级别缓存
- 或重新组织目标函数计算顺序，避免重复构造同一 `log_lik`

#### P2：推断 dtype 可控

建议为单细胞推断增加 `torch_dtype` 选项，并统一到：

- `fit`
- `infer`
- `extract signals`
- server 侧分析接口

默认策略可考虑：

- fit 默认 `float64`
- infer 默认 `float32`
- 允许显式覆盖

#### P3：减少 Python 层回填开销

`extract signals` 当前按 channel、按 gene 做列回填。  
建议改为按 batch 直接写块状切片。

#### P4：复用参考计数

在一个 scope 内，参考计数一般只依赖 cell 子集和 reference gene 集合。  
许多场景下不需要重复从原矩阵做 sum，可以通过预计算向量切片或缓存减少重复工作。

### 10.2 参数控制扩展

建议新增或明确以下参数：

- `init_method`
  - 例如：`uniform`、`posterior_mean`、`prior_file`

- `init_seed`
  - 用于随机初始化或随机扰动初始化

- `warm_start_checkpoint`
  - 从已有 checkpoint 的 prior 初始化

- `grid_strategy`
  - 例如：`linear`、`quantile`、`adaptive`

- `grid_max_method`
  - 控制 `p_grid_max` 的构造策略

- `early_stop_tol`
- `early_stop_patience`

- `align_every`
  - 不必每一步都重新计算对齐项时可降低成本

- `inference_torch_dtype`

- `chunk_policy`
  - 明确 cell chunk / gene batch / sample batch 的策略

### 10.3 数值兼容要求

model 层优化必须保留以下保障：

- checkpoint schema 向后兼容
- 现有 `signal`、`map_p`、`map_mu` 等导出通道语义不变
- 新老实现之间需要有回归测试或数值差异阈值检查

---

## 11. 分阶段执行计划

### Phase 0：公共基础层

目标：

- 新增 `src/prism/io`
- 新增 `src/prism/cli/common`
- 统一 list schema 和 loader

产出：

- `read_gene_list_spec()` 等公共接口
- CLI、server、scripts 不再各自复制基本 helper

### Phase 1：CLI 统一与脚本吸收

目标：

- 将稳定脚本能力并入 CLI
- `scripts/` 重新分层
- 对已完全被替代的脚本直接删除，而不是保留双份实现

产出：

- `prism genes merge`
- `prism genes filter`
- `prism data subset-genes`
- `prism data downsample`
- 扩展后的 `prism genes rank`
- 独立的 `prism plot`
- 独立的 `prism analyze overlap-de`

### Phase 2：model 层性能优化

目标：

- 去掉 fit 入口整体 densify
- 降低重复 likelihood 计算
- 推断 dtype 可控
- 优化 extract 输出回填

产出：

- 更低内存占用
- 更快的拟合和提取速度
- 更明确的性能参数

### Phase 3：高级参数与实验层整理

目标：

- 引入初始化策略、warm start、early stopping
- 评估是否为实验训练脚本建立共享模块

产出：

- 更完整的模型控制面
- 更少的实验脚本跨文件耦合

---

## 12. 第一阶段推荐任务清单

建议按下面顺序开始：

1. 先实现统一的 `GeneListSpec` 读写模块
2. 再把 CLI / server / scripts 中重复的 list helper 收敛过去
3. 扩展 `prism genes rank`，吸收 `calc_gene_list.py` 的核心能力
4. 新增 `prism genes merge`
5. 新增 `prism genes filter`
6. 新增 `prism data subset-genes`
7. 新增 `prism data downsample`
8. 将 plotting 和 analysis 边界从 `checkpoint` 中拆出
9. 删除已完全被 CLI 替代的旧脚本
10. 再进入 model 层性能优化

这个顺序的好处是：

- 用户接口先稳定
- 共享代码先建立
- 后续优化不会继续在分叉接口上返工

---

## 13. 风险与注意事项

### 13.1 风险

- 旧 gene-list JSON 兼容不足会打断现有训练脚本
- 过早删除脚本会影响历史工作流
- 直接修改 model 数值路径可能引入隐性回归

### 13.2 应对策略

- loader 先做强兼容
- 已完全被 CLI 替代的脚本直接删除；remaining scripts 只做归类与去耦
- model 优化必须配套 benchmark 和数值回归检查

---

## 14. 最终结论

PRISM 当前已经具备清晰的核心方法主线，但工程层正处在一个典型的“功能逐渐丰富、接口开始分叉”的阶段。  
这次重构的重点不应是零散修补，而应是先统一共享基础层和文件/参数规范，再逐步完成 CLI 整理、脚本归类和 model 优化。

一句话总结当前建议：

**先统一 schema 和公共 helper，再统一 CLI，再做性能优化。**

---

## 15. 详细 TODO List

下面的 TODO 按模块拆分，尽量做到“每一项都可以独立提交、独立验证、独立回滚”。

状态更新（2026-04-01）：

- 已完成：模块 1、模块 2、模块 3、模块 4、模块 5、模块 6、模块 7、模块 8、模块 9、模块 11、模块 12、模块 14、模块 15、模块 16。
- 按设计跳过：模块 10（保留整体 densify，效率优先）、模块 13（训练脚本不改）。
- 最新决策：
  - 对已被正式 CLI 替代的旧脚本，优先直接删除，不再保留兼容包装。
  - `checkpoint` 不再继续吸纳 plotting / analysis 功能，后续分别进入 `plot` 与 `analyze`。

### 15.1 模块 1：统一 list/schema 与公共 IO

- [x] 新增 `src/prism/io/__init__.py`
- [x] 新增 `src/prism/io/lists.py`
- [x] 定义统一的 `GeneListSpec` 数据结构
- [x] 为 gene-list JSON 定义 `schema_version`
- [x] 支持从纯文本读取 gene list
- [x] 支持从旧 CLI JSON 读取 gene list
- [x] 支持从旧训练脚本 JSON 读取 gene list
- [x] 提供统一的 `read_gene_list()`
- [x] 提供统一的 `read_gene_list_spec()`
- [x] 提供统一的 `write_gene_list_text()`
- [x] 提供统一的 `write_gene_list_spec()`
- [x] 评估并实现 `label list` / `annotation list` 可复用的通用 string-list schema
- [x] 为 text / JSON 读取错误提供一致的异常文案
- [x] 为 `gene_names`、`scores` 长度不一致提供显式校验
- [x] 为重复 gene name 做统一去重策略
- [x] 补充模块级文档字符串，说明兼容策略

验证项：

- [x] 纯文本 gene list 可正常读取
- [x] 旧 `calc_gene_list.py` 风格 JSON 可正常读取
- [x] 旧训练脚本风格 JSON 可正常读取
- [x] 新 schema JSON 可正常读写往返
- [x] string-list text / JSON array / JSON object 可正常读取

### 15.2 模块 2：CLI 公共 helper 收敛

- [x] 新增 `src/prism/cli/common/`
- [x] 将 list 读取逻辑从 `fit/extract/genes/checkpoint` 的 `common.py` 中收敛
- [x] 收敛 `select_matrix()`
- [x] 收敛 `compute_reference_counts()`
- [x] 收敛 `resolve_dtype()`
- [x] 收敛 `resolve_prior_source()`
- [x] 统一 mutually-exclusive 参数检查工具
- [x] 统一 CLI 输出 summary / plan table 的封装

验证项：

- [x] `prism fit priors --help` 行为不变
- [x] `prism extract signals --help` 行为不变
- [x] `prism genes rank --help` 行为不变
- [x] 重复 helper 数量明显下降

### 15.3 模块 3：扩展 `prism genes rank`

- [x] 将 `calc_gene_list.py` 中的 signal 系方法并入 `prism genes rank`
- [x] 支持 `signal-hvg`
- [x] 支持 `signal-variance`
- [x] 支持 `signal-dispersion`
- [x] 支持 `--max-cells`
- [x] 支持 `--seed`
- [x] 支持 `--restrict-genes`
- [x] 统一 gene-list JSON 输出为新 schema
- [ ] 保留旧字段兼容期

验证项：

- [ ] 现有 `hvg` / `lognorm-*` 排名结果保持一致
- [x] 新增 `signal-*` 方法可直接替代旧脚本

### 15.4 模块 4：新增 `prism genes merge`

- [x] 将 `merge_gene_list.py` 逻辑迁入 CLI
- [x] 输入兼容旧 JSON 和新 JSON
- [x] 输出统一为新 schema JSON
- [x] 支持输出纯文本 ranked gene list
- [x] 明确 rank merge 的 tie-break 规则

验证项：

- [x] 两个旧 `calc_gene_list.py` 输出可被新命令直接合并
- [ ] 合并结果与旧脚本一致

### 15.5 模块 5：新增 `prism genes filter`

- [x] 将 `filter_gene_list.py` 逻辑迁入 CLI
- [x] 保留 built-in species nuisance rule sets
- [x] 保留自定义 JSON/YAML 规则扩展
- [x] 支持 sidecar removed gene 输出
- [x] 支持 dry-run
- [x] 输出支持 text 和新 schema JSON

验证项：

- [ ] human / mouse / ecoli / bsub 规则可正常工作
- [x] removed gene sidecar 正常输出

### 15.6 模块 6：新增 `prism data subset-genes`

- [x] 将 `gene_subset_anndata.py` 迁入 CLI
- [x] 输入兼容 text gene list 和 JSON gene list
- [x] `uns` 中写入标准化 provenance
- [x] 明确缺失 gene 的处理策略
- [x] 输出写入改为原子写入

验证项：

- [ ] 与旧脚本输出结果一致
- [x] 旧 gene-list JSON 可直接作为输入

### 15.7 模块 7：新增 `prism data downsample`

- [x] 将 `down_sample_anndata.py` 迁入 CLI
- [x] 支持按 `obs` 列分层采样
- [x] 支持最小每类样本数
- [x] 在 `uns` 中写入采样 provenance

验证项：

- [ ] 与旧脚本采样结果分布一致
- [ ] 输入参数错误时有清晰报错

### 15.8 模块 8：plotting / overlap 收敛

- [x] 评估 `plot_batch_perturbation_grid.py` 与 `checkpoint plot-fg` 的合并方式
- [x] 将 batch × perturbation grid 布局并入独立 `prism plot` 命令组
- [x] 修正 `--x-axis p` 的真实语义
- [x] 统一 gene list / label list 输入
- [x] 统一图和 CSV 的输出风格
- [x] 新增独立 `prism plot` 顶层命令组
- [x] 保留 `checkpoint plot-fg` 兼容别名
- [x] 扩展 `prism plot priors` 支持 `curve-mode`、`y-scale`、`summary-csv`、`stat` 注释与 panel size 控制
- [x] 扩展 `prism plot batch-grid` 支持空 panel 控制、坐标轴显隐、summary 导出与统计注释
- [x] 新增 `prism plot overlap`，提供 overlap/JSD/Wasserstein/best-scale 热图与 CSV 导出
- [x] 将 `checkpoint overlap-de` 收敛到与 `prism plot overlap` 相同的 overlap backend，避免数值实现分叉
- [x] 将 `overlap-de` 的正式入口迁移到 `prism analyze overlap-de`，`checkpoint` 下仅保留兼容别名

验证项：

- [x] overlay / facet / batch-grid 三种布局均能输出
- [x] `mu` 和 `p` 横轴定义与模型语义一致
- [x] overlap heatmap 与 CSV 指标可输出

### 15.9 模块 9：脚本目录整理与残余脚本去耦

说明：根据最新决策，已被正式 CLI 替代的旧脚本将直接删除，不再保留兼容包装。

- [x] 新建 `scripts/data/`
- [x] 新建 `scripts/dist/`
- [x] 新建 `scripts/analysis/`
- [x] 新建 `scripts/experiments/`
- [x] 新建 `scripts/dev/`
- [x] 迁移对应脚本到新目录
- [x] 删除已被 `prism genes` 替代的旧 gene-list 脚本
- [x] 删除已被 `prism data` 替代的旧 data 脚本
- [x] 删除已被 `prism plot` 替代的旧 plotting 脚本
- [x] 在 README 中更新替代入口
- [x] 去掉脚本中不必要的 `sys.path` 注入

验证项：

- [x] 新入口可以完全覆盖已迁入 CLI 的旧功能

### 15.10 模块 10：model 层内存优化

说明：经评估，保留整体 densify 以保证切片效率。此模块按设计跳过。

- [x] ~~去掉 `fit priors` 的入口整体 densify~~ — 按设计保留
- [x] ~~保持 sparse matrix 到 gene-batch 级别~~ — 按设计保留
- [x] ~~对参考计数优先使用 sparse sum~~ — 按设计保留
- [x] ~~审查 `slice_gene_counts()` 的 densify 时机~~ — 按设计保留

验证项：

- [x] 按设计跳过，不适用

### 15.11 模块 11：model 层计算优化

- [x] 评估 `fit_gene_priors()` 内 likelihood 缓存方案
- [x] 减少 step 内重复 `log_binomial_likelihood_grid()` 计算
- [x] 评估 `align` 计算频率控制
- [x] 优化 `extract signals` 的批量回填
- [x] 复用 scope 内参考计数（已确认现有实现已在 scope 级别复用）

验证项：

- [x] 拟合总耗时下降
- [x] `extract signals` 总耗时下降
- [x] 数值回归通过

### 15.12 模块 12：参数控制扩展

- [x] 新增 `init_method`
- [x] 新增 `init_seed`
- [x] 新增 `warm_start_checkpoint`
- [x] 新增 `grid_strategy`
- [x] 新增 `grid_max_method`
- [x] 新增 `early_stop_tol`
- [x] 新增 `early_stop_patience`
- [x] 新增 `inference_torch_dtype`
- [x] 统一 `batch_size` / `sample_batch_size` / `cell_chunk_size` 的文档语义

验证项：

- [x] 参数默认值保持向后兼容
- [x] 新参数在 CLI 帮助中有清晰解释

### 15.13 模块 13：实验训练脚本公共层

说明：经评估，训练脚本不在本轮重构范围内。此模块按设计跳过。

- [x] ~~抽取 `train_gene_mae.py` 中被复用的公共函数~~ — 按设计跳过
- [x] ~~为实验脚本建立共享 `common` 模块~~ — 按设计跳过
- [x] ~~解除跨脚本导入~~ — 按设计跳过
- [x] ~~统一实验脚本的 gene-list 读取入口~~ — 按设计跳过
- [x] ~~统一实验脚本的日志、checkpoint、history 输出格式~~ — 按设计跳过

### 15.14 模块 14：测试与回归保障

- [x] 为 list/schema 读写补充单元测试
- [x] 为 CLI 的关键公共 helper 补充测试
- [x] 为 rank / merge / filter / subset / downsample 补充回归测试
- [x] 为 `plot priors` / `plot batch-grid` / `plot overlap` / `analyze overlap-de` 补充回归测试
- [x] 为 model 核心拟合与推断补充数值回归测试
- [x] 为 checkpoint schema v1/v2 读取兼容补充测试

验证项：

- [x] 关键路径具备最小回归保障
- [x] 新旧 schema 兼容性得到测试覆盖（gene-list / string-list）
- [x] checkpoint schema 兼容性得到测试覆盖

### 15.15 模块 15：`analyze` 命令组与分析型命令边界

- [x] 新增 `src/prism/cli/analyze/`
- [x] 将 `overlap-de` 的正式入口迁移到 `prism analyze overlap-de`
- [x] 保留 `checkpoint overlap-de` 兼容别名
- [x] 为 `analyze` 收敛共享 helper / output 封装（`analyze/common.py`）
- [x] 明确后续分析命令的归类规则：`analyze` 输出表格/指标，`plot` 输出图形
- [x] 新增 `prism analyze checkpoint-summary`

验证项：

- [x] `prism analyze --help` 可正常显示
- [x] `prism analyze overlap-de` 与旧 `checkpoint overlap-de` 行为一致
- [x] `prism analyze checkpoint-summary` 可正常导入

### 15.16 模块 16：plot 二期丰富

- [x] 为 extracted `h5ad` 新增 `signal` / `entropy` / `MI` 分布图（`plot distributions`）
- [x] 支持按 `obs` 分组的 violin / box / hist 图型（`plot distributions --group-key --plot-type`）
- [x] 为 checkpoint 新增 label similarity / summary heatmap（`plot label-summary`）
- [x] 统一 checkpoint 型与 `h5ad` 型绘图命令的输入/输出规范（统一 `--output` / `--palette` 风格）
- [x] 为 plot 增加 palette / ordering / sampling 控制（`--palette` / `--max-genes` / `--seed`）
- [ ] 为 plotting 文档补充功能总览、参数示例和推荐工作流

验证项：

- [x] `plot` 与 `analyze` 的边界在 CLI 帮助中清晰
- [ ] 新增图型具备最小 CLI 回归测试

### 15.17 推荐执行顺序

- [x] 先完成模块 1 的主体
- [x] 再完成模块 2
- [x] 再推进模块 3/4/5/6/7
- [x] 然后处理模块 8/9
- [x] 补模块 14 的基础测试缺口
- [x] 完成模块 10（按设计跳过，保留整体 densify）
- [x] 完成模块 11（likelihood 缓存、align_every、批量回填）
- [x] 完成模块 12（init_method/warm_start/early_stop/grid_max_method/grid_strategy/batch_size 文档）
- [x] 完成模块 15（analyze 共享 helper、checkpoint-summary、归类规则）
- [x] 完成模块 16（distributions、label-summary、palette/sampling 控制）
- [x] 跳过模块 13（训练脚本不改）

### 15.18 每个模块的实施约束

- [ ] 每次只改一个清晰模块，避免横向扩散过大
- [ ] 每次重构前先写明思路、原因、目标、预期效果
- [ ] 每次重构后给出变更摘要和验证结果
- [ ] 对已有用户接口默认保持兼容，必要时通过兼容包装过渡

### 15.19 模块 17：多拟合分布模式与模式锁（binomial / negative_binomial / poisson）

目标：

- [ ] 在保持 `binomial` 为默认理论主路径的前提下，引入两个额外消融拟合分布模式：`negative_binomial`、`poisson`
- [ ] checkpoint 显式保存 `fit_distribution` / `posterior_distribution` / `grid_domain`
- [ ] 禁止不同模式在 downstream 中静默迁移到错误模式
- [ ] `extract` / `infer` / `plot` / `kbulk` / `server` 都能正确识别三种模式

当前已知问题：

- [ ] `fit` 已支持 `NB/poisson`，但 checkpoint 还未把模式锁做成一等元数据
- [ ] `infer` / `Posterior` / `extract` 仍默认假定 `p` 域和 binomial posterior
- [ ] `plot`、`kbulk`、`server` 仍默认假定 `map_p` / `map_mu` 永远存在

实施项：

- [ ] checkpoint：保存并显示 `fit_distribution` / `posterior_distribution` / `grid_domain`
- [ ] checkpoint merge：禁止 merge 不同模式的 checkpoint
- [ ] `infer_posteriors()` 改为三模式分发：binomial / NB / poisson
- [ ] `Posterior` / `extract signals` 改为模式感知输出
- [ ] `poisson` 增加 `map_rate` / `support_rate` 或统一 `support` 输出语义
- [ ] `kbulk` 支持三模式，或对不支持的模式显式报错
- [ ] `plot priors` / `plot overlap` / `label-summary` 对 grid domain 和比较方式做模式检查
- [ ] server analysis / global eval / kbulk comparison 改为模式感知

设计原则：

- [ ] 显式模式锁：不同模式 checkpoint 不能静默互用
- [ ] `binomial/NB` 维持 `grid_domain='p'`
- [ ] `poisson` 使用 `grid_domain='rate'`
- [ ] 对不支持的 channel/plot/kbulk/server 路径给出显式错误，不允许静默 fallback

推荐顺序：

- [ ] 先做 checkpoint 模式锁
- [ ] 再做 infer / posterior / extract
- [ ] 再做 plot
- [ ] 再做 kbulk / server
- [ ] 最后补回归测试

验证项：

- [ ] checkpoint round-trip 不丢模式字段
- [ ] 不同模式 checkpoint merge 会明确报错
- [ ] `extract signals` 能处理三种模式
- [ ] `plot priors` 能处理 `p` 域和 `rate` 域
- [ ] 原有 `binomial` 路径回归全部通过
