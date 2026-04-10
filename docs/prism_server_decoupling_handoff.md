# PRISM Server Decoupling Handoff

## 1. 本次重新探索的结论

这次我没有沿用旧 handoff 的判断，而是重新把 `src/prism/server`、CLI 入口、服务层、旧 SSR 代码和新前端工程全部重新梳理了一遍。重新探索后的结论很明确：

1. `prism serve` 的真实运行时已经是 `FastAPI + Uvicorn`
2. `AppState` 仍然是单实例、单上下文、带缓存的分析状态中心
3. 分析能力真正的核心仍然在 `services/analysis.py`、`services/checkpoints.py`、`services/datasets.py`、`services/figures.py`
4. 旧的 `router/handlers/views/assets` 仍然留在仓库里，但已经不是实际运行链的入口
5. 之前的新前端只停留在骨架，现在已经补齐为可构建、可托管的 React/Vite 应用

因此，`src/prism/server` 现在应该被理解为：

- 后端：一个围绕单上下文分析状态暴露标准 JSON API 的 FastAPI 服务
- 前端：一个由 FastAPI 托管静态产物的 React SPA
- 旧 SSR 代码：仍在仓库中，但属于 legacy 参考实现，不再是主路径

## 2. 真实运行链

### 2.1 CLI 到服务启动

当前入口链如下：

- `src/prism/cli/main.py`
- `src/prism/cli/serve/app.py`
- `src/prism/server/app.py`
- `src/prism/server/api.py`

也就是说，`prism serve` 最终会：

1. 构造 `ServerConfig`
2. 初始化 `AppState`
3. 创建 FastAPI app
4. 通过 Uvicorn 提供服务

### 2.2 后端运行结构

当前后端主路径分成四层：

- `src/prism/server/app.py`
  - 服务启动与 Uvicorn 托管
- `src/prism/server/api.py`
  - FastAPI app 装配、异常处理、中间件、前端静态托管、SPA fallback
- `src/prism/server/api_routes.py`
  - API 路由定义
- `src/prism/server/state.py`
  - 当前数据集 / checkpoint / 缓存的统一状态容器

其中 `api_routes.py` 已经收口为标准的 FastAPI 路由层，并使用：

- `app.state.prism_state`
- `Depends(get_prism_state)`

来访问运行中的分析状态，而不是继续把所有路由逻辑塞在 app factory 里。

### 2.3 前端运行结构

前端现在位于：

- `src/prism/server/frontend/`

当前已经具备：

- `React + Vite + TypeScript`
- `AppShell`
- `DashboardPage`
- `GenePage`
- API client
- context load hook
- gene workspace hook
- URL 状态解析与序列化
- 共享 UI primitives
- 响应式样式体系

构建后的产物位于：

- `src/prism/server/frontend/dist/`

FastAPI 会：

- 挂载 `/assets`
- 对非 `/api/*` 路由做 SPA fallback
- 返回前端 `index.html`

## 3. 当前功能总结

### 3.1 数据与上下文

当前 server 支持：

- 加载本地 `.h5ad`
- 可选加载 checkpoint
- 可选选择 layer
- 维护单实例共享上下文
- 预计算 gene totals、detected counts、cell zero fraction、label columns
- 基于 context key 做缓存隔离

### 3.2 Dashboard 能力

前端 Dashboard 当前支持：

- 加载数据 / checkpoint 表单
- 数据集摘要卡片
- checkpoint 摘要卡片
- label key 展示
- gene browser
- gene browser 的 query / scope / sort / direction / page URL 状态同步
- 直接跳转 gene workspace

### 3.3 Gene Workspace 能力

当前 gene 页面支持：

- raw / checkpoint / fit 三种模式
- global / label 两种 prior source
- label key / label 选择
- 单基因 raw summary
- checkpoint summary
- on-demand fit 参数配置
- kBulk 参数配置
- figure 展示
- URL 深链接恢复

### 3.4 后端分析能力

当前 API 后端已经对接这些核心能力：

- gene browse
- gene search
- gene analysis
- checkpoint posterior
- on-demand fit
- kBulk analysis
- figure 序列化

这些能力仍然复用原有 Python 分析层，不重写算法语义。

## 4. 当前 API 契约

当前运行中的主 API 如下：

- `GET /api/health`
- `GET /api/context`
- `POST /api/context/load`
- `GET /api/genes`
- `GET /api/genes/search`
- `GET /api/gene-analysis`
- `GET /api/kbulk-analysis`

所有 API 统一返回：

- `ok`
- `data`
- `error`
- `meta`

错误处理已统一到 `src/prism/server/api_responses.py`。

## 5. 这次重构实际完成了什么

## 5.1 后端层

这次已经完成的后端收口包括：

- 将 FastAPI app 装配和 API 路由定义拆开
- 新增 `api_contracts.py`
- 新增 `api_routes.py`
- 将路由状态访问切到 `app.state + Depends`
- 保留统一错误响应
- 保留前端静态产物托管与 SPA fallback
- 为新 API 新增 FastAPI 测试

这意味着当前后端已经不再是“有 API 骨架，但路由仍粘在一个文件里”的中间态。

## 5.2 前端层

这次已经补齐的前端包括：

- `src/prism/server/frontend/src/components/AppShell.tsx`
- `src/prism/server/frontend/src/components/ui.tsx`
- `src/prism/server/frontend/src/hooks/useContextSnapshot.ts`
- `src/prism/server/frontend/src/lib/geneUrlState.ts`
- `src/prism/server/frontend/src/pages/DashboardPage.tsx`
- `src/prism/server/frontend/src/pages/GenePage.tsx`
- `src/prism/server/frontend/src/styles.css`

前端现在不再只是入口和 hooks 骨架，而是：

- 真实可构建
- 受控表单
- URL 深链接
- 页面状态与接口契约分层
- 可由 FastAPI 直接托管
- 已修复 `useAsyncResource` 因 `loader` identity 变化导致的重复请求循环
- 已补前端 hook 回归测试，避免 dashboard 再次出现无穷刷新 / 重复 `GET /api/genes`

## 5.3 UI/UX 设计约束

这轮实现继续使用了 `ui-ux-pro-max` skill，并再次确认了当前前端应遵循的设计方向：

- Data-Dense Dashboard
- 蓝色主色 + amber 强调色
- Fira Code / Fira Sans
- 强调受控表单、loading feedback、深链接和数据密度
- 避免装饰化样式和不可过滤的静态面板

当前前端样式就是按这一组约束落地的。

## 6. 仍然保留但不再是主路径的部分

下面这些文件仍然存在：

- `src/prism/server/router.py`
- `src/prism/server/handlers.py`
- `src/prism/server/queries.py`
- `src/prism/server/assets.py`
- `src/prism/server/views/*`

它们当前的定位应当视为：

- legacy SSR / 旧路由实现参考
- 不是 `prism serve` 实际运行主链

这批代码现在没有被删除，主要是为了：

- 保持已有旧测试与参考实现可读
- 避免在本轮前后端解耦过程中引入额外删除风险

如果后续要继续清理，建议单独做一轮 legacy retirement，而不是和本轮 API / frontend 联调混在一起。

## 7. 验证结果

### 7.1 前端构建

已验证：

- `cd src/prism/server/frontend && npm install`
- `cd src/prism/server/frontend && npm test`
- `cd src/prism/server/frontend && npm run build`

结果：

- `2 passed`
- `vite build` 成功

其中新增的前端回归测试覆盖了：

- inline loader 函数在组件重渲染时不应触发重复请求
- dependency key 变化时应按预期重新请求

当前前端已可成功构建，FastAPI 托管所需的 `dist` 已生成。

### 7.2 Python 测试

已验证：

- `uv run python -m pytest -q src/tests/test_server_api.py src/tests/test_server_config.py src/tests/test_server_state.py src/tests/test_cli_serve.py src/tests/test_server_handlers.py src/tests/test_server_router.py src/tests/test_server_views.py`

结果：

- `29 passed`

其中新增的 `src/tests/test_server_api.py` 覆盖了：

- unloaded context
- context load
- gene-analysis query contract
- kbulk-analysis query contract
- SPA fallback

### 7.3 真实运行时验证

已验证：

- `uv run prism serve --host 127.0.0.1 --port 8765`
- `curl --noproxy '*' http://127.0.0.1:8765/`
- `curl --noproxy '*' http://127.0.0.1:8765/gene/TP53`
- `curl --noproxy '*' http://127.0.0.1:8765/api/health`
- `curl --noproxy '*' 'http://127.0.0.1:8765/api/genes?scope=auto&sort_by=total_count&direction=desc&page=1'`

确认结果：

- `/` 与 `/gene/*` 均正确回到 SPA `index.html`
- `/api/health` 与 `/api/genes` 正常返回 JSON
- 空闲观察期内未出现新的后台自发循环请求日志

### 7.4 打包验证

已验证：

- `rm -rf build src/prism.egg-info /tmp/prism-wheelcheck2`
- `uv build --wheel -o /tmp/prism-wheelcheck2`

确认结果：

- wheel 已包含 `prism/server/assets/base.css`
- wheel 已包含 `prism/server/frontend/dist/index.html`
- wheel 已包含 `prism/server/frontend/dist/assets/*`
- wheel 中未再包含 `tests/`

## 8. 当前剩余事项

虽然主链已经打通，但仍有几个明确的后续事项：

### 8.1 旧 SSR 代码退役

当前 legacy SSR 代码还在仓库里，后续如果要继续做结构清理，建议：

1. 先确认没有其他模块依赖旧 `handlers/views`
2. 再单独删除或迁移到 `legacy/`

### 8.2 更强的端到端验证

目前已经完成：

- API 测试
- 前端 hook 回归测试
- 前端构建
- `prism serve` 真实运行时验证

但还没有完成浏览器自动化级别的验证，例如：

- Playwright / Cypress
- `prism serve` 启动后真实页面点击流

### 8.3 发布流程自动化

当前 wheel 已能正确包含前端产物，但后续如果要正式发布，仍建议单独补一轮发布流程自动化，例如：

- 在 CI 里固定执行 frontend build
- 在发布前自动做 wheel smoke check
- 明确 `dist` 由谁生成、何时生成

## 9. 一句话交接结论

`src/prism/server` 的前后端解耦已经推进到“FastAPI API 主链成立、React 前端页面可构建并由后端托管、重复请求循环已修复、核心接口和关键前端 hook 已有测试覆盖、wheel 打包已包含前端产物”的状态。

当前真正的主工作已经不是“继续设计架构”，而是：

- 是否清理 legacy SSR 代码
- 是否补浏览器级 e2e
- 是否把发布流程自动化补齐
