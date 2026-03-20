# PRISM Server Refactor Plan

Status: in progress

## Goals

- Rewrite `src/prism/server` as a single coherent server module with no `legacy` split.
- Preserve the rich visual diagnostics from `server.py`, but promote them into first-class server plot modules.
- Add baseline comparison outputs alongside PRISM-specific analysis.
- Support loading checkpoints so the server can reuse fitted priors and avoid recomputation when possible.
- Move algorithm/report responsibilities into `src/prism/model` where appropriate, while keeping model APIs backward compatible.

## Current Problems

- `src/prism/server` still injects old-page HTML through `legacy_block` and depends on `services/legacy_plots.py`.
- `src/prism/server/services/fitting.py` duplicates parts of `src/prism/model` instead of consuming formal model reports.
- Server analysis objects mix dataset loading, fitting, posterior extraction, baseline metrics, and old diagnostic payloads.
- Checkpoint loading only supports a minimal subset of reusable information.
- Plotting is split between a small SVG layer and copied matplotlib/Plotly functions from `server.py`.

## Target Architecture

### Model

- Keep existing public entrypoints (`fit_pool_scale`, `PriorEngine.fit`, `Posterior.extract`) working.
- Add report-oriented APIs for pool fitting, prior fitting, and posterior summarization.
- Define reusable dataclasses for diagnostics/reports so CLI and server share one implementation.

### Server

- `state.py`: maintain separate dataset state, model/checkpoint state, and cached reports.
- `services/datasets.py`: dataset lookup, slicing, summary, and search only.
- `services/analysis.py`: orchestrate checkpoint reuse, on-demand fitting, posterior extraction, and baseline comparison.
- `plots/*`: first-class plot builders for gene overview, pool fit, prior fit, signal space, posterior gallery, and baseline comparison.
- `views/*`: render structured view models only; no raw HTML block injection.
- `handlers.py`: request parsing and response wiring only.

## Checkpoint Strategy

- Continue supporting existing checkpoint dictionaries containing `engine` and `s_hat`.
- Add optional richer report payloads in new checkpoints, but do not require them.
- Server load path should:
  - reuse checkpoint priors immediately when available;
  - reuse report payloads when present;
  - compute missing reports lazily and cache them in memory.

## Execution Plan

### Phase 1: Model reports

- [x] Add pool-fit report dataclass and API in `src/prism/model/estimator.py`.
- [x] Add prior-fit report dataclasses/APIs in `src/prism/model/engine.py`.
- [x] Add posterior summary/report helpers in `src/prism/model/posterior.py`.
- [x] Re-export new report types from `src/prism/model/__init__.py`.

### Phase 2: Server state and analysis

- [x] Redesign `src/prism/server/state.py` around dataset/model/report caches.
- [x] Replace checkpoint helpers with schema-aware loading utilities.
- [x] Rewrite `src/prism/server/services/analysis.py` against model reports.
- [x] Remove duplicated fitting logic from `src/prism/server/services/fitting.py` or delete the module.

### Phase 3: Plot system

- [x] Replace `services/legacy_plots.py` with first-class plot modules.
- [x] Keep the old `server.py` figure coverage: gene overview, stage 0, optimization trace, init comparison, prior/self-consistency, signal interface, 3D signal space, posterior gallery.
- [x] Add baseline comparison figures and metrics.

### Phase 4: Views and handlers

- [x] Rewrite home and gene pages around the new analysis/view models.
- [x] Remove `legacy_block`, `legacy_fit`, and `legacy.py`.
- [x] Simplify handlers so they no longer know about old/new rendering paths.
- [x] Refresh CSS to support the richer figure layout.

### Phase 5: Validation

- [x] Verify server imports and CLI entrypoints.
- [x] Run focused checks for checkpoint loading and on-demand fitting flows.

## Tracking Notes

- This document is the source of truth for the refactor order.
- As phases are completed, checkboxes should be updated in place.
