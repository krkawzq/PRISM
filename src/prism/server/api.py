from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .api_responses import ApiError, install_exception_handlers, ok_response
from .api_serializers import (
    serialize_context_snapshot,
    serialize_gene_analysis,
    serialize_gene_browse_page,
    serialize_gene_candidate,
    serialize_kbulk_analysis,
)
from .config import ServerConfig
from .frontend import frontend_dist_dir, serve_frontend_asset
from .services.analysis import (
    AnalysisMode,
    BrowseScope,
    BrowseSort,
    GeneFitParams,
    KBulkParams,
    PriorSource,
    browse_gene_candidates,
    build_checkpoint_summary,
    build_dataset_summary,
    build_gene_analysis,
    compute_kbulk_analysis,
    search_gene_candidates,
)
from .state import AppState


def create_api_app(config: ServerConfig, *, state: AppState | None = None) -> FastAPI:
    resolved_state = AppState(config) if state is None else state
    app = FastAPI(
        title="PRISM Server API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.prism_state = resolved_state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_exception_handlers(app)
    _mount_frontend_assets(app)

    @app.get("/api/health")
    def health() -> object:
        loaded = resolved_state.loaded
        return ok_response(
            {
                "status": "ok",
                "loaded": loaded is not None,
                "contextKey": None if loaded is None else loaded.context_key,
            }
        )

    @app.get("/api/context")
    def context() -> object:
        return ok_response(
            serialize_context_snapshot(
                resolved_state,
                dataset_summary=build_dataset_summary(resolved_state),
                checkpoint_summary=build_checkpoint_summary(resolved_state),
            )
        )

    @app.post("/api/context/load")
    def load_context(
        payload: LoadContextPayload,
    ) -> object:
        if not payload.h5ad_path.strip():
            raise ApiError(400, "missing_dataset_path", "h5ad_path is required")
        resolved_state.load(
            h5ad_path=payload.h5ad_path,
            ckpt_path=payload.ckpt_path,
            layer=payload.layer,
        )
        return ok_response(
            serialize_context_snapshot(
                resolved_state,
                dataset_summary=build_dataset_summary(resolved_state),
                checkpoint_summary=build_checkpoint_summary(resolved_state),
            )
        )

    @app.get("/api/genes")
    def browse_genes(
        query: str = "",
        scope: BrowseScope = "auto",
        sort_by: BrowseSort = "total_count",
        direction: Literal["asc", "desc"] = "desc",
        page: int = 1,
    ) -> object:
        if resolved_state.loaded is None:
            return ok_response(
                {
                    "query": query,
                    "scope": scope,
                    "sortBy": sort_by,
                    "descending": direction != "asc",
                    "page": 1,
                    "pageSize": config.browse_page_size,
                    "totalItems": 0,
                    "totalPages": 1,
                    "items": [],
                }
            )
        page_data = browse_gene_candidates(
            resolved_state,
            query=query,
            scope=scope,
            sort_by=sort_by,
            descending=direction != "asc",
            page=page,
        )
        return ok_response(serialize_gene_browse_page(page_data))

    @app.get("/api/genes/search")
    def search_genes(q: str = "", limit: int | None = None) -> object:
        if resolved_state.loaded is None:
            return ok_response({"items": []})
        return ok_response(
            {
                "items": [
                    serialize_gene_candidate(item)
                    for item in search_gene_candidates(resolved_state, q, limit=limit)
                ]
            }
        )

    @app.get("/api/gene-analysis")
    def gene_analysis(
        q: str,
        mode: AnalysisMode = "checkpoint",
        prior_source: PriorSource = "global",
        label_key: str | None = None,
        label: str | None = None,
        scale: float | None = None,
        reference_source: Literal["checkpoint", "dataset"] = "checkpoint",
        n_support_points: int = 512,
        max_em_iterations: int | None = 200,
        convergence_tolerance: float = 1e-6,
        cell_chunk_size: int = 512,
        support_max_from: Literal["observed_max", "quantile"] = "observed_max",
        support_spacing: Literal["linear", "sqrt"] = "linear",
        use_adaptive_support: bool = False,
        adaptive_support_fraction: float = 1.0,
        adaptive_support_quantile_hi: float = 0.99,
        likelihood: Literal["binomial", "negative_binomial", "poisson"] = "binomial",
        nb_overdispersion: float = 0.01,
        torch_dtype: Literal["float32", "float64"] = "float64",
        compile_model: bool = True,
        device: str = "cpu",
    ) -> object:
        analysis = build_gene_analysis(
            resolved_state,
            query=q,
            mode=mode,
            prior_source=prior_source,
            label_key=label_key,
            label=label,
            fit_params=(
                None
                if mode != "fit"
                else GeneFitParams(
                    scale=scale,
                    reference_source=reference_source,
                    n_support_points=n_support_points,
                    max_em_iterations=max_em_iterations,
                    convergence_tolerance=convergence_tolerance,
                    cell_chunk_size=cell_chunk_size,
                    support_max_from=support_max_from,
                    support_spacing=support_spacing,
                    use_adaptive_support=use_adaptive_support,
                    adaptive_support_fraction=adaptive_support_fraction,
                    adaptive_support_quantile_hi=adaptive_support_quantile_hi,
                    likelihood=likelihood,
                    nb_overdispersion=nb_overdispersion,
                    torch_dtype=torch_dtype,
                    compile_model=compile_model,
                    device=device,
                )
            ),
        )
        return ok_response(serialize_gene_analysis(resolved_state, analysis))

    @app.get("/api/kbulk-analysis")
    def kbulk_analysis(
        q: str,
        class_key: str | None = None,
        label_key: str | None = None,
        label: str | None = None,
        k: int = 8,
        n_samples: int = 24,
        sample_seed: int = 0,
        max_classes: int | None = None,
        sample_batch_size: int = 32,
        kbulk_prior_source: PriorSource = "global",
        torch_dtype: Literal["float32", "float64"] = "float64",
        compile_model: bool = True,
        device: str = "cpu",
    ) -> object:
        result = compute_kbulk_analysis(
            resolved_state,
            query=q,
            label_key=label_key,
            label=label,
            params=KBulkParams(
                class_key=class_key,
                k=k,
                n_samples=n_samples,
                sample_seed=sample_seed,
                max_classes=(
                    config.kbulk_default_max_classes
                    if max_classes is None
                    else max_classes
                ),
                sample_batch_size=sample_batch_size,
                kbulk_prior_source=kbulk_prior_source,
                torch_dtype=torch_dtype,
                compile_model=compile_model,
                device=device,
            ),
        )
        return ok_response(serialize_kbulk_analysis(resolved_state, result))

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str = "") -> object:
        if full_path.startswith("api"):
            raise ApiError(404, "not_found", "Not Found")
        return serve_frontend_asset(full_path)

    return app


class LoadContextPayload(BaseModel):
    h5ad_path: str
    ckpt_path: str | None = None
    layer: str | None = None


def _mount_frontend_assets(app: FastAPI) -> None:
    assets_dir = frontend_dist_dir() / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=Path(assets_dir)),
            name="frontend-assets",
        )


__all__ = ["create_api_app"]
