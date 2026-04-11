from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Depends, Request

from .api_contracts import LoadContextPayload
from .api_responses import ApiError, ok_response
from .api_serializers import (
    serialize_context_snapshot,
    serialize_gene_analysis,
    serialize_gene_browse_page,
    serialize_gene_candidate,
    serialize_kbulk_analysis,
)
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


def build_api_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health(state: AppState = Depends(get_prism_state)) -> object:
        loaded = state.loaded
        return ok_response(
            {
                "status": "ok",
                "loaded": loaded is not None,
                "contextKey": None if loaded is None else loaded.context_key,
            }
        )

    @router.get("/context")
    def context(state: AppState = Depends(get_prism_state)) -> object:
        return ok_response(
            serialize_context_snapshot(
                state,
                dataset_summary=build_dataset_summary(state),
                checkpoint_summary=build_checkpoint_summary(state),
            )
        )

    @router.post("/context/load")
    def load_context(
        payload: LoadContextPayload,
        state: AppState = Depends(get_prism_state),
    ) -> object:
        if not payload.h5ad_path.strip():
            raise ApiError(400, "missing_dataset_path", "h5ad_path is required")
        state.load(
            h5ad_path=payload.h5ad_path,
            ckpt_path=payload.ckpt_path,
            layer=payload.layer,
        )
        return ok_response(
            serialize_context_snapshot(
                state,
                dataset_summary=build_dataset_summary(state),
                checkpoint_summary=build_checkpoint_summary(state),
            )
        )

    @router.get("/genes")
    def browse_genes(
        query: str = "",
        scope: BrowseScope = "auto",
        sort_by: BrowseSort = "total_count",
        direction: Literal["asc", "desc"] = "desc",
        page: int = 1,
        state: AppState = Depends(get_prism_state),
    ) -> object:
        if state.loaded is None:
            return ok_response(
                _empty_browse_page(
                    state,
                    query=query,
                    scope=scope,
                    sort_by=sort_by,
                    direction=direction,
                )
            )
        page_data = browse_gene_candidates(
            state,
            query=query,
            scope=scope,
            sort_by=sort_by,
            descending=direction != "asc",
            page=page,
        )
        return ok_response(serialize_gene_browse_page(page_data))

    @router.get("/genes/search")
    def search_genes(
        q: str = "",
        limit: int | None = None,
        state: AppState = Depends(get_prism_state),
    ) -> object:
        if state.loaded is None:
            return ok_response({"items": []})
        return ok_response(
            {
                "items": [
                    serialize_gene_candidate(item)
                    for item in search_gene_candidates(state, q, limit=limit)
                ]
            }
        )

    @router.get("/gene-analysis")
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
        cell_chunk_size: int = 4096,
        support_max_from: Literal["observed_max", "quantile"] = "observed_max",
        support_spacing: Literal["linear", "sqrt"] = "linear",
        support_scale: float = 1.5,
        use_adaptive_support: bool = False,
        adaptive_support_scale: float = 1.5,
        adaptive_support_quantile: float = 0.99,
        likelihood: Literal["binomial", "negative_binomial", "poisson"] = "binomial",
        nb_overdispersion: float = 0.01,
        torch_dtype: Literal["float32", "float64"] = "float64",
        compile_model: bool = True,
        device: str = "cpu",
        state: AppState = Depends(get_prism_state),
    ) -> object:
        analysis = build_gene_analysis(
            state,
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
                    support_scale=support_scale,
                    use_adaptive_support=use_adaptive_support,
                    adaptive_support_scale=adaptive_support_scale,
                    adaptive_support_quantile=adaptive_support_quantile,
                    likelihood=likelihood,
                    nb_overdispersion=nb_overdispersion,
                    torch_dtype=torch_dtype,
                    compile_model=compile_model,
                    device=device,
                )
            ),
        )
        return ok_response(serialize_gene_analysis(state, analysis))

    @router.get("/kbulk-analysis")
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
        state: AppState = Depends(get_prism_state),
    ) -> object:
        result = compute_kbulk_analysis(
            state,
            query=q,
            label_key=label_key,
            label=label,
            params=KBulkParams(
                class_key=class_key,
                k=k,
                n_samples=n_samples,
                sample_seed=sample_seed,
                max_classes=(
                    state.config.kbulk_default_max_classes
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
        return ok_response(serialize_kbulk_analysis(state, result))

    return router


def get_prism_state(request: Request) -> AppState:
    state = getattr(request.app.state, "prism_state", None)
    if state is None:
        raise RuntimeError("PRISM app state is not configured")
    return cast(AppState, state)


def _empty_browse_page(
    state: AppState,
    *,
    query: str,
    scope: BrowseScope,
    sort_by: BrowseSort,
    direction: Literal["asc", "desc"],
) -> dict[str, object]:
    return {
        "query": query,
        "scope": scope,
        "sortBy": sort_by,
        "descending": direction != "asc",
        "page": 1,
        "pageSize": state.config.browse_page_size,
        "totalItems": 0,
        "totalPages": 1,
        "items": [],
    }


__all__ = ["build_api_router", "get_prism_state"]
