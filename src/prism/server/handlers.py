from __future__ import annotations

"""Legacy server-rendered handlers kept for reference and compatibility tests.

The active `prism serve` runtime uses FastAPI routes from `prism.server.api_routes`.
"""

from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from typing import cast

from .assets import build_asset_response
from .queries import GenePageQuery, HomePageQuery, LoadRequestData
from .queries import parse_bool, parse_fit_params, parse_float, parse_int
from .queries import parse_kbulk_params, parse_optional_float, resolve_likelihood
from .queries import resolve_mode, resolve_prior_source
from .router import Request, Response, Router
from .services.analysis import (
    AnalysisMode,
    BrowseScope,
    BrowseSort,
    GeneAnalysis,
    KBulkAnalysis,
    PriorSource,
    browse_gene_candidates,
    build_checkpoint_summary,
    build_dataset_summary,
    build_gene_analysis,
    compute_kbulk_analysis,
    search_gene_candidates,
)
from .services.figures import (
    figure_to_data_uri,
    plot_kbulk_group_comparison,
    plot_objective_trace,
    plot_posterior_gallery,
    plot_prior_overlay,
    plot_raw_overview,
    plot_signal_interface,
)
from .state import AppState
from .views import render_gene_page, render_home_page


@dataclass(frozen=True, slots=True)
class GenePageFigures:
    raw_figure: str
    prior_figure: str | None
    signal_figure: str | None
    gallery_figure: str | None
    objective_figure: str | None


def build_router(state: AppState) -> Router:
    router = Router()
    router.add("GET", "/", partial(handle_home, state=state))
    router.add("GET", "/load", partial(handle_load, state=state))
    router.add("GET", "/gene", partial(handle_gene, state=state))
    router.add("GET", "/api/health", partial(handle_health, state=state))
    router.add("GET", "/api/search", partial(handle_search, state=state))
    router.add_prefix("GET", "/assets/", handle_asset)
    router.add(
        "GET",
        "/favicon.ico",
        lambda _request: Response.text("", status=HTTPStatus.NO_CONTENT),
    )
    return router


def handle_home(request: Request, state: AppState) -> Response:
    query = HomePageQuery.from_request(request)
    loaded = state.loaded
    if loaded is None:
        return Response.html(
            render_home_page(
                dataset_summary=None,
                checkpoint_summary=None,
                gene_browser=None,
                search_query=query.search_query,
                h5ad_path="",
                ckpt_path="",
                layer="",
            )
        )
    browse_sort: BrowseSort = cast(BrowseSort, query.browse_sort)
    browse_scope: BrowseScope = cast(BrowseScope, query.browse_scope)
    return Response.html(
        render_home_page(
            dataset_summary=build_dataset_summary(state),
            checkpoint_summary=build_checkpoint_summary(state),
            gene_browser=browse_gene_candidates(
                state,
                query=query.browse_query,
                sort_by=browse_sort,
                descending=query.browse_dir != "asc",
                page=query.browse_page,
                scope=browse_scope,
            ),
            search_query=query.search_query,
            h5ad_path=str(loaded.dataset.h5ad_path),
            ckpt_path=""
            if loaded.checkpoint is None
            else str(loaded.checkpoint.ckpt_path),
            layer=loaded.dataset.layer or "",
        )
    )


def handle_load(request: Request, state: AppState) -> Response:
    form = LoadRequestData.from_request(request)
    if not form.h5ad_path:
        return _render_load_error(form, "h5ad path is required.")
    try:
        state.load(
            h5ad_path=form.h5ad_path,
            ckpt_path=form.ckpt_path or None,
            layer=form.layer or None,
        )
    except Exception as exc:
        return _render_load_error(form, str(exc))
    return handle_home(Request(method="GET", path="/", raw_path="/", query={}), state)


def handle_gene(request: Request, state: AppState) -> Response:
    if state.loaded is None:
        return Response.redirect("/")
    page_query = _parse_gene_page_query(request, state)
    if not page_query.query:
        return Response.redirect("/")
    analysis, error_message = _load_gene_analysis(state, page_query)
    figures = _build_gene_figures(state, analysis)
    kbulk_analysis, kbulk_figure, kbulk_error = _run_kbulk_analysis(
        state=state, page_query=page_query
    )
    return Response.html(
        render_gene_page(
            analysis=analysis,
            raw_figure=figures.raw_figure,
            prior_figure=figures.prior_figure,
            signal_figure=figures.signal_figure,
            gallery_figure=figures.gallery_figure,
            objective_figure=figures.objective_figure,
            fit_params=page_query.fit_params,
            kbulk_params=page_query.kbulk_params,
            kbulk_analysis=kbulk_analysis,
            kbulk_figure=kbulk_figure,
            error_message=error_message,
            kbulk_error=kbulk_error,
        ),
        status=HTTPStatus.BAD_REQUEST if error_message else HTTPStatus.OK,
    )


def handle_health(_request: Request, state: AppState) -> Response:
    loaded = state.loaded
    if loaded is None:
        return Response.json({"status": "ok", "loaded": False})
    fitted_genes = (
        0 if loaded.checkpoint is None else len(loaded.checkpoint.checkpoint.gene_names)
    )
    return Response.json(
        {
            "status": "ok",
            "loaded": True,
            "n_cells": loaded.n_cells,
            "n_genes": loaded.n_genes,
            "fitted_genes": fitted_genes,
            "has_checkpoint": loaded.checkpoint is not None,
        }
    )


def handle_search(request: Request, state: AppState) -> Response:
    query = (request.first("q") or "").strip()
    items = search_gene_candidates(state, query)
    return Response.json(
        [
            {
                "gene_name": item.gene_name,
                "gene_index": item.gene_index,
                "total_umi": item.total_count,
                "detected_cells": item.detected_cells,
                "detected_fraction": item.detected_fraction,
            }
            for item in items
        ]
    )


def handle_asset(request: Request) -> Response:
    return build_asset_response(request.path.removeprefix("/assets/"))


def _render_load_error(form: LoadRequestData, error_message: str) -> Response:
    return Response.html(
        render_home_page(
            dataset_summary=None,
            checkpoint_summary=None,
            gene_browser=None,
            h5ad_path=form.h5ad_path,
            ckpt_path=form.ckpt_path,
            layer=form.layer,
            error_message=error_message,
        ),
        status=HTTPStatus.BAD_REQUEST,
    )


def _parse_gene_page_query(request: Request, state: AppState) -> GenePageQuery:
    return GenePageQuery(
        query=(request.first("q") or "").strip(),
        mode=cast(AnalysisMode, _resolve_mode(request)),
        prior_source=cast(
            PriorSource,
            _resolve_prior_source(request.first("prior_source")),
        ),
        label_key=(request.first("label_key") or "").strip() or None,
        label=(request.first("label") or "").strip() or None,
        fit_params=_parse_fit_params(request),
        kbulk_params=_parse_kbulk_params(request, state),
        run_kbulk=(request.first("kbulk") or "") == "1",
    )


def _load_gene_analysis(
    state: AppState,
    page_query: GenePageQuery,
) -> tuple[GeneAnalysis, str | None]:
    error_message: str | None = None
    try:
        analysis = build_gene_analysis(
            state,
            query=page_query.query,
            mode=page_query.mode,
            prior_source=page_query.prior_source,
            label_key=page_query.label_key,
            label=page_query.label,
            fit_params=page_query.fit_params if page_query.mode == "fit" else None,
        )
    except Exception as exc:
        error_message = str(exc)
        analysis = build_gene_analysis(
            state,
            query=page_query.query,
            mode="raw",
            prior_source="global",
        )
    return analysis, error_message


def _build_gene_figures(state: AppState, analysis: GeneAnalysis) -> GenePageFigures:
    raw_figure = _cached_figure(
        state, analysis.cache_key, "raw", lambda: plot_raw_overview(analysis)
    )
    prior_figure = (
        None
        if analysis.prior is None
        else _cached_figure(
            state,
            analysis.cache_key,
            "prior",
            lambda: plot_prior_overlay(analysis),
        )
    )
    signal_figure = (
        None
        if analysis.posterior is None
        else _cached_figure(
            state,
            analysis.cache_key,
            "signal",
            lambda: plot_signal_interface(analysis),
        )
    )
    gallery_figure = None
    if (
        analysis.posterior is not None
        and analysis.posterior.posterior_probabilities is not None
    ):
        gallery_figure = _cached_figure(
            state,
            analysis.cache_key,
            "gallery",
            lambda: plot_posterior_gallery(analysis),
        )
    objective_figure = (
        None
        if analysis.fit_result is None
        else _cached_figure(
            state,
            analysis.cache_key,
            "objective",
            lambda: plot_objective_trace(analysis),
        )
    )
    return GenePageFigures(
        raw_figure=raw_figure,
        prior_figure=prior_figure,
        signal_figure=signal_figure,
        gallery_figure=gallery_figure,
        objective_figure=objective_figure,
    )


def _run_kbulk_analysis(
    *,
    state: AppState,
    page_query: GenePageQuery,
) -> tuple[KBulkAnalysis | None, str | None, str | None]:
    if not page_query.run_kbulk:
        return None, None, None
    try:
        kbulk_analysis = compute_kbulk_analysis(
            state,
            query=page_query.query,
            params=page_query.kbulk_params,
            label_key=page_query.label_key,
            label=page_query.label,
        )
        kbulk_figure = _cached_figure(
            state,
            kbulk_analysis.cache_key,
            "kbulk",
            lambda: plot_kbulk_group_comparison(kbulk_analysis),
        )
        return kbulk_analysis, kbulk_figure, None
    except Exception as exc:
        return None, None, str(exc)


def _cached_figure(state: AppState, cache_key: str, figure_name: str, factory) -> str:
    key = state.make_cache_key("figures", cache_key, figure_name)
    return state.get_or_create_cache(
        "figures", key, lambda: figure_to_data_uri(factory())
    )


def _resolve_mode(request: Request) -> AnalysisMode:
    return resolve_mode(request)


def _resolve_prior_source(value: str | None) -> PriorSource:
    return resolve_prior_source(value)


def _parse_fit_params(request: Request):
    return parse_fit_params(request)


def _parse_kbulk_params(request: Request, state: AppState):
    return parse_kbulk_params(
        request,
        kbulk_default_max_classes=state.config.kbulk_default_max_classes,
    )


def _resolve_likelihood(value: str | None):
    return resolve_likelihood(value)


def _parse_bool(value: str | None, *, default: bool) -> bool:
    return parse_bool(value, default=default)


def _parse_int(value: str | None, *, default: int, min_value: int | None = None) -> int:
    return parse_int(value, default=default, min_value=min_value)


def _parse_float(
    value: str | None, *, default: float, min_value: float | None = None
) -> float:
    return parse_float(value, default=default, min_value=min_value)


def _parse_optional_float(value: str | None) -> float | None:
    return parse_optional_float(value)
