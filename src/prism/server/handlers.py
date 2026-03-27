from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from http import HTTPStatus
from importlib import resources
from typing import Literal, cast

from prism.model import OptimizerName, SchedulerName

from .router import Request, Response, Router
from .services.analysis import (
    KBulkParams,
    GeneBrowsePage,
    GeneFitParams,
    analyze_gene,
    browse_gene_candidates,
    build_dataset_summary,
    search_gene_candidates,
    summarize_gene_expression,
)
from .services.datasets import GeneNotFoundError, resolve_gene_query, slice_gene_counts
from .services.figures import (
    plot_gene_overview,
    plot_global_overview,
    plot_kbulk_comparison,
    plot_loss_trace,
    plot_posterior_gallery,
    plot_prior_fit,
    plot_signal_interface,
)
from .services.global_eval import GlobalEvalParams, compute_global_evaluation
from .state import AppState
from .views.gene import render_gene_page, render_gene_pending_page
from .views.home import render_home_page


def build_router(state: AppState) -> Router:
    router = Router()
    router.add("GET", "/", lambda request: handle_home(request, state))
    router.add("GET", "/load", lambda request: handle_load(request, state))
    router.add("GET", "/gene", lambda request: handle_gene(request, state))
    router.add("GET", "/api/health", lambda request: handle_health(request, state))
    router.add("GET", "/api/search", lambda request: handle_search(request, state))
    router.add_prefix("GET", "/assets/", handle_asset)
    router.add("GET", "/favicon.ico", handle_favicon)
    return router


def handle_home(request: Request, state: AppState) -> Response:
    loaded = state.loaded
    query = (request.first("q") or "").strip()
    browse_query = (request.first("browse_q") or query).strip()
    browse_sort = (request.first("browse_sort") or "total_umi").strip()
    browse_dir = (request.first("browse_dir") or "desc").strip().lower()
    browse_scope = (request.first("browse_scope") or "auto").strip().lower()
    browse_page = max(1, int((request.first("browse_page") or "1").strip() or "1"))
    params = _parse_global_eval_params(request)
    if loaded is None:
        return Response.html(
            render_home_page(
                dataset_summary=None,
                gene_browser=None,
                search_query=query,
                h5ad_path="",
                ckpt_path="",
                layer="",
                global_eval_params=params,
            )
        )

    gene_browser = browse_gene_candidates(
        state,
        query=browse_query,
        sort_by=browse_sort,
        descending=browse_dir != "asc",
        page=browse_page,
        scope=browse_scope,
    )
    global_eval = None
    global_figures = None
    if request.first("global_eval") == "1" and loaded.checkpoint is not None:
        global_eval = compute_global_evaluation(state, params=params)
        global_figures = {
            "global_overview": _cached_figure(
                state,
                "home",
                "global_overview",
                lambda: plot_global_overview(global_eval),
            )
        }
    html = render_home_page(
        dataset_summary=build_dataset_summary(state),
        gene_browser=gene_browser,
        search_query=query,
        h5ad_path=str(loaded.dataset.h5ad_path),
        ckpt_path="" if loaded.checkpoint is None else str(loaded.checkpoint.ckpt_path),
        layer=loaded.dataset.layer or "",
        global_eval=global_eval,
        global_eval_params=params,
        global_eval_figures=global_figures,
    )
    return Response.html(html)


def handle_load(request: Request, state: AppState) -> Response:
    h5ad_path = (request.first("h5ad") or "").strip()
    ckpt_path = (request.first("ckpt") or "").strip()
    layer = (request.first("layer") or "").strip() or None
    params = _parse_global_eval_params(request)
    if not h5ad_path:
        return Response.html(
            render_home_page(
                dataset_summary=None,
                gene_browser=None,
                h5ad_path=h5ad_path,
                ckpt_path=ckpt_path,
                layer=layer or "",
                error_message="h5ad path is required.",
                global_eval_params=params,
            ),
            status=HTTPStatus.BAD_REQUEST,
        )
    try:
        loaded = state.load(
            h5ad_path=h5ad_path, ckpt_path=ckpt_path or None, layer=layer
        )
    except Exception as exc:
        return Response.html(
            render_home_page(
                dataset_summary=None,
                gene_browser=None,
                h5ad_path=h5ad_path,
                ckpt_path=ckpt_path,
                layer=layer or "",
                error_message=str(exc),
                global_eval_params=params,
            ),
            status=HTTPStatus.BAD_REQUEST,
        )
    html = render_home_page(
        dataset_summary=build_dataset_summary(state),
        gene_browser=browse_gene_candidates(state),
        h5ad_path=str(loaded.dataset.h5ad_path),
        ckpt_path="" if loaded.checkpoint is None else str(loaded.checkpoint.ckpt_path),
        layer=loaded.dataset.layer or "",
        global_eval=None,
        global_eval_params=params,
    )
    return Response.html(html)


def handle_gene(request: Request, state: AppState) -> Response:
    loaded = state.loaded
    if loaded is None:
        return Response.redirect("/")
    query = (request.first("q") or "").strip()
    if not query:
        return Response.redirect("/")
    fit_params = _parse_fit_params(request) if request.first("fit") == "1" else None
    kbulk_params = (
        _parse_kbulk_params(request) if request.first("kbulk") == "1" else None
    )
    try:
        gene_index = resolve_gene_query(
            query,
            loaded.dataset.gene_names,
            loaded.dataset.gene_names_lower,
            loaded.dataset.gene_to_idx,
            loaded.dataset.gene_lower_to_idx,
        )
    except GeneNotFoundError:
        return Response.html(
            render_home_page(
                dataset_summary=build_dataset_summary(state),
                gene_browser=browse_gene_candidates(state, query=query),
                search_query=query,
                h5ad_path=str(loaded.dataset.h5ad_path),
                ckpt_path=""
                if loaded.checkpoint is None
                else str(loaded.checkpoint.ckpt_path),
                layer=loaded.dataset.layer or "",
                error_message=f"Gene {query!r} not found.",
            ),
            status=HTTPStatus.NOT_FOUND,
        )
    gene_name = str(loaded.dataset.gene_names[gene_index])
    has_checkpoint_prior = loaded.checkpoint is not None and gene_name in set(
        loaded.fitted_gene_names
    )
    if not has_checkpoint_prior and fit_params is None and kbulk_params is None:
        summary = summarize_gene_expression(loaded, gene_index)
        figures = {
            "gene_overview": _cached_figure(
                state,
                f"pending-{gene_name}",
                "gene_overview",
                lambda: plot_gene_overview(
                    summary,
                    slice_gene_counts(loaded.dataset.matrix, gene_index),
                    loaded.dataset.totals,
                ),
            )
        }
        return Response.html(
            render_gene_pending_page(
                gene_name=gene_name,
                gene_index=gene_index,
                summary=summary,
                search_query=query,
                fit_params=None,
                kbulk_params=None if kbulk_params is None else asdict(kbulk_params),
                candidates=search_gene_candidates(state, query, limit=8),
                figures=figures,
                has_checkpoint=loaded.checkpoint is not None,
            )
        )
    analysis = analyze_gene(
        state, query, fit_params=fit_params, kbulk_params=kbulk_params
    )
    figures = {
        "gene_overview": _cached_figure(
            state,
            analysis.cache_key,
            "gene_overview",
            lambda: plot_gene_overview(
                analysis.summary, analysis.counts, analysis.reference_counts
            ),
        ),
        "prior_fit": _cached_figure(
            state, analysis.cache_key, "prior_fit", lambda: plot_prior_fit(analysis)
        ),
        "signal_interface": _cached_figure(
            state,
            analysis.cache_key,
            "signal_interface",
            lambda: plot_signal_interface(analysis),
        ),
        "posterior_gallery": _cached_figure(
            state,
            analysis.cache_key,
            "posterior_gallery",
            lambda: plot_posterior_gallery(analysis),
        ),
    }
    if analysis.fit_result is not None:
        figures["loss_trace"] = _cached_figure(
            state, analysis.cache_key, "loss_trace", lambda: plot_loss_trace(analysis)
        )
    if analysis.kbulk is not None:
        comparison = analysis.kbulk
        figures["kbulk"] = _cached_figure(
            state,
            analysis.cache_key,
            "kbulk",
            lambda: plot_kbulk_comparison(comparison),
        )
    html = render_gene_page(
        analysis=analysis,
        figures=figures,
        search_query=query,
        candidates=search_gene_candidates(state, query, limit=8),
        fit_params=None if fit_params is None else asdict(fit_params),
        kbulk_params=None if kbulk_params is None else asdict(kbulk_params),
        has_checkpoint=loaded.checkpoint is not None,
    )
    return Response.html(html)


def handle_health(_: Request, state: AppState) -> Response:
    loaded = state.loaded
    return Response.json(
        {
            "status": "ok",
            "loaded": loaded is not None,
            "n_cells": 0 if loaded is None else loaded.n_cells,
            "n_genes": 0 if loaded is None else loaded.n_genes,
            "fitted_genes": 0 if loaded is None else len(loaded.fitted_gene_names),
            "has_checkpoint": False
            if loaded is None
            else loaded.checkpoint is not None,
        }
    )


def handle_search(request: Request, state: AppState) -> Response:
    if state.loaded is None:
        return Response.json([])
    query = (request.first("q") or "").strip()
    candidates = search_gene_candidates(state, query, limit=state.config.top_gene_limit)
    return Response.json([asdict(candidate) for candidate in candidates])


def handle_asset(request: Request) -> Response:
    asset_name = request.path.removeprefix("/assets/")
    if not asset_name:
        return Response.text("Not Found", status=HTTPStatus.NOT_FOUND)
    asset_path = resources.files("prism.server.assets").joinpath(asset_name)
    if not asset_path.is_file():
        return Response.text("Not Found", status=HTTPStatus.NOT_FOUND)
    content_type = (
        "text/css; charset=utf-8"
        if asset_name.endswith(".css")
        else "application/octet-stream"
    )
    return Response(
        status=HTTPStatus.OK,
        content_type=content_type,
        body=_read_asset_bytes(asset_name),
        headers={"Cache-Control": "public, max-age=3600"},
    )


def handle_favicon(_: Request) -> Response:
    return Response(status=HTTPStatus.NO_CONTENT, content_type="image/x-icon", body=b"")


def _parse_fit_params(request: Request) -> GeneFitParams:
    def first(name: str, default: str) -> str:
        value = request.first(name)
        return default if value is None or value == "" else value

    optimizer = first("optimizer", "adamw")
    if optimizer not in {"adam", "adamw", "sgd", "rmsprop"}:
        optimizer = "adamw"
    scheduler = first("scheduler", "cosine")
    if scheduler not in {"cosine", "linear", "constant", "step"}:
        scheduler = "cosine"
    torch_dtype: Literal["float64", "float32"] | str = first("torch_dtype", "float64")
    if torch_dtype not in {"float64", "float32"}:
        torch_dtype = "float64"
    reference_mode = first("reference_mode", "checkpoint")
    if reference_mode not in {"checkpoint", "all"}:
        reference_mode = "checkpoint"
    s_raw = request.first("S")
    return GeneFitParams(
        S=None if s_raw is None or s_raw.strip() == "" else float(s_raw),
        reference_mode=cast(Literal["checkpoint", "all"], reference_mode),
        grid_size=int(first("grid_size", "512")),
        sigma_bins=float(first("sigma_bins", "1.0")),
        align_loss_weight=float(first("align_loss_weight", "1.0")),
        lr=float(first("lr", "0.05")),
        n_iter=int(first("n_iter", "100")),
        lr_min_ratio=float(first("lr_min_ratio", "0.1")),
        grad_clip=None
        if first("grad_clip", "") == ""
        else float(first("grad_clip", "0")),
        init_temperature=float(first("init_temperature", "1.0")),
        cell_chunk_size=int(first("cell_chunk_size", "512")),
        optimizer=cast(OptimizerName, optimizer),
        scheduler=cast(SchedulerName, scheduler),
        torch_dtype=cast(Literal["float64", "float32"], torch_dtype),
        device=first("device", "cpu"),
    )


def _parse_global_eval_params(request: Request) -> GlobalEvalParams:
    def first(name: str, default: str) -> str:
        value = request.first(name)
        return default if value is None or value == "" else value

    return GlobalEvalParams(
        max_cells=max(64, int(first("ge_max_cells", "2000"))),
        max_genes=max(8, int(first("ge_max_genes", "256"))),
        gene_batch_size=max(1, int(first("ge_batch", "64"))),
        random_seed=max(0, int(first("ge_seed", "0"))),
    )


def _parse_kbulk_params(request: Request) -> KBulkParams:
    def first(name: str, default: str) -> str:
        value = request.first(name)
        return default if value is None or value == "" else value

    return KBulkParams(
        k=max(2, int(first("kbulk_k", "8"))),
        n_samples=max(1, int(first("kbulk_samples", "24"))),
        max_groups=max(1, int(first("kbulk_groups", "4"))),
        min_cells_per_group=max(4, int(first("kbulk_min_cells", "24"))),
        random_seed=max(0, int(first("kbulk_seed", "0"))),
    )


def _cached_figure(
    state: AppState, analysis_key: str, figure_name: str, factory
) -> str:
    cache_key = state.make_cache_key("figures", analysis_key, figure_name)
    return cast(str, state.get_or_create_cache("figures", cache_key, factory))


@lru_cache(maxsize=32)
def _read_asset_bytes(asset_name: str) -> bytes:
    asset_path = resources.files("prism.server.assets").joinpath(asset_name)
    return asset_path.read_bytes()
