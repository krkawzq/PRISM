from __future__ import annotations

from dataclasses import asdict
from http import HTTPStatus
from importlib import resources
from typing import Literal, cast

from prism.model._typing import OptimizerName, SchedulerName

from .router import Request, Response, Router
from .services.analysis import (
    GeneFitParams,
    analyze_gene,
    build_dataset_summary,
    search_gene_candidates,
    top_fitted_genes,
)
from .services.datasets import GeneNotFoundError
from .services.figures import (
    plot_gene_overview,
    plot_init_comparison,
    plot_loss_trace,
    plot_posterior_gallery,
    plot_prior_fit,
    plot_signal_interface,
    plot_signal_interface_3d_html,
    plot_stage0,
    treatment_block,
)
from .services.global_eval import compute_global_evaluation
from .state import AppState
from .views.gene import render_gene_page
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
    print(f"[prism-server] route / query={request.query}", flush=True)
    loaded = state.loaded
    query = (request.first("q") or "").strip()
    if loaded is None:
        html = render_home_page(
            dataset_summary=None,
            top_genes=[],
            search_query=query,
            h5ad_path="",
            ckpt_path="",
            layer="",
        )
        return Response.html(html)

    search_results = None
    global_eval = None
    if query:
        search_results = search_gene_candidates(
            state, query, limit=state.config.top_gene_limit
        )
    if request.first("global_eval") == "1" and loaded.model.ckpt_path is not None:
        print("[prism-server] requested global evaluation from home page", flush=True)
        try:
            global_eval = compute_global_evaluation(state)
        except Exception as exc:
            print(f"[prism-server] global evaluation failed: {exc}", flush=True)

    html = _cached_html(
        state,
        "home",
        f"query={query}|global_eval={'1' if global_eval is not None else '0'}",
        lambda: render_home_page(
            dataset_summary=build_dataset_summary(state),
            top_genes=top_fitted_genes(state),
            search_query=query,
            search_results=search_results,
            h5ad_path=str(loaded.dataset.h5ad_path),
            ckpt_path=""
            if loaded.model.ckpt_path is None
            else str(loaded.model.ckpt_path),
            layer=loaded.dataset.layer or "",
            global_eval=global_eval,
        ),
    )
    return Response.html(html)


def handle_load(request: Request, state: AppState) -> Response:
    print(f"[prism-server] route /load query={request.query}", flush=True)
    h5ad_path = (request.first("h5ad") or "").strip()
    ckpt_path = (request.first("ckpt") or "").strip()
    layer = (request.first("layer") or "").strip() or None
    if not h5ad_path:
        html = render_home_page(
            dataset_summary=None,
            top_genes=[],
            h5ad_path=h5ad_path,
            ckpt_path=ckpt_path,
            layer=layer or "",
            error_message="h5ad path is required.",
        )
        return Response.html(html, status=HTTPStatus.BAD_REQUEST)

    try:
        loaded = state.load(
            h5ad_path=h5ad_path, ckpt_path=ckpt_path or None, layer=layer
        )
    except Exception as exc:
        html = render_home_page(
            dataset_summary=None,
            top_genes=[],
            h5ad_path=h5ad_path,
            ckpt_path=ckpt_path,
            layer=layer or "",
            error_message=str(exc),
        )
        return Response.html(html, status=HTTPStatus.BAD_REQUEST)

    html = _cached_html(
        state,
        "home",
        "default",
        lambda: render_home_page(
            dataset_summary=build_dataset_summary(state),
            top_genes=top_fitted_genes(state),
            h5ad_path=str(loaded.dataset.h5ad_path),
            ckpt_path=""
            if loaded.model.ckpt_path is None
            else str(loaded.model.ckpt_path),
            layer=loaded.dataset.layer or "",
            global_eval=None,
        ),
    )
    return Response.html(html)


def handle_gene(request: Request, state: AppState) -> Response:
    print(f"[prism-server] route /gene query={request.query}", flush=True)
    loaded = state.loaded
    if loaded is None:
        return Response.redirect("/")
    query = (request.first("q") or "").strip()
    if not query:
        return Response.redirect("/")

    fit_params = None
    if request.first("fit") == "1":
        fit_params = _parse_fit_params(request)
        print(f"[prism-server] parsed fit params={fit_params}", flush=True)

    try:
        analysis = analyze_gene(state, query, fit_params=fit_params)
    except GeneNotFoundError:
        html = render_home_page(
            dataset_summary=build_dataset_summary(state),
            top_genes=top_fitted_genes(state),
            search_query=query,
            search_results=search_gene_candidates(
                state, query, limit=state.config.top_gene_limit
            ),
            h5ad_path=str(loaded.dataset.h5ad_path),
            ckpt_path=""
            if loaded.model.ckpt_path is None
            else str(loaded.model.ckpt_path),
            layer=loaded.dataset.layer or "",
            error_message=f"Gene {query!r} not found.",
        )
        return Response.html(html, status=HTTPStatus.NOT_FOUND)

    figures = {
        "gene_overview": _cached_figure(
            state,
            analysis.cache_key,
            "gene_overview",
            lambda: plot_gene_overview(
                analysis.summary, analysis.counts, analysis.totals
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
        "signal_3d": _cached_figure(
            state,
            analysis.cache_key,
            "signal_3d",
            lambda: plot_signal_interface_3d_html(analysis),
        ),
        "posterior_gallery": _cached_figure(
            state,
            analysis.cache_key,
            "posterior_gallery",
            lambda: plot_posterior_gallery(analysis),
        ),
    }
    if analysis.pool_report is not None:
        figures["stage0"] = _cached_figure(
            state,
            analysis.cache_key,
            "stage0",
            lambda: plot_stage0(analysis.pool_report, analysis.totals),
        )
    if analysis.prior_report is not None:
        figures["loss_trace"] = _cached_figure(
            state,
            analysis.cache_key,
            "loss_trace",
            lambda: plot_loss_trace(analysis.prior_report),
        )
        figures["init_comparison"] = _cached_figure(
            state,
            analysis.cache_key,
            "init_comparison",
            lambda: plot_init_comparison(analysis.prior_report),
        )

    treatment_html = _cached_html(
        state,
        analysis.cache_key,
        "treatment_block",
        lambda: treatment_block(analysis.summary),
    )

    html = _cached_html(
        state,
        analysis.cache_key,
        "gene_page",
        lambda: render_gene_page(
            analysis=analysis,
            figures=figures,
            search_query=query,
            candidates=search_gene_candidates(state, query, limit=8),
            fit_params=None if fit_params is None else asdict(fit_params),
            treatment_block_html=treatment_html,
        ),
    )
    return Response.html(html)


def handle_health(_: Request, state: AppState) -> Response:
    print("[prism-server] route /api/health", flush=True)
    loaded = state.loaded
    return Response.json(
        {
            "status": "ok",
            "loaded": loaded is not None,
            "n_cells": 0 if loaded is None else loaded.n_cells,
            "n_genes": 0 if loaded is None else loaded.n_genes,
            "fitted_genes": 0 if loaded is None else len(loaded.fitted_gene_names),
        }
    )


def handle_search(request: Request, state: AppState) -> Response:
    print(f"[prism-server] route /api/search query={request.query}", flush=True)
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

    if asset_name.endswith(".css"):
        return Response(
            status=HTTPStatus.OK,
            content_type="text/css; charset=utf-8",
            body=asset_path.read_text(encoding="utf-8").encode("utf-8"),
        )
    return Response(
        status=HTTPStatus.OK,
        content_type="application/octet-stream",
        body=asset_path.read_bytes(),
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

    return GeneFitParams(
        r=float(first("r", "0.05")),
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


def _cached_figure(
    state: AppState,
    analysis_key: str,
    figure_name: str,
    factory,
) -> str:
    cache_key = state.make_cache_key("figures", analysis_key, figure_name)
    cached = state.get_cache("figures", cache_key)
    if cached is not None:
        print(f"[prism-server] figure cache hit name={figure_name}", flush=True)
        return cast(str, cached)
    print(f"[prism-server] figure cache miss name={figure_name}", flush=True)
    value = factory()
    state.set_cache("figures", cache_key, value)
    return value


def _cached_html(
    state: AppState,
    analysis_key: str,
    name: str,
    factory,
) -> str:
    cache_key = state.make_cache_key("html", analysis_key, name)
    cached = state.get_cache("html", cache_key)
    if cached is not None:
        print(f"[prism-server] html cache hit name={name}", flush=True)
        return cast(str, cached)
    print(f"[prism-server] html cache miss name={name}", flush=True)
    value = factory()
    state.set_cache("html", cache_key, value)
    return value
