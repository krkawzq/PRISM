from __future__ import annotations

from dataclasses import asdict
from http import HTTPStatus
from importlib import resources

from .router import Request, Response, Router
from .services.analysis import (
    GeneFitParams,
    analyze_gene,
    build_dataset_summary,
    search_gene_candidates,
    top_fitted_genes,
)
from .services.datasets import GeneNotFoundError
from .services.plotting import histogram_svg, multi_line_svg, scatter_svg
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
    if query:
        search_results = search_gene_candidates(
            state, query, limit=state.config.top_gene_limit
        )

    html = render_home_page(
        dataset_summary=build_dataset_summary(state),
        top_genes=top_fitted_genes(state),
        search_query=query,
        search_results=search_results,
        h5ad_path=str(loaded.h5ad_path),
        ckpt_path="" if loaded.ckpt_path is None else str(loaded.ckpt_path),
        layer=loaded.layer or "",
    )
    return Response.html(html)


def handle_load(request: Request, state: AppState) -> Response:
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

    html = render_home_page(
        dataset_summary=build_dataset_summary(state),
        top_genes=top_fitted_genes(state),
        h5ad_path=str(loaded.h5ad_path),
        ckpt_path="" if loaded.ckpt_path is None else str(loaded.ckpt_path),
        layer=loaded.layer or "",
    )
    return Response.html(html)


def handle_gene(request: Request, state: AppState) -> Response:
    loaded = state.loaded
    if loaded is None:
        return Response.redirect("/")
    query = (request.first("q") or "").strip()
    if not query:
        return Response.redirect("/")

    fit_params = None
    if request.first("fit") == "1":
        fit_params = _parse_fit_params(request)

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
            h5ad_path=str(loaded.h5ad_path),
            ckpt_path="" if loaded.ckpt_path is None else str(loaded.ckpt_path),
            layer=loaded.layer or "",
        )
        return Response.html(html, status=HTTPStatus.NOT_FOUND)

    figures = {
        "counts_hist": histogram_svg(
            analysis.counts,
            title="Raw counts",
            color="#155e75",
        ),
        "signal_hist": histogram_svg(
            analysis.signal,
            title="Signal",
            color="#0f766e",
        ),
        "xeff_signal": scatter_svg(
            analysis.x_eff,
            analysis.signal,
            title="X_eff vs signal",
            color="#1d4ed8",
            max_points=state.config.plot_max_points,
        ),
        "confidence_surprisal": scatter_svg(
            analysis.confidence,
            analysis.surprisal,
            title="Confidence vs surprisal",
            color="#c2410c",
            max_points=state.config.plot_max_points,
        ),
        "posterior_curves": multi_line_svg(
            analysis.support,
            analysis.posterior_samples,
            title="Posterior samples",
        ),
    }

    html = render_gene_page(
        analysis=analysis,
        figures=figures,
        search_query=query,
        fit_params=None if fit_params is None else analysis.fit_params,
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
        optimizer=first("optimizer", "adamw"),
        scheduler=first("scheduler", "cosine"),
        torch_dtype=first("torch_dtype", "float64"),
        device=first("device", "cpu"),
    )
