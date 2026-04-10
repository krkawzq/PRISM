from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prism.server.config import ServerConfig
from prism.server.handlers import (
    _parse_bool,
    _parse_fit_params,
    _parse_float,
    _parse_int,
    _parse_kbulk_params,
    _parse_optional_float,
    _resolve_likelihood,
    _resolve_mode,
    _resolve_prior_source,
    build_router,
    handle_asset,
    handle_gene,
    handle_health,
    handle_home,
    handle_search,
)
from prism.server.router import Request
from prism.server.services.analysis import GeneFitParams, KBulkParams
from prism.server.state import AppState


@dataclass
class _Candidate:
    gene_name: str = "GeneA"
    gene_index: int = 1
    total_count: int = 7
    detected_cells: int = 3
    detected_fraction: float = 0.5


def _state_with_loaded() -> AppState:
    state = AppState(ServerConfig())
    state._loaded = SimpleNamespace(  # noqa: SLF001
        dataset=SimpleNamespace(
            h5ad_path=Path("/tmp/data.h5ad"),
            layer="counts",
        ),
        checkpoint=SimpleNamespace(
            ckpt_path=Path("/tmp/model.ckpt"),
            checkpoint=SimpleNamespace(gene_names=["GeneA", "GeneB"]),
        ),
        n_cells=10,
        n_genes=2,
        label_keys=("condition",),
    )
    return state


def test_handler_parse_helpers_cover_clamping_and_modes() -> None:
    request = Request.from_raw_path(
        "GET",
        "/gene?fit=1&prior_source=label&scale=1.5&reference_source=dataset&n_support_points=1&max_em_iterations=7&likelihood=poisson&compile_model=0&k=0&n_samples=0",
    )
    assert _resolve_mode(request) == "fit"
    assert _resolve_prior_source("label") == "label"
    assert _resolve_prior_source("other") == "global"
    fit = _parse_fit_params(request)
    assert isinstance(fit, GeneFitParams)
    assert fit.scale == 1.5
    assert fit.reference_source == "dataset"
    assert fit.n_support_points == 2
    assert fit.max_em_iterations == 7
    assert fit.likelihood == "poisson"
    assert fit.compile_model is False
    state = AppState(ServerConfig())
    kbulk = _parse_kbulk_params(request, state)
    assert isinstance(kbulk, KBulkParams)
    assert kbulk.k == 1
    assert kbulk.n_samples == 1
    assert _resolve_likelihood("negative_binomial") == "negative_binomial"
    assert _parse_bool("yes", default=False) is True
    assert _parse_bool("off", default=True) is False
    assert _parse_int("0", default=2, min_value=1) == 1
    assert _parse_float("-1", default=2.0, min_value=0.0) == 0.0
    assert _parse_optional_float("") is None


def test_parse_fit_params_defaults_support_scales() -> None:
    request = Request.from_raw_path("GET", "/gene")

    params = _parse_fit_params(request)

    assert params.support_scale == 1.5
    assert params.adaptive_support_scale == 1.5


def test_handle_home_renders_loaded_and_unloaded_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unloaded = AppState(ServerConfig())
    response = handle_home(Request.from_raw_path("GET", "/"), unloaded)
    assert response.status == 200
    assert b"Welcome" in response.body

    loaded = _state_with_loaded()
    monkeypatch.setattr(
        "prism.server.handlers.build_dataset_summary",
        lambda _state: {
            "n_cells": 10,
            "n_genes": 2,
            "layer": "counts",
            "h5ad_path": "/tmp/data.h5ad",
            "label_keys": ("condition",),
            "total_count_mean": 2.5,
            "total_count_median": 2.0,
            "total_count_p99": 4.0,
        },
    )
    monkeypatch.setattr(
        "prism.server.handlers.build_checkpoint_summary", lambda _state: None
    )
    monkeypatch.setattr(
        "prism.server.handlers.browse_gene_candidates",
        lambda *args, **kwargs: SimpleNamespace(
            query="",
            scope="all",
            sort_by="total_count",
            descending=True,
            page=1,
            total_pages=1,
            total_items=1,
            items=[_Candidate()],
        ),
    )
    response = handle_home(Request.from_raw_path("GET", "/?browse_q=gene"), loaded)
    assert response.status == 200
    assert b"Dataset Snapshot" in response.body
    assert b"Gene Browser" in response.body


def test_handle_health_and_search_json(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state_with_loaded()
    health = handle_health(Request.from_raw_path("GET", "/api/health"), state)
    assert b'"loaded":true' in health.body
    monkeypatch.setattr(
        "prism.server.handlers.search_gene_candidates",
        lambda *_args, **_kwargs: [_Candidate()],
    )
    search = handle_search(Request.from_raw_path("GET", "/api/search?q=Gene"), state)
    assert b'"gene_name":"GeneA"' in search.body


def test_handle_gene_redirects_and_falls_back_to_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unloaded = AppState(ServerConfig())
    assert (
        handle_gene(Request.from_raw_path("GET", "/gene?q=GeneA"), unloaded).headers[
            "Location"
        ]
        == "/"
    )

    state = _state_with_loaded()
    analysis = SimpleNamespace(
        cache_key="cache-key",
        gene_name="GeneA",
        gene_index=0,
        mode="raw",
        source="raw-only",
        prior_source="global",
        label_key=None,
        label=None,
        available_label_keys=(),
        available_labels=(),
        raw_summary=SimpleNamespace(
            mean_count=1.0,
            median_count=1.0,
            p99_count=2.0,
            detected_fraction=0.5,
            zero_fraction=0.5,
            count_total_correlation=0.1,
        ),
        n_cells=2,
        reference_source="none",
        reference_gene_count=0,
        posterior=None,
        prior=None,
        fit_result=None,
    )
    calls = {"count": 0}

    def fake_build_gene_analysis(*_args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("boom")
        assert kwargs["mode"] == "raw"
        return analysis

    monkeypatch.setattr(
        "prism.server.handlers.build_gene_analysis", fake_build_gene_analysis
    )
    monkeypatch.setattr(
        "prism.server.handlers._cached_figure",
        lambda *_args, **_kwargs: "data:image/png;base64,abc",
    )
    monkeypatch.setattr(
        "prism.server.handlers.render_gene_page",
        lambda **kwargs: kwargs["error_message"] or "ok",
    )
    response = handle_gene(
        Request.from_raw_path("GET", "/gene?q=GeneA&mode=checkpoint"), state
    )
    assert response.status == 400
    assert b"boom" in response.body
    assert calls["count"] == 2


def test_handle_asset_and_router_dispatch() -> None:
    css = handle_asset(Request.from_raw_path("GET", "/assets/base.css"))
    assert css.status == 200
    assert css.content_type.startswith("text/css")
    missing = handle_asset(Request.from_raw_path("GET", "/assets/missing.css"))
    assert missing.status == 404

    router = build_router(AppState(ServerConfig()))
    response = router.dispatch(Request.from_raw_path("GET", "/favicon.ico"))
    assert response.status == 204
