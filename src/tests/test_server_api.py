from __future__ import annotations

from types import SimpleNamespace

from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
import pytest

from prism.server.api import create_api_app
from prism.server.config import ServerConfig
from prism.server.services.analysis import GeneFitParams, KBulkParams
from prism.server.state import AppState


def test_api_unloaded_context_health_and_browse_defaults() -> None:
    config = ServerConfig(browse_page_size=7)
    client = _make_client(AppState(config))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["data"] == {
        "status": "ok",
        "loaded": False,
        "contextKey": None,
    }

    context = client.get("/api/context")
    assert context.status_code == 200
    assert context.json()["data"] == {
        "loaded": False,
        "contextKey": None,
        "dataset": None,
        "checkpoint": None,
    }

    browse = client.get("/api/genes")
    assert browse.status_code == 200
    assert browse.json()["data"] == {
        "query": "",
        "scope": "auto",
        "sortBy": "total_count",
        "descending": True,
        "page": 1,
        "pageSize": 7,
        "totalItems": 0,
        "totalPages": 1,
        "items": [],
    }


def test_api_context_load_requires_dataset_path() -> None:
    client = _make_client(AppState(ServerConfig()))

    response = client.post("/api/context/load", json={"h5ad_path": "   "})

    assert response.status_code == 400
    body = response.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "missing_dataset_path"


def test_api_context_load_returns_serialized_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = AppState(ServerConfig())
    client = _make_client(state)

    def fake_load(*, h5ad_path: str, ckpt_path: str | None, layer: str | None) -> None:
        assert h5ad_path == "/tmp/data.h5ad"
        assert ckpt_path == "/tmp/model.ckpt"
        assert layer == "counts"
        state._loaded = SimpleNamespace(context_key="ctx-123")  # noqa: SLF001

    monkeypatch.setattr(state, "load", fake_load)
    monkeypatch.setattr(
        "prism.server.api_routes.build_dataset_summary",
        lambda _state: {
            "n_cells": 12,
            "n_genes": 3,
            "layer": "counts",
            "h5ad_path": "/tmp/data.h5ad",
            "label_keys": ("condition",),
            "total_count_mean": 4.5,
            "total_count_median": 4.0,
            "total_count_p99": 9.0,
        },
    )
    monkeypatch.setattr(
        "prism.server.api_routes.build_checkpoint_summary",
        lambda _state: None,
    )

    response = client.post(
        "/api/context/load",
        json={
            "h5ad_path": "/tmp/data.h5ad",
            "ckpt_path": "/tmp/model.ckpt",
            "layer": "counts",
        },
    )

    assert response.status_code == 200
    assert response.json()["data"] == {
        "loaded": True,
        "contextKey": "ctx-123",
        "dataset": {
            "nCells": 12,
            "nGenes": 3,
            "layer": "counts",
            "h5adPath": "/tmp/data.h5ad",
            "labelKeys": ["condition"],
            "totalCountMean": 4.5,
            "totalCountMedian": 4.0,
            "totalCountP99": 9.0,
        },
        "checkpoint": None,
    }


def test_api_gene_analysis_and_kbulk_routes_map_query_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = AppState(ServerConfig(kbulk_default_max_classes=9))
    client = _make_client(state)
    captured: dict[str, object] = {}

    def fake_build_gene_analysis(_state: AppState, **kwargs: object) -> object:
        captured["gene_analysis"] = kwargs
        return {"kind": "gene-analysis"}

    def fake_serialize_gene_analysis(_state: AppState, payload: object) -> object:
        return payload

    def fake_compute_kbulk_analysis(_state: AppState, **kwargs: object) -> object:
        captured["kbulk_analysis"] = kwargs
        return {"kind": "kbulk-analysis"}

    def fake_serialize_kbulk_analysis(_state: AppState, payload: object) -> object:
        return payload

    monkeypatch.setattr(
        "prism.server.api_routes.build_gene_analysis",
        fake_build_gene_analysis,
    )
    monkeypatch.setattr(
        "prism.server.api_routes.serialize_gene_analysis",
        fake_serialize_gene_analysis,
    )
    monkeypatch.setattr(
        "prism.server.api_routes.compute_kbulk_analysis",
        fake_compute_kbulk_analysis,
    )
    monkeypatch.setattr(
        "prism.server.api_routes.serialize_kbulk_analysis",
        fake_serialize_kbulk_analysis,
    )

    analysis_response = client.get(
        "/api/gene-analysis",
        params={
            "q": "GeneA",
            "mode": "fit",
            "prior_source": "label",
            "label_key": "condition",
            "label": "treated",
            "scale": 1.25,
            "reference_source": "dataset",
            "n_support_points": 128,
            "max_em_iterations": 11,
            "convergence_tolerance": 1e-5,
            "cell_chunk_size": 64,
            "support_max_from": "quantile",
            "support_spacing": "sqrt",
            "support_scale": 2.5,
            "use_adaptive_support": True,
            "adaptive_support_scale": 1.75,
            "adaptive_support_quantile": 0.95,
            "likelihood": "poisson",
            "nb_overdispersion": 0.2,
            "torch_dtype": "float32",
            "compile_model": False,
            "device": "cuda",
        },
    )

    assert analysis_response.status_code == 200
    assert analysis_response.json()["data"] == {"kind": "gene-analysis"}

    analysis_kwargs = captured["gene_analysis"]
    assert isinstance(analysis_kwargs, dict)
    assert analysis_kwargs["query"] == "GeneA"
    assert analysis_kwargs["mode"] == "fit"
    assert analysis_kwargs["prior_source"] == "label"
    assert analysis_kwargs["label_key"] == "condition"
    assert analysis_kwargs["label"] == "treated"
    fit_params = analysis_kwargs["fit_params"]
    assert isinstance(fit_params, GeneFitParams)
    assert fit_params.reference_source == "dataset"
    assert fit_params.n_support_points == 128
    assert fit_params.support_scale == 2.5
    assert fit_params.use_adaptive_support is True
    assert fit_params.adaptive_support_scale == 1.75
    assert fit_params.likelihood == "poisson"
    assert fit_params.compile_model is False
    assert fit_params.device == "cuda"

    kbulk_response = client.get(
        "/api/kbulk-analysis",
        params={
            "q": "GeneA",
            "class_key": "condition",
            "label_key": "condition",
            "label": "treated",
            "k": 4,
            "n_samples": 6,
            "sample_seed": 3,
            "sample_batch_size": 10,
            "kbulk_prior_source": "label",
            "torch_dtype": "float32",
            "compile_model": False,
            "device": "cuda",
        },
    )

    assert kbulk_response.status_code == 200
    assert kbulk_response.json()["data"] == {"kind": "kbulk-analysis"}

    kbulk_kwargs = captured["kbulk_analysis"]
    assert isinstance(kbulk_kwargs, dict)
    assert kbulk_kwargs["query"] == "GeneA"
    assert kbulk_kwargs["label_key"] == "condition"
    assert kbulk_kwargs["label"] == "treated"
    params = kbulk_kwargs["params"]
    assert isinstance(params, KBulkParams)
    assert params.class_key == "condition"
    assert params.k == 4
    assert params.n_samples == 6
    assert params.max_classes == 9
    assert params.kbulk_prior_source == "label"
    assert params.compile_model is False
    assert params.device == "cuda"


def test_spa_fallback_serves_frontend_and_rejects_api_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(AppState(ServerConfig()))
    monkeypatch.setattr(
        "prism.server.api.serve_frontend_asset",
        lambda path="": HTMLResponse(f"frontend:{path}"),
    )

    spa = client.get("/gene/TP53")
    assert spa.status_code == 200
    assert "frontend:gene/TP53" in spa.text

    missing_api = client.get("/api/missing")
    assert missing_api.status_code == 404
    assert missing_api.json()["error"]["code"] == "not_found"


def _make_client(state: AppState) -> TestClient:
    return TestClient(create_api_app(state.config, state=state))
