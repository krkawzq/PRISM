from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from prism.server.services.analysis import CheckpointSummary, GeneFitParams, KBulkParams
from prism.server.views.gene import render_gene_page
from prism.server.views.home import cast_tuple_str, render_home_page
from prism.server.views.layout import render_loader, render_message, render_nav, render_page, stat_card


def _analysis() -> SimpleNamespace:
    posterior = SimpleNamespace(
        map_scaled_support=np.asarray([[1.0], [2.0]], dtype=np.float64),
        posterior_entropy=np.asarray([[0.1], [0.2]], dtype=np.float64),
        mutual_information=np.asarray([[0.3], [0.4]], dtype=np.float64),
    )
    return SimpleNamespace(
        gene_name="GeneA",
        gene_index=0,
        mode="fit",
        source="fit/global",
        prior_source="global",
        label_key=None,
        label=None,
        available_label_keys=("condition",),
        available_labels=("ctrl", "stim"),
        raw_summary=SimpleNamespace(
            mean_count=1.0,
            median_count=1.0,
            p99_count=2.0,
            detected_fraction=0.5,
            zero_fraction=0.5,
            count_total_correlation=0.1,
        ),
        n_cells=2,
        reference_source="checkpoint",
        reference_gene_count=10,
        posterior=posterior,
        prior=object(),
        fit_result=SimpleNamespace(objective_history=[1.0, 0.5]),
    )


def test_layout_helpers_render_expected_markup() -> None:
    page = render_page(title="Title", body="<p>x</p>")
    assert "<title>Title</title>" in page
    assert "PRISM Server" in render_nav(current_query="GeneA")
    assert "Load Dataset" in render_loader(h5ad_path="data.h5ad")
    assert "notice-error" in render_message("boom", level="error")
    assert "Cells" in stat_card("Cells", "10")


def test_render_home_page_covers_welcome_and_loaded_states() -> None:
    welcome = render_home_page(
        dataset_summary=None,
        checkpoint_summary=None,
        gene_browser=None,
    )
    assert "Welcome" in welcome

    loaded = render_home_page(
        dataset_summary={
            "n_cells": 10,
            "n_genes": 2,
            "layer": "counts",
            "h5ad_path": "/tmp/data.h5ad",
            "label_keys": ("condition",),
            "total_count_mean": 2.0,
            "total_count_median": 2.0,
            "total_count_p99": 4.0,
        },
        checkpoint_summary=CheckpointSummary(
            ckpt_path="/tmp/model.ckpt",
            gene_count=2,
            has_global_prior=True,
            n_label_priors=1,
            label_preview=("ctrl",),
            distribution="binomial",
            support_domain="probability",
            scale=2.0,
            mean_reference_count=4.0,
            n_reference_genes=5,
            n_overlap_reference_genes=4,
            suggested_label_key="condition",
        ),
        gene_browser=SimpleNamespace(
            query="Gene",
            scope="all",
            sort_by="total_count",
            descending=True,
            page=1,
            total_pages=1,
            total_items=1,
            items=[SimpleNamespace(gene_name="GeneA", gene_index=0, total_count=3, detected_cells=2, detected_fraction=1.0)],
        ),
    )
    assert "Dataset Snapshot" in loaded
    assert "Checkpoint Summary" in loaded
    assert "Gene Browser" in loaded
    assert cast_tuple_str(("a", "b")) == ("a", "b")
    assert cast_tuple_str(["a"]) == ()


def test_render_gene_page_renders_sections_and_kbulk_summary() -> None:
    html = render_gene_page(
        analysis=_analysis(),
        raw_figure="data:image/png;base64,raw",
        prior_figure="data:image/png;base64,prior",
        signal_figure="data:image/png;base64,signal",
        gallery_figure="data:image/png;base64,gallery",
        objective_figure="data:image/png;base64,obj",
        fit_params=GeneFitParams(),
        kbulk_params=KBulkParams(),
        kbulk_analysis=SimpleNamespace(
            class_key="condition",
            prior_source="global",
            k=4,
            n_samples=8,
            groups=[
                SimpleNamespace(
                    label="ctrl",
                    n_cells=10,
                    realized_samples=8,
                    mean_signal=1.2,
                    std_signal=0.1,
                    mean_entropy=0.2,
                    std_entropy=0.05,
                )
            ],
        ),
        kbulk_figure="data:image/png;base64,kbulk",
        error_message="boom",
        kbulk_error="kbulk failed",
    )
    assert "Analysis Controls" in html
    assert "On-Demand Fit" in html
    assert "kBulk Analysis" in html
    assert "Gene Summary" in html
    assert "kBulk Summary" in html
    assert "boom" in html
    assert "kbulk failed" in html
