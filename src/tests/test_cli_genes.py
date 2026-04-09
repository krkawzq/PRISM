from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from prism.cli.genes.common import (
    RankingResult,
    compute_signal_ranking,
    merge_gene_lists,
    subset_gene_list,
)
from prism.cli.genes.filter import filter_genes_command
from prism.io import write_gene_list


def test_ranking_result_validates_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        RankingResult(
            gene_names=np.array(["a", "b"]),
            scores=np.array([1.0]),
        )


def test_compute_signal_ranking_supports_sparse_layers() -> None:
    adata = ad.AnnData(
        X=np.zeros((2, 2), dtype=np.float64),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    adata.layers["signal"] = sparse.csr_matrix([[0.0, 1.0], [0.0, 3.0]])
    gene_names, scores, metadata = compute_signal_ranking(
        adata,
        method="signal-variance",
    )
    assert gene_names.tolist() == ["g1", "g2"]
    assert scores.shape == (2,)
    assert metadata["layer"] == "signal"


def test_filter_genes_command_rejects_empty_output(tmp_path: Path) -> None:
    input_path = tmp_path / "genes.txt"
    output_path = tmp_path / "kept.txt"
    write_gene_list(input_path, ["MALAT1"])
    with pytest.raises(ValueError, match="removed all genes"):
        filter_genes_command(
            input_genes=input_path,
            output_path=output_path,
            species="human",
        )


def test_merge_gene_lists_union_preserves_rank_sum_order() -> None:
    ordered, metadata = merge_gene_lists(
        [["g1", "g2", "g3"], ["g2", "g3", "g4"]],
        method="rank-sum",
        gene_set_mode="union",
    )
    assert ordered == ["g2", "g1", "g3", "g4"]
    assert metadata["gene_set_mode"] == "union"


def test_subset_gene_list_reports_filtered_size() -> None:
    selected, metadata = subset_gene_list(
        ["g1", "g2", "g3", "g4"],
        start=1,
        end=None,
        top_k=2,
        intersect_genes=["g2", "g3", "g4"],
        exclude_genes=["g4"],
    )
    assert selected == ["g3"]
    assert metadata["n_after_filtering"] == 2
    assert metadata["n_output_genes"] == 1
