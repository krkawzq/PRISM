from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from prism.cli.data.downsample import stratified_sample_indices
from prism.cli.data.subset_genes import subset_genes_command
from prism.io import write_gene_list


def test_stratified_sample_indices_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="label column is empty"):
        stratified_sample_indices(
            pd.Series([], dtype="object"),
            fraction=0.5,
            seed=0,
            per_class_min=1,
        )


def test_stratified_sample_indices_retains_each_class_minimum() -> None:
    sampled = stratified_sample_indices(
        pd.Series(["a", "a", "b"]),
        fraction=0.5,
        seed=0,
        per_class_min=1,
    )
    assert sampled.tolist() == [1, 2]


def test_subset_genes_command_writes_subset_h5ad(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    input_path = tmp_path / "input.h5ad"
    genes_path = tmp_path / "genes.txt"
    output_path = tmp_path / "subset.h5ad"
    adata.write_h5ad(input_path)
    write_gene_list(genes_path, ["g2"])

    result = subset_genes_command(
        input_path=input_path,
        genes_path=genes_path,
        output_path=output_path,
    )

    assert result == 0
    subset = ad.read_h5ad(output_path)
    assert subset.var_names.tolist() == ["g2"]
    assert subset.uns["gene_subset"]["n_selected_genes"] == 1
