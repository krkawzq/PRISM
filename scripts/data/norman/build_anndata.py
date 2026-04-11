from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from prism.io.anndata import write_h5ad


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_output_path() -> Path:
    return project_root() / "data" / "norman" / "01_counts" / "norman_counts.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an AnnData object from Cell Ranger v2 matrix-market outputs "
            "and compute CellPopulation-style summary statistics."
        )
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        required=True,
        help="Path to matrix.mtx or matrix.mtx.gz.",
    )
    parser.add_argument(
        "--barcodes-path",
        type=Path,
        required=True,
        help="Path to barcodes.tsv or barcodes.tsv.gz.",
    )
    parser.add_argument(
        "--genes-path",
        type=Path,
        required=True,
        help="Path to genes.tsv or genes.tsv.gz from Cell Ranger v2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output h5ad path.",
    )
    parser.add_argument(
        "--filter",
        dest="apply_filter",
        action="store_true",
        help="Filter cells by total UMI count >= --umi-threshold.",
    )
    parser.add_argument(
        "--no-filter",
        dest="apply_filter",
        action="store_false",
        help="Do not filter cells after loading counts.",
    )
    parser.set_defaults(apply_filter=False)
    parser.add_argument(
        "--umi-threshold",
        type=int,
        default=2000,
        help="Minimum per-cell UMI count retained when --filter is enabled.",
    )
    parser.add_argument(
        "--dataset-name",
        default="NormanWeissman2019",
        help="Stored in adata.uns for provenance.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def read_barcodes(path: Path) -> pd.DataFrame:
    barcodes = pd.read_csv(path, sep="\t", header=None, names=["cell_barcode"])
    if barcodes.empty:
        raise ValueError(f"No barcodes found in {path}")
    return barcodes


def read_genes(path: Path) -> pd.DataFrame:
    genes = pd.read_csv(path, sep="\t", header=None)
    if genes.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 columns in Cell Ranger v2 genes.tsv, got {genes.shape[1]} from {path}"
        )
    genes = genes.iloc[:, :2].copy()
    genes.columns = ["gene_id", "gene_name"]
    if genes.empty:
        raise ValueError(f"No genes found in {path}")
    return genes


def read_matrix(path: Path) -> sparse.csr_matrix:
    matrix = mmread(path)
    if not sparse.issparse(matrix):
        matrix = sparse.coo_matrix(matrix)
    # Cell Ranger matrix market stores genes x barcodes; CellPopulation uses cells x genes.
    matrix = matrix.transpose().tocsr().astype(np.int32)
    return matrix


def compute_gem_group(index: pd.Index) -> pd.Series:
    def parse_barcode(barcode: str) -> int | pd.NA:
        parts = str(barcode).rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return pd.NA

    gem_group = pd.Series((parse_barcode(x) for x in index), index=index, name="gem_group")
    return gem_group.astype("Int64")


def build_obs(barcodes: pd.DataFrame, matrix: sparse.csr_matrix) -> pd.DataFrame:
    obs = barcodes.set_index("cell_barcode").copy()
    umi_count = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.int64)
    obs["UMI_count"] = umi_count
    obs["gem_group"] = compute_gem_group(obs.index)
    return obs


def maybe_filter_cells(
    matrix: sparse.csr_matrix,
    obs: pd.DataFrame,
    *,
    apply_filter: bool,
    umi_threshold: int,
) -> tuple[sparse.csr_matrix, pd.DataFrame, dict[str, int]]:
    original_n_obs = matrix.shape[0]

    if not apply_filter:
        return matrix, obs, {"n_obs_before": original_n_obs, "n_obs_after": original_n_obs}

    keep_mask = obs["UMI_count"].to_numpy() >= umi_threshold
    filtered_matrix = matrix[keep_mask]
    filtered_obs = obs.loc[keep_mask].copy()
    return filtered_matrix, filtered_obs, {
        "n_obs_before": original_n_obs,
        "n_obs_after": filtered_matrix.shape[0],
    }


def compute_var_stats(matrix: sparse.csr_matrix, genes: pd.DataFrame) -> pd.DataFrame:
    n_obs = matrix.shape[0]
    if n_obs == 0:
        raise ValueError("No cells remain after filtering.")

    mean = np.asarray(matrix.mean(axis=0)).reshape(-1)
    mean_sq = np.asarray(matrix.astype(np.float64).power(2).mean(axis=0)).reshape(-1)
    var = np.clip(mean_sq - np.square(mean), a_min=0.0, a_max=None)
    std = np.sqrt(var)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = std / mean
        fano = var / mean

    cv[mean == 0] = np.nan
    fano[mean == 0] = np.nan

    var_df = genes.set_index("gene_id").copy()
    var_df["gene_name"] = var_df["gene_name"].astype(str)
    var_df["mean"] = mean
    var_df["std"] = std
    var_df["cv"] = cv
    var_df["fano"] = fano
    var_df["in_matrix"] = True
    return var_df


def build_anndata(
    matrix_path: Path,
    barcodes_path: Path,
    genes_path: Path,
    *,
    apply_filter: bool,
    umi_threshold: int,
    dataset_name: str,
) -> tuple[ad.AnnData, dict[str, int]]:
    barcodes = read_barcodes(barcodes_path)
    genes = read_genes(genes_path)
    matrix = read_matrix(matrix_path)

    expected_shape = (len(barcodes), len(genes))
    if matrix.shape != expected_shape:
        raise ValueError(
            "Matrix shape does not match barcode / gene files: "
            f"matrix={matrix.shape}, expected={expected_shape}"
        )

    obs = build_obs(barcodes, matrix)
    matrix, obs, filter_stats = maybe_filter_cells(
        matrix,
        obs,
        apply_filter=apply_filter,
        umi_threshold=umi_threshold,
    )
    var = compute_var_stats(matrix, genes)

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["source"] = "Cell Ranger v2 matrix market"
    adata.uns["dataset_name"] = dataset_name
    adata.uns["matrix_path"] = str(matrix_path.expanduser().resolve())
    adata.uns["barcodes_path"] = str(barcodes_path.expanduser().resolve())
    adata.uns["genes_path"] = str(genes_path.expanduser().resolve())
    adata.uns["filter_applied"] = bool(apply_filter)
    adata.uns["umi_threshold"] = int(umi_threshold)
    adata.uns["compat_source"] = "Perturbseq_GI CellPopulation.from_file (matrix-only subset)"
    adata.uns["cellpopulation_fields"] = {
        "obs": ["UMI_count", "gem_group"],
        "var": ["gene_name", "mean", "std", "cv", "fano", "in_matrix"],
    }
    adata.uns["build_stats"] = filter_stats
    return adata, filter_stats


def main() -> None:
    args = parse_args()

    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --force to overwrite."
        )

    adata, stats = build_anndata(
        args.matrix_path.expanduser().resolve(),
        args.barcodes_path.expanduser().resolve(),
        args.genes_path.expanduser().resolve(),
        apply_filter=args.apply_filter,
        umi_threshold=args.umi_threshold,
        dataset_name=args.dataset_name,
    )

    write_h5ad(adata, output_path)
    print(
        "Wrote "
        f"{adata.n_obs:,} cells x {adata.n_vars:,} genes to {output_path} "
        f"(before filter: {stats['n_obs_before']:,}, after filter: {stats['n_obs_after']:,})"
    )


if __name__ == "__main__":
    main()
