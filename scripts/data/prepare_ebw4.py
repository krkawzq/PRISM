from __future__ import annotations

import argparse
import gzip
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


GEO_ACCESSION = "GSE231935"
EBW4_SAMPLE = "GSM7306273"
GEO_DOWNLOAD_URL = (
    f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={GEO_ACCESSION}&format=file"
)

HOST_EXTRA_CHR = {
    "MG1655": {"Lambda"},
}


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return project_root() / "data" / "ebw4"


def default_raw_dir() -> Path:
    return project_root() / "data" / "raw" / "ebw4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download eBW4 raw files and build per-species h5ad outputs."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir(),
        help="Directory for the GEO tarball and extracted GSM7306273 csv.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_data_dir(),
        help="Directory where h5ad files are written.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the GEO tarball even if it already exists.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_ebw4_members(tar_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(tar_path, "r") as archive:
        members = [
            member
            for member in archive.getmembers()
            if Path(member.name).name.startswith(f"{EBW4_SAMPLE}_")
        ]
        if not members:
            raise RuntimeError(f"No {EBW4_SAMPLE} members found in {tar_path}")
        for member in members:
            member_path = output_dir / Path(member.name).name
            if member_path.exists() and member.size == member_path.stat().st_size:
                extracted.append(member_path)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Failed to extract {member.name}")
            with source, member_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(member_path)
    return sorted(extracted)


def grouped_expression_files(input_dir: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(input_dir.glob("GSM7306273_*_*.csv.gz")):
        name = path.name
        if "expression_filtered_" in name:
            suffix = name.replace(
                "GSM7306273_expression_filtered_", "", 1
            ).removesuffix(".csv.gz")
            groups[suffix]["expression"] = path
        elif "cell_index_" in name:
            suffix = name.replace("GSM7306273_cell_index_", "", 1).removesuffix(
                ".csv.gz"
            )
            groups[suffix]["cell_index"] = path
        elif "gene_index_" in name:
            suffix = name.replace("GSM7306273_gene_index_", "", 1).removesuffix(
                ".csv.gz"
            )
            groups[suffix]["gene_index"] = path
    complete = {
        suffix: files
        for suffix, files in groups.items()
        if {"expression", "cell_index", "gene_index"}.issubset(files)
    }
    if not complete:
        raise RuntimeError(f"No complete eBW4 file groups found in {input_dir}")
    return complete


def read_gene_index(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")


def read_cell_index(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")


def allowed_chrs(species: str) -> set[str]:
    return {species, *HOST_EXTRA_CHR.get(species, set())}


def build_sparse_matrix(
    expression_path: Path,
    selected_rows: np.ndarray,
    selected_cols: np.ndarray,
) -> sp.csr_matrix:
    row_lookup = np.full(selected_rows.max() + 1, -1, dtype=np.int64)
    row_lookup[selected_rows] = np.arange(selected_rows.size, dtype=np.int64)

    data: list[int] = []
    rows: list[int] = []
    cols: list[int] = []

    with gzip.open(expression_path, "rt", newline="") as handle:
        header = handle.readline()
        if not header:
            raise RuntimeError(f"Empty expression file: {expression_path}")
        for input_row, line in enumerate(handle):
            if input_row >= row_lookup.size:
                break
            output_row = row_lookup[input_row]
            if output_row < 0:
                continue
            values = np.fromstring(line.strip(), sep=",", dtype=np.int32)
            subset = values[selected_cols]
            nonzero = np.flatnonzero(subset)
            if nonzero.size == 0:
                continue
            rows.extend([int(output_row)] * int(nonzero.size))
            cols.extend(nonzero.tolist())
            data.extend(subset[nonzero].tolist())

    shape = (selected_rows.size, selected_cols.size)
    return sp.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.int32)


def build_species_anndata(
    files: dict[str, Path], suffix: str, species: str
) -> ad.AnnData:
    cell_df = read_cell_index(files["cell_index"])
    gene_df = read_gene_index(files["gene_index"])

    row_mask = cell_df["identity"] == species
    if not row_mask.any():
        raise ValueError(f"No cells for species {species} in {suffix}")

    gene_mask = gene_df["chr"].isin(list(allowed_chrs(species)))
    if not bool(gene_mask.to_numpy().any()):
        raise ValueError(f"No genes for species {species} in {suffix}")

    selected_rows = np.flatnonzero(row_mask.to_numpy())
    selected_cols = gene_df.loc[gene_mask, "gene_index"].to_numpy(dtype=np.int64)
    selected_gene_df = (
        gene_df.loc[gene_mask].sort_values("gene_index").reset_index(drop=True)
    )

    with gzip.open(files["expression"], "rt", newline="") as handle:
        expression_header = handle.readline().strip().split(",")
    gene_names = [expression_header[idx] for idx in selected_cols]
    if gene_names != selected_gene_df["Name"].tolist():
        raise RuntimeError(f"Gene order mismatch in {suffix} for {species}")

    matrix = build_sparse_matrix(files["expression"], selected_rows, selected_cols)

    obs = cell_df.loc[row_mask].reset_index(drop=True).copy()
    obs.insert(0, "species", species)
    obs.insert(1, "source_dataset", suffix)
    obs.index = [f"{suffix}:{idx}" for idx in obs["cell_index"].tolist()]

    var = selected_gene_df.loc[:, ["chr", "Geneid", "Name", "gene_index"]].copy()
    var.rename(columns={"chr": "genome", "Name": "gene_name"}, inplace=True)
    var.index = var["Geneid"].astype(str).tolist()

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.uns["source"] = "GSE231935 / GSM7306273 (eBW4)"
    adata.uns["species"] = species
    adata.uns["source_dataset"] = suffix
    return adata


def concat_species(groups: dict[str, dict[str, Path]]) -> dict[str, ad.AnnData]:
    species_to_parts: dict[str, list[ad.AnnData]] = defaultdict(list)
    for suffix, files in sorted(groups.items()):
        cell_df = read_cell_index(files["cell_index"])
        for species in sorted(cell_df["identity"].unique()):
            species_to_parts[species].append(
                build_species_anndata(files, suffix, species)
            )

    merged: dict[str, ad.AnnData] = {}
    for species, parts in species_to_parts.items():
        merged[species] = ad.concat(parts, axis=0, join="outer", merge="same")
        merged[species].uns["source"] = "GSE231935 / GSM7306273 (eBW4)"
        merged[species].uns["species"] = species
        merged[species].uns["source_datasets"] = [
            part.uns["source_dataset"] for part in parts
        ]
    return merged


def prepare_ebw4(raw_dir: Path, output_dir: Path, force_download: bool) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = raw_dir / f"{GEO_ACCESSION}_RAW.tar"
    if force_download or not tar_path.exists():
        print(f"Downloading {GEO_ACCESSION} to {tar_path} ...")
        download_file(GEO_DOWNLOAD_URL, tar_path)
    else:
        print(f"Using existing tarball: {tar_path}")

    extracted = extract_ebw4_members(tar_path, raw_dir)
    print(f"Extracted {len(extracted)} eBW4 files to {raw_dir}")

    groups = grouped_expression_files(raw_dir)
    species_adatas = concat_species(groups)
    for species, adata in sorted(species_adatas.items()):
        output_path = output_dir / f"ebw4_{species.lower()}.h5ad"
        adata.write_h5ad(output_path)
        print(
            f"Wrote {species}: {adata.n_obs} cells x {adata.n_vars} genes -> {output_path}"
        )


def main() -> None:
    args = parse_args()
    prepare_ebw4(
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
