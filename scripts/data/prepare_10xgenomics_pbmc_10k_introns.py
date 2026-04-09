from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad


BASE_URL = "https://cf.10xgenomics.com/samples/cell-exp/6.1.2"
DATASET_NAME = "10k Human PBMCs (3' v3.1, Chromium X)"
DOWNLOAD_HEADERS = {
    "User-Agent": "Wget/1.21.4",
    "Accept": "*/*",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    url: str
    raw_filename: str
    output_filename: str


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="without_introns",
        label="without introns",
        url=(
            f"{BASE_URL}/10k_PBMC_3p_nextgem_Chromium_X/"
            "10k_PBMC_3p_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
        ),
        raw_filename="10k_PBMC_3p_nextgem_Chromium_X_filtered_feature_bc_matrix.h5",
        output_filename="pbmc_10k_without_introns_shared_barcodes.h5ad",
    ),
    DatasetSpec(
        key="with_introns",
        label="with introns",
        url=(
            f"{BASE_URL}/10k_PBMC_3p_nextgem_Chromium_X_intron/"
            "10k_PBMC_3p_nextgem_Chromium_X_intron_filtered_feature_bc_matrix.h5"
        ),
        raw_filename=(
            "10k_PBMC_3p_nextgem_Chromium_X_intron_filtered_feature_bc_matrix.h5"
        ),
        output_filename="pbmc_10k_with_introns_shared_barcodes.h5ad",
    ),
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_dir() -> Path:
    return project_root() / "data" / "10xgenomics" / "introns"


def default_raw_dir() -> Path:
    return project_root() / "data" / "raw" / "10xgenomics" / "introns"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download 10x Genomics PBMC 10k matrices with and without intronic "
            "reads, keep only shared cell barcodes, and write aligned h5ad files."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir(),
        help="Directory where the original 10x h5 downloads are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where the filtered h5ad outputs are written.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the source h5 files even if they already exist.",
    )
    return parser.parse_args()


def format_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def download_file(url: str, destination: Path, *, force: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Using existing download: {destination} ({format_mb(destination)})")
        return destination

    temp_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()

    print(f"Downloading {url} -> {destination}")
    try:
        request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
        with urllib.request.urlopen(request) as response, temp_path.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"Downloaded {destination} ({format_mb(destination)})")
    return destination


def read_10x_h5(path: Path) -> "ad.AnnData":
    import scanpy as sc

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Variable names are not unique.*",
            category=UserWarning,
        )
        adata = sc.read_10x_h5(path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def shared_barcodes(
    without_introns: "ad.AnnData", with_introns: "ad.AnnData"
) -> list[str]:
    with_introns_lookup = set(with_introns.obs_names)
    return [
        barcode
        for barcode in without_introns.obs_names
        if barcode in with_introns_lookup
    ]


def annotate_adata(
    adata: "ad.AnnData",
    *,
    spec: DatasetSpec,
    raw_path: Path,
    shared_barcode_count: int,
) -> "ad.AnnData":
    adata.uns["source"] = "10x Genomics"
    adata.uns["dataset_name"] = DATASET_NAME
    adata.uns["dataset_variant"] = spec.key
    adata.uns["download_url"] = spec.url
    adata.uns["raw_h5_path"] = str(raw_path)
    adata.uns["shared_barcode_count"] = shared_barcode_count
    return adata


def write_h5ad_atomic(adata: "ad.AnnData", output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()
    try:
        adata.write_h5ad(temp_path)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def prepare_pbmc_10k_introns(
    *, raw_dir: Path, output_dir: Path, force_download: bool
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    for spec in DATASETS:
        downloaded[spec.key] = download_file(
            spec.url,
            raw_dir / spec.raw_filename,
            force=force_download,
        )

    print(f"\nReading {DATASET_NAME} 10x matrices ...")
    adata_without = read_10x_h5(downloaded["without_introns"])
    adata_with = read_10x_h5(downloaded["with_introns"])

    print(
        f"Loaded without introns: {adata_without.n_obs:,} cells x "
        f"{adata_without.n_vars:,} genes"
    )
    print(
        f"Loaded with introns   : {adata_with.n_obs:,} cells x "
        f"{adata_with.n_vars:,} genes"
    )

    barcodes = shared_barcodes(adata_without, adata_with)
    if not barcodes:
        raise RuntimeError("No shared cell barcodes were found between the two inputs")

    without_only = adata_without.n_obs - len(barcodes)
    with_only = adata_with.n_obs - len(barcodes)

    adata_without = adata_without[barcodes].copy()
    adata_with = adata_with[barcodes].copy()

    print(f"Shared cell barcodes  : {len(barcodes):,}")
    print(f"Only without introns  : {without_only:,}")
    print(f"Only with introns     : {with_only:,}")

    by_key = {
        "without_introns": adata_without,
        "with_introns": adata_with,
    }

    for spec in DATASETS:
        output_path = output_dir / spec.output_filename
        prepared = annotate_adata(
            by_key[spec.key],
            spec=spec,
            raw_path=downloaded[spec.key],
            shared_barcode_count=len(barcodes),
        )
        write_h5ad_atomic(prepared, output_path)
        print(
            f"Wrote {spec.label:16s}: "
            f"{prepared.n_obs:,} cells x {prepared.n_vars:,} genes -> {output_path}"
        )
    print()


def main() -> None:
    args = parse_args()
    prepare_pbmc_10k_introns(
        raw_dir=args.raw_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
