from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import anndata as ad

from _10xgenomics_common import (
    DatasetSpec,
    annotate_adata,
    default_output_dir,
    default_raw_dir,
    download_dataset,
    read_10x_h5,
    write_h5ad_atomic,
)


SINGLE_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="jejunum_5k_nuclei",
        dataset_name="5k Human Jejunum Nuclei (Chromium Nuclei Isolation Kit)",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/7.0.0/"
            "5k_human_jejunum_CNIK_3pv3/"
            "5k_human_jejunum_CNIK_3pv3_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="tissues",
        tissue="Jejunum",
        material_type="Nuclei",
        approx_cells="~5k",
        platform="NextGEM",
        notes="Only nuclei dataset in this collection",
        output_filename="jejunum_5k_nuclei.h5ad",
        output_subdir="jejunum",
        raw_subdir="jejunum",
    ),
    DatasetSpec(
        key="lung_a549_crispr_5k",
        dataset_name="5k A549 Lung Carcinoma + CRISPR Pool",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.0.0/"
            "SC3_v3_NextGem_DI_CRISPR_A549_5K_SC3_v3_NextGem_DI_CRISPR_A549_5K/"
            "SC3_v3_NextGem_DI_CRISPR_A549_5K_SC3_v3_NextGem_DI_CRISPR_A549_5K_count_sample_feature_bc_matrix.h5"
        ),
        analysis_category="tissues",
        tissue="Lung",
        material_type="Cells",
        approx_cells="~5k",
        platform="NextGEM",
        notes="Cell line perturbation dataset",
        output_filename="a549_crispr_5k.h5ad",
        output_subdir="lung",
        raw_subdir="lung",
    ),
    DatasetSpec(
        key="breast_idc_750_lt",
        dataset_name="750 Sorted Cells from Human Invasive Ductal Carcinoma, LT",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.0.0/"
            "Breast_Cancer_3p_LT/Breast_Cancer_3p_LT_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="tissues",
        tissue="Breast",
        material_type="Cells",
        approx_cells="~750",
        platform="NextGEM",
        notes="FACS sorted, low throughput",
        output_filename="idc_750_lt.h5ad",
        output_subdir="breast",
        raw_subdir="breast",
    ),
    DatasetSpec(
        key="breast_idc_7500",
        dataset_name="7.5k Sorted Cells from Human Invasive Ductal Carcinoma",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.0.0/"
            "Breast_Cancer_3p/Breast_Cancer_3p_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="tissues",
        tissue="Breast",
        material_type="Cells",
        approx_cells="~7.5k",
        platform="NextGEM",
        notes="FACS sorted",
        output_filename="idc_7500.h5ad",
        output_subdir="breast",
        raw_subdir="breast",
    ),
    DatasetSpec(
        key="brain_glioblastoma_200_lt",
        dataset_name="200 Sorted Cells from Human Glioblastoma Multiforme, LT",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.0.0/"
            "Brain_Tumor_3p_LT/Brain_Tumor_3p_LT_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="tissues",
        tissue="Brain",
        material_type="Cells",
        approx_cells="~200",
        platform="NextGEM",
        notes="FACS sorted, very low cell count",
        output_filename="glioblastoma_200_lt.h5ad",
        output_subdir="brain",
        raw_subdir="brain",
    ),
    DatasetSpec(
        key="brain_glioblastoma_2k",
        dataset_name="2k Sorted Cells from Human Glioblastoma Multiforme",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.0.0/"
            "Brain_Tumor_3p/Brain_Tumor_3p_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="tissues",
        tissue="Brain",
        material_type="Cells",
        approx_cells="~2k",
        platform="NextGEM",
        notes="FACS sorted",
        output_filename="glioblastoma_2k.h5ad",
        output_subdir="brain",
        raw_subdir="brain",
    ),
    DatasetSpec(
        key="lymph_node_hodgkins_lymphoma_wta",
        dataset_name="Hodgkin's Lymphoma, Dissociated Tumor: Whole Transcriptome",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/4.0.0/"
            "Parent_NGSC3_DI_HodgkinsLymphoma/"
            "Parent_NGSC3_DI_HodgkinsLymphoma_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="tissues",
        tissue="Lymph Node",
        material_type="Cells",
        approx_cells="~5-10k",
        platform="NextGEM",
        notes="Whole transcriptome assay",
        output_filename="hodgkins_lymphoma_wta.h5ad",
        output_subdir="lymph_node",
        raw_subdir="lymph_node",
    ),
)


@dataclass(frozen=True)
class MultiSampleDataset:
    key: str
    dataset_name: str
    sample_urls: tuple[str, ...]
    tissue: str
    material_type: str
    approx_cells: str
    platform: str
    notes: str
    output_filename: str
    output_subdir: str
    raw_subdir: str


MULTI_SAMPLE_DATASETS: tuple[MultiSampleDataset, ...] = (
    MultiSampleDataset(
        key="lung_nsclc_dtc_20k_intronic",
        dataset_name="20k NSCLC DTCs from 7 donors (with intronic reads)",
        sample_urls=tuple(
            (
                "https://cf.10xgenomics.com/samples/cell-exp/6.1.2/"
                f"20k_NSCLC_DTC_3p_nextgem_intron_donor_{donor}/"
                f"20k_NSCLC_DTC_3p_nextgem_intron_donor_{donor}_count_sample_feature_bc_matrix.h5"
            )
            for donor in range(1, 8)
        ),
        tissue="Lung",
        material_type="Cells",
        approx_cells="~20k",
        platform="NextGEM",
        notes="7 donors, intronic reads included; concatenated from sample-level outputs",
        output_filename="nsclc_dtc_20k_intronic.h5ad",
        output_subdir="lung",
        raw_subdir="lung/nsclc_dtc_20k_intronic",
    ),
    MultiSampleDataset(
        key="lung_nsclc_dtc_40k_ht",
        dataset_name="40k NSCLC DTCs from 7 donors, HT",
        sample_urls=tuple(
            (
                "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
                f"40k_NSCLC_DTC_3p_HT_nextgem_donor_{donor}/"
                f"40k_NSCLC_DTC_3p_HT_nextgem_donor_{donor}_count_sample_feature_bc_matrix.h5"
            )
            for donor in range(1, 8)
        ),
        tissue="Lung",
        material_type="Cells",
        approx_cells="~40k",
        platform="NextGEM",
        notes="7 donors, high-throughput run; concatenated from sample-level outputs",
        output_filename="nsclc_dtc_40k_ht.h5ad",
        output_subdir="lung",
        raw_subdir="lung/nsclc_dtc_40k_ht",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 10x non-blood tissue datasets and write h5ad files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir("tissues"),
        help="Directory where the original 10x h5 downloads are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("tissues"),
        help="Directory where the processed h5ad outputs are written.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the source h5 files even if they already exist.",
    )
    return parser.parse_args()


def build_single_dataset(
    spec: DatasetSpec,
    *,
    raw_root: Path,
    output_root: Path,
    force_download: bool,
) -> None:
    raw_path = download_dataset(spec, raw_root, force_download=force_download)
    print(f"Reading {spec.dataset_name} ...")
    adata = read_10x_h5(raw_path)
    annotate_adata(adata, spec=spec, raw_path=raw_path)
    output_path = output_root / spec.output_subdir / spec.output_filename
    write_h5ad_atomic(adata, output_path)
    print(
        f"Wrote {spec.key:28s}: "
        f"{adata.n_obs:,} cells x {adata.n_vars:,} genes -> {output_path}"
    )


def sample_id_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    stem = name.removesuffix(".h5")
    return stem.removesuffix("_count_sample_feature_bc_matrix")


def build_multi_sample_dataset(
    spec: MultiSampleDataset,
    *,
    raw_root: Path,
    output_root: Path,
    force_download: bool,
) -> None:
    raw_paths: list[Path] = []
    parts: list[ad.AnnData] = []
    sample_ids: list[str] = []

    for url in spec.sample_urls:
        sample_id = sample_id_from_url(url)
        sample_spec = DatasetSpec(
            key=sample_id,
            dataset_name=spec.dataset_name,
            url=url,
            analysis_category="tissues",
            tissue=spec.tissue,
            material_type=spec.material_type,
            approx_cells=spec.approx_cells,
            platform=spec.platform,
            notes=spec.notes,
            output_filename="unused.h5ad",
            raw_subdir=spec.raw_subdir,
        )
        raw_path = download_dataset(sample_spec, raw_root, force_download=force_download)
        raw_paths.append(raw_path)
        sample_ids.append(sample_id)

        print(f"Reading {spec.dataset_name} / {sample_id} ...")
        adata_part = read_10x_h5(raw_path)
        adata_part.obs_names = [f"{sample_id}:{barcode}" for barcode in adata_part.obs_names]
        adata_part.obs["sample_id"] = sample_id
        parts.append(adata_part)

    merged = ad.concat(parts, axis=0, join="outer", merge="same")
    merged.uns["source"] = "10x Genomics"
    merged.uns["dataset_name"] = spec.dataset_name
    merged.uns["dataset_key"] = spec.key
    merged.uns["analysis_category"] = "tissues"
    merged.uns["tissue"] = spec.tissue
    merged.uns["material_type"] = spec.material_type
    merged.uns["approx_cells"] = spec.approx_cells
    merged.uns["platform"] = spec.platform
    merged.uns["notes"] = spec.notes
    merged.uns["download_urls"] = list(spec.sample_urls)
    merged.uns["raw_h5_paths"] = [str(path) for path in raw_paths]
    merged.uns["source_sample_ids"] = sample_ids
    merged.uns["source_sample_count"] = len(sample_ids)
    merged.uns["license"] = "CC BY 4.0"

    output_path = output_root / spec.output_subdir / spec.output_filename
    write_h5ad_atomic(merged, output_path)
    print(
        f"Wrote {spec.key:28s}: "
        f"{merged.n_obs:,} cells x {merged.n_vars:,} genes -> {output_path}"
    )


def main() -> None:
    args = parse_args()
    raw_root = args.raw_dir.resolve()
    output_root = args.output_dir.resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    for index, spec in enumerate(SINGLE_DATASETS):
        build_single_dataset(
            spec,
            raw_root=raw_root,
            output_root=output_root,
            force_download=args.force_download,
        )
        if index != len(SINGLE_DATASETS) - 1 or MULTI_SAMPLE_DATASETS:
            print()

    for index, spec in enumerate(MULTI_SAMPLE_DATASETS):
        build_multi_sample_dataset(
            spec,
            raw_root=raw_root,
            output_root=output_root,
            force_download=args.force_download,
        )
        if index != len(MULTI_SAMPLE_DATASETS) - 1:
            print()


if __name__ == "__main__":
    main()
