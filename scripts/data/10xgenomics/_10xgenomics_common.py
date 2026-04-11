from __future__ import annotations

import os
import subprocess
import shutil
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse


DOWNLOAD_HEADERS = {
    "User-Agent": "Wget/1.21.4",
    "Accept": "*/*",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_name: str
    url: str
    analysis_category: str
    tissue: str
    material_type: str
    approx_cells: str
    platform: str
    notes: str
    output_filename: str
    output_subdir: str = ""
    raw_subdir: str = ""
    raw_filename: str | None = None

    def resolved_raw_filename(self) -> str:
        if self.raw_filename is not None:
            return self.raw_filename
        return Path(urlparse(self.url).path).name


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_output_dir(category: str) -> Path:
    return project_root() / "data" / "10xgenomics" / category


def default_raw_dir(category: str) -> Path:
    return project_root() / "data" / "raw" / "10xgenomics" / category


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
        try:
            request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
            with (
                urllib.request.urlopen(request) as response,
                temp_path.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
        except (HTTPError, URLError):
            if temp_path.exists():
                temp_path.unlink()
            subprocess.run(
                [
                    "curl",
                    "-fSL",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "5",
                    "-o",
                    str(temp_path),
                    url,
                ],
                check=True,
            )
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"Downloaded {destination} ({format_mb(destination)})")
    return destination


def read_10x_h5(path: Path):
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


def annotate_adata(adata, *, spec: DatasetSpec, raw_path: Path, extra_uns=None):
    adata.uns["source"] = "10x Genomics"
    adata.uns["dataset_name"] = spec.dataset_name
    adata.uns["dataset_key"] = spec.key
    adata.uns["analysis_category"] = spec.analysis_category
    adata.uns["tissue"] = spec.tissue
    adata.uns["material_type"] = spec.material_type
    adata.uns["approx_cells"] = spec.approx_cells
    adata.uns["platform"] = spec.platform
    adata.uns["notes"] = spec.notes
    adata.uns["download_url"] = spec.url
    adata.uns["raw_h5_path"] = str(raw_path)
    adata.uns["license"] = "CC BY 4.0"
    if extra_uns:
        for key, value in extra_uns.items():
            adata.uns[key] = value
    return adata


def write_h5ad_atomic(adata, output_path: Path) -> None:
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


def download_dataset(
    spec: DatasetSpec, raw_root: Path, *, force_download: bool
) -> Path:
    return download_file(
        spec.url,
        raw_root / spec.raw_subdir / spec.resolved_raw_filename(),
        force=force_download,
    )


def build_dataset(
    spec: DatasetSpec,
    *,
    raw_root: Path,
    output_root: Path,
    force_download: bool,
    extra_uns=None,
) -> tuple[Path, Path]:
    raw_path = download_dataset(spec, raw_root, force_download=force_download)
    print(f"Reading {spec.dataset_name} ...")
    adata = read_10x_h5(raw_path)
    annotate_adata(adata, spec=spec, raw_path=raw_path, extra_uns=extra_uns)
    output_path = output_root / spec.output_subdir / spec.output_filename
    write_h5ad_atomic(adata, output_path)
    print(
        f"Wrote {spec.key:28s}: "
        f"{adata.n_obs:,} cells x {adata.n_vars:,} genes -> {output_path}"
    )
    return raw_path, output_path


def prepare_simple_category(
    specs: tuple[DatasetSpec, ...] | list[DatasetSpec],
    *,
    raw_root: Path,
    output_root: Path,
    force_download: bool,
) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(specs):
        build_dataset(
            spec,
            raw_root=raw_root,
            output_root=output_root,
            force_download=force_download,
        )
        if index != len(specs) - 1:
            print()


def ensure_relative_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Refusing to replace existing non-symlink: {link_path}")

    relative_target = os.path.relpath(target, link_path.parent)
    link_path.symlink_to(relative_target)
    print(f"Linked {link_path} -> {relative_target}")
