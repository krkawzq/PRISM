from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import tarfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

from prism.io.anndata import write_h5ad


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "cell_cycle"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "cell_cycle"
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DEFAULT_DATASETS = [
    "mNeurosphere",
    "mHippNPC",
    "mPancreas",
    "mRetina",
    "mESC",
    "mHSC",
    "HeLa1",
    "HeLa2",
    "hESC",
    "hU2OS",
    "hiPSCs",
]


def log(message: str) -> None:
    print(f"[cell-cycle-prepare] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one or more cell-cycle datasets into AnnData files."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help=f"Dataset names to build. Default: all ({', '.join(DEFAULT_DATASETS)}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where {dataset}.h5ad files will be written.",
    )
    return parser.parse_args()


def dataset_raw_dir(dataset_name: str) -> Path:
    return RAW_ROOT / dataset_name


def output_path(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / f"{dataset_name}.h5ad"


def clean_string(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        text = text[1:-1]
    return text


def unique_index(values: Iterable[object]) -> pd.Index:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for value in values:
        base = clean_string(value) or "NA"
        count = seen.get(base, 0)
        unique.append(base if count == 0 else f"{base}_{count}")
        seen[base] = count + 1
    return pd.Index(unique)


def preferred_var_names(
    var: pd.DataFrame,
    *,
    symbol_columns: Iterable[str] = (
        "gene_symbol",
        "symbol",
        "Gene",
        "AssociatedGeneName",
    ),
    fallback_columns: Iterable[str] = (
        "ensembl_id",
        "Accession",
        "EnsemblGeneID",
        "gene_id",
    ),
) -> pd.Index:
    for column in symbol_columns:
        if column in var.columns:
            values = [clean_string(value) for value in var[column].tolist()]
            if any(values):
                return unique_index(
                    [value if value else f"gene_{idx}" for idx, value in enumerate(values)]
                )
    for column in fallback_columns:
        if column in var.columns:
            return unique_index(
                [
                    clean_string(value) or f"gene_{idx}"
                    for idx, value in enumerate(var[column].tolist())
                ]
            )
    return unique_index([f"gene_{idx}" for idx in range(var.shape[0])])


def finalize_obs(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    out.index = unique_index(out.index.astype(str))
    out.index.name = None
    return out


def finalize_var(var: pd.DataFrame) -> pd.DataFrame:
    out = var.copy()
    out.index = preferred_var_names(out)
    out.index.name = None
    return out


def finalize_adata(
    X: sp.spmatrix | np.ndarray,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    *,
    dataset_name: str,
    species: str,
    matrix_kind: str,
    source_files: Iterable[str],
    notes: str,
    layers: dict[str, sp.spmatrix | np.ndarray] | None = None,
    obsm: dict[str, np.ndarray] | None = None,
    uns_extra: dict[str, object] | None = None,
) -> ad.AnnData:
    adata = ad.AnnData(X=X, obs=finalize_obs(obs), var=finalize_var(var))
    if layers:
        for key, value in layers.items():
            adata.layers[key] = value
    if obsm:
        for key, value in obsm.items():
            adata.obsm[key] = value
    adata.uns["dataset_name"] = dataset_name
    adata.uns["species"] = species
    adata.uns["matrix_kind"] = matrix_kind
    adata.uns["source_files"] = list(source_files)
    adata.uns["notes"] = notes
    if uns_extra:
        adata.uns.update(uns_extra)
    return adata


def read_tsv(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def mmread_path(path: Path) -> sp.csr_matrix:
    matrix = mmread(path)
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    return matrix.tocsr()


def read_gene_by_cell_text_sparse(
    handle: io.TextIOBase,
    *,
    delimiter: str = "\t",
    dtype: np.dtype = np.float32,
) -> tuple[list[str], list[str], sp.csr_matrix]:
    header = handle.readline().rstrip("\n").split(delimiter)
    if len(header) < 2:
        raise RuntimeError("Expression matrix header is malformed.")
    barcodes = [clean_string(value) for value in header[1:]]
    genes: list[str] = []
    data: list[float] = []
    rows: list[int] = []
    cols: list[int] = []
    for gene_idx, line in enumerate(handle):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split(delimiter)
        genes.append(clean_string(parts[0]))
        values = np.fromstring(delimiter.join(parts[1:]), sep=delimiter, dtype=dtype)
        nonzero = np.flatnonzero(values)
        if nonzero.size == 0:
            continue
        rows.extend(nonzero.tolist())
        cols.extend([gene_idx] * int(nonzero.size))
        data.extend(values[nonzero].tolist())
    matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(barcodes), len(genes)),
        dtype=dtype,
    )
    return genes, barcodes, matrix


def read_dense_csv_cells_by_gene_sparse(
    path: Path,
    *,
    index_column: str | None = None,
    chunksize: int = 128,
    dtype: np.dtype = np.float32,
) -> tuple[list[str], list[str], sp.csr_matrix]:
    obs_names: list[str] = []
    blocks: list[sp.csr_matrix] = []
    var_names: list[str] | None = None
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if index_column is None:
            index_column = str(chunk.columns[0])
        if var_names is None:
            var_names = [clean_string(value) for value in chunk.columns[1:]]
        obs_names.extend(chunk[index_column].astype(str).tolist())
        blocks.append(sp.csr_matrix(chunk.iloc[:, 1:].to_numpy(dtype=dtype)))
    if var_names is None:
        raise RuntimeError(f"No data found in {path}")
    return obs_names, var_names, sp.vstack(blocks, format="csr")


def read_simple_gene_table(lines: list[str]) -> pd.DataFrame:
    rows = [line.split("\t") for line in lines if line]
    if not rows:
        raise RuntimeError("Empty gene table.")
    if len(rows[0]) == 2:
        return pd.DataFrame(rows, columns=["ensembl_id", "gene_symbol"])
    if len(rows[0]) == 3:
        return pd.DataFrame(rows, columns=["feature_id", "ensembl_id", "gene_symbol"])
    raise RuntimeError(f"Unexpected gene table width: {len(rows[0])}")


def read_quoted_tsv(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quotechar='"')
        for row in reader:
            rows.append(row)
    return rows


def dataframe_from_rows(rows: list[list[str]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def parse_boolean_series(series: pd.Series) -> pd.Series:
    mapping = {"TRUE": True, "FALSE": False, "True": True, "False": False}
    return series.map(lambda value: mapping.get(str(value), value))


def excel_col_to_idx(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if match is None:
        raise RuntimeError(f"Unexpected Excel cell reference: {cell_ref}")
    label = match.group(1)
    result = 0
    for char in label:
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result


def read_shared_strings(xlsx_path: Path) -> list[str]:
    with zipfile.ZipFile(xlsx_path) as archive:
        text = archive.read("xl/sharedStrings.xml").decode("utf-8", errors="ignore")
    return [clean_string(value) for value in re.findall(r"<t>(.*?)</t>", text)]


def parse_mhsc_sheet1(
    xlsx_path: Path,
) -> tuple[list[str], list[str], list[str], np.ndarray, list[str]]:
    shared = read_shared_strings(xlsx_path)
    gene_symbols: list[str] = []
    transcripts: list[str] = []
    cell_names: list[str] | None = None
    source_cell_ids: list[str] | None = None
    rows: list[np.ndarray] = []
    with zipfile.ZipFile(xlsx_path) as archive:
        with archive.open("xl/worksheets/sheet1.xml") as handle:
            for _, elem in ET.iterparse(handle, events=("end",)):
                if elem.tag != f"{XML_NS}row":
                    continue
                row_num = int(elem.attrib.get("r", "0"))
                values: dict[int, str] = {}
                for cell in elem.findall(f"{XML_NS}c"):
                    ref = cell.attrib.get("r")
                    if ref is None:
                        continue
                    col_idx = excel_col_to_idx(ref)
                    value_elem = cell.find(f"{XML_NS}v")
                    if value_elem is None or value_elem.text is None:
                        continue
                    value = value_elem.text
                    if cell.attrib.get("t") == "s":
                        value = shared[int(value)]
                    values[col_idx] = clean_string(value)
                if row_num == 1:
                    source_cell_ids = [
                        values[col_idx] for col_idx in sorted(values) if col_idx >= 3
                    ]
                elif row_num == 2:
                    cell_names = [
                        values[col_idx] for col_idx in sorted(values) if col_idx >= 3
                    ]
                elif row_num >= 3:
                    if cell_names is None:
                        raise RuntimeError("Cell header row missing in mHSC workbook.")
                    gene_symbols.append(values.get(1, ""))
                    transcripts.append(values.get(2, ""))
                    row = np.zeros(len(cell_names), dtype=np.float32)
                    for col_idx, value in values.items():
                        if col_idx <= 2:
                            continue
                        row[col_idx - 3] = np.float32(float(value))
                    rows.append(row)
                elem.clear()
    if cell_names is None or source_cell_ids is None:
        raise RuntimeError("Failed to parse mHSC workbook headers.")
    return cell_names, source_cell_ids, gene_symbols, np.vstack(rows).T, transcripts


def save_adata(adata: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_h5ad(adata, path)
    log(f"wrote {path} with shape {adata.shape}")


def _to_object_string(series: pd.Series) -> pd.Series:
    out = series.astype("string")
    return out.where(out.notna(), None).astype(object)


def _read_dge_member(
    tar_path: Path, member_name: str
) -> tuple[list[str], list[str], sp.csr_matrix]:
    with tarfile.open(tar_path, "r") as archive:
        member = archive.getmember(member_name)
        stream = archive.extractfile(member)
        if stream is None:
            raise RuntimeError(f"Failed to extract {member_name}")
        with stream, gzip.open(stream, "rt") as handle:
            return read_gene_by_cell_text_sparse(handle, delimiter="\t", dtype=np.float32)


def _align_sparse_columns(
    matrix: sp.csr_matrix, current_names: list[str], target_names: list[str]
) -> sp.csr_matrix:
    name_to_target = {name: idx for idx, name in enumerate(target_names)}
    coo = matrix.tocoo()
    new_cols = np.fromiter(
        (name_to_target[current_names[col_idx]] for col_idx in coo.col),
        dtype=np.int64,
        count=coo.nnz,
    )
    return sp.csr_matrix(
        (coo.data, (coo.row, new_cols)),
        shape=(matrix.shape[0], len(target_names)),
    )


def _infer_hsc_annotations(cell_name: str) -> dict[str, object]:
    match = re.match(r"^(young|old)_(LT_HSC|ST_HSC|MPP)_(\d+)$", cell_name)
    if match is None:
        return {
            "sample": pd.NA,
            "age_group": pd.NA,
            "cell_type": pd.NA,
            "source_cell_number": pd.NA,
        }
    age_group, subtype, number = match.groups()
    return {
        "sample": f"{age_group}_{subtype}",
        "age_group": age_group,
        "cell_type": subtype,
        "source_cell_number": int(number),
    }


def _parse_retina_barcode_table(path: Path, *, nfi: bool) -> pd.DataFrame:
    rows = read_quoted_tsv(path)
    data_rows = rows[1:]
    columns = (
        [
            "barcode",
            "raw_barcode",
            "sample",
            "age",
            "num_genes_expressed",
            "Total_mRNAs",
            "genotype",
            "umap_coord1",
            "umap_coord2",
            "umap_coord3",
            "cell_type",
        ]
        if nfi
        else [
            "barcode",
            "raw_barcode",
            "sample",
            "age",
            "num_genes_expressed",
            "Total_mRNAs",
            "umap_cluster",
            "umap_coord1",
            "umap_coord2",
            "umap_coord3",
            "used_for_pseudotime",
            "cell_type",
        ]
    )
    obs = dataframe_from_rows(data_rows, columns)
    if "used_for_pseudotime" in obs.columns:
        obs["used_for_pseudotime"] = parse_boolean_series(obs["used_for_pseudotime"])
    obs["num_genes_expressed"] = obs["num_genes_expressed"].astype(np.int32)
    obs["Total_mRNAs"] = obs["Total_mRNAs"].astype(np.int32)
    obs["umap_coord1"] = obs["umap_coord1"].astype(np.float32)
    obs["umap_coord2"] = obs["umap_coord2"].astype(np.float32)
    obs["umap_coord3"] = obs["umap_coord3"].astype(np.float32)
    obs.index = obs["barcode"].astype(str)
    return obs


def _parse_retina_gene_table(path: Path, *, nfi: bool) -> pd.DataFrame:
    rows = read_quoted_tsv(path)
    data_rows = rows[1:]
    columns = (
        [
            "ensembl_id",
            "gene_symbol",
            "num_cells_expressed",
            "mean_expr",
            "percent_detection",
            "use_for_ordering",
        ]
        if nfi
        else ["ensembl_id", "gene_id", "gene_symbol", "num_cells_expressed"]
    )
    var = dataframe_from_rows(data_rows, columns)
    var["num_cells_expressed"] = var["num_cells_expressed"].astype(np.int32)
    if nfi:
        var["mean_expr"] = var["mean_expr"].astype(np.float32)
        var["percent_detection"] = var["percent_detection"].astype(np.float32)
        var["use_for_ordering"] = parse_boolean_series(var["use_for_ordering"])
    var.index = var["ensembl_id"].astype(str)
    return var


def _build_hela(
    dataset_name: str, member_names: tuple[str, str], genotype: str | None = None
) -> ad.AnnData:
    raw_dir = dataset_raw_dir(dataset_name)
    tar_path = next(raw_dir.glob("*.tar"))
    exon_genes, exon_barcodes, exon_matrix = _read_dge_member(tar_path, member_names[0])
    intron_genes, intron_barcodes, intron_matrix = _read_dge_member(
        tar_path, member_names[1]
    )
    if exon_barcodes != intron_barcodes:
        raise RuntimeError(f"{dataset_name} exon/intron barcode lists do not match.")
    target_genes = sorted(set(exon_genes) | set(intron_genes))
    exon_matrix = _align_sparse_columns(exon_matrix, exon_genes, target_genes)
    intron_matrix = _align_sparse_columns(intron_matrix, intron_genes, target_genes)
    total_matrix = (exon_matrix + intron_matrix).tocsr()
    obs = pd.DataFrame(
        {"barcode": exon_barcodes, "sample": dataset_name, "species": "human"},
        index=exon_barcodes,
    )
    if genotype is not None:
        obs["genotype"] = genotype
    var = pd.DataFrame({"gene_symbol": target_genes}, index=target_genes)
    return finalize_adata(
        total_matrix,
        obs,
        var,
        dataset_name=dataset_name,
        species="human",
        matrix_kind="counts",
        source_files=[tar_path.name, *member_names],
        notes=(
            "Combined exon and intron DGE tables into X=exon+intron; "
            "the original exon and intron matrices are kept in layers."
        ),
        layers={"exon_counts": exon_matrix, "intron_counts": intron_matrix},
    )


def build_mNeurosphere() -> ad.AnnData:
    raw_dir = dataset_raw_dir("mNeurosphere")
    matrix_path = raw_dir / "GSE171636_mNeuroProcessedCounts.tsv.gz"
    pheno_path = raw_dir / "GSE171636_mNeuroPheno.tsv.gz"
    expr = pd.read_csv(matrix_path, sep="\t")
    gene_ids = expr.iloc[:, 0].astype(str).tolist()
    cell_ids = expr.columns[1:].astype(str).tolist()
    X = expr.iloc[:, 1:].T.to_numpy(dtype=np.float32)
    obs = read_tsv(pheno_path).set_index("cellid").reindex(cell_ids)
    obs.index = cell_ids
    obs["barcode"] = obs.index.astype(str)
    obs["sample"] = obs["sample"].astype("string")
    obs["species"] = "mouse"
    obs["cell_type"] = obs["type"].astype("string")
    obs["total_umis"] = obs["TotalUMIs"].astype(np.float32)
    obs["cell_cycle_phase"] = obs["CCStage"].astype("string")
    obs["cell_cycle_phase_method"] = "SchwabeCC"
    obs["cell_cycle_position"] = obs["tricyclePosition"].astype(np.float32)
    obs["cyclone_phase"] = obs["cyclone"].astype("string")
    obs["seurat_phase"] = obs["SeuratCC"].astype("string")
    obs = obs[
        [
            "barcode",
            "sample",
            "species",
            "cell_type",
            "total_umis",
            "cell_cycle_phase",
            "cell_cycle_phase_method",
            "cell_cycle_position",
            "cyclone_phase",
            "seurat_phase",
        ]
    ]
    var = pd.DataFrame({"ensembl_id": gene_ids}, index=gene_ids)
    return finalize_adata(
        X,
        obs,
        var,
        dataset_name="mNeurosphere",
        species="mouse",
        matrix_kind="processed_expression",
        source_files=[matrix_path.name, pheno_path.name],
        notes=(
            "Author-provided processed expression matrix; values are not raw integer UMI counts."
        ),
    )


def build_mHippNPC() -> ad.AnnData:
    raw_dir = dataset_raw_dir("mHippNPC")
    matrix_path = raw_dir / "GSE190514_Hipp_logcount.mtx.gz"
    row_path = raw_dir / "GSE190514_Hipp_rowData.tsv.gz"
    col_path = raw_dir / "GSE190514_Hipp_colData.tsv.gz"
    matrix = mmread_path(matrix_path)
    var = read_tsv(row_path)
    obs = read_tsv(col_path).set_index("cellid")
    if matrix.shape == (var.shape[0], obs.shape[0]):
        X = matrix.T.tocsr()
    elif matrix.shape == (obs.shape[0], var.shape[0]):
        X = matrix.tocsr()
    else:
        raise RuntimeError(f"Unexpected mHippNPC matrix shape: {matrix.shape}")
    obs["barcode"] = obs.index.astype(str)
    obs["sample"] = obs["sample"].astype("string")
    obs["species"] = "mouse"
    obs["day"] = obs["day"].astype("Int64")
    obs["total_umis"] = obs["TotalUMIs"].astype(np.float32)
    obs["ks_label"] = obs["ks"].astype("Int64")
    obs = obs[["barcode", "sample", "species", "day", "total_umis", "ks_label"]]
    var = var.rename(
        columns={
            "Accession": "ensembl_id",
            "Gene": "gene_symbol",
            "GeneType": "gene_type",
        }
    )
    var.index = var["ensembl_id"].astype(str)
    return finalize_adata(
        X,
        obs,
        var,
        dataset_name="mHippNPC",
        species="mouse",
        matrix_kind="logcounts",
        source_files=[matrix_path.name, row_path.name, col_path.name],
        notes="Author-provided logcount matrix; values are not raw integer UMI counts.",
    )


def build_mPancreas() -> ad.AnnData:
    raw_dir = dataset_raw_dir("mPancreas")
    tar_path = raw_dir / "GSE132188_RAW.tar"
    parts: list[ad.AnnData] = []
    with tarfile.open(tar_path, "r") as outer:
        for member in sorted(
            (m for m in outer.getmembers() if m.isfile()), key=lambda m: m.name
        ):
            match = re.search(r"_(E\d+_\d+)_counts", Path(member.name).name)
            if match is None:
                continue
            sample_label = match.group(1)
            sample_day = sample_label.replace("_", ".")
            blob = outer.extractfile(member)
            if blob is None:
                raise RuntimeError(f"Failed to read {member.name}")
            with tarfile.open(fileobj=io.BytesIO(blob.read()), mode="r:gz") as inner:
                inner_members = inner.getmembers()
                gene_member = next(
                    m for m in inner_members if m.name.endswith("genes.tsv")
                )
                barcode_member = next(
                    m for m in inner_members if m.name.endswith("barcodes.tsv")
                )
                matrix_member = next(
                    m for m in inner_members if m.name.endswith("matrix.mtx")
                )
                genes = read_simple_gene_table(
                    inner.extractfile(gene_member)
                    .read()
                    .decode("utf-8")
                    .splitlines()  # type: ignore[union-attr]
                )
                barcodes = inner.extractfile(barcode_member).read().decode(
                    "utf-8"
                ).splitlines()  # type: ignore[union-attr]
                matrix = sp.csr_matrix(mmread(inner.extractfile(matrix_member))).T  # type: ignore[arg-type]
            obs = pd.DataFrame(
                {
                    "barcode": barcodes,
                    "sample": sample_label,
                    "day": sample_day,
                    "species": "mouse",
                },
                index=[f"{barcode}-{sample_label}" for barcode in barcodes],
            )
            if "ensembl_id" not in genes.columns:
                genes["ensembl_id"] = genes["feature_id"].astype(str)
            var = genes.copy()
            var.index = var["ensembl_id"].astype(str)
            parts.append(ad.AnnData(X=matrix, obs=obs, var=var))
    combined = ad.concat(parts, axis=0, join="outer", merge="first", index_unique=None)
    return finalize_adata(
        combined.X.tocsr(),
        combined.obs,
        combined.var,
        dataset_name="mPancreas",
        species="mouse",
        matrix_kind="counts",
        source_files=[tar_path.name],
        notes="Concatenated raw 10x count matrices from all developmental-stage tarballs in GEO.",
    )


def build_mRetina() -> ad.AnnData:
    raw_dir = dataset_raw_dir("mRetina")
    tenx_matrix = mmread_path(raw_dir / "GSE118614_10x_aggregate.mtx.gz")
    tenx_obs = _parse_retina_barcode_table(raw_dir / "GSE118614_barcodes.tsv.gz", nfi=False)
    tenx_var = _parse_retina_gene_table(raw_dir / "GSE118614_genes.tsv.gz", nfi=False)
    if tenx_matrix.shape != (tenx_obs.shape[0], tenx_var.shape[0]):
        raise RuntimeError(f"Unexpected mRetina 10x shape: {tenx_matrix.shape}")
    tenx_obs["source_subset"] = "10x_aggregate"
    ad_tenx = ad.AnnData(X=tenx_matrix, obs=tenx_obs, var=tenx_var)

    nfi_matrix = mmread_path(raw_dir / "GSE118614_NFI_aggregate.mtx.gz")
    nfi_obs = _parse_retina_barcode_table(raw_dir / "GSE118614_NFI_barcodes.tsv.gz", nfi=True)
    nfi_var = _parse_retina_gene_table(raw_dir / "GSE118614_NFI_genes.tsv.gz", nfi=True)
    if nfi_matrix.shape == (nfi_var.shape[0], nfi_obs.shape[0]):
        nfi_matrix = nfi_matrix.T.tocsr()
    if nfi_matrix.shape != (nfi_obs.shape[0], nfi_var.shape[0]):
        raise RuntimeError(f"Unexpected mRetina NFI shape: {nfi_matrix.shape}")
    nfi_obs["used_for_pseudotime"] = pd.NA
    nfi_obs["source_subset"] = "NFI_aggregate"
    ad_nfi = ad.AnnData(X=nfi_matrix, obs=nfi_obs, var=nfi_var)

    combined = ad.concat([ad_tenx, ad_nfi], axis=0, join="outer", merge="first", index_unique=None)
    obs = combined.obs.copy()
    obs["barcode"] = obs.index.astype(str)
    obs["species"] = "mouse"
    obs["raw_barcode"] = _to_object_string(obs["raw_barcode"])
    obs["sample"] = _to_object_string(obs["sample"])
    obs["age"] = _to_object_string(obs["age"])
    if "genotype" in obs.columns:
        obs["genotype"] = _to_object_string(obs["genotype"])
    if "source_subset" in obs.columns:
        obs["source_subset"] = _to_object_string(obs["source_subset"])
    obs["cell_type"] = _to_object_string(obs["cell_type"])
    obs["used_for_pseudotime"] = obs["used_for_pseudotime"].astype("boolean")
    obs["total_umis"] = obs["Total_mRNAs"].astype(np.float32)
    obs["n_genes_by_counts"] = obs["num_genes_expressed"].astype(np.float32)
    obsm = {
        "X_source_umap": obs[
            ["umap_coord1", "umap_coord2", "umap_coord3"]
        ].to_numpy(dtype=np.float32)
    }
    return finalize_adata(
        combined.X.tocsr(),
        obs,
        combined.var,
        dataset_name="mRetina",
        species="mouse",
        matrix_kind="counts",
        source_files=[
            "GSE118614_10x_aggregate.mtx.gz",
            "GSE118614_barcodes.tsv.gz",
            "GSE118614_genes.tsv.gz",
            "GSE118614_NFI_aggregate.mtx.gz",
            "GSE118614_NFI_barcodes.tsv.gz",
            "GSE118614_NFI_genes.tsv.gz",
        ],
        notes=(
            "Combined the 10x aggregate matrix and the NFI aggregate matrix; "
            "the NFI matrix was transposed from gene-by-cell to cell-by-gene."
        ),
        obsm=obsm,
    )


def build_mESC() -> ad.AnnData:
    raw_dir = dataset_raw_dir("mESC")
    phase_files = {
        "G1": raw_dir / "G1_singlecells_counts.txt",
        "S": raw_dir / "S_singlecells_counts.txt",
        "G2M": raw_dir / "G2M_singlecells_counts.txt",
    }
    obs_parts: list[pd.DataFrame] = []
    matrix_blocks: list[np.ndarray] = []
    var: pd.DataFrame | None = None
    for phase, path in phase_files.items():
        df = pd.read_csv(path, sep="\t")
        metadata = df.iloc[:, :4].copy()
        if var is None:
            var = metadata.rename(
                columns={
                    "EnsemblGeneID": "ensembl_id",
                    "EnsemblTranscriptID": "ensembl_transcript_id",
                    "AssociatedGeneName": "gene_symbol",
                    "GeneLength": "gene_length",
                }
            )
            var.index = var["ensembl_id"].astype(str)
        cell_names = [column.removesuffix("_count") for column in df.columns[4:]]
        matrix_blocks.append(df.iloc[:, 4:].to_numpy(dtype=np.float32).T)
        obs_parts.append(
            pd.DataFrame(
                {
                    "barcode": cell_names,
                    "sample": phase,
                    "species": "mouse",
                    "cell_cycle_phase": phase,
                    "cell_cycle_method": "FACS",
                },
                index=cell_names,
            )
        )
    if var is None:
        raise RuntimeError("mESC files are empty.")
    return finalize_adata(
        np.vstack(matrix_blocks),
        pd.concat(obs_parts, axis=0),
        var,
        dataset_name="mESC",
        species="mouse",
        matrix_kind="counts",
        source_files=[path.name for path in phase_files.values()],
        notes="Counts were taken directly from the phase-specific Fluidigm C1 tables without filtering or normalization.",
    )


def build_mHSC() -> ad.AnnData:
    raw_dir = dataset_raw_dir("mHSC")
    xlsx_path = raw_dir / "GSE59114_C57BL6_GEO_all.xlsx"
    cell_names, source_cell_ids, gene_symbols, matrix, transcripts = parse_mhsc_sheet1(
        xlsx_path
    )
    obs_rows = []
    for cell_name, source_cell_id in zip(cell_names, source_cell_ids, strict=True):
        row = _infer_hsc_annotations(cell_name)
        row["barcode"] = cell_name
        row["source_cell_id"] = source_cell_id
        row["species"] = "mouse"
        obs_rows.append(row)
    obs = pd.DataFrame(obs_rows, index=cell_names)
    obs["sample"] = obs["sample"].astype("string")
    obs["age_group"] = obs["age_group"].astype("string")
    obs["cell_type"] = obs["cell_type"].astype("string")
    obs["source_cell_id"] = obs["source_cell_id"].astype("string")
    obs["source_cell_number"] = obs["source_cell_number"].astype("Int64")
    var = pd.DataFrame(
        {"gene_symbol": gene_symbols, "ucsc_transcripts": transcripts},
        index=[f"row_{idx}" for idx in range(len(gene_symbols))],
    )
    return finalize_adata(
        matrix.astype(np.float32, copy=False),
        obs,
        var,
        dataset_name="mHSC",
        species="mouse",
        matrix_kind="microarray_normalized_expression",
        source_files=[xlsx_path.name],
        notes=(
            "Parsed directly from the GEO workbook XML. Values are normalized expression "
            "intensities, not raw sequencing counts."
        ),
    )


def build_HeLa1() -> ad.AnnData:
    return _build_hela(
        "HeLa1",
        (
            "GSM4224315_out_gene_exon_tagged.dge_exonssf002_WT.txt.gz",
            "GSM4224315_out_gene_exon_tagged.dge_intronssf002_WT.txt.gz",
        ),
        genotype="WT",
    )


def build_HeLa2() -> ad.AnnData:
    return _build_hela(
        "HeLa2",
        (
            "GSM4226257_out_gene_exon_tagged.dge_exonsds_046.txt.gz",
            "GSM4226257_out_gene_exon_tagged.dge_intronsds_046.txt.gz",
        ),
    )


def build_hESC() -> ad.AnnData:
    raw_dir = dataset_raw_dir("hESC")
    matrix_path = raw_dir / "GSE64016_H1andFUCCI_normalized_EC.csv.gz"
    expr = pd.read_csv(matrix_path)
    gene_symbols = expr.iloc[:, 0].astype(str).tolist()
    cell_ids = expr.columns[1:].astype(str).tolist()
    X = expr.iloc[:, 1:].T.to_numpy(dtype=np.float32)
    prefixes = [cell_id.split("_", 1)[0] for cell_id in cell_ids]
    phase_map = {"G1": "G1", "S": "S", "G2": "G2M"}
    obs = pd.DataFrame(
        {
            "barcode": cell_ids,
            "sample": prefixes,
            "species": "human",
            "source_prefix": prefixes,
            "cell_cycle_phase": [phase_map.get(prefix, pd.NA) for prefix in prefixes],
            "cell_cycle_method": [
                "FACS" if prefix in phase_map else pd.NA for prefix in prefixes
            ],
        },
        index=cell_ids,
    )
    var = pd.DataFrame({"gene_symbol": gene_symbols}, index=gene_symbols)
    return finalize_adata(
        X,
        obs,
        var,
        dataset_name="hESC",
        species="human",
        matrix_kind="normalized_expression",
        source_files=[matrix_path.name],
        notes="The only downloaded expression matrix is the author-provided normalized expression table.",
    )


def build_hU2OS() -> ad.AnnData:
    raw_dir = dataset_raw_dir("hU2OS")
    counts_path = raw_dir / "GSE146773_Counts.csv.gz"
    ercc_path = raw_dir / "GSE146773_Counts.csv.ercc.csv.gz"
    fucci_path = raw_dir / "GSE146773_fucci_coords.csv.gz"
    obs_names, gene_ids, X = read_dense_csv_cells_by_gene_sparse(
        counts_path, dtype=np.float32
    )
    ercc_obs_names, ercc_gene_ids, ercc_matrix = read_dense_csv_cells_by_gene_sparse(
        ercc_path, dtype=np.float32
    )
    if obs_names != ercc_obs_names:
        raise RuntimeError("hU2OS ERCC matrix is not aligned to the main counts matrix.")
    obs = pd.DataFrame(index=obs_names)
    obs["barcode"] = obs.index.astype(str)
    obs["sample"] = "hU2OS"
    obs["species"] = "human"
    fucci = pd.read_csv(fucci_path).set_index("cell").reindex(obs.index)
    obs["cell_cycle_phase"] = fucci["phase_by_facs_gating"].astype("string")
    obs["cell_cycle_method"] = np.where(
        fucci["phase_by_facs_gating"].notna(), "FUCCI", pd.NA
    )
    obs["cell_cycle_position"] = fucci["polar_coord_phi"].astype(np.float32)
    obs["fucci_time_hours"] = fucci["fucci_time_hrs"].astype(np.float32)
    obs["fucci_green_raw"] = fucci["raw_green530"].astype(np.float32)
    obs["fucci_red_raw"] = fucci["raw_red585"].astype(np.float32)
    obs["fucci_green_rescaled"] = fucci["green530_lognorm_rescale"].astype(np.float32)
    obs["fucci_red_rescaled"] = fucci["red585_lognorm_rescale"].astype(np.float32)
    var = pd.DataFrame({"ensembl_id": gene_ids}, index=gene_ids)
    obsm = {
        "X_fucci_rescaled": fucci[
            ["green530_lognorm_rescale", "red585_lognorm_rescale"]
        ].to_numpy(dtype=np.float32)
    }
    return finalize_adata(
        X,
        obs,
        var,
        dataset_name="hU2OS",
        species="human",
        matrix_kind="counts_like_expression",
        source_files=[counts_path.name, ercc_path.name, fucci_path.name],
        notes=(
            "The downloaded 'Counts' matrix contains non-integer values for some genes, "
            "so X preserves the source values exactly rather than coercing them to integers."
        ),
        obsm=obsm,
        uns_extra={
            "ercc_gene_ids": ercc_gene_ids,
            "ercc_matrix_shape": list(ercc_matrix.shape),
        },
    )


def build_hiPSCs() -> ad.AnnData:
    raw_dir = dataset_raw_dir("hiPSCs")
    counts_path = raw_dir / "GSE121265_fucci-counts.txt.gz"
    ann_path = raw_dir / "GSE121265_fucci-annotation.txt.gz"
    with gzip.open(counts_path, "rt") as handle:
        gene_ids, cell_ids, X = read_gene_by_cell_text_sparse(
            handle, delimiter="\t", dtype=np.float32
        )
    obs = pd.read_csv(ann_path, sep="\t")
    obs.index = (
        obs["chip_id"].astype(str)
        + "."
        + obs["experiment"].astype(str)
        + "."
        + obs["well"].astype(str)
    )
    obs = obs.reindex(cell_ids)
    obs["barcode"] = obs.index.astype(str)
    obs["sample"] = obs["chip_id"].astype("string")
    obs["species"] = "human"
    obs["experiment"] = obs["experiment"].astype("Int64")
    obs["cell_qc_pass"] = obs["filter_all"].astype("boolean")
    obs["fucci_egfp_reads"] = obs["reads_egfp"].astype(np.float32)
    obs["fucci_mcherry_reads"] = obs["reads_mcherry"].astype(np.float32)
    obs["fucci_egfp_molecules"] = obs["mol_egfp"].astype(np.float32)
    obs["fucci_mcherry_molecules"] = obs["mol_mcherry"].astype(np.float32)
    obs["total_reads"] = obs["raw"].astype(np.float32)
    obs["total_umis"] = obs["umi"].astype(np.float32)
    obs["mapped_reads"] = obs["mapped"].astype(np.float32)
    obs["cell_cycle_method"] = "FUCCI_transgene_counts"
    obs = obs[
        [
            "barcode",
            "sample",
            "species",
            "experiment",
            "well",
            "chip_id",
            "cell_qc_pass",
            "total_reads",
            "total_umis",
            "mapped_reads",
            "fucci_egfp_reads",
            "fucci_mcherry_reads",
            "fucci_egfp_molecules",
            "fucci_mcherry_molecules",
            "cell_cycle_method",
        ]
    ]
    var = pd.DataFrame({"gene_symbol": gene_ids}, index=gene_ids)
    obsm = {
        "X_fucci_transgene_molecules": obs[
            ["fucci_egfp_molecules", "fucci_mcherry_molecules"]
        ].to_numpy(dtype=np.float32)
    }
    return finalize_adata(
        X,
        obs,
        var,
        dataset_name="hiPSCs",
        species="human",
        matrix_kind="counts",
        source_files=[counts_path.name, ann_path.name],
        notes=(
            "The text annotations do not contain the original FUCCI imaging coordinates, "
            "so this build stores the FUCCI transgene read and molecule summaries that "
            "were downloaded alongside the count matrix."
        ),
        obsm=obsm,
    )


BUILDERS = {
    "mNeurosphere": build_mNeurosphere,
    "mHippNPC": build_mHippNPC,
    "mPancreas": build_mPancreas,
    "mRetina": build_mRetina,
    "mESC": build_mESC,
    "mHSC": build_mHSC,
    "HeLa1": build_HeLa1,
    "HeLa2": build_HeLa2,
    "hESC": build_hESC,
    "hU2OS": build_hU2OS,
    "hiPSCs": build_hiPSCs,
}


def main() -> None:
    args = parse_args()
    unknown = [name for name in args.datasets if name not in BUILDERS]
    if unknown:
        raise KeyError(
            f"Unknown dataset(s): {', '.join(unknown)}. Valid names: {', '.join(DEFAULT_DATASETS)}"
        )
    for dataset_name in args.datasets:
        log(f"Preparing {dataset_name}")
        adata = BUILDERS[dataset_name]()
        save_adata(adata, output_path(args.output_dir, dataset_name))


if __name__ == "__main__":
    main()
