#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html.parser
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


USER_AGENT = "Mozilla/5.0"
ENA_FIELDS = (
    "study_accession,run_accession,experiment_accession,sample_accession,"
    "sample_title,experiment_title,library_layout,library_strategy,"
    "instrument_platform,fastq_ftp,fastq_md5,submitted_ftp,submitted_md5,sra_ftp"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "cell_cycle"
PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(message: str) -> None:
    print(f"[cell-cycle-download] {message}", flush=True)


def build_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def proxyless_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in PROXY_ENV_VARS:
        env.pop(key, None)
    return env


def fetch_text_via_curl(url: str, *, timeout: int) -> str:
    completed = subprocess.run(
        [
            "curl",
            "-fL",
            "--noproxy",
            "*",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--max-time",
            str(timeout),
            "-H",
            f"User-Agent: {USER_AGENT}",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=proxyless_env(),
    )
    return completed.stdout


def fetch_text(url: str, *, max_retries: int = 4, timeout: int = 120) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with DIRECT_OPENER.open(build_request(url), timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            try:
                log(f"urllib failed for {url}; trying curl fallback")
                return fetch_text_via_curl(url, timeout=timeout)
            except Exception as curl_exc:  # noqa: BLE001
                last_error = curl_exc
                if attempt == max_retries:
                    break
            delay = min(30, 2 ** (attempt - 1))
            log(f"Retrying {url} in {delay}s after: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def fetch_json(url: str, *, max_retries: int = 4, timeout: int = 120):
    return json.loads(fetch_text(url, max_retries=max_retries, timeout=timeout))


class DirectoryListingParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


@dataclass(frozen=True)
class DownloadItem:
    url: str
    filename: str
    source: str
    md5: str | None = None
    category: str = "processed"
    relative_destination: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    species: str
    cell_type: str
    geo_accession: str | None = None
    biostudies_accession: str | None = None
    ena_filter: Callable[[dict[str, str]], bool] | None = None
    supplementary_filter: Callable[[DownloadItem], bool] | None = None
    notes: str = ""


def geo_series_prefix(accession: str) -> str:
    match = re.fullmatch(r"(GSE)(\d+)", accession)
    if not match:
        raise ValueError(f"Unsupported GEO accession: {accession}")
    digits = match.group(2)
    if len(digits) <= 3:
        return f"{match.group(1)}nnn"
    return f"{match.group(1)}{digits[:-3]}nnn"


def geo_quick_text(accession: str) -> str:
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&targ=self&form=text&view=quick"
    return fetch_text(url)


def geo_sra_study(accession: str) -> str | None:
    text = geo_quick_text(accession)
    match = re.search(
        r"!Series_relation = SRA: https://www\.ncbi\.nlm\.nih\.gov/sra\?term=(SRP\d+)",
        text,
    )
    if match:
        return match.group(1)
    return None


def list_geo_supplementary(accession: str) -> list[DownloadItem]:
    prefix = geo_series_prefix(accession)
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/suppl/"
    try:
        html = fetch_text(base)
    except RuntimeError as exc:
        if "404" in str(exc):
            return []
        raise

    parser = DirectoryListingParser()
    parser.feed(html)

    items: list[DownloadItem] = []
    for href in parser.links:
        parsed_href = urllib.parse.urlparse(href)
        if (
            href.endswith("/")
            or href.startswith("/")
            or parsed_href.scheme in {"http", "https"}
            or href == "Parent Directory"
        ):
            continue
        url = urllib.parse.urljoin(base, href)
        items.append(
            DownloadItem(
                url=url,
                filename=Path(href).name,
                source=f"GEO supplementary ({accession})",
            )
        )
    return items


def list_biostudies_files(accession: str) -> list[DownloadItem]:
    url = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}"
    raw_json = fetch_text(url)
    paths = sorted(set(re.findall(r'"path"\s*:\s*"([^"]+)"', raw_json)))
    return [
        DownloadItem(
            url=f"https://www.ebi.ac.uk/biostudies/files/{accession}/{path}",
            filename=Path(path).name,
            source=f"BioStudies ({accession})",
        )
        for path in paths
    ]


def normalize_remote_url(url: str) -> str:
    if not url:
        return url
    if "://" not in url:
        return f"https://{url}"
    if url.startswith("ftp://"):
        return "https://" + url.removeprefix("ftp://")
    return url


def list_ena_raw_files(
    study_accession: str, row_filter: Callable[[dict[str, str]], bool] | None
) -> tuple[list[DownloadItem], list[dict[str, str]]]:
    url = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport"
        f"?accession={study_accession}&result=read_run&fields={ENA_FIELDS}"
        "&format=tsv&download=true"
    )
    raw_tsv = fetch_text(url)
    rows = list(csv.DictReader(raw_tsv.splitlines(), delimiter="\t"))
    if row_filter is not None:
        rows = [row for row in rows if row_filter(row)]

    items: list[DownloadItem] = []
    for row in rows:
        for field_name, md5_field in (
            ("fastq_ftp", "fastq_md5"),
            ("submitted_ftp", "submitted_md5"),
        ):
            paths = [path for path in row.get(field_name, "").split(";") if path]
            md5s = [md5 for md5 in row.get(md5_field, "").split(";") if md5]
            for index, path in enumerate(paths):
                filename = Path(path).name
                md5 = md5s[index] if index < len(md5s) else None
                items.append(
                    DownloadItem(
                        url=normalize_remote_url(path),
                        filename=filename,
                        source=f"ENA {study_accession}:{row['run_accession']}",
                        md5=md5,
                        category="raw_fastq",
                    )
                )
            if paths:
                break
    return items, rows


def sample_title_contains(text: str) -> Callable[[dict[str, str]], bool]:
    needle = text.lower()

    def predicate(row: dict[str, str]) -> bool:
        return needle in row.get("sample_title", "").lower()

    return predicate


def sample_title_not_contains(text: str) -> Callable[[dict[str, str]], bool]:
    needle = text.lower()

    def predicate(row: dict[str, str]) -> bool:
        return needle not in row.get("sample_title", "").lower()

    return predicate


def sample_title_startswith(
    prefixes: tuple[str, ...],
) -> Callable[[dict[str, str]], bool]:
    lowered = tuple(prefix.lower() for prefix in prefixes)

    def predicate(row: dict[str, str]) -> bool:
        return row.get("sample_title", "").lower().startswith(lowered)

    return predicate


def library_layout_is(layout: str) -> Callable[[dict[str, str]], bool]:
    wanted = layout.upper()

    def predicate(row: dict[str, str]) -> bool:
        return row.get("library_layout", "").upper() == wanted

    return predicate


def all_of(
    *predicates: Callable[[dict[str, str]], bool],
) -> Callable[[dict[str, str]], bool]:
    def predicate(row: dict[str, str]) -> bool:
        return all(func(row) for func in predicates)

    return predicate


def regex_sample_title(pattern: str) -> Callable[[dict[str, str]], bool]:
    compiled = re.compile(pattern, re.IGNORECASE)

    def predicate(row: dict[str, str]) -> bool:
        return bool(compiled.search(row.get("sample_title", "")))

    return predicate


def filename_contains(text: str) -> Callable[[DownloadItem], bool]:
    needle = text.lower()

    def predicate(item: DownloadItem) -> bool:
        return needle in item.filename.lower()

    return predicate


def filename_not_contains(text: str) -> Callable[[DownloadItem], bool]:
    needle = text.lower()

    def predicate(item: DownloadItem) -> bool:
        return needle not in item.filename.lower()

    return predicate


def filename_not_in(names: tuple[str, ...]) -> Callable[[DownloadItem], bool]:
    blocked = {name.lower() for name in names}

    def predicate(item: DownloadItem) -> bool:
        return item.filename.lower() not in blocked

    return predicate


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="mNeurosphere",
        species="mouse",
        cell_type="E14.5 cortical neurosphere",
        geo_accession="GSE171636",
    ),
    DatasetSpec(
        name="mHippNPC",
        species="mouse",
        cell_type="primary hippocampal NPC",
        geo_accession="GSE190514",
        ena_filter=sample_title_startswith(("day0sample2", "day0sample4")),
        notes="Restricted to day 0 samples, matching the dataset used in the tricycle paper.",
    ),
    DatasetSpec(
        name="mPancreas",
        species="mouse",
        cell_type="developing pancreas",
        geo_accession="GSE132188",
    ),
    DatasetSpec(
        name="mHSC",
        species="mouse",
        cell_type="hematopoietic stem cells",
        geo_accession="GSE59114",
        ena_filter=sample_title_contains("C57BL6"),
        supplementary_filter=filename_contains("C57BL6"),
        notes="Restricted to C57BL/6 samples, matching the subset described in the paper.",
    ),
    DatasetSpec(
        name="mRetina",
        species="mouse",
        cell_type="retina",
        geo_accession="GSE118614",
        ena_filter=sample_title_not_contains("Cell "),
        supplementary_filter=filename_not_contains("Smart"),
        notes="Restricted to 10x-style runs and non-Smart-seq supplementary files.",
    ),
    DatasetSpec(
        name="mESC",
        species="mouse",
        cell_type="embryonic stem cells",
        biostudies_accession="E-MTAB-2805",
        notes="BioStudies hosts processed count tables plus IDF/SDRF files for this dataset.",
    ),
    DatasetSpec(
        name="HeLa1",
        species="human",
        cell_type="HeLa set 1",
        geo_accession="GSE142277",
        ena_filter=regex_sample_title(r"wild[- ]?type"),
        notes="Restricted to the wild-type sample; AGO2KO is excluded.",
    ),
    DatasetSpec(
        name="HeLa2",
        species="human",
        cell_type="HeLa set 2",
        geo_accession="GSE142356",
        ena_filter=regex_sample_title(r"wild[- ]?type"),
    ),
    DatasetSpec(
        name="hESC",
        species="human",
        cell_type="embryonic stem cells",
        geo_accession="GSE64016",
        ena_filter=regex_sample_title(r"fucci"),
        notes="Restricted to the H1-FUCCI / FACS-sorted subset used in the tricycle paper.",
    ),
    DatasetSpec(
        name="hU2OS",
        species="human",
        cell_type="U-2 OS cells",
        geo_accession="GSE146773",
        supplementary_filter=filename_not_in(("GSE146773_RAW.tar",)),
        notes="Keep processed count/TPM/FUCCI files; exclude the large per-cell RAW.tar bundle by default.",
    ),
    DatasetSpec(
        name="hiPSCs",
        species="human",
        cell_type="induced pluripotent stem cells",
        geo_accession="GSE121265",
    ),
)


DATASET_LOOKUP = {spec.name: spec for spec in DATASETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download processed cell-cycle datasets into data/raw/cell_cycle/{name}/. "
            "By default ENA/SRA raw FASTQ files are skipped; only GEO supplementary files "
            "or BioStudies-hosted processed outputs are downloaded."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[spec.name for spec in DATASETS],
        help="Dataset names to download. Default: all known datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root output directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the files that would be downloaded.",
    )
    parser.add_argument(
        "--include-raw",
        dest="include_raw",
        action="store_true",
        help="Include ENA/SRA raw FASTQ files. Disabled by default.",
    )
    parser.add_argument(
        "--skip-raw",
        dest="include_raw",
        action="store_false",
        help="Deprecated alias. Raw FASTQ files are skipped by default.",
    )
    parser.add_argument(
        "--skip-supplementary",
        action="store_true",
        help="Skip GEO supplementary or BioStudies-hosted files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported dataset names and exit.",
    )
    parser.set_defaults(include_raw=False)
    return parser.parse_args()


def ensure_supported_datasets(names: list[str]) -> list[DatasetSpec]:
    missing = [name for name in names if name not in DATASET_LOOKUP]
    if missing:
        raise SystemExit(f"Unknown dataset(s): {', '.join(sorted(missing))}")
    return [DATASET_LOOKUP[name] for name in names]


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classify_download_item(item: DownloadItem) -> DownloadItem:
    filename = item.filename.lower()

    if filename == "filelist.txt":
        return DownloadItem(
            url=item.url,
            filename=item.filename,
            source=item.source,
            md5=item.md5,
            category="metadata_listing",
            relative_destination="_metadata",
        )

    if filename.endswith((".fastq.gz", ".fq.gz")):
        category = "raw_fastq"
    elif filename.endswith((".tar", ".tar.gz")):
        category = "processed_archive"
    elif filename.endswith((".mtx", ".mtx.gz")):
        category = "processed_matrix"
    elif filename.endswith(("barcodes.tsv.gz", "genes.tsv.gz", "features.tsv.gz")):
        category = "processed_index"
    elif any(
        token in filename
        for token in ("annotation", "pheno", "coldata", "rowdata", "coords", "idf", "sdrf")
    ):
        category = "processed_annotation"
    elif any(token in filename for token in ("counts", "logcount", "tpm", "aggregate")):
        category = "processed_matrix"
    elif filename.endswith((".rds", ".rds.gz", ".h5ad", ".h5ad.h5")):
        category = "processed_object"
    elif filename.endswith((".xlsx", ".csv", ".csv.gz", ".txt", ".txt.gz", ".tsv", ".tsv.gz")):
        category = "processed_table"
    else:
        category = item.category

    return DownloadItem(
        url=item.url,
        filename=item.filename,
        source=item.source,
        md5=item.md5,
        category=category,
        relative_destination=item.relative_destination,
    )


def download_file(item: DownloadItem, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        log(f"Using existing file: {destination}")
        return

    if force and destination.exists():
        destination.unlink()

    log(f"Downloading {item.url} -> {destination}")
    subprocess.run(
        [
            "curl",
            "-fL",
            "--noproxy",
            "*",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            item.url,
        ],
        check=True,
        env=proxyless_env(),
    )


def download_dataset(spec: DatasetSpec, args: argparse.Namespace) -> None:
    dataset_dir = args.output_root / spec.name
    metadata_dir = dataset_dir / "_metadata"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    all_items: list[DownloadItem] = []
    source_notes: dict[str, object] = {
        "name": spec.name,
        "species": spec.species,
        "cell_type": spec.cell_type,
        "geo_accession": spec.geo_accession,
        "biostudies_accession": spec.biostudies_accession,
        "notes": spec.notes,
    }

    if spec.geo_accession is not None:
        quick_text = geo_quick_text(spec.geo_accession)
        write_text(metadata_dir / f"{spec.geo_accession}_quick.txt", quick_text)

        if args.include_raw:
            study_accession = geo_sra_study(spec.geo_accession)
            if study_accession is None:
                log(f"No SRA study found for {spec.name} ({spec.geo_accession})")
            else:
                raw_items, rows = list_ena_raw_files(study_accession, spec.ena_filter)
                all_items.extend(raw_items)
                source_notes["sra_study"] = study_accession
                write_json(metadata_dir / f"{study_accession}_ena_runs.json", rows)

        if not args.skip_supplementary:
            supp_items = list_geo_supplementary(spec.geo_accession)
            if spec.supplementary_filter is not None:
                supp_items = [
                    item for item in supp_items if spec.supplementary_filter(item)
                ]
            all_items.extend(supp_items)

    if spec.biostudies_accession is not None and not args.skip_supplementary:
        biostudies_items = list_biostudies_files(spec.biostudies_accession)
        all_items.extend(biostudies_items)

    deduped: dict[str, DownloadItem] = {}
    for item in all_items:
        classified = classify_download_item(item)
        deduped[classified.filename] = classified
    resolved_items = [deduped[name] for name in sorted(deduped)]

    manifest = [
        {
            "category": item.category,
            "filename": item.filename,
            "url": item.url,
            "source": item.source,
            "md5": item.md5,
            "relative_destination": item.relative_destination or ".",
        }
        for item in resolved_items
    ]
    write_json(metadata_dir / "download_manifest.json", manifest)
    write_json(metadata_dir / "dataset_info.json", source_notes)

    log(f"{spec.name}: resolved {len(resolved_items)} files")
    if args.dry_run:
        for item in resolved_items:
            destination = item.relative_destination or "."
            print(
                f"{spec.name}\t{item.filename}\t{item.category}\t{destination}\t"
                f"{item.source}\t{item.url}"
            )
        return

    for item in resolved_items:
        relative_dir = Path(item.relative_destination) if item.relative_destination else Path()
        download_file(item, dataset_dir / relative_dir / item.filename, force=args.force)


def main() -> None:
    args = parse_args()
    if args.list:
        for spec in DATASETS:
            print(spec.name)
        return

    specs = ensure_supported_datasets(args.datasets)
    for index, spec in enumerate(specs):
        log(f"Preparing {spec.name}")
        download_dataset(spec, args)
        if index != len(specs) - 1:
            print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
