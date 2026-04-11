#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import socket
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


FIGSHARE_ARTICLE_ID = "20029387"
DOWNLOAD_URL_BASE = "https://ndownloader.figshare.com/files"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "replogle"
CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 12


@dataclass(frozen=True)
class RemoteFile:
    file_id: str
    filename: str
    expected_md5: str
    size_bytes: int

    @property
    def url(self) -> str:
        return f"{DOWNLOAD_URL_BASE}/{self.file_id}"


FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        file_id="35773219",
        filename="K562_essential_raw_singlecell_01.h5ad",
        expected_md5="4f1122ce1c7f13299a68df6459a266d3",
        size_bytes=10661879995,
    ),
    RemoteFile(
        file_id="35775507",
        filename="K562_gwps_raw_singlecell_01.h5ad",
        expected_md5="887e3e6a8c8df6eadf7a3030a53c9546",
        size_bytes=65830941948,
    ),
    RemoteFile(
        file_id="35775581",
        filename="rpe1_raw_singlecell_01.h5ad",
        expected_md5="74765fa87635467a869ea972356ae0e7",
        size_bytes=95350546,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download raw single-cell h5ad files for the three Replogle et al. 2022 "
            f"Perturb-seq experiments from Figshare (article {FIGSHARE_ARTICLE_ID})."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where the files will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart downloads even if local files already exist.",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip MD5 verification after download.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retry attempts per file. Default: {DEFAULT_MAX_RETRIES}",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Socket timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[replogle-download] {message}", flush=True)


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["bytes", "KB", "MB", "GB", "TB"]
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "bytes":
        return f"{int(value)} {unit}"
    return f"{value:.2f} {unit}"


def render_progress(
    filename: str,
    downloaded_bytes: int,
    total_bytes: int,
    *,
    started_at: float,
) -> None:
    if total_bytes <= 0:
        return

    ratio = max(0.0, min(1.0, downloaded_bytes / total_bytes))
    width = 30
    filled = int(ratio * width)
    bar = "=" * filled + "." * (width - filled)
    elapsed = max(time.time() - started_at, 1e-6)
    rate = downloaded_bytes / elapsed
    percent = ratio * 100.0
    line = (
        f"\r[replogle-download] {filename} "
        f"[{bar}] {percent:6.2f}% "
        f"{human_size(downloaded_bytes)}/{human_size(total_bytes)} "
        f"{human_size(int(rate))}/s"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def finish_progress() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_md5(path: Path, expected_md5: str) -> bool:
    actual_md5 = file_md5(path)
    if actual_md5 == expected_md5:
        log(f"  MD5 verified: {actual_md5}")
        return True
    log(f"  MD5 MISMATCH: expected={expected_md5} actual={actual_md5}")
    return False


def should_retry(exc: BaseException) -> bool:
    transient_http = {408, 409, 425, 429, 500, 502, 503, 504}
    if isinstance(exc, error.HTTPError):
        return exc.code in transient_http or 300 <= exc.code < 400
    return isinstance(
        exc,
        (
            error.URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            ssl.SSLError,
            OSError,
        ),
    )


def sleep_before_retry(attempt: int) -> None:
    delay = min(60, 2 ** (attempt - 1))
    log(f"  Retrying in {delay}s...")
    time.sleep(delay)


def build_request(url: str, start_byte: int) -> request.Request:
    headers = {"User-Agent": "Mozilla/5.0"}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"
    return request.Request(url, headers=headers)


def stream_download(
    remote_file: RemoteFile,
    destination: Path,
    *,
    timeout_seconds: int,
    max_retries: int,
) -> None:
    attempt = 0

    while True:
        current_size = destination.stat().st_size if destination.exists() else 0
        if current_size == remote_file.size_bytes:
            return
        if current_size > remote_file.size_bytes:
            log(f"  Local file is larger than expected for {remote_file.filename}; restarting.")
            destination.unlink()
            current_size = 0

        attempt += 1
        if attempt > max_retries:
            raise RuntimeError(
                f"Exceeded {max_retries} retries while downloading {remote_file.filename}"
            )

        if current_size > 0:
            log(
                f"Resuming {remote_file.filename} from "
                f"{human_size(current_size)} / {human_size(remote_file.size_bytes)} "
                f"(attempt {attempt}/{max_retries})"
            )
        else:
            log(
                f"Downloading {remote_file.filename} ({human_size(remote_file.size_bytes)}) "
                f"(attempt {attempt}/{max_retries})"
            )

        req = build_request(remote_file.url, current_size)

        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                status = getattr(response, "status", None)
                if current_size > 0 and status == 200:
                    log("  Server ignored resume request; restarting from byte 0.")
                    destination.unlink(missing_ok=True)
                    current_size = 0
                    continue
                if status not in {200, 206}:
                    raise error.HTTPError(
                        remote_file.url,
                        status or 0,
                        f"unexpected status: {status}",
                        hdrs=response.headers,
                        fp=None,
                    )

                destination.parent.mkdir(parents=True, exist_ok=True)
                mode = "ab" if current_size > 0 else "wb"
                started_at = time.time() - (current_size / max(1, CHUNK_SIZE))
                bytes_written = current_size
                last_update = 0.0
                with destination.open(mode) as handle:
                    for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                        handle.write(chunk)
                        bytes_written += len(chunk)
                        now = time.time()
                        if now - last_update >= 0.5 or bytes_written == remote_file.size_bytes:
                            render_progress(
                                remote_file.filename,
                                bytes_written,
                                remote_file.size_bytes,
                                started_at=started_at,
                            )
                            last_update = now
                finish_progress()

            final_size = destination.stat().st_size
            if final_size < remote_file.size_bytes:
                raise IOError(
                    f"incomplete download for {remote_file.filename}: "
                    f"{final_size} < {remote_file.size_bytes}"
                )
            if final_size > remote_file.size_bytes:
                raise IOError(
                    f"download exceeded expected size for {remote_file.filename}: "
                    f"{final_size} > {remote_file.size_bytes}"
                )
            return
        except KeyboardInterrupt:
            finish_progress()
            raise
        except BaseException as exc:  # noqa: BLE001
            finish_progress()
            if not should_retry(exc):
                raise
            log(f"  Download interrupted: {exc}")
            sleep_before_retry(attempt)


def maybe_skip_existing_file(
    remote_file: RemoteFile,
    destination: Path,
    *,
    skip_md5: bool,
) -> bool:
    if not destination.exists() or destination.stat().st_size == 0:
        return False

    file_size = destination.stat().st_size
    if file_size < remote_file.size_bytes:
        return False

    if file_size > remote_file.size_bytes:
        log(f"Existing file too large for {remote_file.filename}; deleting and restarting.")
        destination.unlink()
        return False

    if skip_md5:
        log(f"Skipping existing file without MD5 check: {remote_file.filename}")
        return True

    log(f"Checking existing file: {remote_file.filename}")
    if verify_md5(destination, remote_file.expected_md5):
        return True

    log("  Existing file failed MD5 check. Deleting and re-downloading...")
    destination.unlink()
    return False


def download_one(
    remote_file: RemoteFile,
    output_dir: Path,
    *,
    force: bool,
    skip_md5: bool,
    max_retries: int,
    timeout_seconds: int,
) -> None:
    destination = output_dir / remote_file.filename

    if force and destination.exists():
        log(f"Removing existing file due to --force: {remote_file.filename}")
        destination.unlink()

    if maybe_skip_existing_file(remote_file, destination, skip_md5=skip_md5):
        return

    if destination.exists() and destination.stat().st_size < remote_file.size_bytes:
        log(
            f"Found partial download for {remote_file.filename}: "
            f"{human_size(destination.stat().st_size)} / {human_size(remote_file.size_bytes)}"
        )

    stream_download(
        remote_file,
        destination,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )

    if not skip_md5:
        if not verify_md5(destination, remote_file.expected_md5):
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 verification failed for {remote_file.filename}")

    log(f"  Done: {remote_file.filename}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Downloading {len(FILES)} raw single-cell h5ad files to {output_dir}")
    log("")

    for remote_file in FILES:
        download_one(
            remote_file,
            output_dir,
            force=args.force,
            skip_md5=args.skip_md5,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
        )
        log("")

    log("Finished downloading Replogle et al. 2022 raw single-cell files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
