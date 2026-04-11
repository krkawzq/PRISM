#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[2]
console = Console()
error_console = Console(stderr=True)


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    rank: int
    gpu_id: str
    shard_output: Path
    log_path: Path
    command: list[str]


@dataclass(frozen=True, slots=True)
class WorkerResult:
    rank: int
    gpu_id: str
    return_code: int
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch sharded `prism fit priors` jobs with `uv`, one shard per worker, "
            "then merge the resulting checkpoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5ad",
        required=True,
        type=Path,
        help="Input h5ad file passed to `prism fit priors`.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Prefix used for the run directory and merged checkpoint path.",
    )
    parser.add_argument(
        "--world-size",
        required=True,
        type=int,
        help="Number of shards to launch.",
    )
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated GPU ids. Ranks are assigned round-robin.",
    )
    parser.add_argument(
        "--reference-genes",
        required=True,
        type=Path,
        help="Reference gene-list text file.",
    )
    parser.add_argument(
        "--fit-genes",
        required=True,
        type=Path,
        help="Gene-list text file describing the genes to fit.",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help="Optional merged checkpoint path. Defaults to <output-prefix>.merged.pkl.",
    )
    parser.add_argument(
        "--allow-partial-merge",
        action="store_true",
        help="Pass --allow-partial to `prism checkpoint merge`.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device forwarded to `prism fit priors`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without launching them.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to `prism fit priors` after `--`.",
    )
    args = parser.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    return args


def resolve_gpu_list(raw_value: str) -> list[str]:
    gpu_list = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not gpu_list:
        raise ValueError("--gpus cannot be empty")
    return gpu_list


def resolve_run_paths(output_prefix: Path, merged_output: Path | None) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = output_prefix.expanduser().resolve()
    run_dir = Path(f"{output_prefix}.dist_{timestamp}")
    merged_path = (
        merged_output.expanduser().resolve()
        if merged_output is not None
        else Path(f"{output_prefix}.merged.pkl").resolve()
    )
    return run_dir, merged_path


def build_worker_specs(args: argparse.Namespace, *, run_dir: Path, gpu_list: list[str]) -> list[WorkerSpec]:
    if args.world_size < 1:
        raise ValueError("--world-size must be >= 1")
    shard_dir = run_dir / "shards"
    log_dir = run_dir / "logs"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = args.h5ad.expanduser().resolve()
    reference_genes = args.reference_genes.expanduser().resolve()
    fit_genes = args.fit_genes.expanduser().resolve()
    extra_args = [str(value) for value in args.extra_args]
    worker_specs: list[WorkerSpec] = []
    for rank in range(args.world_size):
        gpu_id = gpu_list[rank % len(gpu_list)]
        shard_output = shard_dir / f"rank{rank}.pkl"
        log_path = log_dir / f"rank{rank}.log"
        command = [
            "uv",
            "run",
            "prism",
            "fit",
            "priors",
            str(h5ad_path),
            "--output",
            str(shard_output),
            "--reference-genes",
            str(reference_genes),
            "--fit-genes",
            str(fit_genes),
            "--device",
            str(args.device),
            "--shard",
            f"{rank}/{args.world_size}",
            *extra_args,
        ]
        worker_specs.append(
            WorkerSpec(
                rank=rank,
                gpu_id=gpu_id,
                shard_output=shard_output,
                log_path=log_path,
                command=command,
            )
        )
    return worker_specs


def write_log_header(handle: TextIO, spec: WorkerSpec) -> None:
    handle.write(f"[prism-dist] rank={spec.rank} gpu={spec.gpu_id}\n")
    handle.write(f"[prism-dist] cwd={REPO_ROOT}\n")
    handle.write(f"[prism-dist] command={shlex.join(spec.command)}\n\n")
    handle.flush()


def print_intro(
    *,
    run_dir: Path,
    merged_output: Path,
    gpu_list: list[str],
    worker_count: int,
    dry_run: bool,
) -> None:
    table = Table(show_header=False, box=None)
    table.add_row("Repo root", str(REPO_ROOT))
    table.add_row("Run dir", str(run_dir))
    table.add_row("Merged output", str(merged_output))
    table.add_row("Workers", str(worker_count))
    table.add_row("GPUs", ", ".join(gpu_list))
    table.add_row("Mode", "dry-run" if dry_run else "execute")
    console.print(Panel(table, title="PRISM Distributed Fit", border_style="cyan"))


def print_worker_plan(worker_specs: list[WorkerSpec], *, include_commands: bool) -> None:
    table = Table(title="Shard Plan")
    table.add_column("Rank", justify="right")
    table.add_column("GPU", justify="right")
    table.add_column("Shard")
    table.add_column("Checkpoint", overflow="fold")
    table.add_column("Log", overflow="fold")
    if include_commands:
        table.add_column("Command", overflow="fold")
    for spec in worker_specs:
        row = [
            str(spec.rank),
            spec.gpu_id,
            next(
                (
                    spec.command[idx + 1]
                    for idx, token in enumerate(spec.command)
                    if token == "--shard" and idx + 1 < len(spec.command)
                ),
                "",
            ),
            str(spec.shard_output),
            str(spec.log_path),
        ]
        if include_commands:
            row.append(shlex.join(spec.command))
        table.add_row(*row)
    console.print(table)


def print_worker_results(results: list[WorkerResult]) -> None:
    table = Table(title="Worker Status")
    table.add_column("Rank", justify="right")
    table.add_column("GPU", justify="right")
    table.add_column("Status")
    table.add_column("Log", overflow="fold")
    for result in results:
        status = "[green]ok[/green]" if result.return_code == 0 else f"[red]failed ({result.return_code})[/red]"
        table.add_row(
            str(result.rank),
            result.gpu_id,
            status,
            str(result.log_path),
        )
    console.print(table)


def print_merge_plan(command: list[str], *, merged_output: Path) -> None:
    table = Table(show_header=False, box=None)
    table.add_row("Merged output", str(merged_output))
    table.add_row("Command", shlex.join(command))
    console.print(Panel(table, title="Merge Checkpoints", border_style="blue"))


def print_failure_summary(results: list[WorkerResult]) -> None:
    failed = [result for result in results if result.return_code != 0]
    table = Table(title="Shard Failures")
    table.add_column("Rank", justify="right")
    table.add_column("GPU", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("Log", overflow="fold")
    for result in failed:
        table.add_row(
            str(result.rank),
            result.gpu_id,
            str(result.return_code),
            str(result.log_path),
        )
    error_console.print(
        Panel(
            table,
            title="Distributed Fit Failed",
            border_style="red",
        )
    )


def print_success(message: str) -> None:
    console.print(Panel(message, border_style="green"))


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def launch_workers(worker_specs: list[WorkerSpec], *, dry_run: bool) -> tuple[int, list[WorkerResult]]:
    print_worker_plan(worker_specs, include_commands=dry_run)
    if dry_run:
        return 0, []

    processes: list[subprocess.Popen[str]] = []
    log_handles: list[TextIO] = []
    results: list[WorkerResult] = []
    interrupted = False

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        error_console.print(
            f"[bold red][prism-dist][/bold red] received signal {signum}; terminating workers"
        )
        terminate_processes(processes)

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    try:
        for spec in worker_specs:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = spec.gpu_id
            env.setdefault("PYTHONUNBUFFERED", "1")
            log_handle = spec.log_path.open("w", encoding="utf-8")
            write_log_header(log_handle, spec)
            console.print(
                "[cyan][prism-dist][/cyan] "
                f"launch rank={spec.rank} gpu={spec.gpu_id} log={spec.log_path}"
            )
            process = subprocess.Popen(
                spec.command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(process)
            log_handles.append(log_handle)

        for spec, process in zip(worker_specs, processes, strict=True):
            return_code = process.wait()
            results.append(
                WorkerResult(
                    rank=spec.rank,
                    gpu_id=spec.gpu_id,
                    return_code=return_code,
                    log_path=spec.log_path,
                )
            )
            status = "[green]ok[/green]" if return_code == 0 else f"[red]failed ({return_code})[/red]"
            console.print(
                "[cyan][prism-dist][/cyan] "
                f"rank={spec.rank} {status}"
            )
        print_worker_results(results)
        if interrupted:
            return 130, results
        if any(result.return_code != 0 for result in results):
            print_failure_summary(results)
            return 1, results
        return 0, results
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        terminate_processes(processes)
        for handle in log_handles:
            handle.close()


def merge_checkpoints(
    worker_specs: list[WorkerSpec],
    *,
    merged_output: Path,
    allow_partial_merge: bool,
    dry_run: bool,
) -> int:
    command = [
        "uv",
        "run",
        "prism",
        "checkpoint",
        "merge",
        *[str(spec.shard_output) for spec in worker_specs],
        "--output",
        str(merged_output),
    ]
    if allow_partial_merge:
        command.append("--allow-partial")

    print_merge_plan(command, merged_output=merged_output)
    if dry_run:
        return 0
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(result.returncode)


def main() -> int:
    args = parse_args()
    gpu_list = resolve_gpu_list(args.gpus)
    run_dir, merged_output = resolve_run_paths(args.output_prefix, args.merged_output)
    worker_specs = build_worker_specs(args, run_dir=run_dir, gpu_list=gpu_list)

    print_intro(
        run_dir=run_dir,
        merged_output=merged_output,
        gpu_list=gpu_list,
        worker_count=len(worker_specs),
        dry_run=bool(args.dry_run),
    )

    launch_status, _ = launch_workers(worker_specs, dry_run=bool(args.dry_run))
    if launch_status != 0:
        error_console.print(
            "[bold red][prism-dist][/bold red] at least one shard failed; skip merge"
        )
        return launch_status

    merge_status = merge_checkpoints(
        worker_specs,
        merged_output=merged_output,
        allow_partial_merge=bool(args.allow_partial_merge),
        dry_run=bool(args.dry_run),
    )
    if merge_status != 0:
        error_console.print("[bold red][prism-dist][/bold red] merge failed")
        return merge_status

    print_success("[bold green]Distributed fit complete[/bold green]")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        error_console.print(Panel(str(exc), title="run_fit_distributed failed", border_style="red"))
        raise SystemExit(1) from exc
