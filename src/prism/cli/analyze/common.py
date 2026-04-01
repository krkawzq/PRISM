"""Shared helpers for analyze commands."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from prism.cli.common import print_elapsed, print_key_value_table, print_saved_path

console = Console()


def print_analysis_plan(title: str = "Analysis Plan", **values: object) -> None:
    print_key_value_table(console, title=title, values=values)


def print_analysis_summary(
    *,
    output_path: Path | None = None,
    elapsed_sec: float | None = None,
    title: str = "Analysis Summary",
    **values: object,
) -> None:
    if values:
        print_key_value_table(console, title=title, values=values)
    if output_path is not None:
        print_saved_path(console, output_path)
    if elapsed_sec is not None:
        print_elapsed(console, elapsed_sec)


def build_summary_table(title: str, rows: list[tuple[str, str]]) -> Table:
    table = Table(title=title)
    table.add_column("Field")
    table.add_column("Value")
    for field, value in rows:
        table.add_row(field, value)
    return table


__all__ = [
    "build_summary_table",
    "console",
    "print_analysis_plan",
    "print_analysis_summary",
]
