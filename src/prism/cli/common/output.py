from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _iter_rows(values: Mapping[str, object] | Iterable[tuple[str, object]]):
    if isinstance(values, Mapping):
        return list(values.items())
    return list(values)


def build_key_value_table(
    *,
    title: str,
    values: Mapping[str, object] | Iterable[tuple[str, object]],
    show_header: bool = True,
    value_overflow: str = "fold",
    box=None,
) -> Table:
    table = Table(title=title, show_header=show_header, box=box)
    table.add_column("Field")
    table.add_column("Value", overflow=value_overflow)
    for key, value in _iter_rows(values):
        table.add_row(str(key), str(value))
    return table


def print_key_value_table(
    console: Console,
    *,
    title: str,
    values: Mapping[str, object] | Iterable[tuple[str, object]],
    show_header: bool = True,
    value_overflow: str = "fold",
    box=None,
) -> None:
    console.print(
        build_key_value_table(
            title=title,
            values=values,
            show_header=show_header,
            value_overflow=value_overflow,
            box=box,
        )
    )


def print_saved_path(console: Console, path: str | Path) -> None:
    console.print(f"[bold green]Saved[/bold green] {Path(path)}")


def print_elapsed(console: Console, elapsed_sec: float) -> None:
    console.print(f"[bold green]Elapsed[/bold green] {elapsed_sec:.2f}s")


__all__ = [
    "build_key_value_table",
    "print_elapsed",
    "print_key_value_table",
    "print_saved_path",
]
