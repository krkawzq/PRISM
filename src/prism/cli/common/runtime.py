from __future__ import annotations

from pathlib import Path
from typing import cast

import typer

try:
    from rich.console import Console
except ModuleNotFoundError:
    class Console:  # type: ignore[no-redef]
        def print(self, *args: object, **kwargs: object) -> None:
            print(*args)

console = Console()


def create_typer_app(*, help: str, name: str | None = None) -> typer.Typer:
    return typer.Typer(
        name=name,
        help=help,
        add_completion=False,
        no_args_is_help=True,
        rich_markup_mode="rich",
    )


def unwrap_typer_value(value: object) -> object:
    return getattr(value, "default", value)


def resolve_path(value: str | Path | object) -> Path:
    resolved = unwrap_typer_value(value)
    return Path(cast(str | Path, resolved)).expanduser().resolve()


def resolve_optional_path(value: Path | None | object) -> Path | None:
    resolved = unwrap_typer_value(value)
    if resolved is None:
        return None
    return Path(cast(str | Path, resolved)).expanduser().resolve()


def resolve_bool(value: bool | object) -> bool:
    return bool(unwrap_typer_value(value))


def resolve_int(value: int | str | object) -> int:
    return int(cast(int | str, unwrap_typer_value(value)))


def resolve_optional_int(value: int | str | None | object) -> int | None:
    resolved = unwrap_typer_value(value)
    if resolved is None:
        return None
    return int(cast(int | str, resolved))


def resolve_float(value: float | int | str | object) -> float:
    return float(cast(float | int | str, unwrap_typer_value(value)))


def resolve_optional_float(value: float | int | str | None | object) -> float | None:
    resolved = unwrap_typer_value(value)
    if resolved is None:
        return None
    return float(cast(float | int | str, resolved))


def resolve_str(value: str | object) -> str:
    return str(unwrap_typer_value(value))


def resolve_optional_str(value: str | None | object) -> str | None:
    resolved = unwrap_typer_value(value)
    if resolved is None:
        return None
    return str(resolved)


__all__ = [
    "console",
    "create_typer_app",
    "resolve_bool",
    "resolve_float",
    "resolve_int",
    "resolve_optional_float",
    "resolve_optional_int",
    "resolve_optional_path",
    "resolve_optional_str",
    "resolve_path",
    "resolve_str",
    "unwrap_typer_value",
]
