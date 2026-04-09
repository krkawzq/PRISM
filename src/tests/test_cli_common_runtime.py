from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import prism.cli as cli
from prism.cli.common import (
    create_typer_app,
    resolve_bool,
    resolve_float,
    resolve_int,
    resolve_optional_float,
    resolve_optional_int,
    resolve_optional_path,
    resolve_optional_str,
    resolve_path,
    resolve_str,
    unwrap_typer_value,
)


@dataclass
class _DummyOption:
    default: object


def test_runtime_value_resolution_helpers() -> None:
    value = _DummyOption(default="7")
    assert unwrap_typer_value(value) == "7"
    assert resolve_int(value) == 7
    assert resolve_float(_DummyOption(default="1.5")) == 1.5
    assert resolve_bool(_DummyOption(default=1)) is True
    assert resolve_str(_DummyOption(default="alpha")) == "alpha"
    assert resolve_optional_int(_DummyOption(default=None)) is None
    assert resolve_optional_float(_DummyOption(default=None)) is None
    assert resolve_optional_str(_DummyOption(default=None)) is None


def test_runtime_path_resolution_helpers(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    assert resolve_path(target) == target.resolve()
    assert resolve_optional_path(_DummyOption(default=None)) is None
    assert resolve_optional_path(_DummyOption(default=target)) == target.resolve()


def test_create_typer_app_uses_project_defaults() -> None:
    app = create_typer_app(name="demo", help="demo help")
    assert app.info.name == "demo"
    assert app.info.help == "demo help"
    assert app.info.no_args_is_help is True
    assert app.rich_markup_mode == "rich"


def test_cli_package_exports_all_current_entrypoints() -> None:
    for name in (
        "app",
        "main",
        "checkpoint_app",
        "data_app",
        "extract_app",
        "fit_app",
        "genes_app",
        "plot_app",
        "serve_command",
    ):
        assert hasattr(cli, name)
