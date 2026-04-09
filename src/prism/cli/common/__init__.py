from __future__ import annotations

from importlib import import_module

__all__ = [
    "build_key_value_table",
    "console",
    "create_typer_app",
    "normalize_choice",
    "print_elapsed",
    "print_key_value_table",
    "print_saved_path",
    "resolve_bool",
    "resolve_float",
    "resolve_int",
    "resolve_numpy_dtype",
    "resolve_optional_float",
    "resolve_optional_int",
    "resolve_optional_path",
    "resolve_optional_str",
    "resolve_path",
    "resolve_prior_source",
    "resolve_str",
    "unwrap_typer_value",
]


def __getattr__(name: str) -> object:
    runtime_names = {
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
    }
    output_names = {
        "build_key_value_table",
        "print_elapsed",
        "print_key_value_table",
        "print_saved_path",
    }
    validator_names = {"normalize_choice", "resolve_numpy_dtype", "resolve_prior_source"}
    if name in runtime_names:
        module = import_module(".runtime", __name__)
        return getattr(module, name)
    if name in output_names:
        module = import_module(".output", __name__)
        return getattr(module, name)
    if name in validator_names:
        module = import_module(".validators", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
