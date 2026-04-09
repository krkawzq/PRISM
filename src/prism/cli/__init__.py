from __future__ import annotations

from importlib import import_module

__all__ = [
    "app",
    "checkpoint_app",
    "data_app",
    "extract_app",
    "fit_app",
    "genes_app",
    "main",
    "plot_app",
]


def __getattr__(name: str) -> object:
    if name == "app" or name == "main":
        module = import_module(".main", __name__)
        return getattr(module, name)
    mapping = {
        "checkpoint_app": ".checkpoint",
        "data_app": ".data",
        "extract_app": ".extract",
        "fit_app": ".fit",
        "genes_app": ".genes",
        "plot_app": ".plot",
    }
    if name not in mapping:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(mapping[name], __name__)
    return getattr(module, name)
