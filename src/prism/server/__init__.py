from __future__ import annotations

from importlib import import_module

__all__ = ["ServerApp", "ServerConfig", "create_api_app", "run_server"]


def __getattr__(name: str) -> object:
    if name == "ServerConfig":
        module = import_module(".config", __name__)
        return getattr(module, name)
    if name == "create_api_app":
        module = import_module(".api", __name__)
        return getattr(module, name)
    if name in {"ServerApp", "run_server"}:
        module = import_module(".app", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
