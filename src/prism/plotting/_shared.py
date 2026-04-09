from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module

PLOT_EXPORTS = frozenset(
    {
        "plot_batch_grid_figure",
        "plot_prior_facet_figure",
        "plot_prior_overlay_figure",
        "plt",
    }
)


def dedupe_names(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = str(value).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        ordered.append(stripped)
    return ordered


def resolve_plot_export(name: str, *, package: str) -> object:
    if name not in PLOT_EXPORTS:
        raise AttributeError
    module = import_module(".prior_plots", package)
    return getattr(module, name)
