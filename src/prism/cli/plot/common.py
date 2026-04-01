from __future__ import annotations

from prism.cli.common import option_sequence, option_value
from prism.cli.common.validators import normalize_choice


def resolve_order_mode(value, *, name: str) -> str:
    return normalize_choice(
        str(option_value(value)),
        supported=("input", "alpha", "metric"),
        option_name=name,
    )


__all__ = ["option_sequence", "option_value", "resolve_order_mode"]
