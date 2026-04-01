from __future__ import annotations

from collections.abc import Sequence

from typer.models import OptionInfo


def option_value(value):
    if isinstance(value, OptionInfo):
        return value.default
    return value


def option_sequence(value) -> list[str] | None:
    resolved = option_value(value)
    if resolved is None:
        return None
    if isinstance(resolved, str):
        return [resolved]
    if isinstance(resolved, Sequence):
        return list(resolved)
    raise ValueError(f"expected a sequence value, got {type(resolved)!r}")


def _has_value(value: object) -> bool:
    resolved = option_value(value)
    if resolved is None:
        return False
    if isinstance(resolved, str):
        return bool(resolved.strip())
    if isinstance(resolved, Sequence) and not isinstance(
        resolved, (str, bytes, bytearray)
    ):
        return len(resolved) > 0
    return True


def ensure_mutually_exclusive(*pairs: tuple[str, object]) -> None:
    present = [name for name, value in pairs if _has_value(value)]
    if len(present) <= 1:
        return
    raise ValueError(f"{' and '.join(present)} are mutually exclusive")


__all__ = ["ensure_mutually_exclusive", "option_sequence", "option_value"]
