"""Plain-text readers and writers for gene/string lists."""

from pathlib import Path


def _normalize(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for v in values:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def read_list(path: str | Path) -> list[str]:
    """Read a newline-separated list from a file, stripping whitespace, blank lines, and duplicates."""
    return _normalize(Path(path).read_text(encoding="utf-8").splitlines())


def write_list(path: str | Path, items: list[str]) -> None:
    """Write a list to a file as newline-separated values."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(_normalize(items)) + "\n", encoding="utf-8")


def read_gene_list(path: str | Path) -> list[str]:
    return read_list(path)


def write_gene_list(path: str | Path, items: list[str]) -> None:
    write_list(path, items)


def read_string_list(path: str | Path) -> list[str]:
    return read_list(path)


def write_string_list(path: str | Path, items: list[str]) -> None:
    write_list(path, items)


__all__ = [
    "read_gene_list",
    "read_list",
    "read_string_list",
    "write_gene_list",
    "write_list",
    "write_string_list",
]
