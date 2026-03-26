from __future__ import annotations

import json
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Callable
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    path: str
    raw_path: str
    query: dict[str, list[str]]

    @classmethod
    def from_raw_path(cls, method: str, raw_path: str) -> "Request":
        parsed = urlparse(raw_path)
        return cls(
            method=method.upper(),
            path=parsed.path,
            raw_path=raw_path,
            query=parse_qs(parsed.query, keep_blank_values=False),
        )

    def first(self, key: str, default: str | None = None) -> str | None:
        values = self.query.get(key)
        if not values:
            return default
        return values[0]


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    content_type: str
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def html(cls, body: str, status: int = HTTPStatus.OK) -> "Response":
        return cls(
            status=status,
            content_type="text/html; charset=utf-8",
            body=body.encode("utf-8"),
        )

    @classmethod
    def text(cls, body: str, status: int = HTTPStatus.OK) -> "Response":
        return cls(
            status=status,
            content_type="text/plain; charset=utf-8",
            body=body.encode("utf-8"),
        )

    @classmethod
    def json(cls, payload: object, status: int = HTTPStatus.OK) -> "Response":
        return cls(
            status=status,
            content_type="application/json; charset=utf-8",
            body=json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )

    @classmethod
    def redirect(cls, location: str, status: int = HTTPStatus.FOUND) -> "Response":
        return cls(
            status=status,
            content_type="text/plain; charset=utf-8",
            body=b"",
            headers={"Location": location},
        )


Handler = Callable[[Request], Response]


class Router:
    def __init__(self) -> None:
        self._exact: dict[tuple[str, str], Handler] = {}
        self._prefix: list[tuple[str, str, Handler]] = []

    def add(self, method: str, path: str, handler: Handler) -> None:
        self._exact[(method.upper(), path)] = handler

    def add_prefix(self, method: str, prefix: str, handler: Handler) -> None:
        self._prefix.append((method.upper(), prefix, handler))

    def dispatch(self, request: Request) -> Response:
        exact = self._exact.get((request.method, request.path))
        if exact is not None:
            return exact(request)

        for method, prefix, handler in self._prefix:
            if method == request.method and request.path.startswith(prefix):
                return handler(request)

        return Response.text("Not Found", status=HTTPStatus.NOT_FOUND)
