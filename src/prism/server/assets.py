from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from .router import Response

ASSET_DIR = Path(__file__).resolve().parent / "assets"
_ASSET_CONTENT_TYPES = {
    "base.css": "text/css; charset=utf-8",
}


def build_asset_response(asset_name: str) -> Response:
    content_type = _ASSET_CONTENT_TYPES.get(asset_name)
    if content_type is None:
        return Response.text("Not Found", status=HTTPStatus.NOT_FOUND)
    asset_path = ASSET_DIR / asset_name
    if not asset_path.is_file():
        return Response.text("Not Found", status=HTTPStatus.NOT_FOUND)
    return Response(
        status=HTTPStatus.OK,
        content_type=content_type,
        body=asset_path.read_bytes(),
    )


__all__ = ["build_asset_response"]
