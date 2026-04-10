from __future__ import annotations

from pydantic import BaseModel


class LoadContextPayload(BaseModel):
    h5ad_path: str
    ckpt_path: str | None = None
    layer: str | None = None


__all__ = ["LoadContextPayload"]
