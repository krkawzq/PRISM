from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .services.datasets import GeneNotFoundError


@dataclass(slots=True)
class ApiError(Exception):
    status_code: int
    code: str
    message: str
    details: dict[str, Any] | None = None


def ok_response(
    data: Any,
    *,
    meta: dict[str, Any] | None = None,
    status_code: int = HTTPStatus.OK,
) -> JSONResponse:
    return JSONResponse(
        status_code=int(status_code),
        content={
            "ok": True,
            "data": data,
            "error": None,
            "meta": meta or {},
        },
    )


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(
            exc.status_code,
            exc.code,
            exc.message,
            details=exc.details,
        )

    @app.exception_handler(GeneNotFoundError)
    async def handle_gene_not_found(
        _request: Request,
        exc: GeneNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            HTTPStatus.NOT_FOUND,
            "gene_not_found",
            str(exc),
        )

    @app.exception_handler(FileNotFoundError)
    async def handle_file_not_found(
        _request: Request,
        exc: FileNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            HTTPStatus.BAD_REQUEST,
            "file_not_found",
            str(exc),
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(
            HTTPStatus.BAD_REQUEST,
            _classify_value_error(exc),
            str(exc),
        )

    @app.exception_handler(RuntimeError)
    async def handle_runtime_error(
        _request: Request,
        exc: RuntimeError,
    ) -> JSONResponse:
        code = (
            "dataset_not_loaded"
            if "dataset is not loaded" in str(exc)
            else "runtime_error"
        )
        status_code = (
            HTTPStatus.CONFLICT
            if code == "dataset_not_loaded"
            else HTTPStatus.BAD_REQUEST
        )
        return _error_response(status_code, code, str(exc))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "internal_error",
            str(exc) or exc.__class__.__name__,
        )


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=int(status_code),
        content={
            "ok": False,
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            },
            "meta": {},
        },
    )


def _classify_value_error(exc: ValueError) -> str:
    message = str(exc)
    if "kBulk" in message or "kbulk" in message:
        return "invalid_kbulk_request"
    if "checkpoint" in message:
        return "invalid_checkpoint_request"
    if "label" in message:
        return "invalid_label_request"
    return "invalid_request"


__all__ = ["ApiError", "install_exception_handlers", "ok_response"]
