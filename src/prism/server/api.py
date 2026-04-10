from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api_responses import ApiError, install_exception_handlers
from .api_routes import build_api_router
from .config import ServerConfig
from .frontend import frontend_dist_dir, serve_frontend_asset
from .state import AppState


def create_api_app(config: ServerConfig, *, state: AppState | None = None) -> FastAPI:
    resolved_state = AppState(config) if state is None else state
    app = FastAPI(
        title="PRISM Server API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.prism_state = resolved_state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_exception_handlers(app)
    _mount_frontend_assets(app)
    app.include_router(build_api_router())

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str = "") -> object:
        if full_path.startswith("api"):
            raise ApiError(404, "not_found", "Not Found")
        return serve_frontend_asset(full_path)

    return app


def _mount_frontend_assets(app: FastAPI) -> None:
    assets_dir = frontend_dist_dir() / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=Path(assets_dir)),
            name="frontend-assets",
        )


__all__ = ["create_api_app"]
