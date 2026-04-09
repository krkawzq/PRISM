from __future__ import annotations

import inspect

from prism.cli.serve.app import serve_command
from prism.server import ServerConfig


def test_serve_command_signature_only_exposes_bind_options() -> None:
    parameters = tuple(inspect.signature(serve_command).parameters)
    assert parameters == ("host", "port")


def test_serve_command_builds_minimal_server_config(
    monkeypatch,
) -> None:
    captured: list[ServerConfig] = []

    def fake_run_server(config: ServerConfig) -> None:
        captured.append(config)

    monkeypatch.setattr("prism.server.run_server", fake_run_server)

    serve_command(host="0.0.0.0", port=9000)

    assert captured == [ServerConfig(host="0.0.0.0", port=9000)]
