from __future__ import annotations

import pytest

from prism.server.config import ServerConfig


def test_server_config_accepts_defaults() -> None:
    config = ServerConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 8000


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"port": 0}, "port must be in"),
        ({"browse_page_size": 0}, "browse_page_size"),
        ({"top_gene_limit": 0}, "top_gene_limit"),
        ({"inference_batch_size": 0}, "inference_batch_size"),
        ({"posterior_gallery_cells": 0}, "posterior_gallery_cells"),
        ({"kbulk_default_max_classes": 0}, "kbulk_default_max_classes"),
    ],
)
def test_server_config_validates_ranges(
    kwargs: dict[str, int],
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match=pattern):
        ServerConfig(**kwargs)
