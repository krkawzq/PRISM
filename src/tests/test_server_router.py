from __future__ import annotations

from prism.server.router import Request, Response, Router


def test_request_parses_query_and_first_value() -> None:
    request = Request.from_raw_path("get", "/gene?q=GeneA&q=GeneB&mode=fit")
    assert request.method == "GET"
    assert request.path == "/gene"
    assert request.first("q") == "GeneA"
    assert request.first("mode") == "fit"
    assert request.first("missing", "fallback") == "fallback"


def test_response_helpers_encode_payloads() -> None:
    assert Response.html("<b>x</b>").content_type.startswith("text/html")
    assert Response.text("plain").body == b"plain"
    json_response = Response.json({"x": 1})
    assert json_response.content_type.startswith("application/json")
    assert json_response.body == b'{"x":1}'
    redirect = Response.redirect("/next")
    assert redirect.headers["Location"] == "/next"


def test_router_dispatches_exact_and_prefix_routes() -> None:
    router = Router()
    router.add("GET", "/exact", lambda _request: Response.text("exact"))
    router.add_prefix("GET", "/assets/", lambda _request: Response.text("prefix"))

    exact = router.dispatch(Request.from_raw_path("GET", "/exact"))
    prefixed = router.dispatch(Request.from_raw_path("GET", "/assets/base.css"))
    missing = router.dispatch(Request.from_raw_path("GET", "/missing"))

    assert exact.body == b"exact"
    assert prefixed.body == b"prefix"
    assert missing.status == 404
