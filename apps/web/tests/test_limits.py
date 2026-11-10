"""Negative Content-Length is rejected, and source files are not served."""

from __future__ import annotations

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from server import Handler, parse_content_length, ContentLengthError

from test_proxy import _serve


def test_parse_content_length_rejects_missing_negative_and_junk():
    for raw in (None, "", "nope", "-1", "-5"):
        try:
            parse_content_length(raw, 100)
        except ContentLengthError as exc:
            assert exc.status == 400
        else:
            raise AssertionError(raw)
    try:
        parse_content_length("101", 100)
    except ContentLengthError as exc:
        assert exc.status == 413
    else:
        raise AssertionError("expected 413")
    assert parse_content_length("0", 100) == 0


def test_negative_content_length_does_not_read_the_body(monkeypatch):
    monkeypatch.delenv("WEB_ACCESS_TOKEN", raising=False)
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    try:
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        connection.putrequest("POST", "/translate")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "-1")
        connection.endheaders()
        response = connection.getresponse()
        body = response.read()
        assert response.status == 400
        assert b"Content-Length" in body

        missing = HTTPConnection("127.0.0.1", port, timeout=3)
        missing.putrequest("POST", "/translate")
        missing.putheader("Content-Type", "application/json")
        missing.endheaders()
        missed = missing.getresponse()
        assert missed.status == 400
        missed.read()

        source = HTTPConnection("127.0.0.1", port, timeout=3)
        source.request("GET", "/server.py")
        hidden = source.getresponse()
        hidden.read()
        assert hidden.status == 404
    finally:
        web.shutdown()
