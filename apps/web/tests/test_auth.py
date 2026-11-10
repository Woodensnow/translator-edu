"""Token cookie, wrong-token copy, and lockout."""

from __future__ import annotations

import json
import urllib.parse
from http.client import HTTPConnection

from auth import AuthGuard, tokens_equal
from server import GUARD, Handler, ensure_token

from test_proxy import _serve
from http.server import ThreadingHTTPServer


TOKEN = "correct-token"


def _conn(port: int) -> HTTPConnection:
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    return connection


def test_compare_is_exact():
    assert tokens_equal("abc", "abc")
    assert not tokens_equal("abc", "abcd")
    assert not tokens_equal("abc", "abd")


def test_production_refuses_to_start_without_a_token():
    try:
        ensure_token({"WEB_REQUIRE_AUTH": "1", "WEB_ACCESS_TOKEN": ""})
    except SystemExit as exc:
        assert "openssl rand -hex 32" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
    try:
        ensure_token({"WEB_REQUIRE_AUTH": "1", "WEB_ACCESS_TOKEN": "replace-with-openssl-rand-hex-32"})
    except SystemExit as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("placeholder must be refused")
    try:
        ensure_token({"WEB_ACCESS_TOKEN": "short-token"})
    except SystemExit as exc:
        assert "32" in str(exc)
    else:
        raise AssertionError("short token must be refused")
    assert ensure_token({"WEB_ACCESS_TOKEN": "a" * 32}) == "a" * 32


def test_missing_and_wrong_token_are_401_and_right_token_sets_a_cookie(monkeypatch):
    monkeypatch.setenv("WEB_ACCESS_TOKEN", TOKEN)
    GUARD.reset()
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    try:
        bare = _conn(port)
        bare.request("GET", "/config.json")
        missing = bare.getresponse()
        missing.read()
        assert missing.status == 401

        wrong = _conn(port)
        body = urllib.parse.urlencode({"token": "nope"}).encode()
        wrong.request("POST", "/login", body=body, headers={"Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(len(body))})
        denied = wrong.getresponse()
        page = denied.read().decode()
        assert denied.status == 401
        assert "口令不对" in page
        assert "Set-Cookie" not in {key.title(): value for key, value in denied.headers.items()} or "Max-Age=0" in denied.headers.get("Set-Cookie", "")

        ok = _conn(port)
        body = urllib.parse.urlencode({"token": TOKEN}).encode()
        ok.request("POST", "/login", body=body, headers={"Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(len(body))})
        allowed = ok.getresponse()
        allowed.read()
        cookie = allowed.headers.get("Set-Cookie", "")
        assert allowed.status == 303
        assert "HttpOnly" in cookie
        assert "SameSite=Strict" in cookie
        assert "Secure" not in cookie
        assert "web_access=" in cookie

        authed = _conn(port)
        authed.request("GET", "/config.json", headers={"Cookie": cookie.split(";", 1)[0]})
        opened = authed.getresponse()
        payload = json.loads(opened.read().decode())
        assert opened.status == 200
        assert "translate:8080" not in json.dumps(payload)
        assert payload["course"] == "deep-learning"
    finally:
        web.shutdown()
        GUARD.reset()


def test_lockout_after_repeated_wrong_tokens():
    clock = {"now": 0.0}
    guard = AuthGuard(limit=5, base_delay=60, clock=lambda: clock["now"])
    for _ in range(4):
        assert guard.check("10.0.0.8", "wrong", TOKEN) == "bad"
    assert guard.check("10.0.0.8", "wrong", TOKEN) == "locked"
    assert guard.check("10.0.0.8", TOKEN, TOKEN) == "ok"
    for _ in range(4):
        assert guard.check("10.0.0.8", "wrong", TOKEN) == "bad"
    assert guard.check("10.0.0.8", "wrong", TOKEN) == "locked"
    assert guard.retry_after("10.0.0.8") == 60
    clock["now"] = 61
    assert guard.check("10.0.0.8", "wrong", TOKEN) == "bad"
    for _ in range(3):
        guard.check("10.0.0.8", "wrong", TOKEN)
    assert guard.check("10.0.0.8", "wrong", TOKEN) == "locked"
    assert guard.retry_after("10.0.0.8") == 120


def test_http_lockout_returns_429(monkeypatch):
    monkeypatch.setenv("WEB_ACCESS_TOKEN", TOKEN)
    GUARD.reset()
    GUARD.limit = 5
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    try:
        statuses = []
        for _ in range(6):
            connection = _conn(port)
            body = urllib.parse.urlencode({"token": "nope"}).encode()
            connection.request(
                "POST",
                "/login",
                body=body,
                headers={"Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(len(body))},
            )
            response = connection.getresponse()
            response.read()
            statuses.append(response.status)
        assert statuses[:4] == [401, 401, 401, 401]
        assert statuses[4:] == [429, 429]
        still = _conn(port)
        still.request("GET", "/config.json", headers={"Cookie": f"web_access={TOKEN}"})
        opened = still.getresponse()
        opened.read()
        assert opened.status == 200
    finally:
        web.shutdown()
        GUARD.reset()
        GUARD.limit = 5


def test_wrong_api_token_is_rate_limited_and_a_valid_token_is_not(monkeypatch):
    monkeypatch.setenv("WEB_ACCESS_TOKEN", TOKEN)
    GUARD.reset()
    web = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    _serve(web)
    port = web.server_address[1]
    try:
        quiet = []
        for _ in range(8):
            connection = _conn(port)
            connection.request("GET", "/config.json")
            response = connection.getresponse()
            response.read()
            quiet.append(response.status)
        assert quiet == [401] * 8

        statuses = []
        for _ in range(6):
            connection = _conn(port)
            connection.request("GET", "/results", headers={"X-Access-Token": "wrong-token"})
            response = connection.getresponse()
            response.read()
            statuses.append(response.status)
        assert statuses[:4] == [401, 401, 401, 401]
        assert statuses[4:] == [429, 429]

        allowed = _conn(port)
        allowed.request("GET", "/config.json", headers={"X-Access-Token": TOKEN})
        opened = allowed.getresponse()
        opened.read()
        assert opened.status == 200

        again = _conn(port)
        again.request("GET", "/results", headers={"X-Access-Token": "wrong-token"})
        follow = again.getresponse()
        follow.read()
        assert follow.status == 401
    finally:
        web.shutdown()
        GUARD.reset()
