"""Shared-token gate for the captions page.

v1 is plain HTTP on a public IP, so the cookie is HttpOnly and SameSite=Strict
without Secure. TODO(https): add Secure when Caddy terminates TLS.

Compare with hmac.compare_digest on SHA-256 digests so a short guess does not
raise and the compare itself does not leak the length.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
import urllib.parse
from dataclasses import dataclass


COOKIE = "web_access"
# Thirty days. The browser sends it on later visits without asking again.
COOKIE_MAX_AGE = 30 * 24 * 60 * 60


def tokens_equal(presented: str, expected: str) -> bool:
    left = hashlib.sha256(presented.encode("utf-8")).digest()
    right = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(left, right)


def configured_token() -> str:
    return os.environ.get("WEB_ACCESS_TOKEN", "").strip()


PLACEHOLDER = "replace-with-openssl-rand-hex-32"
MIN_TOKEN_LENGTH = 32


def ensure_token(env: dict) -> str:
    """Return the configured token. Production refuses to boot without one.

    The example file's placeholder and any token shorter than 32 characters
    are rejected even outside production, so copying ``.env.example`` unchanged
    cannot become the live password.
    """

    token = str(env.get("WEB_ACCESS_TOKEN", "")).strip()
    require = str(env.get("WEB_REQUIRE_AUTH", "")).strip().lower() in {"1", "true", "yes", "on"}
    if token and (token == PLACEHOLDER or len(token) < MIN_TOKEN_LENGTH):
        raise SystemExit(
            "WEB_ACCESS_TOKEN must be at least 32 characters and must not be the "
            ".env.example placeholder. Generate one with: openssl rand -hex 32"
        )
    if require and not token:
        raise SystemExit("WEB_ACCESS_TOKEN is required. Generate one with: openssl rand -hex 32")
    return token


@dataclass
class _Failure:
    count: int = 0
    strikes: int = 0
    locked_until: float = 0.0
    updated: float = 0.0


class AuthGuard:
    """Five wrong login tokens lock further guesses from that IP.

    A request that already carries the real token is accepted even while the
    IP is locked, so one shared classroom NAT cannot shut Tim out. Entries
    expire and the map is capped so it cannot grow without bound.
    """

    def __init__(
        self,
        limit: int = 5,
        base_delay: float = 60.0,
        clock=time.monotonic,
        max_entries: int = 10_000,
        idle_ttl: float = 600.0,
    ) -> None:
        self.limit = limit
        self.base_delay = base_delay
        self.clock = clock
        self.max_entries = max_entries
        self.idle_ttl = idle_ttl
        self._failures: dict[str, _Failure] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()

    def check(self, ip: str, presented: str | None, expected: str) -> str:
        """Return ok, missing, bad, or locked.

        ``locked`` is only for a wrong or missing guess while the IP is in
        the login penalty window. The correct token always returns ``ok``.
        """

        now = self.clock()
        with self._lock:
            self._prune(now)
            if presented and tokens_equal(presented, expected):
                self._failures.pop(ip, None)
                return "ok"
            state = self._failures.get(ip)
            if state is not None and state.locked_until > now:
                return "locked"
            if presented is None or presented == "":
                return "missing"
            count = 0 if state is None else state.count
            strikes = 0 if state is None else state.strikes
            count += 1
            if count >= self.limit:
                strikes += 1
                delay = min(self.base_delay * (2 ** (strikes - 1)), 15 * 60)
                self._failures[ip] = _Failure(count=0, strikes=strikes, locked_until=now + delay, updated=now)
                return "locked"
            self._failures[ip] = _Failure(count=count, strikes=strikes, locked_until=0.0, updated=now)
            return "bad"

    def _prune(self, now: float) -> None:
        stale = []
        for ip, state in self._failures.items():
            if state.locked_until and state.locked_until <= now:
                if now - state.locked_until > self.idle_ttl:
                    stale.append(ip)
            elif not state.locked_until and now - state.updated > self.idle_ttl:
                stale.append(ip)
        for ip in stale:
            self._failures.pop(ip, None)
        overflow = len(self._failures) - self.max_entries
        if overflow > 0:
            oldest = sorted(self._failures, key=lambda ip: self._failures[ip].updated)
            for ip in oldest[:overflow]:
                self._failures.pop(ip, None)

    def retry_after(self, ip: str) -> int:
        state = self._failures.get(ip)
        if state is None:
            return 0
        return max(0, int(state.locked_until - self.clock()))


def cookie_value(header: str | None) -> str | None:
    if not header:
        return None
    for part in header.split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name == COOKIE:
            return urllib.parse.unquote(value)
    return None


def presented_token(headers) -> str | None:
    direct = headers.get("X-Access-Token")
    if direct:
        return direct.strip()
    authorization = headers.get("Authorization") or ""
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return cookie_value(headers.get("Cookie"))


def set_cookie_header(token: str) -> str:
    # TODO(https): append "; Secure" when the page is served over HTTPS.
    quoted = urllib.parse.quote(token, safe="")
    return f"{COOKIE}={quoted}; HttpOnly; SameSite=Strict; Path=/; Max-Age={COOKIE_MAX_AGE}"


def clear_cookie_header() -> str:
    return f"{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
