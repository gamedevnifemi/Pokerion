"""Visitor sessions via an HttpOnly cookie.

The session id is the scope for everything a visitor owns: their matches,
their replays, their trained strategies, their one-at-a-time training job.
Cookie rather than IP (shared NATs collide) and rather than localStorage
(the server needs it on every request, including the first).
"""

from __future__ import annotations

import os
import re
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

SESSION_COOKIE = "pokerion_sid"
COOKIE_MAX_AGE = 180 * 24 * 3600  # ~6 months

# Secure cookies are opt-IN, not opt-out. A Secure cookie is silently dropped
# by any client on plain HTTP, and because `is_new` is derived from the
# incoming cookie, a dropped cookie means a fresh session id on every single
# request — the visitor loses their match, replays and trained strategy
# continuously. Defaulting this on broke local dev and the test suite the
# moment it was introduced; production sets it explicitly.
_SECURE_COOKIES = os.environ.get("POKERION_SECURE_COOKIES", "").lower() in ("1", "true")

# The cookie value becomes a database primary key, so it is not free-form.
# Without this an attacker can send a multi-kilobyte id per request and write
# arbitrary rows into `sessions` — and can fixate a victim's session by
# choosing its id for them.
_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def new_session_id() -> str:
    return secrets.token_hex(16)


def is_valid_session_id(value: str | None) -> bool:
    return bool(value) and bool(_SESSION_ID_RE.match(value))


def client_key(request_or_ws) -> str:
    """Throttling identity: the real client IP, not the cookie.

    Sessions are free to mint (any request without a cookie gets a new one), so
    per-session limits are not limits at all. Cloudflare sets CF-Connecting-IP;
    the security group only admits Cloudflare, so on the deployed path that
    header is trustworthy. Locally it is absent and we fall back to the peer.
    """
    headers = request_or_ws.headers
    ip = headers.get("cf-connecting-ip") or headers.get("x-forwarded-for", "").split(",")[0].strip()
    if not ip:
        client = getattr(request_or_ws, "client", None)
        ip = client.host if client else "unknown"
    return ip


class SessionMiddleware(BaseHTTPMiddleware):
    """Ensure every HTTP request carries a session id.

    Note: BaseHTTPMiddleware does not run for WebSockets — WS handlers read
    the cookie themselves (see routes/training.py).
    """

    def __init__(self, app, repo_getter):
        super().__init__(app)
        self._repo_getter = repo_getter  # late-bound: repo exists after lifespan startup

    # Endpoints that must not write a session row: the container healthcheck and
    # CI's deploy verification poll these constantly and carry no cookie, so
    # touching the session table here grows it forever for no reason.
    _NO_SESSION_PATHS = frozenset({"/api/version", "/api/games"})

    async def dispatch(self, request: Request, call_next):
        raw = request.cookies.get(SESSION_COOKIE)
        # A malformed or absent cookie gets a fresh id rather than being trusted
        # verbatim — this is what closes session fixation.
        is_new = not is_valid_session_id(raw)
        sid = new_session_id() if is_new else raw
        request.state.session_id = sid
        request.state.client_key = client_key(request)

        repo = self._repo_getter()
        if (
            repo is not None
            and request.url.path.startswith("/api")
            and request.url.path not in self._NO_SESSION_PATHS
        ):
            repo.touch_session(sid)

        response = await call_next(request)
        if is_new:
            response.set_cookie(
                SESSION_COOKIE,
                sid,
                max_age=COOKIE_MAX_AGE,
                httponly=True,
                samesite="lax",
                secure=_SECURE_COOKIES,
            )
        return response
