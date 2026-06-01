# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Per-request context: ID and client ID propagated via contextvars.

Goal: when a customer reports "às vezes parou" we can grep the logs for one
request and see exactly what happened. Today logs are linear and you have to
guess by timestamp. With request_id, `grep req=abc12345 logs/proxy.log` returns
every line for that one request, across the entire pipeline.

Trade-off: contextvars are async-safe (each task gets its own copy), but they
require the FastAPI middleware to SET them, and the logging.Filter to READ them.
Both are below — wire once in main.py and every log line in the process gets
req= and client= fields automatically.
"""
import contextvars
import logging
import uuid
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)
_client_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "client_id", default=None
)


def get_request_id() -> Optional[str]:
    return _request_id.get()


def get_client_id() -> Optional[str]:
    return _client_id.get()


def set_client_id(value: Optional[str]) -> None:
    """Auth dependencies call this after resolving the caller."""
    _client_id.set(value)


class RequestContextFilter(logging.Filter):
    """Stamps every log record with req= and client= fields.

    Defaults to '-' when unset so formatters don't crash on early-startup logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or "-"
        record.client_id = _client_id.get() or "-"
        return True


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Generates request_id on entry (or accepts inbound X-Request-ID) and echoes
    it back to the client in the response header."""

    async def dispatch(self, request: Request, call_next) -> Response:
        inbound = request.headers.get("x-request-id")
        rid = inbound if inbound and len(inbound) <= 64 else uuid.uuid4().hex[:12]
        token_req = _request_id.set(rid)
        token_client = _client_id.set(None)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            _request_id.reset(token_req)
            _client_id.reset(token_client)
