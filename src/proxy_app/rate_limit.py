# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Sliding-window rate limit per API key (in-memory).

Problem this solves: a buggy client looping at full speed could burn through
your upstream keys in minutes. The admin_db has daily_limit, but a daily limit
fires too late — by the time it hits, you already paid the cost.

Approach: sliding window of timestamps per client_id. O(1) check, O(N) memory
where N is requests in window. Trims expired stamps on each check.

Caveat: in-memory means per-worker. With gunicorn -w 4 the effective limit is
4×limit. That's acceptable as a first line — it stops the runaway-loop case
without needing Redis. If you ever need a hard global limit, swap the store.

Config:
- PROXY_RATE_LIMIT_PER_MINUTE  (default 120, set to 0 to disable)
- PROXY_RATE_LIMIT_BURST       (default 20: extra burst above per-minute rate
                                allowed within any 1-second slice)
"""
import asyncio
import logging
import os
import time
from collections import deque
from typing import Deque, Dict, Optional

from fastapi import HTTPException, Request


_LOG = logging.getLogger("proxy.rate_limit")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class _RateLimiter:
    def __init__(self) -> None:
        self._window: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, client_id: str, per_minute: int, burst: int) -> None:
        """Raise HTTPException(429) if over limit. Cheap when under."""
        if per_minute <= 0:
            return
        now = time.monotonic()
        cutoff = now - 60.0
        async with self._lock:
            stamps = self._window.setdefault(client_id, deque())
            # Drop expired stamps from the left (deque is ordered by insertion).
            while stamps and stamps[0] < cutoff:
                stamps.popleft()
            # Per-minute check.
            if len(stamps) >= per_minute:
                retry = max(1, int(60 - (now - stamps[0])))
                _LOG.warning(
                    "rate limit per-minute hit: client=%s used=%d/%d",
                    client_id, len(stamps), per_minute,
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded ({per_minute}/min). Retry in ~{retry}s.",
                    headers={"Retry-After": str(retry)},
                )
            # Burst check (last 1 second).
            recent = sum(1 for s in stamps if s >= now - 1.0)
            if burst > 0 and recent >= burst:
                _LOG.warning(
                    "rate limit burst hit: client=%s recent=%d/%d",
                    client_id, recent, burst,
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"Burst limit exceeded ({burst}/sec). Slow down.",
                    headers={"Retry-After": "1"},
                )
            stamps.append(now)


# Singleton — one limiter per process.
_LIMITER = _RateLimiter()


async def enforce_rate_limit(request: Request, client_id: Optional[str]) -> None:
    """Call from request handlers (or middleware) after auth resolved client_id."""
    per_minute = _env_int("PROXY_RATE_LIMIT_PER_MINUTE", 120)
    burst = _env_int("PROXY_RATE_LIMIT_BURST", 20)
    if per_minute <= 0:
        return
    # Root key is unlimited (admin/operator).
    if client_id in (None, "root"):
        return
    await _LIMITER.check(client_id or "anonymous", per_minute, burst)
