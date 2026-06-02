# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel
"""
Fair usage windows (load shedding) for a single shared GPU.

Problem: a daily token limit alone doesn't stop everyone from hammering the GPU
AT THE SAME TIME, which overloads it. This adds a *window*: when too many
distinct clients are active at once (overload), the clients who have already
used the largest share of their daily limit get their window closed for a while,
so the people who barely used anything keep flowing. It's honest — the user is
told it's a WINDOW limit due to high demand, not that they hit their own quota.

Design mirrors rate_limit.py: in-memory, per-process, O(1)-ish, no deps. Fully
OFF by default (USAGE_WINDOW=off) so nothing changes until you opt in.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Deque, Dict, Optional
from collections import deque

from fastapi import HTTPException

log = logging.getLogger("proxy_app.usage_window")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _enabled() -> bool:
    return os.getenv("USAGE_WINDOW", "off").strip().lower() in {"1", "true", "yes", "on"}


class _UsageWindow:
    def __init__(self) -> None:
        # client_id -> last-seen monotonic timestamp (for "active" counting)
        self._active: Dict[str, float] = {}
        # client_id -> monotonic time until which the window is closed for them
        self._paused_until: Dict[str, float] = {}

    def _active_count(self, now: float, active_window_s: float) -> int:
        # Trim stale, return how many distinct clients are active right now.
        stale = [cid for cid, ts in self._active.items() if now - ts > active_window_s]
        for cid in stale:
            self._active.pop(cid, None)
        return len(self._active)

    def check(self, client_id: str, usage_fraction: float) -> None:
        """Raise HTTP 429 with an honest message if this client's window is
        closed. Closes the window when the server is overloaded AND this client
        has already used a large share of their daily limit. Never raises for
        clients who used little — they keep priority under load."""
        if not _enabled() or not client_id:
            return

        now = time.monotonic()
        active_window_s = _env_float("USAGE_WINDOW_ACTIVE_SECONDS", 60.0)
        overload_clients = _env_int("USAGE_WINDOW_MAX_CLIENTS", 4)
        heavy_fraction = _env_float("USAGE_WINDOW_HEAVY_FRACTION", 0.5)
        pause_minutes = _env_int("USAGE_WINDOW_PAUSE_MINUTES", 60)

        # 1. Still serving an existing pause? Reject with time remaining.
        pu = self._paused_until.get(client_id)
        if pu is not None:
            if now < pu:
                self._reject(pu - now)
            else:
                self._paused_until.pop(client_id, None)

        # 2. Register this client as active and measure concurrency.
        self._active[client_id] = now
        active = self._active_count(now, active_window_s)

        # 3. Not overloaded → let everyone through (window stays open).
        if active <= overload_clients:
            return

        # 4. Overloaded. Pause only the heavy users (≥ heavy_fraction of their
        #    daily limit). Light users keep flowing — that's the fairness rule.
        if usage_fraction >= heavy_fraction:
            until = now + pause_minutes * 60
            self._paused_until[client_id] = until
            log.warning(
                "Usage window closed: client=%s used %.0f%% of daily limit, "
                "%d clients active (>%d) — paused %dmin",
                client_id, usage_fraction * 100, active, overload_clients, pause_minutes,
            )
            self._reject(pause_minutes * 60.0)

    def _reject(self, seconds_left: float) -> None:
        mins = max(1, int(round(seconds_left / 60)))
        raise HTTPException(
            status_code=429,
            detail=(
                f"Limite da janela de uso atingido: o servidor está com muita gente "
                f"usando agora. Como você já usou boa parte do seu limite de hoje, sua "
                f"janela reabre em ~{mins} min. Quem usou menos tem prioridade no horário "
                f"de pico. Tente novamente em seguida."
            ),
        )


window = _UsageWindow()


async def enforce_usage_window(client_id: Optional[str], usage_fraction: float) -> None:
    """Entry point used by the API-key dependencies. Safe no-op when disabled."""
    if not client_id:
        return
    window.check(client_id, usage_fraction)
