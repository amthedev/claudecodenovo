# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import random
import time
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

from ...types import CredentialState, SelectionContext, RotationMode
from .base import RoutingStrategy

lib_logger = logging.getLogger("rotator_library")


class LowestLatencyStrategy(RoutingStrategy):
    """
    Selects the credential with the lowest average latency.

    Maintains a rolling window of recent (timestamp, latency_ms) samples
    per credential. Credentials with fewer than min_samples fall back to
    random selection so new keys get a fair chance to build history.

    Pattern adopted from LiteLLM's LowestLatencyLoggingHandler.
    """

    def __init__(self, window_seconds: int = 60, min_samples: int = 3):
        self._window_seconds = window_seconds
        self._min_samples = min_samples
        self._history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=20)
        )

    @property
    def name(self) -> str:
        return "lowest_latency"

    @property
    def mode(self) -> RotationMode:
        return RotationMode.LOWEST_LATENCY

    def on_success(
        self,
        stable_id: str,
        provider: str,
        model: str,
        latency_ms: float,
    ) -> None:
        self._history[stable_id].append((time.time(), latency_ms))

    def select(
        self,
        context: SelectionContext,
        states: Dict[str, CredentialState],
    ) -> Optional[str]:
        if not context.candidates:
            return None

        if len(context.candidates) == 1:
            return context.candidates[0]

        priorities = context.priorities or {
            c: states[c].priority for c in context.candidates
        }
        min_priority = min(priorities.get(c, 999) for c in context.candidates)
        tier = [c for c in context.candidates if priorities.get(c, 999) == min_priority]

        cutoff = time.time() - self._window_seconds
        scored: List[Tuple[float, str]] = []
        cold: List[str] = []

        for cred_id in tier:
            recent = [lat for ts, lat in self._history.get(cred_id, []) if ts > cutoff]
            if len(recent) >= self._min_samples:
                scored.append((sum(recent) / len(recent), cred_id))
            else:
                cold.append(cred_id)

        if scored:
            scored.sort()
            selected = scored[0][1]
            lib_logger.debug(
                f"LowestLatency: selected {selected} "
                f"(avg {scored[0][0]:.0f}ms from {len(scored)} scored)"
            )
            return selected

        return random.choice(cold) if cold else None
