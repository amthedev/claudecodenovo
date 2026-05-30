# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from abc import ABC, abstractmethod
from typing import Dict, Optional

from ...types import CredentialState, SelectionContext, RotationMode


class RoutingStrategy(ABC):
    """
    Base class for credential selection strategies.

    Each strategy decides which credential to use given a set of candidates.
    Hooks (on_success, on_failure) allow strategies to maintain their own
    state across requests without coupling to the broader UsageManager.

    Pattern adopted from LiteLLM's router_strategy design.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def mode(self) -> RotationMode: ...

    @abstractmethod
    def select(
        self,
        context: SelectionContext,
        states: Dict[str, CredentialState],
    ) -> Optional[str]: ...

    def on_success(
        self,
        stable_id: str,
        provider: str,
        model: str,
        latency_ms: float,
    ) -> None:
        """Called after a successful request. Override to track state."""

    def on_failure(
        self,
        stable_id: str,
        provider: str,
        model: str,
        error_type: str,
    ) -> None:
        """Called after a failed request. Override to track state."""

    def pre_call_check(
        self,
        stable_id: str,
        provider: str,
        model: str,
    ) -> bool:
        """Return False to block this credential before the call. Default: allow all."""
        return True
