# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Deployment config loader for proxy_config.json model_list.

Allows declaring credentials explicitly in proxy_config.json instead of
relying solely on env var naming conventions. Credentials defined in
model_list are injected into os.environ so the existing CredentialManager
and api_keys discovery logic picks them up automatically.

Example proxy_config.json:
    {
        "env": { "PROXY_API_KEY": "my-key" },
        "model_list": [
            {
                "model_name": "gpt-4o",
                "provider": "openai",
                "api_key": "sk-xxx",
                "rpm": 1000,
                "priority": 1
            },
            {
                "model_name": "gpt-4o",
                "provider": "openai",
                "api_key": "sk-yyy",
                "rpm": 500,
                "priority": 2
            },
            {
                "model_name": "gemini-2.5-pro",
                "provider": "gemini",
                "api_key_env": "MY_GEMINI_KEY"
            }
        ]
    }

Pattern adopted from LiteLLM's model_list / Deployment concept.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")

# Maps provider name → env var prefix used for API keys
_PROVIDER_ENV_PREFIX: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cohere": "COHERE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "nanogpt": "NANOGPT_API_KEY",
    "chutes": "CHUTES_API_KEY",
}


@dataclass
class DeploymentEntry:
    model_name: str
    provider: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    rpm: Optional[int] = None
    tpm: Optional[int] = None
    priority: Optional[int] = None


def load_deployment_config(config_path: Path) -> List[DeploymentEntry]:
    """Parse model_list from proxy_config.json. Returns [] if absent or unreadable."""
    if not config_path.exists():
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        lib_logger.warning(f"deployment_config: could not read {config_path}: {exc}")
        return []

    raw_list = data.get("model_list")
    if not raw_list:
        return []

    entries: List[DeploymentEntry] = []
    for i, item in enumerate(raw_list):
        try:
            entries.append(
                DeploymentEntry(
                    model_name=item["model_name"],
                    provider=item["provider"].lower(),
                    api_key=item.get("api_key"),
                    api_key_env=item.get("api_key_env"),
                    rpm=item.get("rpm"),
                    tpm=item.get("tpm"),
                    priority=item.get("priority"),
                )
            )
        except (KeyError, TypeError) as exc:
            lib_logger.warning(f"deployment_config: skipping model_list[{i}]: {exc}")

    return entries


def apply_deployment_config(entries: List[DeploymentEntry]) -> None:
    """
    Inject credentials from model_list entries into os.environ.

    Keys become PROVIDER_API_KEY, PROVIDER_API_KEY_2, PROVIDER_API_KEY_3, …
    matching the naming convention that CredentialManager and api_keys
    discovery already expect. Existing env vars are never overwritten.
    """
    if not entries:
        return

    # Track next index per provider to generate numbered env vars
    next_index: Dict[str, int] = {}

    for entry in entries:
        provider = entry.provider
        env_prefix = _PROVIDER_ENV_PREFIX.get(provider)
        if not env_prefix:
            lib_logger.warning(
                f"deployment_config: unknown provider '{provider}', skipping"
            )
            continue

        api_key = entry.api_key
        if api_key is None and entry.api_key_env:
            api_key = os.environ.get(entry.api_key_env)
        if not api_key:
            continue

        idx = next_index.get(provider, 1)
        next_index[provider] = idx + 1
        env_var = env_prefix if idx == 1 else f"{env_prefix}_{idx}"

        if os.environ.get(env_var):
            lib_logger.debug(f"deployment_config: {env_var} already set, skipping")
            continue

        os.environ[env_var] = api_key
        lib_logger.info(
            f"deployment_config: injected {provider} key as {env_var} "
            f"(model={entry.model_name})"
        )

    if next_index:
        summary = ", ".join(f"{p}:{n - 1}" for p, n in next_index.items())
        lib_logger.info(f"deployment_config: applied model_list ({summary} keys)")
