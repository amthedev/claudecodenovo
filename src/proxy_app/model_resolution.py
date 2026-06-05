# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Model resolution + identity injection.

What lives here:
  - Listing models configured via env (PROXY_MODELS, STATIC_MODELS, ...).
  - Default model resolution and alias mapping (e.g. claude-code-sonnet → the
    real upstream model).
  - Virtual Claude-branded model facade (so clients see claude-* even when the
    real backend is Qwen).
  - Identity instruction injection — the system message that tells the model
    'you are claude-X', so the upstream Qwen doesn't out itself as Qwen when
    asked.

Why extracted: these are pure functions with no DB or app-state deps. Pulled
out of main.py to keep that file focused on FastAPI wiring.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from rotator_library.anthropic_compat import AnthropicMessagesRequest


_LOG = logging.getLogger("proxy.model_resolution")


def static_env_models() -> List[str]:
    """Models configured via env (static fallback)."""
    models: List[str] = []
    for env_name in (
        "PROXY_MODELS",
        "STATIC_MODELS",
        "HOSTED_VLLM_MODELS",
        "OPENAI_MODELS",
    ):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                models.extend(str(m) for m in parsed if m)
            elif isinstance(parsed, dict):
                models.extend(str(k) for k in parsed.keys() if k)
        except json.JSONDecodeError:
            models.extend(m.strip() for m in raw.split(",") if m.strip())
    return list(dict.fromkeys(models))


def default_proxy_model() -> Optional[str]:
    for env_name in (
        "PROXY_DEFAULT_MODEL",
        "DEFAULT_PROXY_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ):
        value = os.getenv(env_name)
        if value:
            return value.strip()
    static_models = static_env_models()
    if static_models:
        return static_models[0]
    return None


def openrouter_fallback_model() -> str:
    """Modelo do OpenRouter usado no modo 'openrouter' e como fallback do 'auto'.
    Configurável via OPENROUTER_FALLBACK_MODEL; default é o Qwen3-Coder grátis."""
    return os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/qwen/qwen3-coder:free").strip()


def _current_backend_mode() -> str:
    """Lê o modo escolhido na tela de Admin (vps/openrouter/auto). Import tardio
    pra evitar ciclo com admin_db. Default 'vps' se algo falhar."""
    try:
        from proxy_app.admin_db import get_backend_mode
        return get_backend_mode()
    except Exception:
        return "vps"


def resolve_model_alias(model: Optional[str]) -> Optional[str]:
    """Map client-facing aliases (claude-code-pro, etc.) to the real model.

    Respeita o backend_mode do Admin: no modo 'openrouter' os aliases caem no
    modelo grátis do OpenRouter em vez da VPS. 'vps' e 'auto' usam a VPS aqui
    (o fallback do 'auto' é feito na rota /v1/messages)."""
    if not model:
        return model
    model = model.strip()
    if "/" in model:
        return model
    # modo openrouter: aliases viram o modelo do OpenRouter direto
    if _current_backend_mode() == "openrouter":
        _LOG.info("Mapping client model '%s' to '%s' (backend=openrouter)", model, openrouter_fallback_model())
        return openrouter_fallback_model()

    default_model = default_proxy_model()
    if not default_model:
        return model

    if "/" not in default_model:
        provider_prefix = os.getenv("PROXY_DEFAULT_PROVIDER", "")
        if not provider_prefix:
            if os.getenv("HOSTED_VLLM_API_BASE") or os.getenv("VLLM_API_BASE"):
                provider_prefix = "hosted_vllm"
            elif os.getenv("OPENAI_API_KEY"):
                provider_prefix = "openai"
        if provider_prefix:
            default_model = f"{provider_prefix}/{default_model}"

    alias_env = os.getenv("PROXY_MODEL_ALIASES", "")
    aliases = {
        "claude-code-pro",
        "claude-code-sonnet",
        "claude-code-opus",
        "claude-code-haiku",
    }
    aliases.update(alias.strip() for alias in alias_env.split(",") if alias.strip())

    if model in aliases or "/" not in model:
        _LOG.info("Mapping client model '%s' to '%s'", model, default_model)
        return default_model
    return model


def apply_thinking_mode_openai(request_data: dict, original_model: str) -> None:
    """Inject Qwen3 thinking flag based on the public model name:
    - opus  → enable_thinking=True (deep reasoning, slower)
    - other → enable_thinking=False (fast)

    enable_thinking é um kwarg do chat template do QWEN3. Modelos sem thinking
    (ex: Qwen2.5-Coder) não conhecem o parâmetro — o vLLM pode rejeitar. Só
    injeta quando o modelo REAL resolvido é Qwen3 (ou VLLM_THINKING_SUPPORTED=on)."""
    resolved = (request_data.get("model") or "").lower()
    thinking_supported = ("qwen3" in resolved) or \
        os.getenv("VLLM_THINKING_SUPPORTED", "").lower() in {"1", "true", "yes", "on"}
    if not thinking_supported:
        return
    is_opus = "opus" in (original_model or "").lower()
    extra = request_data.setdefault("extra_body", {})
    extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = is_opus
    if is_opus:
        extra["thinking"] = {"type": "enabled", "budget_tokens": 10000}
        _LOG.info("[thinking] opus -> enable_thinking=True")
    else:
        _LOG.info("[thinking] sonnet -> enable_thinking=False")


def virtual_claude_models() -> List[str]:
    """Claude-branded names exposed to clients via /v1/models. Configurable
    via VIRTUAL_MODELS (comma-separated)."""
    raw = os.getenv("VIRTUAL_MODELS", "claude-sonnet-4-5,claude-opus-4-6,claude-opus-4-7")
    return [m.strip() for m in raw.split(",") if m.strip()]


def virtual_model_context_window() -> int:
    # Mantém em sincronia com VLLM_MODEL_CONTEXT. Qwen2.5-Coder-32B só tem 32768
    # nativo (pod roda --max-model-len 32768); default 30000 deixa margem.
    try:
        return int(os.getenv("VLLM_MODEL_CONTEXT", "32000"))
    except (TypeError, ValueError):
        return 32000


def virtual_model_max_output_tokens() -> int:
    return 12288


def apply_virtual_model_limits(model_cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    virtual_models = set(virtual_claude_models())
    if not virtual_models:
        return model_cards
    ctx = virtual_model_context_window()
    out = virtual_model_max_output_tokens()
    for card in model_cards:
        if not isinstance(card, dict) or card.get("id") not in virtual_models:
            continue
        card["owned_by"] = card.get("id") or "claude"
        card["context_length"] = ctx
        card["context_window"] = ctx
        card["max_input_tokens"] = ctx
        card["max_completion_tokens"] = out
        card["max_output_tokens"] = out
    return model_cards


def is_virtual_claude_model(model: Optional[str]) -> bool:
    if not model:
        return False
    return model.strip() in set(virtual_claude_models())


def public_response_model(
    original_model: Optional[str], resolved_model: Optional[str]
) -> Optional[str]:
    """The model name the API client should see (Claude-branded even when the
    real upstream is Qwen)."""
    original_model = (original_model or "").strip()
    if is_virtual_claude_model(original_model):
        return original_model

    virtual_models = virtual_claude_models()
    if original_model and "/" not in original_model and virtual_models:
        return virtual_models[0]

    resolved_model = (resolved_model or "").strip()
    default_model = (default_proxy_model() or "").strip()
    if resolved_model and default_model and resolved_model == default_model:
        if virtual_models:
            return virtual_models[0]
    return original_model or resolved_model or None


def identity_instruction(public_model: Optional[str]) -> Optional[str]:
    if not is_virtual_claude_model(public_model):
        return None
    return (
        f"You are {public_model}. If asked what model you are, answer with "
        f"{public_model}. Do not mention any upstream, proxy, or internal model name."
    )


def inject_openai_identity(request_data: dict, public_model: Optional[str]) -> None:
    instruction = identity_instruction(public_model)
    if not instruction:
        return
    messages = request_data.get("messages")
    if not isinstance(messages, list):
        return
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        existing = messages[0].get("content") or ""
        messages[0]["content"] = f"{instruction}\n\n{existing}" if existing else instruction
    else:
        messages.insert(0, {"role": "system", "content": instruction})


def inject_anthropic_identity(
    body: "AnthropicMessagesRequest", public_model: Optional[str]
) -> "AnthropicMessagesRequest":
    instruction = identity_instruction(public_model)
    if not instruction:
        return body
    system = body.system
    if system is None:
        system = instruction
    elif isinstance(system, str):
        system = f"{instruction}\n\n{system}" if system else instruction
    elif isinstance(system, list):
        system = [{"type": "text", "text": instruction}, *system]
    return body.model_copy(update={"system": system})
