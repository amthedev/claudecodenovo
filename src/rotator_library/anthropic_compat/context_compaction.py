# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Intelligent context compaction for text-only backends with a hard context ceiling.

Qwen3-32B has a 40960-token ceiling shared by input AND output. In long
conversations the input grows until little room is left for the output — and
since the model emits its <think> reasoning BEFORE the tool call, that little
room gets eaten by reasoning and the model stops (finish_reason=length) before
ever emitting the action. Users see "it read everything, understood, then stopped
when told to execute".

Fix: when the conversation gets close to the ceiling, summarize the OLD middle of
the conversation with one model call (preserving decisions, files touched, values,
pending items), keeping the system message and the recent tail intact. This frees
guaranteed space for the model to both reason and act.

Token counting is an estimate (chars/ratio) — much better than the old
char-budget and good enough with a safety margin. Falls back safely to plain
truncation if the summarization call fails.
"""
import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Optional

from .models import AnthropicMessagesRequest

if TYPE_CHECKING:
    from ..client.rotating_client import RotatingClient

_SUMMARY_MARKER = "[Resumo da conversa anterior]"
_SUMMARY_CACHE: OrderedDict[tuple[str, ...], str] = OrderedDict()
# Average chars per token for mixed PT/EN/code. Conservative (low) so we
# over-estimate tokens slightly and compact a bit earlier rather than too late.
_CHARS_PER_TOKEN = 3.3


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _block_text(block: Any) -> str:
    """Readable text from any Anthropic content block (text/tool_use/tool_result)."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        bt = block.get("type")
        if bt == "text":
            return str(block.get("text") or "")
        if bt == "tool_use":
            return f"[chamou {block.get('name','')}({block.get('input',{})})]"
        if bt == "tool_result":
            c = block.get("content")
            if isinstance(c, list):
                return " ".join(_block_text(b) for b in c)
            return f"[resultado: {c}]"
        if bt == "image":
            return "[imagem]"
        return str(block)
    # Pydantic block object
    bt = getattr(block, "type", None)
    if bt == "text":
        return str(getattr(block, "text", "") or "")
    return str(block)


def _message_text(message: Any) -> str:
    role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else "")
    content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
    if isinstance(content, str):
        body = content
    elif isinstance(content, list):
        body = "\n".join(_block_text(b) for b in content)
    else:
        body = str(content or "")
    return f"{role}: {body}"


def _system_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(_block_text(b) for b in system)
    return str(system)


def _estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN)


def _request_input_tokens(request: AnthropicMessagesRequest) -> int:
    total = _estimate_tokens(_system_text(request.system))
    for m in (request.messages or []):
        total += _estimate_tokens(_message_text(m))
    return total


def _message_fingerprint(message: Any) -> str:
    return hashlib.sha256(_message_text(message).encode("utf-8")).hexdigest()


def _find_cached_summary(messages: List[Any]) -> tuple[int, Optional[str]]:
    fingerprints = tuple(_message_fingerprint(message) for message in messages)
    best_key: Optional[tuple[str, ...]] = None
    for key in _SUMMARY_CACHE:
        if len(key) <= len(fingerprints) and fingerprints[: len(key)] == key:
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key is None:
        return 0, None
    summary = _SUMMARY_CACHE[best_key]
    _SUMMARY_CACHE.move_to_end(best_key)
    return len(best_key), summary


def _cache_summary(messages: List[Any], summary: str) -> None:
    key = tuple(_message_fingerprint(message) for message in messages)
    if not key:
        return
    _SUMMARY_CACHE[key] = summary
    _SUMMARY_CACHE.move_to_end(key)
    max_entries = max(1, _env_int("VLLM_CONTEXT_CACHE_MAX_ENTRIES", 64))
    while len(_SUMMARY_CACHE) > max_entries:
        _SUMMARY_CACHE.popitem(last=False)


def _summary_input_text(messages: List[Any], max_chars: int) -> str:
    text = "\n\n".join(_message_text(m) for m in messages)
    if len(text) <= max_chars:
        return text
    head_chars = max_chars // 3
    tail_chars = max_chars - head_chars
    omitted = len(text) - head_chars - tail_chars
    return (
        f"{text[:head_chars]}\n\n"
        f"[... {omitted} caracteres antigos omitidos para resumir com rapidez ...]\n\n"
        f"{text[-tail_chars:]}"
    )


async def compact_context_if_needed(
    request: AnthropicMessagesRequest,
    client: "RotatingClient",
    *,
    log: Optional[logging.Logger] = None,
) -> AnthropicMessagesRequest:
    """If the conversation exceeds the input budget, summarize the old middle and
    keep system + recent tail intact. Returns the request unchanged when it fits."""
    log = log or logging.getLogger("rotator_library")
    if os.getenv("VLLM_CONTEXT_COMPACTION", "on").lower() in {"off", "0", "false", "no"}:
        return request

    model_context = _env_int("VLLM_MODEL_CONTEXT", 40960)
    output_reserve = _env_int("VLLM_CONTEXT_OUTPUT_RESERVE", 12288)
    keep_tail_tokens = _env_int("VLLM_CONTEXT_KEEP_TAIL_TOKENS", 8000)
    margin = 1024
    input_budget = max(2000, model_context - output_reserve - margin)

    total = _request_input_tokens(request)
    if total <= input_budget:
        return request  # fits — common path, no model call

    messages = list(request.messages or [])
    if len(messages) <= 2:
        return request  # nothing to compact (single exchange too big is handled by truncation)

    # Keep the recent tail intact (current task). Walk from the end accumulating
    # tokens until we hit keep_tail_tokens; everything before that is the "middle".
    tail: List[Any] = []
    tail_tokens = 0
    split_idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        t = _estimate_tokens(_message_text(messages[i]))
        if tail and tail_tokens + t > keep_tail_tokens:
            break
        tail.insert(0, messages[i])
        tail_tokens += t
        split_idx = i
    middle = messages[:split_idx]
    if not middle:
        # Tail alone already exceeds budget — nothing old to summarize; let the
        # downstream char-truncation guard handle it.
        return request

    has_previous_summary = bool(
        middle and _SUMMARY_MARKER in _message_text(middle[0])
    )
    previous_summary_msg = middle[0] if has_previous_summary else None
    cached_prefix_len = 0
    if not has_previous_summary:
        cached_prefix_len, cached_summary = _find_cached_summary(middle)
        if cached_summary:
            previous_summary_msg = {
                "role": "user",
                "content": f"{_SUMMARY_MARKER}\n{cached_summary}",
            }
    new_middle = (
        middle[1:]
        if has_previous_summary
        else middle[cached_prefix_len:]
    )
    if not new_middle:
        if previous_summary_msg:
            log.info("Context compaction: reusing cached summary")
            return request.model_copy(update={"messages": [previous_summary_msg, *tail]})
        return request

    summary_tokens = max(256, _env_int("VLLM_CONTEXT_SUMMARY_TOKENS", 1000))
    refresh_min_tokens = max(
        summary_tokens,
        _env_int("VLLM_CONTEXT_REFRESH_MIN_TOKENS", 2500),
    )
    new_middle_tokens = sum(_estimate_tokens(_message_text(m)) for m in new_middle)
    if previous_summary_msg and new_middle_tokens < refresh_min_tokens:
        log.info(
            "Context compaction: reusing prior summary; dropping %d stale msgs (~%d tok)",
            len(new_middle),
            new_middle_tokens,
        )
        return request.model_copy(update={"messages": [previous_summary_msg, *tail]})

    # Cap what we feed the summarizer so the summary call itself fits comfortably.
    max_summary_input_chars = int(input_budget * _CHARS_PER_TOKEN * 0.7)
    messages_to_summarize = (
        [previous_summary_msg, *new_middle]
        if previous_summary_msg
        else new_middle
    )
    convo_text = _summary_input_text(messages_to_summarize, max_summary_input_chars)

    summary_prompt = (
        "Você é um arquivista técnico. Resuma a conversa abaixo de forma DETALHADA e "
        "ESTRUTURADA, para que outro agente possa continuar a tarefa sem ter visto o "
        "original. Não seja econômico com informação importante — prefira preservar "
        "demais a perder. Organize em seções com estes títulos (omita os que não se "
        "aplicarem):\n"
        "## Objetivo — o que o usuário quer no geral.\n"
        "## Arquivos — cada arquivo criado/editado/lido, com caminho e o que mudou.\n"
        "## Decisões técnicas — escolhas feitas e o porquê (libs, abordagens, nomes).\n"
        "## Valores e configs — números, chaves, parâmetros, comandos exatos usados.\n"
        "## Estado atual — o que já está pronto e funcionando.\n"
        "## Pendências — o que falta fazer / próximos passos combinados.\n"
        "Não invente nada; registre só o que realmente aconteceu na conversa. "
        "Responda apenas com o resumo estruturado.\n\n"
        f"=== CONVERSA ===\n{convo_text}"
    )
    try:
        model = request.model
        summary_timeout = max(
            1, _env_int("VLLM_CONTEXT_SUMMARY_TIMEOUT_SECONDS", 45)
        )
        response = await asyncio.wait_for(
            client.acompletion(
                model=model,
                messages=[{"role": "user", "content": summary_prompt}],
                stream=False,
                max_tokens=summary_tokens,
            ),
            timeout=summary_timeout,
        )
        if isinstance(response, dict) and "choices" not in response:
            raise RuntimeError(f"summary error payload: {response.get('error')}")
        choices = response["choices"] if isinstance(response, dict) else response.choices
        msg = choices[0]["message"] if isinstance(choices[0], dict) else choices[0].message
        summary = (msg.get("content") if isinstance(msg, dict) else msg.content) or ""
        summary = summary.strip()
        if not summary:
            raise RuntimeError("empty summary")
    except Exception as e:
        log.warning("Context compaction summary failed (%s); leaving truncation to guard.", e)
        if previous_summary_msg:
            return request.model_copy(update={"messages": [previous_summary_msg, *tail]})
        return request  # safe fallback: char-truncation downstream still applies

    if not has_previous_summary:
        _cache_summary(middle, summary)
    log.info(
        "Context compaction: %d msgs (~%d tok) -> summary + %d tail msgs",
        len(messages_to_summarize), total, len(tail),
    )
    summary_msg = {"role": "user", "content": f"{_SUMMARY_MARKER}\n{summary}"}
    new_messages = [summary_msg, *tail]
    return request.model_copy(update={"messages": new_messages})
