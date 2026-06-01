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

Token counting is REAL (litellm token_counter on the OpenAI-translated request,
including the tool schemas), with a char-based estimate as fallback when the model
can't be mapped. Falls back safely to bounded recent-context truncation if the
summarization call fails or times out.
"""
import asyncio
import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Optional

from .models import AnthropicMessagesRequest

if TYPE_CHECKING:
    from ..client.rotating_client import RotatingClient

# Dedicated semaphore so summary calls don't pile onto the single GPU all at
# once. The MAIN request that triggered this summary is already holding a slot
# of the backend concurrency semaphore (in client/anthropic.py), so we must NOT
# reuse that one here — N main requests each spawning a summary would deadlock
# waiting for slots none of them will release until their summary returns.
# A small separate cap (default 1) means summaries are serialized among
# themselves and add at most 1 extra concurrent stream to the GPU instead of
# one-per-in-flight-request. Env VLLM_CONTEXT_SUMMARY_CONCURRENCY (0 = no cap).
_SUMMARY_SEMAPHORE: Optional[asyncio.Semaphore] = None
_SUMMARY_SEMAPHORE_LOCK = asyncio.Lock()


async def _get_summary_semaphore() -> asyncio.Semaphore:
    global _SUMMARY_SEMAPHORE
    if _SUMMARY_SEMAPHORE is not None:
        return _SUMMARY_SEMAPHORE
    async with _SUMMARY_SEMAPHORE_LOCK:
        if _SUMMARY_SEMAPHORE is None:
            try:
                cap = int(os.environ.get("VLLM_CONTEXT_SUMMARY_CONCURRENCY", "1"))
            except (TypeError, ValueError):
                cap = 1
            _SUMMARY_SEMAPHORE = asyncio.Semaphore(10_000 if cap <= 0 else cap)
        return _SUMMARY_SEMAPHORE


_SUMMARY_MARKER = "[Resumo da conversa anterior]"
_FALLBACK_SUMMARY = (
    f"{_SUMMARY_MARKER}\n"
    "[Mensagens antigas omitidas automaticamente porque o resumo não ficou pronto "
    "a tempo. Continue usando o contexto recente preservado abaixo.]"
)
_SUMMARY_CACHE: OrderedDict[tuple[str, ...], str] = OrderedDict()
_SUMMARY_FAILURE_CACHE: OrderedDict[tuple[str, ...], float] = OrderedDict()
# Chars-per-token used ONLY as a fallback when real token counting (litellm) is
# unavailable. Code/JSON pack denser than prose (~2.5-3 chars/token), so we use a
# low divisor to OVER-estimate and compact a bit early rather than too late — the
# opposite mistake (under-estimating) brings back "thinks and stops" because the
# real request overflows the shared ceiling.
_CHARS_PER_TOKEN = 2.8


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


def _estimate_request_tokens(request: AnthropicMessagesRequest) -> int:
    """Cheap char-based estimate of the whole request, INCLUDING the tool schemas.

    Claude Code ships ~15 tool definitions (Read/Edit/Bash/...) that are easily
    2-4k tokens of JSON. Ignoring them made the budget optimistic and let the real
    request overflow the ceiling — the exact 'thinks and stops' bug this module
    fixes. Counting them (even roughly) closes that gap.
    """
    total = _estimate_tokens(_system_text(request.system))
    for m in (request.messages or []):
        total += _estimate_tokens(_message_text(m))
    total += _estimate_tools_tokens(request)
    return total


# Backwards-compat alias: tests import the old name.
_request_input_tokens = _estimate_request_tokens


def _estimate_tools_tokens(request: AnthropicMessagesRequest) -> int:
    tools = getattr(request, "tools", None)
    if not tools:
        return 0
    import json

    try:
        serialized = json.dumps(
            [t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t for t in tools]
        )
    except Exception:
        serialized = str(tools)
    return _estimate_tokens(serialized)


def _count_request_tokens(
    request: AnthropicMessagesRequest,
    client: "RotatingClient",
    log: logging.Logger,
) -> int:
    """Real token count via litellm (messages + system + tools), with a safe
    fallback to the char estimate. litellm is always present where this code runs
    (the whole handler depends on it), but a model litellm can't map would raise —
    hence the fallback.

    DEFENSIVE: returns max(real, estimate). litellm's token_counter uses a generic
    tokenizer (not Qwen's exact one), so it can SUBESTIMATE for code/JSON. A
    50-token-real-but-15k-estimate message would otherwise sneak past the budget,
    overflow the ceiling, and bring back "thinks and stops". Trusting the higher
    of the two costs a tiny bit of over-compaction in edge cases and prevents the
    much worse failure mode of under-compaction."""
    estimate = _estimate_request_tokens(request)
    try:
        from .translator import (
            anthropic_to_openai_messages,
            anthropic_to_openai_tools,
        )

        messages = request.model_dump(exclude_none=True).get("messages", [])
        openai_messages = anthropic_to_openai_messages(messages, request.system)
        total = client.token_count(model=request.model, messages=openai_messages)

        tools = getattr(request, "tools", None)
        if tools:
            openai_tools = anthropic_to_openai_tools(
                [t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else t for t in tools]
            )
            if openai_tools:
                import json

                total += client.token_count(
                    model=request.model, text=json.dumps(openai_tools)
                )
        return max(int(total), estimate)
    except Exception as e:
        log.debug("Real token count unavailable (%r); using char estimate.", e)
        return estimate


def _messages_input_tokens(system: Any, messages: List[Any]) -> int:
    return _estimate_tokens(_system_text(system)) + sum(
        _estimate_tokens(_message_text(message)) for message in messages
    )


def _truncate_message_tail(message: Any, max_tokens: int) -> Any:
    max_chars = max(256, int(max_tokens * _CHARS_PER_TOKEN))
    role = (
        message.get("role", "")
        if isinstance(message, dict)
        else getattr(message, "role", "")
    )
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", "")
    )
    if isinstance(content, str):
        text = content
    else:
        text = "\n".join(_block_text(block) for block in (content or []))
    role_prefix = f"{role}: "
    if len(role_prefix) + len(text) <= max_chars:
        return message
    prefix = "[... conteúdo antigo omitido ...]\n"
    tail_chars = max(0, max_chars - len(role_prefix) - len(prefix))
    tail_text = text[-tail_chars:] if tail_chars else ""
    trimmed = f"{prefix}{tail_text}"
    if isinstance(message, dict):
        return {**message, "content": trimmed}
    return message.model_copy(update={"content": trimmed})


def _fallback_to_recent_context(
    request: AnthropicMessagesRequest,
    recent_messages: List[Any],
    input_budget: int,
    log: logging.Logger,
    *,
    summary_message: Optional[Any] = None,
) -> AnthropicMessagesRequest:
    """Guarantee a bounded request when hidden summarization is unavailable.

    Previously this was CATASTROPHIC: production logs showed `keeping 1/27 recent
    msgs` over and over. Tail_budget got squeezed by the tools reserve down to
    ~5k tokens — that fits one message and dumps the rest. The model then had
    no idea what the user wanted and answered 'what would you like?'.

    New behavior:
    1. Preserve at least the last MIN_KEEP_MESSAGES (default 8) messages REGARDLESS
       of budget — better to slightly overshoot the input ceiling than to nuke
       the conversation. The model degrades gracefully on slight overshoot;
       losing 26/27 messages destroys the session entirely.
    2. If even the minimum tail exceeds the budget hard, truncate the OLDEST kept
       message rather than dropping (keeps coherence of recent turns).
    3. Per-message truncation only kicks in for the head of the kept window,
       NOT the most recent message (that's the active task).
    """
    summary_message = summary_message or {
        "role": "user",
        "content": _FALLBACK_SUMMARY,
    }
    kept = list(recent_messages)
    original_count = len(kept)
    min_keep = _env_int("VLLM_CONTEXT_FALLBACK_MIN_KEEP", 8)

    # Pop oldest until we fit OR we're at min_keep messages — whichever comes first.
    while len(kept) > max(1, min_keep):
        candidate = [summary_message, *kept]
        if _messages_input_tokens(request.system, candidate) <= input_budget:
            break
        kept.pop(0)

    candidate = [summary_message, *kept]
    if kept and _messages_input_tokens(request.system, candidate) > input_budget:
        # Still overflowing with min_keep messages — truncate the oldest kept
        # message (its tail is preserved by _truncate_message_tail). This
        # preserves the count of turns the model sees and the recency of the
        # task. Better to lose detail in old turn than to lose the turn itself.
        fixed_tokens = _messages_input_tokens(request.system, [summary_message] + kept[1:])
        available_tokens = max(512, input_budget - fixed_tokens)
        kept[0] = _truncate_message_tail(kept[0], available_tokens)
        candidate = [summary_message, *kept]

    log.warning(
        "Context compaction fallback: keeping %d/%d recent msgs within ~%d tok budget",
        len(kept),
        original_count,
        input_budget,
    )
    return request.model_copy(update={"messages": candidate})


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


def _cache_summary_failure(messages: List[Any]) -> None:
    # Was 300 (5 min) — a single transient hiccup poisoned every subsequent
    # request for 5 minutes, dropping them all into the brutal recent-context
    # fallback. 30s is enough cooldown for genuine outages without ruining
    # the next 50 user turns after one timeout.
    ttl = max(0, _env_int("VLLM_CONTEXT_FAILURE_CACHE_SECONDS", 30))
    key = tuple(_message_fingerprint(message) for message in messages)
    if ttl <= 0 or not key:
        return
    _SUMMARY_FAILURE_CACHE[key] = time.monotonic() + ttl
    _SUMMARY_FAILURE_CACHE.move_to_end(key)
    max_entries = max(1, _env_int("VLLM_CONTEXT_CACHE_MAX_ENTRIES", 64))
    while len(_SUMMARY_FAILURE_CACHE) > max_entries:
        _SUMMARY_FAILURE_CACHE.popitem(last=False)


def _has_cached_summary_failure(messages: List[Any]) -> bool:
    fingerprints = tuple(_message_fingerprint(message) for message in messages)
    now = time.monotonic()
    matched_key: Optional[tuple[str, ...]] = None
    for key, expires_at in list(_SUMMARY_FAILURE_CACHE.items()):
        if expires_at <= now:
            _SUMMARY_FAILURE_CACHE.pop(key, None)
            continue
        if len(key) <= len(fingerprints) and fingerprints[: len(key)] == key:
            if matched_key is None or len(key) > len(matched_key):
                matched_key = key
    if matched_key is None:
        return False
    _SUMMARY_FAILURE_CACHE.move_to_end(matched_key)
    return True


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
    # OUTPUT_RESERVE is what we save for the response (think + tool call). Used
    # only as the EMERGENCY trigger now — we no longer pre-emptively compact.
    output_reserve = _env_int("VLLM_CONTEXT_OUTPUT_RESERVE", 4000)
    # KEEP_TAIL: keep this much of the recent conversation INTACT (no summary,
    # no truncation). 16k = roughly the last ~25 messages of a typical session.
    keep_tail_tokens = _env_int("VLLM_CONTEXT_KEEP_TAIL_TOKENS", 16000)
    margin = 1024
    input_budget = max(2000, model_context - output_reserve - margin)

    # Fast-path: do a CHEAP char-based estimate first. If the request is
    # clearly under-budget (with 30% margin to account for chars/token
    # variance), skip the expensive token_count call entirely. This shaves
    # ~50-200ms off every short conversation, which is the common case.
    estimate = _estimate_request_tokens(request)
    if estimate < input_budget * 0.7:
        return request  # safely fits — common path, ZERO model calls

    total = _count_request_tokens(request, client, log)
    if total <= input_budget:
        return request  # fits — common path, no model call

    # If we got here, the request really doesn't fit. Log it so you can spot
    # when compaction is firing on real customers — if it's frequent, the
    # budget is wrong, not the user's behaviour.
    log.warning(
        "Context compaction TRIGGERED: real=%d tok > budget=%d tok (model_ctx=%d, "
        "out_reserve=%d). Conversation has grown beyond the model's ceiling; "
        "summarizing the old middle to preserve the recent task intact.",
        total, input_budget, model_context, output_reserve,
    )

    # The tool schemas (~2-4k tokens normal; up to ~30k for Claude Code with
    # MCP) are re-attached AFTER compaction, downstream. Reserve their space.
    # Plus reserve room for the summary itself (~3k by default) so the final
    # request = system + tools + summary + tail still fits the input budget.
    tools_reserve = _estimate_tools_tokens(request)
    summary_reserve = max(256, _env_int("VLLM_CONTEXT_SUMMARY_TOKENS", 3000))
    tail_budget = max(2000, input_budget - tools_reserve - summary_reserve)

    messages = list(request.messages or [])
    if len(messages) <= 2:
        # Special case: 1 or 2 messages but they're HUGE (e.g. Claude Code
        # pastes 60k tokens of project context into a single "oi" message).
        # The fallback can't help — there's nothing to drop. We need to
        # truncate the CONTENT of the message itself instead.
        # _truncate_message_tail keeps the END of the message (the recent
        # task) and trims the head. For typical Claude Code requests, the
        # head is the auto-attached project context which the user wasn't
        # explicitly asking about anyway.
        if messages:
            per_msg_budget = max(2000, tail_budget // max(1, len(messages)))
            truncated = [
                _truncate_message_tail(m, per_msg_budget) for m in messages
            ]
            log.warning(
                "Context compaction: %d giant message(s) truncated to ~%d tok each "
                "(no middle to summarize — single oversized message case)",
                len(messages), per_msg_budget,
            )
            return request.model_copy(update={"messages": truncated})
        return _fallback_to_recent_context(request, messages, tail_budget, log)

    # Effective keep_tail: cap by both the user-configured value AND by the
    # actual tail_budget. Previously the env was 16000 but tail_budget could
    # be only 2000 (Claude Code w/ lots of tools) — the tail loop greedily
    # filled to 16k, leaving `middle` empty, which dropped us into the
    # fallback WITHOUT ever attempting a summary. That's why Feitoza's log
    # kept showing "keeping N/N recent msgs" with no "summary applied" line.
    effective_keep_tail = max(2000, min(keep_tail_tokens, tail_budget))

    # Keep the recent tail intact (current task). Walk from the end accumulating
    # tokens until we hit effective_keep_tail; everything before that is the "middle".
    # CRITICAL: stop accumulating BEFORE consuming all messages — we need at
    # least ONE message left in the middle to have something to summarize.
    tail: List[Any] = []
    tail_tokens = 0
    split_idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        t = _estimate_tokens(_message_text(messages[i]))
        # Stop if adding would overflow tail budget AND tail isn't empty.
        # Also stop if we'd swallow ALL but the first message — we need at
        # least one middle message to summarize, otherwise this whole pass
        # is wasted and we degrade to the brutal fallback.
        if tail and (tail_tokens + t > effective_keep_tail or i == 0):
            break
        tail.insert(0, messages[i])
        tail_tokens += t
        split_idx = i
    middle = messages[:split_idx]
    if not middle:
        return _fallback_to_recent_context(request, tail, tail_budget, log)

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

    summary_tokens = max(256, _env_int("VLLM_CONTEXT_SUMMARY_TOKENS", 3000))
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
    if _has_cached_summary_failure(messages_to_summarize):
        log.info("Context compaction: summary cooldown active; using recent-context fallback")
        return _fallback_to_recent_context(
            request,
            tail,
            tail_budget,
            log,
            summary_message=previous_summary_msg,
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
        # Was 15s default — but summarizing ~25k tokens of input on Qwen3-32B
        # easily takes 20-40s (model has to read everything + generate 3k tokens
        # of output). 15s timed out silently, poisoned the cooldown cache (now
        # 30s instead of 5min after recent fix), and dropped the next handful
        # of turns into the brutal recent-context fallback. 60s gives the
        # model real time to produce a useful summary.
        summary_timeout = max(
            1, _env_int("VLLM_CONTEXT_SUMMARY_TIMEOUT_SECONDS", 60)
        )
        # Wait for a summary slot OUTSIDE the wait_for so queue time doesn't
        # eat the generation timeout — the timeout should cover the model call,
        # not the wait for a free GPU slot.
        summary_sem = await _get_summary_semaphore()
        async with summary_sem:
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
        log.warning("Context compaction summary failed (%r); using recent-context fallback.", e)
        _cache_summary_failure(messages_to_summarize)
        if previous_summary_msg:
            return _fallback_to_recent_context(
                request,
                tail,
                tail_budget,
                log,
                summary_message=previous_summary_msg,
            )
        return _fallback_to_recent_context(request, tail, tail_budget, log)

    if not has_previous_summary:
        _cache_summary(middle, summary)
    log.info(
        "Context compaction: %d msgs (~%d tok) -> summary + %d tail msgs",
        len(messages_to_summarize), total, len(tail),
    )
    summary_msg = {"role": "user", "content": f"{_SUMMARY_MARKER}\n{summary}"}
    # The summary itself takes space (up to summary_tokens). Make sure summary +
    # system + tail actually fits the budget; if a long summary plus a long tail
    # still overflow, shrink the tail (reusing the same bounded-fallback logic)
    # rather than handing the backend an over-budget request.
    candidate = [summary_msg, *tail]
    if _messages_input_tokens(request.system, candidate) > tail_budget:
        return _fallback_to_recent_context(
            request, tail, tail_budget, log, summary_message=summary_msg
        )
    return request.model_copy(update={"messages": candidate})
