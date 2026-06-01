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
import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional

from .models import AnthropicMessagesRequest

if TYPE_CHECKING:
    from ..client.rotating_client import RotatingClient

_SUMMARY_MARKER = "[Resumo da conversa anterior]"
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

    # Don't re-summarize an existing summary: if the first middle message is
    # already our summary, only summarize what came after it.
    already = 0
    if middle and _SUMMARY_MARKER in _message_text(middle[0]):
        already = 1

    to_summarize = middle[already:]
    if not to_summarize:
        return request

    convo_text = "\n\n".join(_message_text(m) for m in to_summarize)
    # Cap what we feed the summarizer so the summary call itself fits comfortably.
    max_summary_input_chars = int(input_budget * _CHARS_PER_TOKEN * 0.7)
    if len(convo_text) > max_summary_input_chars:
        convo_text = convo_text[-max_summary_input_chars:]

    summary_prompt = (
        "Resuma a conversa abaixo de forma concisa, preservando o que importa para "
        "continuar a tarefa: decisões tomadas, arquivos criados/editados (com caminhos), "
        "valores/configurações definidos, comandos executados e pendências. Não invente "
        "nada; só registre o que aconteceu. Responda apenas com o resumo.\n\n"
        f"=== CONVERSA ===\n{convo_text}"
    )
    try:
        model = request.model
        response = await client.acompletion(
            model=model,
            messages=[{"role": "user", "content": summary_prompt}],
            stream=False,
            max_tokens=_env_int("VLLM_CONTEXT_SUMMARY_TOKENS", 1500),
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
        return request  # safe fallback: char-truncation downstream still applies

    log.info(
        "Context compaction: %d msgs (~%d tok) -> summary + %d tail msgs",
        len(to_summarize), total, len(tail),
    )
    kept_summary_block = middle[:already]  # preserve a prior summary if present
    summary_msg = {"role": "user", "content": f"{_SUMMARY_MARKER}\n{summary}"}
    new_messages = [*kept_summary_block, summary_msg, *tail]
    return request.model_copy(update={"messages": new_messages})
