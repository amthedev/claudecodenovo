# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Optional second pass: the model critiques and improves its own answer.

Idea: after the first response comes back, send (original question + first
answer) into a SECOND call asking the model to find weaknesses and produce a
better version. Costs ~2x tokens + ~2x latency. Default OFF — opt-in via
VLLM_SELF_CRITIQUE=on.

When NOT to use:
  - Responses that contain tool_use blocks (in flight; we'd break the agent
    loop if we modified them).
  - Empty or trivial responses (nothing to critique).
  - Errors (passed through unchanged).

When OK to use:
  - Plain text responses on non-streaming requests, where the gain in quality
    justifies the cost — analysis, code review, multi-step reasoning.
"""
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..client.rotating_client import RotatingClient

_LOG = logging.getLogger("rotator_library.self_critique")
_CRITIQUE_PROMPT = (
    "Voce é o mesmo agente que acabou de responder a pergunta abaixo. "
    "Reanalise sua propria resposta de forma critica:\n"
    "- Esta correta tecnicamente? Algum fato, calculo ou caminho de codigo errado?\n"
    "- Esta completa? Faltou um passo ou um caso de borda?\n"
    "- Esta clara? Algum trecho confuso, redundante, ou que poderia explicar melhor?\n"
    "- Atende exatamente o que foi pedido, sem extrapolar?\n"
    "Reescreva a resposta MELHORADA. Se sua resposta original ja estava boa, "
    "devolva ela quase identica (mantenha o que estava bom; nao reescreva por "
    "reescrever). Responda APENAS com a resposta final melhorada, sem meta-"
    "comentario sobre o que voce mudou."
)


def is_enabled() -> bool:
    return os.getenv("VLLM_SELF_CRITIQUE", "off").lower() in {"on", "1", "true", "yes"}


def _extract_text_content(anthropic_response: Dict[str, Any]) -> Optional[str]:
    """Return joined text from text blocks, or None if the response has tool_use
    blocks (don't critique mid-agent-loop) or is empty."""
    content = anthropic_response.get("content")
    if not isinstance(content, list) or not content:
        return None
    has_tool = any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in content
    )
    if has_tool:
        return None
    texts = [
        str(b.get("text") or "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    joined = "\n".join(t for t in texts if t.strip())
    return joined if joined.strip() else None


def _user_question(request: Any) -> str:
    """Pull the most recent user-side text from the original request."""
    msgs = getattr(request, "messages", None) or []
    for m in reversed(msgs):
        role = m.role if hasattr(m, "role") else m.get("role")
        if role != "user":
            continue
        content = m.content if hasattr(m, "content") else m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for b in content:
                bt = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                if bt == "text":
                    parts.append(b.get("text") if isinstance(b, dict) else getattr(b, "text", "") or "")
            if parts:
                return "\n".join(parts)
    return ""


async def maybe_critique_response(
    anthropic_response: Dict[str, Any],
    request: Any,
    client: "RotatingClient",
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Return an improved response if self-critique is enabled AND applicable;
    otherwise the original response unchanged. Never raises — falls back to
    original on any failure (cost was already paid; don't break the request)."""
    log = log or _LOG
    if not is_enabled():
        return anthropic_response

    original_text = _extract_text_content(anthropic_response)
    if not original_text:
        return anthropic_response  # tool_use, empty, or unsupported shape

    question = _user_question(request)
    if not question.strip():
        return anthropic_response

    critique_input = (
        f"=== PERGUNTA DO USUARIO ===\n{question}\n\n"
        f"=== SUA RESPOSTA ORIGINAL ===\n{original_text}\n\n"
        f"=== INSTRUCAO ===\n{_CRITIQUE_PROMPT}"
    )
    try:
        resp = await client.acompletion(
            model=request.model,
            messages=[{"role": "user", "content": critique_input}],
            stream=False,
            max_tokens=min(8000, len(original_text) * 2 // 3 + 1024),
        )
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
        else:
            choices = getattr(resp, "choices", None) or []
        if not choices:
            return anthropic_response
        choice0 = choices[0]
        msg = choice0["message"] if isinstance(choice0, dict) else choice0.message
        improved = (msg.get("content") if isinstance(msg, dict) else msg.content) or ""
        improved = improved.strip()
        if not improved or len(improved) < len(original_text) // 5:
            # Suspect output: too short, model probably misunderstood the task.
            log.info("Self-critique returned suspiciously short text; keeping original.")
            return anthropic_response
        log.info("Self-critique applied (orig=%d chars, improved=%d chars).",
                 len(original_text), len(improved))
    except Exception as e:
        log.warning("Self-critique failed (%r); keeping original response.", e)
        return anthropic_response

    # Rebuild the content: keep non-text blocks (none expected here, since we
    # skipped tool_use above), replace the joined text with the improved version.
    new_response = dict(anthropic_response)
    new_response["content"] = [{"type": "text", "text": improved}]
    return new_response
