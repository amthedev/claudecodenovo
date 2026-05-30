# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Contextual image captioning for text-only backends.

When the target model cannot see images (e.g. a text-only Qwen3 on vLLM), this
module turns each Anthropic image block into a text block by asking an external
vision model (via OpenRouter) to describe it. The user's own request in the same
message is passed to the vision model so the description is focused on what the
user actually wants (transcribe an error, copy a site's layout, etc.) rather than
a generic caption.

On any failure the image is replaced with a short placeholder and the request
continues — the agent never hard-stops because of an image.
"""

import asyncio
import logging
import os
import re
from typing import TYPE_CHECKING, Any, List, Optional

from .models import (
    AnthropicImageBlock,
    AnthropicMessagesRequest,
    AnthropicTextBlock,
)

if TYPE_CHECKING:
    from ..client.rotating_client import RotatingClient

_FAILURE_PLACEHOLDER = "[Imagem recebida mas não pôde ser processada]"

# Defesa conservadora contra vazamento de identidade do VLM. A prevenção principal
# é o _VISION_IDENTITY_GUARD no prompt; aqui só limpamos vazamentos óbvios sem
# arriscar destruir a descrição real.
_THINK_TAG_RE = re.compile(r"(?is)<think>.*?</think>\s*")
# Nomes de provider/modelo de visão que não devem aparecer. Substituição neutra,
# sem repetir artigo (evita "pelo o sistema").
_PROVIDER_NAME_RE = re.compile(r"(?i)\bqwen[\w.\-]*\b|\bopenrouter\b")
# Uma LINHA inteira que é só auto-apresentação ("Como modelo X, ...") é descartada,
# mas só quando há mais conteúdo depois — nunca apagamos a única linha de descrição.
_IDENTITY_LINE_RE = re.compile(
    r"(?i)^\s*(?:as|sou|como|enquanto|i am|i'm)\b[^\n]*"
    r"\b(?:qwen|openrouter|model[oa]?|assistant|ia|ai)\b[^\n]*$"
)


def _sanitize_description(text: str) -> str:
    """Remove reasoning leaks and model/provider self-identification from a caption."""
    text = _THINK_TAG_RE.sub("", text).strip()
    lines = text.split("\n")
    # Drop pure self-identification lines, but keep at least the substantive content.
    kept = [ln for ln in lines if not _IDENTITY_LINE_RE.match(ln.strip())]
    if kept and any(ln.strip() for ln in kept):
        text = "\n".join(kept)
    # Neutralize stray provider/model names anywhere in the text.
    text = _PROVIDER_NAME_RE.sub("o analisador de imagem", text)
    return text.strip()


def _block_type(block: Any) -> Optional[str]:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_source(block: Any) -> Optional[dict]:
    source = block.get("source") if isinstance(block, dict) else getattr(block, "source", None)
    if source is None:
        return None
    if isinstance(source, dict):
        return source
    return source.model_dump() if hasattr(source, "model_dump") else dict(source)


def _is_pdf_document(block: Any) -> bool:
    if _block_type(block) != "document":
        return False
    source = _block_source(block) or {}
    media_type = (source.get("media_type") or "").lower()
    return "pdf" in media_type or (not media_type and bool(source.get("data")))


def _extract_pdf_text(b64_data: str, max_chars: int, log: logging.Logger) -> Optional[str]:
    """Extract text from a base64-encoded PDF. Returns None if not extractable."""
    import base64

    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf não instalado — não é possível extrair texto de PDF.")
        return None
    try:
        from io import BytesIO

        raw = base64.b64decode(b64_data)
        reader = PdfReader(BytesIO(raw))
        parts = []
        total = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue
            parts.append(page_text)
            total += len(page_text)
            if total >= max_chars:
                break
        text = "\n\n".join(parts).strip()
        if not text:
            return None  # scanned/image-only PDF — no extractable text
        return text[:max_chars]
    except Exception as e:
        log.warning("Falha ao extrair texto do PDF: %s", e)
        return None


def _image_data_uri(block: Any) -> Optional[str]:
    """Build a data: URI from an Anthropic image block (base64 or url source)."""
    source = block.get("source") if isinstance(block, dict) else getattr(block, "source", None)
    if source is None:
        return None
    if not isinstance(source, dict):
        source = source.model_dump() if hasattr(source, "model_dump") else dict(source)
    if source.get("type") == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        if not data:
            return None
        return f"data:{media_type};base64,{data}"
    # url source
    return source.get("url") or None


def _text_of(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("text") or "")
    return str(getattr(block, "text", "") or "")


def _user_context_from_content(content: List[Any]) -> str:
    """Concatenate the text the user wrote alongside the image(s)."""
    parts = [
        _text_of(b) for b in content if _block_type(b) == "text" and _text_of(b).strip()
    ]
    return "\n".join(parts).strip()


# Instrução anti-identidade: o VLM (ex: Qwen-VL no OpenRouter) não pode revelar o
# que é nem narrar raciocínio — a descrição vira contexto do modelo principal e
# não deve vazar o provider/modelo de visão para o cliente.
_VISION_IDENTITY_GUARD = (
    "Responda APENAS com a descrição da imagem. Não se apresente, não diga qual "
    "modelo ou IA você é, não mencione provedor, OpenRouter, Qwen ou qualquer nome "
    "de modelo, e não inclua raciocínio, preâmbulo ou tags <think>. "
)


def _build_vision_prompt(user_context: str) -> str:
    if user_context:
        return (
            f"O usuário enviou esta imagem junto com o seguinte pedido:\n"
            f'"{user_context}"\n\n'
            "Descreva a imagem de forma útil para atender exatamente esse pedido. "
            "Se houver texto, código ou mensagens de erro, transcreva-os com fidelidade. "
            "Se for uma tela, site ou interface, descreva o layout, os elementos e os textos "
            "visíveis. Seja completo e objetivo — sua descrição será a única informação que o "
            "assistente principal terá sobre a imagem.\n\n" + _VISION_IDENTITY_GUARD
        )
    return (
        "Descreva esta imagem de forma completa e fiel. Se houver texto, código ou erros, "
        "transcreva-os. Se for uma interface ou site, descreva o layout e os elementos. "
        "Sua descrição será a única informação que o assistente principal terá sobre a imagem.\n\n"
        + _VISION_IDENTITY_GUARD
    )


async def _describe_image(
    data_uri: str,
    user_context: str,
    client: "RotatingClient",
    vision_model: str,
    max_tokens: int,
    log: logging.Logger,
) -> str:
    """Call the vision model for a single image. Returns description or placeholder."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _build_vision_prompt(user_context)},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]
    try:
        response = await client.acompletion(
            model=vision_model,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
        )
        # acompletion may return a dict (error) or a litellm response object.
        if isinstance(response, dict) and "choices" not in response:
            log.warning("Vision model returned error payload: %s", response.get("error"))
            return _FAILURE_PLACEHOLDER
        choices = (
            response["choices"] if isinstance(response, dict) else response.choices
        )
        message = choices[0]["message"] if isinstance(choices[0], dict) else choices[0].message
        text = (message.get("content") if isinstance(message, dict) else message.content) or ""
        text = _sanitize_description(text)
        if not text:
            return _FAILURE_PLACEHOLDER
        return f"[Imagem analisada: {text}]"
    except Exception as e:
        log.warning("Image captioning failed via %s: %s", vision_model, e)
        return _FAILURE_PLACEHOLDER


async def caption_images_in_request(
    request: AnthropicMessagesRequest,
    client: "RotatingClient",
    *,
    vision_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> AnthropicMessagesRequest:
    """
    Replace image blocks with contextual text descriptions for text-only backends.

    Returns the request unchanged (cheaply) when there are no images.
    """
    log = log or logging.getLogger("rotator_library")
    vision_model = vision_model or os.getenv(
        "VISION_MODEL", "openrouter/qwen/qwen-2.5-vl-7b-instruct"
    )
    if max_tokens is None:
        try:
            max_tokens = int(os.getenv("VISION_MAX_TOKENS", "1024"))
        except ValueError:
            max_tokens = 1024

    try:
        pdf_max_chars = int(os.getenv("PDF_MAX_CHARS", "20000"))
    except ValueError:
        pdf_max_chars = 20000

    messages = request.messages or []

    # Map of (message_index, block_index) -> replacement text.
    replacement_by_pos: dict = {}
    # Image jobs go to the VLM (async); PDFs are extracted locally (sync).
    image_jobs = []  # (m_idx, b_idx, data_uri, user_context)

    for m_idx, message in enumerate(messages):
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        user_context = _user_context_from_content(content)
        for b_idx, block in enumerate(content):
            btype = _block_type(block)
            if btype == "image":
                data_uri = _image_data_uri(block)
                if data_uri:
                    image_jobs.append((m_idx, b_idx, data_uri, user_context))
            elif _is_pdf_document(block):
                source = _block_source(block) or {}
                pdf_text = None
                if source.get("type") == "base64" or source.get("data"):
                    pdf_text = _extract_pdf_text(
                        source.get("data", ""), pdf_max_chars, log
                    )
                if pdf_text:
                    replacement_by_pos[(m_idx, b_idx)] = (
                        f"[Conteúdo do PDF anexado]\n{pdf_text}"
                    )
                else:
                    replacement_by_pos[(m_idx, b_idx)] = (
                        "[PDF anexado, mas não foi possível extrair texto "
                        "(pode ser um PDF digitalizado/sem texto).]"
                    )

    if image_jobs:
        log.info("Captioning %d image(s) via %s", len(image_jobs), vision_model)
        descriptions = await asyncio.gather(
            *(
                _describe_image(uri, ctx, client, vision_model, max_tokens, log)
                for (_, _, uri, ctx) in image_jobs
            )
        )
        for (m_idx, b_idx, _, _), desc in zip(image_jobs, descriptions):
            replacement_by_pos[(m_idx, b_idx)] = desc

    if not replacement_by_pos:
        return request  # no images/PDFs — common path, zero cost

    # Rebuild messages with image/PDF blocks swapped for text blocks.
    new_messages = list(messages)
    for m_idx, message in enumerate(new_messages):
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        if not any((m_idx, b_idx) in replacement_by_pos for b_idx in range(len(content))):
            continue
        new_content = []
        for b_idx, block in enumerate(content):
            desc = replacement_by_pos.get((m_idx, b_idx))
            if desc is not None:
                new_content.append(AnthropicTextBlock(text=desc))
            else:
                new_content.append(block)
        new_messages[m_idx] = message.model_copy(update={"content": new_content})

    return request.model_copy(update={"messages": new_messages})
