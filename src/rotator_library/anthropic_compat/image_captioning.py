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
from typing import TYPE_CHECKING, Any, List, Optional

from .models import (
    AnthropicImageBlock,
    AnthropicMessagesRequest,
    AnthropicTextBlock,
)

if TYPE_CHECKING:
    from ..client.rotating_client import RotatingClient

_FAILURE_PLACEHOLDER = "[Imagem recebida mas não pôde ser processada]"


def _block_type(block: Any) -> Optional[str]:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


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


def _build_vision_prompt(user_context: str) -> str:
    if user_context:
        return (
            f"O usuário enviou esta imagem junto com o seguinte pedido:\n"
            f'"{user_context}"\n\n'
            "Descreva a imagem de forma útil para atender exatamente esse pedido. "
            "Se houver texto, código ou mensagens de erro, transcreva-os com fidelidade. "
            "Se for uma tela, site ou interface, descreva o layout, os elementos e os textos "
            "visíveis. Seja completo e objetivo — sua descrição será a única informação que o "
            "assistente principal terá sobre a imagem."
        )
    return (
        "Descreva esta imagem de forma completa e fiel. Se houver texto, código ou erros, "
        "transcreva-os. Se for uma interface ou site, descreva o layout e os elementos. "
        "Sua descrição será a única informação que o assistente principal terá sobre a imagem."
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
        text = text.strip()
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

    messages = request.messages or []

    # Collect every image occurrence with enough info to caption and replace it.
    # job = (message_index, block_index, data_uri, user_context)
    jobs = []
    for m_idx, message in enumerate(messages):
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        user_context = _user_context_from_content(content)
        for b_idx, block in enumerate(content):
            if _block_type(block) != "image":
                continue
            data_uri = _image_data_uri(block)
            if data_uri:
                jobs.append((m_idx, b_idx, data_uri, user_context))

    if not jobs:
        return request  # no images — common path, zero cost

    log.info("Captioning %d image(s) via %s", len(jobs), vision_model)
    descriptions = await asyncio.gather(
        *(
            _describe_image(uri, ctx, client, vision_model, max_tokens, log)
            for (_, _, uri, ctx) in jobs
        )
    )

    # Rebuild messages with image blocks swapped for text blocks.
    new_messages = list(messages)
    replacement_by_pos = {(j[0], j[1]): desc for j, desc in zip(jobs, descriptions)}
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
