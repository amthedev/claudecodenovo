# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Format translation functions between Anthropic and OpenAI API formats.

This module provides functions to convert requests and responses between
Anthropic's Messages API format and OpenAI's Chat Completions API format.
This enables any OpenAI-compatible provider to work with Anthropic clients.
"""

import json
import os
import re
import uuid
import hashlib
from typing import Any, Dict, List, Optional, Union

from .models import AnthropicMessagesRequest

MIN_THINKING_SIGNATURE_LENGTH = 100
OPENAI_MAX_TOOL_NAME_LENGTH = 64
TOOL_NAME_HASH_LENGTH = 8
TOOL_NAME_PREFIX_LENGTH = OPENAI_MAX_TOOL_NAME_LENGTH - TOOL_NAME_HASH_LENGTH - 1

# =============================================================================
# THINKING BUDGET TO REASONING EFFORT MAPPING
# =============================================================================

# Budget thresholds for reasoning effort levels (based on token counts)
# These map Anthropic's budget_tokens to OpenAI-style reasoning_effort levels
THINKING_BUDGET_THRESHOLDS = {
    "minimal": 4096,
    "low": 8192,
    "low_medium": 12288,
    "medium": 16384,
    "medium_high": 24576,
    "high": 32768,
}

# Providers that support granular reasoning effort levels (low_medium, medium_high, etc.)
# Other providers will receive simplified levels (low, medium, high)
GRANULAR_REASONING_PROVIDERS = set()

# Hosted vLLM defaults, sized for Qwen3-32B's real context ceiling of 40960
# tokens (max_position_embeddings in config.json — vLLM refuses higher). Budget:
# ~40k total -> reserve 12k for output, leaving ~28k tokens (~95k chars) for
# input. Every value is overridable via env to match your own --max-model-len.
# Too-low values truncate history/file-reads and make the agent look
# "superficial" or loop on large edits.
VLLM_MAX_OUTPUT_TOKENS = 12288
VLLM_MAX_INPUT_CHARS = 95000
VLLM_MAX_MESSAGE_CHARS = 40000
VLLM_MAX_TOOL_RESULT_CHARS = 20000
# Was 16000. Doubled because long technical analyses (code review, multi-file
# explanation) legitimately go above 16k chars and the truncation message in
# the middle of a perfectly valid answer makes the model look broken.
VLLM_MAX_RESPONSE_TEXT_CHARS = 32000
VLLM_MAX_TOOLS = 16
# All system-prompt templates live in prompts.py (pure data, no logic). The
# helpers in this file (e.g. _inject_workspace_path_prompt) consume them.
from .prompts import (  # noqa: E402
    VLLM_SENSITIVE_WORKSPACE_PROMPT,
    VLLM_TOOL_USE_SYSTEM_PROMPT,
    VLLM_TEXTUAL_TOOL_PROMPT,
    VLLM_NATIVE_AGENT_PROMPT,
    VLLM_WORKSPACE_PATH_MARKER,
    VLLM_WORKSPACE_PATH_PROMPT,
    VLLM_NATIVE_TOOL_ALLOWLIST_MARKER,
    VLLM_MANDATORY_TOOL_MARKER,
    VLLM_AGENT_FLOW_PROMPT,
    VLLM_MANDATORY_TOOL_PROMPT,
    VLLM_CREATE_FILE_TOOL_PROMPT,
    VLLM_INSPECT_PROJECT_TOOL_PROMPT,
    VLLM_RUN_COMMAND_TOOL_PROMPT,
)
VLLM_TOOL_PRIORITY = {
    "create": 0,
    "update": 1,
    "write": 2,
    "edit": 3,
    "multiedit": 4,
    "read": 5,
    "ls": 6,
    "glob": 7,
    "grep": 8,
    "bash": 9,
    "todowrite": 10,
    "task": 11,
}


_TEXTUAL_TOOL_CALL_START_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=[\"']?([A-Za-z0-9_.:-]+)[\"']?>",
    re.DOTALL,
)
_TEXTUAL_TOOL_CALL_END_RE = re.compile(
    r"</function>\s*(?:</tool_call>)?",
    re.DOTALL,
)
_TEXTUAL_TOOL_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.:-]+)>\s*(.*?)(?=(?:</parameter>\s*)?<parameter=|</function>|\Z)",
    re.DOTALL,
)
_EXECUTE_TOOL_CALL_RE = re.compile(
    r"<execute>\s*(\{.*?\})\s*</execute>",
    re.DOTALL | re.IGNORECASE,
)
# Hermes/Qwen canonical format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
# This is what Qwen2.5-Coder actually emits most often, so parse it before the
# <function=...> markup form below.
_JSON_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_FILE_PATH_TOOL_NAMES = {
    "create",
    "edit",
    "multiedit",
    "read",
    "update",
    "write",
}
_THINK_TAG_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
# These markers MUST be specific enough to ONLY match Qwen3's leaked reasoning
# (think-style meta-commentary), NOT normal answers in English. Earlier the list
# included "i need to ", "let me ", "i'll " — those are perfectly valid response
# openers ("Let me explain step by step.", "I'll add the function and run the
# tests."). The strip pass was deleting LEGITIMATE first paragraphs and
# leaving incoherent answers.
#
# What we keep: phrases that are unambiguously about the meta-task ("the user
# asked X so I should Y", "system reminders", "allowed scope"). Removed all
# generic "I X" openers that double as normal English.
_REASONING_PREAMBLE_MARKERS = (
    "since the user ",
    "since they ",
    "system reminders",
    "allowed scope",
    "straightforward language switch",
)
_CREATE_OR_EDIT_MARKERS = (
    "crie",
    "criar",
    "cria",
    "faca",
    "fassa",
    "faça",
    "faz",
    "implemente",
    "implementa",
    "implementar",
    "adicione",
    "adiciona",
    "adicionar",
    "edite",
    "editar",
    "edita",
    "altere",
    "altera",
    "alterar",
    "modifique",
    "modifica",
    "atualize",
    "atualiza",
    "atualizar",
    "corrija",
    "corrige",
    "conserte",
    "conserta",
    "consertar",
    "arrume",
    "arruma",
    "refatore",
    "refatora",
    "refatorar",
    "renomeie",
    "renomeia",
    "remova",
    "remove",
    "remover",
    "delete",
    "apague",
    "salve",
    "salvar",
    "escreva",
    "gere",
    "gera",
    "gerar",
    "write ",
    "create ",
    "edit ",
    "modify ",
    "fix ",
    "add ",
    "update ",
    "refactor",
    "rename ",
    "remove ",
    "delete ",
    "generate ",
)
_PROJECT_INSPECTION_MARKERS = (
    "analise",
    "analisa",
    "analisar",
    "leia",
    "ler ",
    "leio",
    "veja oq",
    "veja o que",
    "veja como",
    "mostre",
    "mostra",
    "explique",
    "explica",
    "explicar",
    "entenda",
    "entender",
    "revise",
    "revisa",
    "revisar",
    "procure",
    "procura",
    "procurar",
    "busque",
    "busca",
    "buscar",
    "encontre",
    "encontra",
    "encontrar",
    "ache",
    "acha",
    "onde esta",
    "onde fica",
    "o que e",
    "oq e",
    "o que faz",
    "como funciona",
    "projeto",
    "repo",
    "repositorio",
    "repository",
    "estrutura",
    "codebase",
    "search ",
    "find ",
    "grep",
    "look ",
    "read ",
    "explain",
    "review",
    "understand",
    "where is",
    "show me",
)
_RUN_COMMAND_MARKERS = (
    "rode",
    "rodar",
    "roda ",
    "execute",
    "executar",
    "executa",
    "testa",
    "teste",
    "testar",
    "compile",
    "compilar",
    "instale",
    "instala",
    "instalar",
    "build",
    "deploy",
    "commit",
    "comite",
    "comita",
    "push",
    "pull",
    "git ",
    "npm ",
    "pip ",
    "yarn ",
    "run ",
    "install ",
    "lint",
    "format",
)
_SAFE_PROJECT_FIND_EXCLUDES = (
    "-not -path '*/.*' "
    "-not -path '*/node_modules/*' "
    "-not -path '*/__pycache__/*' "
    "-not -path '*/venv/*' "
    "-not -path '*/.git/*' "
    "-not -iname '.env' "
    "-not -iname '.env.*' "
    "-not -iname '*secret*' "
    "-not -iname '*token*' "
    "-not -iname '*credential*' "
    "-not -iname '*credentials*' "
    "-not -iname '*api_key*' "
    "-not -iname '*apikey*' "
    "-not -iname '*.pem' "
    "-not -iname '*.key' "
    "-not -iname '*.crt' "
    "-not -iname '*.p12' "
    "-not -iname '*.pfx'"
)
_PYTHON_REQUEST_MARKERS = (
    "python",
    "py ",
    " py",
    "oython",
    "python3",
)


def _normalize_tool_input_for_anthropic(tool_name: str, tool_input: Any) -> Any:
    if not isinstance(tool_input, dict):
        return tool_input
    if (
        str(tool_name or "").lower() in _FILE_PATH_TOOL_NAMES
        and "file_path" not in tool_input
        and "path" in tool_input
    ):
        tool_input = dict(tool_input)
        tool_input["file_path"] = tool_input.pop("path")
    return tool_input


def _normalize_tool_arguments_for_anthropic(tool_name: str, arguments: str) -> str:
    if not arguments:
        return arguments
    try:
        tool_input = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    normalized = _normalize_tool_input_for_anthropic(tool_name, tool_input)
    if normalized is tool_input:
        return arguments
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _parse_textual_tool_calls(text: str) -> tuple[str, List[dict]]:
    """
    Convert common text-emitted tool call markup into Anthropic tool_use blocks.

    Some OpenAI-compatible coding models emit Claude-style tool calls as text,
    for example:
        <tool_call><function=Bash><parameter=command>ls</parameter></function></tool_call>
    Claude Code only executes real Anthropic tool_use blocks, so normalize here.
    """
    if not text:
        return text, []

    tool_blocks: List[dict] = []
    cleaned_execute_parts: List[str] = []
    last_execute_pos = 0

    for match in _EXECUTE_TOOL_CALL_RE.finditer(text):
        cleaned_execute_parts.append(text[last_execute_pos : match.start()])
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            cleaned_execute_parts.append(match.group(0))
            last_execute_pos = match.end()
            continue

        tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
        tool_input = payload.get("input") or payload.get("arguments") or {}
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                tool_input = {}
        tool_input = _normalize_tool_input_for_anthropic(tool_name, tool_input)

        if tool_name and isinstance(tool_input, dict):
            tool_blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
        else:
            cleaned_execute_parts.append(match.group(0))
        last_execute_pos = match.end()

    cleaned_execute_parts.append(text[last_execute_pos:])
    text = "".join(cleaned_execute_parts)

    # Hermes/Qwen canonical JSON form: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    cleaned_json_parts: List[str] = []
    last_json_pos = 0
    for match in _JSON_TOOL_CALL_RE.finditer(text):
        cleaned_json_parts.append(text[last_json_pos : match.start()])
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            cleaned_json_parts.append(match.group(0))
            last_json_pos = match.end()
            continue

        tool_name = str(payload.get("name") or payload.get("tool") or "").strip()
        tool_input = payload.get("arguments")
        if tool_input is None:
            tool_input = payload.get("input") or {}
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                tool_input = {}
        tool_input = _normalize_tool_input_for_anthropic(tool_name, tool_input)

        if tool_name and isinstance(tool_input, dict):
            tool_blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
        else:
            cleaned_json_parts.append(match.group(0))
        last_json_pos = match.end()
    cleaned_json_parts.append(text[last_json_pos:])
    text = "".join(cleaned_json_parts)

    if "<function=" not in text:
        return text.strip(), tool_blocks

    cleaned_parts: List[str] = []

    def parse_tool_body(tool_name: str, body: str) -> Optional[dict]:
        tool_input: Dict[str, Any] = {}

        for param_match in _TEXTUAL_TOOL_PARAM_RE.finditer(body):
            key = param_match.group(1).strip()
            value = param_match.group(2)
            value = re.sub(r"</parameter>\s*$", "", value, flags=re.DOTALL).strip()
            if (
                (value.startswith("{") and value.endswith("}"))
                or (value.startswith("[") and value.endswith("]"))
            ):
                try:
                    tool_input[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            tool_input[key] = value

        tool_input = _normalize_tool_input_for_anthropic(tool_name, tool_input)

        # Accept tool calls even with empty args dict — tools like LS, pwd, or
        # ListFiles legitimately take no parameters. The previous `and tool_input`
        # check treated {} as missing and rejected those entirely.
        if tool_name and isinstance(tool_input, dict):
            return {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:12]}",
                "name": tool_name,
                "input": tool_input,
            }
        return None

    last_pos = 0
    for match in _TEXTUAL_TOOL_CALL_START_RE.finditer(text):
        if match.start() < last_pos:
            continue
        cleaned_parts.append(text[last_pos : match.start()])

        tool_name = match.group(1).strip()
        body_start = match.end()
        end_match = _TEXTUAL_TOOL_CALL_END_RE.search(text, body_start)
        if end_match:
            body_end = end_match.start()
            block_end = end_match.end()
        else:
            body_end = len(text)
            block_end = len(text)

        block = parse_tool_body(tool_name, text[body_start:body_end])
        if block:
            tool_blocks.append(block)
        else:
            cleaned_parts.append(text[match.start() : block_end])
        last_pos = block_end
        if not end_match:
            break

    cleaned_parts.append(text[last_pos:])
    cleaned_text = "".join(cleaned_parts).strip()
    return cleaned_text, tool_blocks


def _truncate_openai_tool_name(name: str) -> str:
    """Keep Anthropic tool names compatible with OpenAI's 64-char function limit."""
    if len(name) <= OPENAI_MAX_TOOL_NAME_LENGTH:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:TOOL_NAME_HASH_LENGTH]
    return f"{name[:TOOL_NAME_PREFIX_LENGTH]}_{digest}"


def _map_tool_name_for_openai(
    name: str,
    tool_name_mapping: Optional[Dict[str, str]] = None,
) -> str:
    mapped = _truncate_openai_tool_name(name)
    if tool_name_mapping is not None and mapped != name:
        tool_name_mapping[mapped] = name
    return mapped


def _restore_tool_name(
    name: str,
    tool_name_mapping: Optional[Dict[str, str]] = None,
) -> str:
    if not tool_name_mapping:
        return name
    return tool_name_mapping.get(name, name)


def _vllm_response_language_instruction() -> str:
    # Default is empty: let the model follow the user's language naturally.
    # Set PROXY_RESPONSE_LANGUAGE=pt-BR (or any locale) in env to force a language.
    language = os.getenv("PROXY_RESPONSE_LANGUAGE", "").strip()
    if not language:
        return ""
    return (
        f"Write normal user-facing responses in {language} unless the user "
        "explicitly requests another language."
    )


def _budget_to_reasoning_effort(budget_tokens: int, model: str) -> str:
    """
    Map Anthropic thinking budget_tokens to a reasoning_effort level.

    Args:
        budget_tokens: The thinking budget in tokens from the Anthropic request
        model: The model name (used to determine if provider supports granular levels)

    Returns:
        A reasoning_effort level string (e.g., "low", "medium", "high")
    """
    # Determine granular level based on budget
    if budget_tokens <= THINKING_BUDGET_THRESHOLDS["minimal"]:
        granular_level = "minimal"
    elif budget_tokens <= THINKING_BUDGET_THRESHOLDS["low"]:
        granular_level = "low"
    elif budget_tokens <= THINKING_BUDGET_THRESHOLDS["low_medium"]:
        granular_level = "low_medium"
    elif budget_tokens <= THINKING_BUDGET_THRESHOLDS["medium"]:
        granular_level = "medium"
    elif budget_tokens <= THINKING_BUDGET_THRESHOLDS["medium_high"]:
        granular_level = "medium_high"
    else:
        granular_level = "high"

    # Check if provider supports granular levels
    provider = model.split("/")[0].lower() if "/" in model else ""
    if provider in GRANULAR_REASONING_PROVIDERS:
        return granular_level

    # Simplify to basic levels for non-granular providers
    simplify_map = {
        "minimal": "low",
        "low": "low",
        "low_medium": "medium",
        "medium": "medium",
        "medium_high": "high",
        "high": "high",
    }
    return simplify_map.get(granular_level, "medium")


def _reorder_assistant_content(content: List[dict]) -> List[dict]:
    """
    Reorder assistant message content blocks to ensure correct order:
    1. Thinking blocks come first (required when thinking is enabled)
    2. Text blocks come in the middle (filtering out empty ones)
    3. Tool_use blocks come at the end (required before tool_result)

    This matches Anthropic's expected ordering and prevents API errors.
    """
    if not isinstance(content, list) or len(content) <= 1:
        return content

    thinking_blocks = []
    text_blocks = []
    tool_use_blocks = []
    other_blocks = []

    for block in content:
        if not isinstance(block, dict):
            other_blocks.append(block)
            continue

        block_type = block.get("type", "")

        if block_type in ("thinking", "redacted_thinking"):
            # Sanitize thinking blocks - remove cache_control and other extra fields
            sanitized = {
                "type": block_type,
                "thinking": block.get("thinking", ""),
            }
            if block.get("signature"):
                sanitized["signature"] = block["signature"]
            thinking_blocks.append(sanitized)

        elif block_type == "tool_use":
            tool_use_blocks.append(block)

        elif block_type == "text":
            # Only keep text blocks with meaningful content
            text = block.get("text", "")
            if text and text.strip():
                text_blocks.append(block)

        else:
            # Other block types (images, documents, etc.) go in the text position
            other_blocks.append(block)

    # Reorder: thinking → other → text → tool_use
    return thinking_blocks + other_blocks + text_blocks + tool_use_blocks


def anthropic_to_openai_messages(
    anthropic_messages: List[dict],
    system: Optional[Union[str, List[dict]]] = None,
    tool_name_mapping: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """
    Convert Anthropic message format to OpenAI format.

    Key differences:
    - Anthropic: system is a separate field, content can be string or list of blocks
    - OpenAI: system is a message with role="system", content is usually string

    Args:
        anthropic_messages: List of messages in Anthropic format
        system: Optional system message (string or list of text blocks)

    Returns:
        List of messages in OpenAI format
    """
    openai_messages = []

    # Handle system message
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # System can be list of text blocks in Anthropic format
            system_text = " ".join(
                block.get("text", "")
                for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if system_text:
                openai_messages.append({"role": "system", "content": system_text})

    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Reorder assistant content blocks to ensure correct order:
            # thinking → text → tool_use
            if role == "assistant":
                content = _reorder_assistant_content(content)

            # Handle content blocks
            openai_content = []
            tool_calls = []
            reasoning_content = ""
            thinking_signature = ""

            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "text")

                    if block_type == "text":
                        text = block.get("text", "")
                        # Claude Code can replay empty text blocks next to tool_use.
                        # Anthropic accepts that, but OpenAI-compatible providers
                        # often reject empty content blocks.
                        if isinstance(text, str) and text.strip():
                            openai_content.append({"type": "text", "text": text})
                    elif block_type == "image":
                        # Convert Anthropic image format to OpenAI
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                                    },
                                }
                            )
                        elif source.get("type") == "url":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": source.get("url", "")},
                                }
                            )
                    elif block_type == "document":
                        # Convert Anthropic document format (e.g. PDF) to OpenAI
                        # Documents are treated similarly to images with appropriate mime type
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{source.get('media_type', 'application/pdf')};base64,{source.get('data', '')}"
                                    },
                                }
                            )
                        elif source.get("type") == "url":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": source.get("url", "")},
                                }
                            )
                    elif block_type == "thinking":
                        signature = block.get("signature", "")
                        if (
                            signature
                            and len(signature) >= MIN_THINKING_SIGNATURE_LENGTH
                        ):
                            thinking_text = block.get("thinking", "")
                            if thinking_text:
                                reasoning_content += thinking_text
                            thinking_signature = signature
                    elif block_type == "redacted_thinking":
                        signature = block.get("signature", "")
                        if (
                            signature
                            and len(signature) >= MIN_THINKING_SIGNATURE_LENGTH
                        ):
                            thinking_signature = signature
                    elif block_type == "tool_use":
                        # Anthropic tool_use -> OpenAI tool_calls
                        tool_name = _map_tool_name_for_openai(
                            str(block.get("name", "")),
                            tool_name_mapping,
                        )
                        tool_calls.append(
                            {
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                        )
                    elif block_type == "tool_result":
                        # Tool results become separate messages in OpenAI format
                        # Content can be string, or list of text/image blocks
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, str):
                            # Simple string content
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": tool_content,
                                }
                            )
                        elif isinstance(tool_content, list):
                            # List of content blocks - may include text and images
                            tool_content_parts = []
                            for b in tool_content:
                                if not isinstance(b, dict):
                                    continue
                                b_type = b.get("type", "")
                                if b_type == "text":
                                    tool_content_parts.append(
                                        {"type": "text", "text": b.get("text", "")}
                                    )
                                elif b_type == "image":
                                    # Convert Anthropic image format to OpenAI format
                                    source = b.get("source", {})
                                    if source.get("type") == "base64":
                                        tool_content_parts.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                                                },
                                            }
                                        )
                                    elif source.get("type") == "url":
                                        tool_content_parts.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": source.get("url", "")
                                                },
                                            }
                                        )

                            # If we only have text parts, join them as a string for compatibility
                            # Otherwise use the array format for multimodal content
                            if all(p.get("type") == "text" for p in tool_content_parts):
                                combined_text = " ".join(
                                    p.get("text", "") for p in tool_content_parts
                                )
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": combined_text,
                                    }
                                )
                            elif tool_content_parts:
                                # Multimodal content (includes images)
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": tool_content_parts,
                                    }
                                )
                            else:
                                # Empty content
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": "",
                                    }
                                )
                        else:
                            # Fallback for unexpected content type
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": str(tool_content)
                                    if tool_content
                                    else "",
                                }
                            )
                        continue  # Don't add to current message

            # Build the message
            if tool_calls:
                # Assistant message with tool calls
                msg_dict = {"role": role}
                if openai_content:
                    # If there's text content alongside tool calls
                    text_parts = [
                        c.get("text", "")
                        for c in openai_content
                        if c.get("type") == "text"
                    ]
                    msg_dict["content"] = " ".join(text_parts) if text_parts else None
                else:
                    msg_dict["content"] = None
                if reasoning_content:
                    msg_dict["reasoning_content"] = reasoning_content
                if thinking_signature:
                    msg_dict["thinking_signature"] = thinking_signature
                msg_dict["tool_calls"] = tool_calls
                openai_messages.append(msg_dict)
            elif openai_content:
                # Check if it's just text or mixed content
                if len(openai_content) == 1 and openai_content[0].get("type") == "text":
                    msg_dict = {
                        "role": role,
                        "content": openai_content[0].get("text", ""),
                    }
                    if reasoning_content:
                        msg_dict["reasoning_content"] = reasoning_content
                    if thinking_signature:
                        msg_dict["thinking_signature"] = thinking_signature
                    openai_messages.append(msg_dict)
                else:
                    msg_dict = {"role": role, "content": openai_content}
                    if reasoning_content:
                        msg_dict["reasoning_content"] = reasoning_content
                    if thinking_signature:
                        msg_dict["thinking_signature"] = thinking_signature
                    openai_messages.append(msg_dict)
            elif reasoning_content:
                msg_dict = {"role": role, "content": ""}
                msg_dict["reasoning_content"] = reasoning_content
                if thinking_signature:
                    msg_dict["thinking_signature"] = thinking_signature
                openai_messages.append(msg_dict)

    return openai_messages


def anthropic_to_openai_tools(
    anthropic_tools: Optional[List[dict]],
    tool_name_mapping: Optional[Dict[str, str]] = None,
) -> Optional[List[dict]]:
    """
    Convert Anthropic tool definitions to OpenAI format.

    Args:
        anthropic_tools: List of tools in Anthropic format

    Returns:
        List of tools in OpenAI format, or None if no tools provided
    """
    if not anthropic_tools:
        return None

    openai_tools = []
    for index, tool in enumerate(anthropic_tools):
        input_schema = tool.get("input_schema")
        if not isinstance(input_schema, dict):
            # Server-side tools (e.g. Anthropic's web_search) ship without an
            # input_schema because the provider executes them — vLLM can't, so
            # we must skip. But we now LOG the skip explicitly, instead of
            # silently dropping (which previously made tool counts mismatch
            # and the model later "complained the tool wasn't available").
            import logging
            logging.getLogger("rotator_library").info(
                "Skipping tool %r — no input_schema (likely a server-side tool unsupported by this backend).",
                tool.get("name"),
            )
            continue
        raw_name = str(tool.get("name") or "").strip()
        if not raw_name:
            raw_name = f"proxy_unnamed_tool_{index}"
        name = _map_tool_name_for_openai(raw_name, tool_name_mapping)
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": input_schema,
                },
            }
        )
    return openai_tools


def anthropic_to_openai_tool_choice(
    anthropic_tool_choice: Optional[dict],
    tool_name_mapping: Optional[Dict[str, str]] = None,
) -> Optional[Union[str, dict]]:
    """
    Convert Anthropic tool_choice to OpenAI format.

    Args:
        anthropic_tool_choice: Tool choice in Anthropic format

    Returns:
        Tool choice in OpenAI format
    """
    if not anthropic_tool_choice:
        return None

    choice_type = anthropic_tool_choice.get("type", "auto")

    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        name = _map_tool_name_for_openai(
            str(anthropic_tool_choice.get("name", "")),
            tool_name_mapping,
        )
        return {
            "type": "function",
            "function": {"name": name},
        }
    elif choice_type == "none":
        return "none"

    return "auto"


def _strip_vllm_rejected_fields(value: Any) -> Any:
    """Remove Anthropic/OpenAI extras that strict vLLM schemas reject."""
    rejected_keys = {
        "cache_control",
        "thinking_signature",
        "reasoning_content",
        "citations",
    }
    if isinstance(value, list):
        return [_strip_vllm_rejected_fields(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _strip_vllm_rejected_fields(child)
            for key, child in value.items()
            if key not in rejected_keys
        }
    return value


def _neutralize_unsupported_media_for_vllm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Safety net: text-only vLLM rejects image_url blocks (images/PDFs). The proxy
    normally replaces those with text before this point (image_captioning), but if
    that didn't run (e.g. pypdf missing, captioning disabled), an image_url block
    here would 400 the request. Replace any residual image_url block with a short
    text note so the request still succeeds instead of failing with "Load failed".
    """
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        new_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                new_blocks.append(
                    {
                        "type": "text",
                        "text": "[Anexo de imagem/PDF não processável por este modelo]",
                    }
                )
            else:
                new_blocks.append(block)
        message["content"] = new_blocks
    return messages


def _env_int(names: List[str], default: int, minimum: int = 0) -> int:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        try:
            return max(minimum, int(value))
        except ValueError:
            return default
    return default


def _vllm_max_output_tokens() -> int:
    return _env_int(
        ["HOSTED_VLLM_MAX_TOKENS", "VLLM_MAX_OUTPUT_TOKENS"],
        VLLM_MAX_OUTPUT_TOKENS,
        minimum=1,
    )


def _vllm_max_input_chars() -> int:
    explicit = _env_int(
        ["HOSTED_VLLM_MAX_INPUT_CHARS", "VLLM_MAX_INPUT_CHARS"],
        VLLM_MAX_INPUT_CHARS,
        minimum=1000,
    )
    # The char-truncation guard must stay WIDER than the token-based context
    # compaction budget — compaction (smart) runs first; this guard (blind) is the
    # last-resort net. If they cross (e.g. someone raises VLLM_MODEL_CONTEXT for a
    # bigger model without raising this), the blind guard would chop the recent
    # tail that compaction just preserved. Derive a floor from the token budget so
    # they never cross, regardless of model size.
    model_context = _env_int(["VLLM_MODEL_CONTEXT"], 40960)
    output_reserve = _env_int(["VLLM_CONTEXT_OUTPUT_RESERVE"], 12288)
    input_budget_tokens = max(2000, model_context - output_reserve - 1024)
    # ~2.8 chars/token + 20% headroom so the guard sits above compaction's output.
    floor_chars = int(input_budget_tokens * 2.8 * 1.2)
    return max(explicit, floor_chars)


def _vllm_max_message_chars() -> int:
    return _env_int(
        ["HOSTED_VLLM_MAX_MESSAGE_CHARS", "VLLM_MAX_MESSAGE_CHARS"],
        VLLM_MAX_MESSAGE_CHARS,
        minimum=1000,
    )


def _vllm_max_tool_result_chars() -> int:
    return _env_int(
        ["HOSTED_VLLM_MAX_TOOL_RESULT_CHARS", "VLLM_MAX_TOOL_RESULT_CHARS"],
        VLLM_MAX_TOOL_RESULT_CHARS,
        minimum=500,
    )


def _message_char_size(message: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(message, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(message))


def _compact_text_middle(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    # Try to cut at line boundaries so we don't slice a function/struct in
    # half. Falls back to character cut if no newlines are nearby. Improves
    # how readable the kept context is to the model — splitting mid-function
    # makes the model think the code is malformed.
    head_target = max(1, max_chars // 2)
    tail_target = max(1, max_chars - head_target)
    # Snap head to the last newline within ±200 chars of head_target.
    head_cut = head_target
    for i in range(min(head_target + 200, len(text)), max(0, head_target - 200), -1):
        if i < len(text) and text[i] == "\n":
            head_cut = i
            break
    # Snap tail to the first newline within ±200 chars of (len-tail_target).
    tail_start_target = len(text) - tail_target
    tail_start = tail_start_target
    for i in range(max(0, tail_start_target - 200), min(len(text), tail_start_target + 200)):
        if text[i] == "\n":
            tail_start = i + 1
            break
    if tail_start <= head_cut:
        # Snapping collapsed the window; fall back to raw char cut.
        head_cut = head_target
        tail_start = len(text) - tail_target
    omitted = max(0, tail_start - head_cut)
    return (
        f"{text[:head_cut]}\n\n"
        f"[... {omitted} characters omitted from {label} to fit hosted vLLM context ...]\n\n"
        f"{text[tail_start:]}"
    )


def _sanitize_vllm_response_text(text: str) -> str:
    """
    Bound repetitive vLLM output before returning it to Anthropic clients.

    Local models can occasionally loop through paraphrases until max_tokens.
    The client only needs the useful prefix, not thousands of repeated lines.
    """
    text = _strip_vllm_reasoning_preamble(text)
    if len(text) <= VLLM_MAX_RESPONSE_TEXT_CHARS:
        return text
    return (
        text[:VLLM_MAX_RESPONSE_TEXT_CHARS].rstrip()
        + "\n\n[Resposta interrompida pelo proxy porque o modelo começou a gerar "
        "texto excessivamente longo ou repetitivo.]"
    )


def _looks_like_reasoning_preamble(paragraph: str) -> bool:
    lowered = paragraph.strip().lower()
    return any(marker in lowered for marker in _REASONING_PREAMBLE_MARKERS)


def _strip_vllm_reasoning_preamble(text: str) -> str:
    if not text:
        return text

    text = _THINK_TAG_RE.sub("", text).strip()
    if os.getenv("HOSTED_VLLM_STRIP_REASONING_PREAMBLE", "true").lower() in {
        "false",
        "0",
        "no",
    }:
        return text

    paragraphs = re.split(r"\n\s*\n", text)
    while len(paragraphs) > 1 and _looks_like_reasoning_preamble(paragraphs[0]):
        paragraphs.pop(0)
    return "\n\n".join(paragraphs).strip()


def _compact_content_for_vllm(content: Any, max_chars: int, label: str) -> Any:
    if isinstance(content, str):
        return _compact_text_middle(content, max_chars, label)
    if isinstance(content, list):
        compacted = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                block_copy = dict(block)
                block_copy["text"] = _compact_text_middle(
                    str(block_copy.get("text") or ""), max_chars, label
                )
                compacted.append(block_copy)
            else:
                compacted.append(block)
        return compacted
    return content


def _compact_large_messages_for_vllm(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    max_tool_result_chars = _vllm_max_tool_result_chars()
    max_message_chars = _vllm_max_message_chars()
    compacted = []
    for message in messages:
        message_copy = dict(message)
        role = message_copy.get("role")
        if role == "tool":
            message_copy["content"] = _compact_content_for_vllm(
                message_copy.get("content", ""),
                max_tool_result_chars,
                "tool output",
            )
        elif _message_char_size(message_copy) > max_message_chars:
            message_copy["content"] = _compact_content_for_vllm(
                message_copy.get("content", ""),
                max_message_chars,
                f"{role or 'message'} content",
            )
        compacted.append(message_copy)
    return compacted


def _trim_message_content(message: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    trimmed = dict(message)
    content = trimmed.get("content")
    if isinstance(content, str) and len(content) > max_chars:
        trimmed["content"] = _compact_text_middle(content, max_chars, "message")
    elif isinstance(content, list) and _message_char_size(trimmed) > max_chars:
        trimmed["content"] = _compact_content_for_vllm(content[-8:], max_chars, "message")
    return trimmed


def _drop_orphan_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove tool_use/tool_result pairs that were broken by truncation.

    After dropping old messages, a `tool` result can be left without the
    `assistant` tool_call that produced it (or an assistant tool_call without its
    result). vLLM/OpenAI reject this, and an inconsistent history makes the model
    forget what it already did and re-run the same tool calls — an infinite loop
    (e.g. re-searching the same symbols over and over). This pass keeps only tool
    results whose call id appears in a preceding assistant tool_calls, and only
    assistant tool_calls whose result follows — keeping the history coherent.
    """
    # Pass 1 (order-aware): for each tool result, mark the call_id only if it
    # appears in an ASSISTANT tool_calls BEFORE this result. A reversed order
    # (result before its call) is semantically broken — both sides get dropped.
    seen_call_ids: set = set()
    result_ids: set = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id"):
                    seen_call_ids.add(tc["id"])
        elif m.get("role") == "tool" and m.get("tool_call_id"):
            if m["tool_call_id"] in seen_call_ids:
                result_ids.add(m["tool_call_id"])

    cleaned: List[Dict[str, Any]] = []
    available_call_ids: set = set()
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            # Keep only the tool_calls that have a matching result downstream.
            kept_calls = [
                tc for tc in message["tool_calls"]
                if tc.get("id") in result_ids
            ]
            if kept_calls:
                msg = dict(message)
                msg["tool_calls"] = kept_calls
                available_call_ids.update(tc.get("id") for tc in kept_calls)
                cleaned.append(msg)
            elif (message.get("content") or "").strip():
                # No valid tool_calls but has text — keep as a plain assistant turn.
                msg = dict(message)
                msg.pop("tool_calls", None)
                cleaned.append(msg)
            # else: assistant with only orphan tool_calls → drop entirely
        elif role == "tool":
            # Keep the result only if its call id is still present upstream.
            if message.get("tool_call_id") in available_call_ids:
                cleaned.append(message)
            # else: orphan result → drop
        else:
            cleaned.append(message)
    return cleaned


def _truncate_messages_for_vllm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Last-resort blind char-based truncation. The PRIMARY budget controller is
    the token-based compaction in context_compaction.py (runs BEFORE this for
    Anthropic+vLLM requests). This guard exists for two reasons:

    1. OpenAI-compatible clients (Cursor / vanilla /v1/chat/completions) skip
       the Anthropic pipeline entirely — compaction doesn't run for them, so we
       still need *some* size cap.
    2. Belt-and-suspenders: if compaction is disabled or fails open, this
       prevents an unbounded payload from reaching the backend.

    Why chars and not tokens here: this function used to be the only line of
    defense and char-counting has zero dependencies. Now that compaction runs
    first for the heavy case, the precise unit matters less; _vllm_max_input_chars
    derives a floor from VLLM_MODEL_CONTEXT so this guard always sits ABOVE the
    compaction budget — they never disagree on what fits."""
    messages = _compact_large_messages_for_vllm(messages)
    max_chars = _vllm_max_input_chars()
    if sum(_message_char_size(message) for message in messages) <= max_chars:
        # Even without truncation, ensure tool pairs are coherent — Claude Code
        # may send histories the model finds confusing if a pair is incomplete.
        return _drop_orphan_tool_messages(messages)

    system_budget = max(4000, max_chars // 3)
    system_messages = [
        _trim_message_content(m, system_budget)
        for m in messages
        if m.get("role") == "system"
    ]
    conversation = [m for m in messages if m.get("role") != "system"]

    kept_reversed: List[Dict[str, Any]] = []
    used_chars = sum(_message_char_size(message) for message in system_messages)
    omitted_count = 0

    for message in reversed(conversation):
        message_size = _message_char_size(message)
        if kept_reversed and used_chars + message_size > max_chars:
            omitted_count += 1
            continue
        if not kept_reversed and message_size > max_chars:
            trimmed = dict(message)
            content = trimmed.get("content")
            if isinstance(content, str):
                trimmed["content"] = _compact_text_middle(content, max_chars, "message")
            elif isinstance(content, list):
                trimmed["content"] = _compact_content_for_vllm(
                    content[-8:], max_chars, "message"
                )
            message = trimmed
            message_size = _message_char_size(message)
        kept_reversed.append(message)
        used_chars += message_size

    kept = list(reversed(kept_reversed))
    if omitted_count:
        notice = {
            "role": "system",
            "content": (
                f"{omitted_count} older conversation message(s) were omitted "
                "because the hosted vLLM context window is smaller than Claude's."
            ),
        }
        system_messages = [*system_messages, notice]

    # Drop tool pairs broken by the truncation above to keep history coherent.
    return _drop_orphan_tool_messages([*system_messages, *kept])


def _vllm_max_tools() -> int:
    return _env_int(
        ["HOSTED_VLLM_MAX_TOOLS", "VLLM_MAX_TOOLS"],
        VLLM_MAX_TOOLS,
    )


def _tool_name(tool: Dict[str, Any]) -> str:
    function = tool.get("function") or {}
    return str(function.get("name") or "").lower()


def _prioritize_tools_for_vllm(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed_tools = list(enumerate(tools))
    indexed_tools.sort(
        key=lambda item: (
            VLLM_TOOL_PRIORITY.get(_tool_name(item[1]), 100),
            item[0],
        )
    )
    return [tool for _, tool in indexed_tools]


def _compact_schema_for_vllm(value: Any, max_description_chars: int = 160) -> Any:
    if isinstance(value, list):
        return [_compact_schema_for_vllm(item, max_description_chars) for item in value]
    if isinstance(value, dict):
        compacted = {}
        for key, child in value.items():
            if key in {"examples", "default", "$defs", "definitions"}:
                continue
            if key == "description" and isinstance(child, str):
                compacted[key] = child[:max_description_chars]
                continue
            compacted[key] = _compact_schema_for_vllm(child, max_description_chars)
        return compacted
    return value


def _compact_tools_for_vllm(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    max_tools = _vllm_max_tools()
    if max_tools == 0:
        return []

    compacted_tools = []
    for tool in _prioritize_tools_for_vllm(tools)[:max_tools]:
        tool_copy = dict(tool)
        function = dict(tool_copy.get("function") or {})
        description = function.get("description")
        if isinstance(description, str):
            function["description"] = description[:512]
        if "parameters" in function:
            function["parameters"] = _compact_schema_for_vllm(function["parameters"])
        tool_copy["function"] = function
        compacted_tools.append(tool_copy)
    return compacted_tools


def _env_flag_enabled(*names: str, default: bool = False) -> bool:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        return value.lower() in {"1", "true", "yes", "on"}
    return default


def _vllm_native_tools_enabled() -> bool:
    return _env_flag_enabled("HOSTED_VLLM_NATIVE_TOOLS", "VLLM_NATIVE_TOOLS")


def _force_tool_fallback_enabled() -> bool:
    return _env_flag_enabled(
        "ANTHROPIC_COMPAT_FORCE_TOOL_FALLBACK",
        "HOSTED_VLLM_FORCE_TOOL_FALLBACK",
        "VLLM_FORCE_TOOL_FALLBACK",
    )


def _format_vllm_textual_tool_prompt(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return "\n".join(
            part
            for part in [VLLM_TOOL_USE_SYSTEM_PROMPT, _vllm_response_language_instruction()]
            if part
        )

    lines = [
        VLLM_TOOL_USE_SYSTEM_PROMPT,
        _vllm_response_language_instruction(),
        "",
        VLLM_TEXTUAL_TOOL_PROMPT,
        "",
        "Available tools:",
    ]
    for tool in tools:
        function = tool.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        description = str(function.get("description") or "").strip()
        schema = function.get("parameters") or {}
        try:
            schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            schema_text = str(schema)
        schema_text = _compact_text_middle(schema_text, 900, f"{name} schema")
        lines.append(f"- {name}: {description}\n  input_schema: {schema_text}")
    return "\n".join(lines).strip()


def _content_to_vllm_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(str(block.get("text") or ""))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _latest_user_text(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _content_to_vllm_text(message.get("content", ""))
    return ""


def _is_tool_result_message(message: Dict[str, Any]) -> bool:
    """True se a mensagem 'user' é na verdade um resultado de ferramenta."""
    content = message.get("content")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def _resolve_persistent_user_text(messages: List[Dict[str, Any]]) -> str:
    """
    Retorna o texto da última mensagem HUMANA REAL, pulando tool_results.

    Isso permite que a intenção (inspect/create/run) persista entre turnos:
    depois que uma ferramenta roda e o resultado volta, a 'última mensagem'
    é o tool_result — mas a intenção original do humano ainda vale.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        if _is_tool_result_message(message):
            continue  # pula tool results, continua procurando o pedido humano
        return _content_to_vllm_text(message.get("content", ""))
    return ""


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _is_whole_project_scope(user_text: str) -> bool:
    """True quando o pedido é sobre o projeto/codebase inteiro (não conceito geral)."""
    lowered = user_text.lower()
    scope_markers = (
        "projeto", "repo", "repositorio", "repository", "codebase",
        "estrutura", "todo o", "toda a", "todos os", "todas as",
        "tudo", "all files", "whole project", "entire", "arquivos",
        "files", "meu codigo", "meu código", "o codigo", "o código",
    )
    return any(m in lowered for m in scope_markers)


def _classify_tool_intent(user_text: str) -> Optional[str]:
    if not user_text.strip():
        return None

    lowered = user_text.lower()
    if _contains_any_marker(lowered, _RUN_COMMAND_MARKERS):
        return "run"
    if _contains_any_marker(lowered, _CREATE_OR_EDIT_MARKERS):
        return "create"
    if _contains_any_marker(lowered, _PROJECT_INSPECTION_MARKERS):
        return "inspect"
    return None


def _append_system_instruction(
    openai_request: Dict[str, Any],
    instruction: str,
) -> None:
    messages = openai_request.setdefault("messages", [])
    if messages and messages[0].get("role") == "system":
        existing = _existing_system_text(messages[0].get("content"))
        if VLLM_MANDATORY_TOOL_MARKER not in existing:
            messages[0]["content"] = f"{existing}\n\n{instruction}".strip()
        return
    messages.insert(0, {"role": "system", "content": instruction})


def _inject_vllm_mandatory_tool_instruction(
    openai_request: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
    attach_fallback: bool = True,
) -> None:
    if not tools:
        return

    intent = _classify_tool_intent(
        _resolve_persistent_user_text(openai_request.get("messages", []))
    )
    if intent is None:
        return

    intent_prompt = {
        "create": VLLM_CREATE_FILE_TOOL_PROMPT,
        "inspect": VLLM_INSPECT_PROJECT_TOOL_PROMPT,
        "run": VLLM_RUN_COMMAND_TOOL_PROMPT,
    }[intent]
    _append_system_instruction(
        openai_request,
        f"{VLLM_MANDATORY_TOOL_PROMPT}\n{intent_prompt}",
    )
    openai_request["_vllm_tool_intent"] = intent
    openai_request["_vllm_previous_tool_count"] = _count_prior_tool_calls(
        openai_request.get("messages", [])
    )
    if attach_fallback:
        fallback = _build_vllm_forced_tool_call(intent, tools, openai_request)
        if fallback:
            openai_request["_vllm_forced_tool_call"] = fallback


def _attach_mandatory_tool_fallback(
    openai_request: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
) -> None:
    if not tools or openai_request.get("_vllm_forced_tool_call"):
        return

    intent = _classify_tool_intent(
        _resolve_persistent_user_text(openai_request.get("messages", []))
    )
    if intent is None:
        return

    openai_request["_vllm_tool_intent"] = intent
    openai_request["_vllm_previous_tool_count"] = _count_prior_tool_calls(
        openai_request.get("messages", [])
    )
    fallback = _build_vllm_forced_tool_call(intent, tools, openai_request)
    if fallback:
        openai_request["_vllm_forced_tool_call"] = fallback


def _count_prior_tool_calls(messages: List[Dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        if message.get("role") == "assistant":
            count += len(message.get("tool_calls") or [])
    return count


def _available_tool_name(
    tools: List[Dict[str, Any]],
    candidates: tuple[str, ...],
) -> Optional[str]:
    by_lower = {}
    for tool in tools:
        function = tool.get("function") or {}
        name = str(function.get("name") or "").strip()
        if name:
            by_lower[name.lower()] = name
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found:
            return found
    return None


def _tool_call(name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"toolu_proxy_{uuid.uuid4().hex[:12]}",
        "name": name,
        "input": tool_input,
    }


def _slugify_filename(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    value = re.sub(r"_+", "_", value)
    return value or "script"


def _infer_create_file_path(user_text: str) -> str:
    lowered = user_text.lower()
    explicit = re.search(
        r"([A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|jsx|html|css|go|rs))",
        user_text,
    )
    if explicit:
        return explicit.group(1)

    if "calculadora" in lowered or "calculator" in lowered:
        return "calculadora.py"
    if (
        "cronometro" in lowered
        or "cronômetro" in lowered
        or "stopwatch" in lowered
        or "timer" in lowered
    ):
        return "cronometro.py"
    if (
        "cobrinha" in lowered
        or "snake" in lowered
        or "jogo da cobra" in lowered
    ):
        return "snake_game.py"

    stem = re.sub(
        r"\b(fa[çc]a|fassa|crie|criar|um|uma|em|de|do|da|no|na|com|avancado|avançado|simples|python|oython)\b",
        " ",
        lowered,
    )
    stem = _slugify_filename(stem)
    extension = (
        ".py" if _contains_any_marker(lowered, _PYTHON_REQUEST_MARKERS) else ".txt"
    )
    return f"{stem}{extension}"


def _write_from_model_text_tool_call(
    write_tool_name: str,
    file_path: str,
) -> Dict[str, Any]:
    tool_call = _tool_call(write_tool_name, {"file_path": file_path, "content": ""})
    tool_call["_proxy_strategy"] = "write_from_model_text"
    tool_call["_proxy_file_path_hint"] = file_path
    return tool_call


def _build_vllm_forced_tool_call(
    intent: str,
    tools: List[Dict[str, Any]],
    openai_request: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    previous_count = int(openai_request.get("_vllm_previous_tool_count") or 0)
    # Usa o pedido humano original (persistente), não o último tool_result
    user_text = _resolve_persistent_user_text(openai_request.get("messages", []))
    lowered = user_text.lower()

    bash = _available_tool_name(tools, ("Bash", "bash"))
    ls_tool = _available_tool_name(tools, ("LS", "ls", "List"))
    write = _available_tool_name(tools, ("Write", "Create", "Update", "write"))

    if intent == "create":
        is_calculator = "calculadora" in lowered or "calculator" in lowered
        is_stopwatch = (
            "cronometro" in lowered
            or "cronômetro" in lowered
            or "stopwatch" in lowered
            or "timer" in lowered
        )
        if previous_count == 0 and write:
            return _write_from_model_text_tool_call(
                write, _infer_create_file_path(user_text)
            )
        if is_calculator and 0 < previous_count <= 1 and bash:
            return _tool_call(
                bash,
                {
                    "command": (
                        "python3 -m py_compile calculadora.py && "
                        "printf '10 + 5\\n' | python3 calculadora.py"
                    ),
                    "description": "Verify the Python calculator",
                },
            )
        if is_stopwatch and 0 < previous_count <= 1 and bash:
            return _tool_call(
                bash,
                {
                    "command": (
                        "python3 -m py_compile cronometro.py && "
                        "printf 'iniciar\\nstatus\\npausar\\nzerar\\nsair\\n' | "
                        "python3 cronometro.py"
                    ),
                    "description": "Verify the Python stopwatch",
                },
            )
        if previous_count == 0 and ls_tool:
            return _tool_call(ls_tool, {"path": "."})

    if intent == "inspect":
        # Só lê o projeto INTEIRO quando o pedido tem escopo de projeto
        # (ex: "analise o projeto", "leia todos os arquivos"). Para pedidos
        # direcionados (ex: "procure X", "explique o login"), deixa o modelo
        # escolher Grep/Read direcionado via prompt — não força read-all.
        if previous_count == 0 and bash and _is_whole_project_scope(user_text):
            safe_excludes = _SAFE_PROJECT_FIND_EXCLUDES
            return _tool_call(
                bash,
                {
                    "command": (
                        "pwd; echo '===== ESTRUTURA ====='; "
                        "find . -maxdepth 4 -type f "
                        f"{safe_excludes} | sort | head -200; "
                        "echo '===== CONTEUDO ====='; "
                        "for f in $(find . -maxdepth 4 -type f "
                        "\\( -name '*.py' -o -name '*.js' -o -name '*.ts' "
                        "-o -name '*.tsx' -o -name '*.jsx' -o -name '*.md' "
                        "-o -name '*.json' -o -name '*.txt' -o -name '*.html' "
                        "-o -name '*.css' -o -name '*.go' -o -name '*.rs' "
                        "-o -name '*.java' -o -name '*.sh' \\) "
                        f"{safe_excludes} | head -40); do echo \"===== $f =====\"; "
                        "sed -n '1,400p' \"$f\"; echo; done"
                    ),
                    "description": "Read entire project for analysis",
                },
            )
        # Pedido de inspeção direcionado: lista o diretório como ponto de partida
        if previous_count == 0 and ls_tool:
            return _tool_call(ls_tool, {"path": "."})

    if intent == "run" and bash:
        if "calculadora" in lowered or "calculator" in lowered:
            return _tool_call(
                bash,
                {
                    "command": "printf '10 + 5\\n' | python3 calculadora.py",
                    "description": "Run the calculator",
                },
            )
        if (
            "cronometro" in lowered
            or "cronômetro" in lowered
            or "stopwatch" in lowered
            or "timer" in lowered
        ):
            return _tool_call(
                bash,
                {
                    "command": (
                        "printf 'iniciar\\nstatus\\npausar\\nsair\\n' | "
                        "python3 cronometro.py"
                    ),
                    "description": "Run the stopwatch",
                },
            )
        return _tool_call(
            bash,
            {"command": "pwd && ls -la", "description": "Inspect current directory"},
        )

    return None


def _tool_call_to_textual_block(tool_call: Dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    name = str(function.get("name") or "").strip()
    if not name:
        return ""

    raw_arguments = function.get("arguments") or {}
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError:
            arguments = {"arguments": raw_arguments}
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = {"arguments": raw_arguments}

    params = []
    for key, value in arguments.items():
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, ensure_ascii=False)
        else:
            value_text = "" if value is None else str(value)
        params.append(f"<parameter={key}>{value_text}</parameter>")

    return f"<tool_call><function={name}>{''.join(params)}</function></tool_call>"


def _linearize_vllm_tool_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    linearized = []
    for message in messages:
        message_copy = dict(message)
        if message_copy.get("role") == "tool":
            tool_result = _content_to_vllm_text(message_copy.get("content", ""))
            tool_call_id = message_copy.get("tool_call_id") or "unknown"
            linearized.append(
                {
                    "role": "user",
                    "content": f"Tool result for {tool_call_id}:\n{tool_result}",
                }
            )
            continue

        tool_calls = message_copy.pop("tool_calls", None) or []
        if tool_calls:
            text = _content_to_vllm_text(message_copy.get("content", ""))
            textual_calls = [
                block
                for block in (_tool_call_to_textual_block(tc) for tc in tool_calls)
                if block
            ]
            message_copy["content"] = "\n".join(
                part for part in [text, *textual_calls] if part
            )

        linearized.append(message_copy)
    return linearized


def _inject_vllm_tool_use_prompt(
    openai_request: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if not tools:
        return
    prompt = _format_vllm_textual_tool_prompt(tools)
    messages = openai_request.setdefault("messages", [])
    if messages and messages[0].get("role") == "system":
        existing = _existing_system_text(messages[0].get("content"))
        if VLLM_TOOL_USE_SYSTEM_PROMPT not in existing:
            # Only append when the existing system is proxy-injected content only
            # (the boundary marker). If the user supplied a real system prompt
            # (no boundary marker = came from the original request), appending
            # more proxy instructions causes conflicts and makes the model ignore
            # the user's actual instructions.
            proxy_only = "Sensitive workspace boundary:" in existing and len(existing) < 600
            user_supplied = "Sensitive workspace boundary:" not in existing and len(existing) > 50
            if not user_supplied:
                messages[0]["content"] = f"{existing}\n\n{prompt}".strip()
        return
    messages.insert(0, {"role": "system", "content": prompt})


def _existing_system_text(content: Any) -> str:
    """Coerce a system message's content into a plain string for concatenation.

    Some clients send `content` as a list of blocks (multimodal-style: list of
    {"type":"text","text":...}). The naive f"{content}\\n\\n{prompt}" produces a
    str() of the list literal — pure garbage — which then becomes the system
    message. This helper joins text blocks and ignores non-text safely.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _inject_sensitive_workspace_boundary(openai_request: Dict[str, Any]) -> None:
    """Always present the secrets boundary, regardless of tool/intent heuristics."""
    messages = openai_request.setdefault("messages", [])
    marker = "Sensitive workspace boundary:"
    if messages and messages[0].get("role") == "system":
        existing = _existing_system_text(messages[0].get("content"))
        if marker not in existing:
            messages[0]["content"] = (
                f"{existing}\n\n{VLLM_SENSITIVE_WORKSPACE_PROMPT}".strip()
            )
        return
    messages.insert(
        0, {"role": "system", "content": VLLM_SENSITIVE_WORKSPACE_PROMPT}
    )


def _inject_native_agent_prompt(openai_request: Dict[str, Any]) -> None:
    """Append the behavior-only agentic prompt once (idempotent across multi-turn
    requests — Claude Code resends history, so we must not stack it every turn)."""
    marker = "autonomous coding agent operating inside an editor"
    messages = openai_request.setdefault("messages", [])
    if messages and messages[0].get("role") == "system":
        existing = _existing_system_text(messages[0].get("content"))
        if marker not in existing:
            messages[0]["content"] = f"{existing}\n\n{VLLM_NATIVE_AGENT_PROMPT}".strip()
        return
    messages.insert(0, {"role": "system", "content": VLLM_NATIVE_AGENT_PROMPT})


def _inject_workspace_path_prompt(openai_request: Dict[str, Any]) -> None:
    messages = openai_request.setdefault("messages", [])
    if messages and messages[0].get("role") == "system":
        existing = _existing_system_text(messages[0].get("content"))
        if VLLM_WORKSPACE_PATH_MARKER not in existing:
            messages[0]["content"] = f"{existing}\n\n{VLLM_WORKSPACE_PATH_PROMPT}".strip()
        return
    messages.insert(0, {"role": "system", "content": VLLM_WORKSPACE_PATH_PROMPT})


def _inject_native_tool_allowlist(
    openai_request: Dict[str, Any],
    tools: List[Dict[str, Any]],
) -> None:
    names = [
        str((tool.get("function") or {}).get("name") or "").strip()
        for tool in tools
    ]
    names = [name for name in names if name]
    if not names:
        return
    # Common synonym hints — when the model says "Search isn't available", it's
    # usually because it knows another agent that uses a different name. List the
    # canonical mapping explicitly so it tries the right one instead of giving up.
    hints = []
    name_set = {n.lower() for n in names}
    # Each entry: (aliases, list-of-canonical-names). Hint is emitted only if
    # ALL canonical names actually exist in the allowlist — otherwise we'd be
    # promising the model a tool that isn't there (e.g. "filesystem → Read/Write"
    # when only Read exists).
    synonyms = [
        (("search", "find"), ["Grep"]),
        (("listdir", "list", "dir"), ["LS"]),
        (("filesystem", "fs"), ["Read", "Write"]),
        (("shell", "terminal", "exec", "execute", "run"), ["Bash"]),
        (("editfile", "modify"), ["Edit"]),
        (("createfile", "newfile"), ["Write"]),
    ]
    for keys, targets in synonyms:
        if all(t.lower() in name_set for t in targets):
            hints.append(f"{'/'.join(keys)} → {'/'.join(targets)}")
    hint_text = (
        f" Common aliases for tools you might be tempted to invent: {'; '.join(hints)}."
        if hints else ""
    )
    prompt = (
        f"{VLLM_NATIVE_TOOL_ALLOWLIST_MARKER} {', '.join(names)}. "
        "Call only these exact tool names. If the user asks for something and "
        "you think a tool is missing, FIRST check this list — the tool you want "
        "is probably here under a different name. Never say 'I don't have that "
        "tool' without first trying the closest match from this list. Only ask "
        f"the user when no tool here can plausibly do the job.{hint_text}"
    )
    messages = openai_request.setdefault("messages", [])
    if messages and messages[0].get("role") == "system":
        existing = _existing_system_text(messages[0].get("content"))
        if VLLM_NATIVE_TOOL_ALLOWLIST_MARKER not in existing:
            messages[0]["content"] = f"{existing}\n\n{prompt}".strip()
        return
    messages.insert(0, {"role": "system", "content": prompt})


def _sanitize_openai_request_for_vllm(openai_request: Dict[str, Any]) -> None:
    # OpenAI-compatible local/vLLM servers commonly reject Anthropic-only
    # sampling/thinking fields. Keep the request strict for Claude Code.
    _inject_sensitive_workspace_boundary(openai_request)
    openai_request.pop("top_k", None)
    # vLLM não suporta reasoning_effort — remover sempre independente do valor
    openai_request.pop("reasoning_effort", None)
    # Thinking mode: defaults to ON now (was OFF). Customer feedback: turning
    # <think> reasoning on across the board makes the model noticeably smarter
    # at the cost of ~1.5x tokens — worth it for an editor agent. The legacy
    # /think on, /think off, /auto slash commands still override per-process
    # via the SQLite-backed _thinking_mode (see proxy_app.admin_db).
    _default_think = os.getenv("VLLM_DEFAULT_THINKING", "on").lower() in {
        "on", "1", "true", "yes"
    }
    try:
        import sys as _sys
        _pm = _sys.modules.get("proxy_app.main") or _sys.modules.get("main")
        _mode = getattr(_pm, "_thinking_mode", None) if _pm else None
    except Exception:
        _mode = None
    _is_opus = "opus" in openai_request.get("model", "").lower()
    if _mode == "on":
        _enable_think = True
    elif _mode == "off":
        _enable_think = False
    elif _mode == "auto":
        _enable_think = _is_opus
    else:
        # No explicit slash-command setting → follow the env-controlled default.
        _enable_think = _default_think
    extra_body = openai_request.setdefault("extra_body", {})
    chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
    # Respect the client's explicit enable_thinking if it already set one. Only
    # fill in our computed default when the field is absent. Previous version
    # always overwrote, ignoring the client's intent.
    if "enable_thinking" not in chat_template_kwargs:
        chat_template_kwargs["enable_thinking"] = _enable_think
    # frequency_penalty=0.2 was previously hardcoded here but hurts code quality:
    # variable names and keywords must repeat, and the penalty caused the model to
    # invent alternatives, hallucinate, and produce inconsistent output. Removed.
    # Set VLLM_FREQUENCY_PENALTY in env if you need a non-zero value.
    _fp = os.getenv("VLLM_FREQUENCY_PENALTY")
    if _fp is not None:
        try:
            openai_request.setdefault("frequency_penalty", float(_fp))
        except ValueError:
            pass
    max_output_tokens = _vllm_max_output_tokens()
    requested_max_tokens = openai_request.get("max_tokens")
    # Floor when tools are present: Qwen3's <think> reasoning is emitted BEFORE the
    # tool call, so a low max_tokens makes the model "read, understand, then stop"
    # (finish_reason=length) before it ever emits the action. Guarantee enough room
    # for reasoning + the tool call. Configurable via VLLM_TOOL_MIN_TOKENS.
    if openai_request.get("tools"):
        try:
            tool_floor = int(os.getenv("VLLM_TOOL_MIN_TOKENS", "4096"))
        except ValueError:
            tool_floor = 4096
        tool_floor = min(tool_floor, max_output_tokens)  # never exceed the hard cap
        if not requested_max_tokens or requested_max_tokens < tool_floor:
            openai_request["max_tokens"] = tool_floor
            requested_max_tokens = tool_floor
    if requested_max_tokens and requested_max_tokens > max_output_tokens:
        openai_request["max_tokens"] = max_output_tokens

    openai_request["messages"] = _strip_vllm_rejected_fields(
        openai_request.get("messages", [])
    )
    openai_request["messages"] = _neutralize_unsupported_media_for_vllm(
        openai_request.get("messages", [])
    )

    native_tools_enabled = _vllm_native_tools_enabled()
    tools = openai_request.get("tools")
    if tools:
        tools = _strip_vllm_rejected_fields(tools)
        tools = _compact_tools_for_vllm(tools)
        openai_request["_vllm_allowed_tool_names"] = [
            str((tool.get("function") or {}).get("name") or "")
            for tool in tools
            if (tool.get("function") or {}).get("name")
        ]
        if not tools:
            openai_request.pop("tools", None)
            openai_request.pop("tool_choice", None)
        elif native_tools_enabled:
            # NATIVE mode: the vLLM server already emits real OpenAI tool_calls
            # (started with --enable-auto-tool-choice --tool-call-parser). Pass
            # the tools through CLEANLY and inject NOTHING else. Injecting the
            # textual tool prompt here (the old behavior) taught the model to ALSO
            # emit <tool_call> text, so it flip-flopped between native and textual
            # formats — the root cause of the intermittent "doesn't edit / acts as
            # chat / hallucinates" reports. The Portuguese-keyword intent heuristic
            # is likewise NOT run in native mode (it guessed wrong and pushed the
            # model toward the wrong action). Only attach the forced-tool fallback
            # if the operator explicitly opted in.
            openai_request["tools"] = tools
            # Behavior-only nudge so the model acts instead of asking ("intern
            # mode"). No output-format text here — that caused the flip-flop.
            # Opt-out via VLLM_NATIVE_AGENT_PROMPT=off if ever undesired.
            if os.getenv("VLLM_NATIVE_AGENT_PROMPT", "on").lower() not in {"off", "0", "false", "no"}:
                _inject_native_agent_prompt(openai_request)
            _inject_workspace_path_prompt(openai_request)
            _inject_native_tool_allowlist(openai_request, tools)
            if _force_tool_fallback_enabled():
                _attach_mandatory_tool_fallback(openai_request, tools)
            if openai_request.get("tool_choice") not in (None, "auto"):
                # vLLM's OpenAI server is picky about forced/required tool choice,
                # while Claude Code still works with auto tool selection.
                openai_request["tool_choice"] = "auto"
        else:
            # FALLBACK mode (no native tool parser on the vLLM server): teach the
            # model the textual tool-call format and linearize tool messages.
            _inject_workspace_path_prompt(openai_request)
            _inject_vllm_tool_use_prompt(openai_request, tools)
            # Inject the behavior-only agent prompt here TOO. Previous version
            # only did this in native mode and the fallback could fall back to
            # "intern mode" silently. The behavior prompt has zero output-format
            # text so it doesn't conflict with the textual-tool injection.
            if os.getenv("VLLM_NATIVE_AGENT_PROMPT", "on").lower() not in {"off", "0", "false", "no"}:
                _inject_native_agent_prompt(openai_request)
            if _force_tool_fallback_enabled():
                _inject_vllm_mandatory_tool_instruction(openai_request, tools)
            openai_request.pop("tools", None)
            openai_request.pop("tool_choice", None)
            openai_request["messages"] = _linearize_vllm_tool_messages(
                openai_request.get("messages", [])
            )
    elif not native_tools_enabled:
        openai_request["messages"] = _linearize_vllm_tool_messages(
            openai_request.get("messages", [])
        )

    openai_request["messages"] = _truncate_messages_for_vllm(
        openai_request.get("messages", [])
    )


def _is_complete_tool_json(tool_use_block: dict) -> bool:
    """Return True if a tool_use's input is a sane, fully-parsed dict.

    A truncated tool_use (cut by max_tokens) has `input` either as a string
    (failed parse fallback) or missing keys. If the input came back as a real
    dict (even empty) the call was structurally OK and worth executing."""
    inp = tool_use_block.get("input")
    return isinstance(inp, dict)


def openai_to_anthropic_response(
    openai_response: dict,
    original_model: str,
    tool_name_mapping: Optional[Dict[str, str]] = None,
    allowed_tool_names: Optional[set[str]] = None,
) -> dict:
    """
    Convert OpenAI chat completion response to Anthropic Messages format.

    Args:
        openai_response: Response from OpenAI-compatible API
        original_model: The model name requested by the client

    Returns:
        Response in Anthropic Messages format
    """
    # vLLM occasionally returns {"choices": [], "error": ...} on partial failures.
    # `choices[0]` would IndexError and crash the whole request. Empty/missing
    # choices → degrade to a graceful empty Anthropic response with end_turn.
    choices = openai_response.get("choices") or [{}]
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    usage = openai_response.get("usage") or {}

    # Build content blocks
    content_blocks = []

    # Add thinking content block if reasoning_content is present
    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        thinking_signature = message.get("thinking_signature", "")
        signature = (
            thinking_signature
            if thinking_signature
            and len(thinking_signature) >= MIN_THINKING_SIGNATURE_LENGTH
            else ""
        )
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": reasoning_content,
                "signature": signature,
            }
        )

    # Add text content if present
    text_content = message.get("content")
    textual_tool_blocks = []
    if text_content:
        text_content = _sanitize_vllm_response_text(text_content)
        text_content, textual_tool_blocks = _parse_textual_tool_calls(text_content)
        for block in textual_tool_blocks:
            if isinstance(block, dict):
                block["name"] = _restore_tool_name(
                    str(block.get("name", "")),
                    tool_name_mapping,
                )
        textual_tool_blocks = [
            block
            for block in textual_tool_blocks
            if allowed_tool_names is None or block.get("name") in allowed_tool_names
        ]
        if text_content:
            content_blocks.append({"type": "text", "text": text_content})

    # Add tool use blocks if present
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_data = {}

        restored_name = _restore_tool_name(
            str(func.get("name", "")),
            tool_name_mapping,
        )
        input_data = _normalize_tool_input_for_anthropic(restored_name, input_data)
        if allowed_tool_names is not None and restored_name not in allowed_tool_names:
            continue
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": restored_name,
                "input": input_data,
            }
        )

    if textual_tool_blocks:
        content_blocks.extend(textual_tool_blocks)

    # Map finish_reason to stop_reason
    finish_reason = choice.get("finish_reason", "end_turn")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")
    has_tool_use = any(block.get("type") == "tool_use" for block in content_blocks)
    if finish_reason in {"tool_calls", "function_call"} and not has_tool_use:
        stop_reason = "end_turn"
    if textual_tool_blocks:
        stop_reason = "tool_use"
    # finish_reason=length WITH a tool_use: only signal tool_use if the tool's
    # JSON arguments are actually complete. A truncated tool_use has broken
    # arguments and Claude Code would retry it forever (the edit-same-file
    # loop). When args are valid JSON we trust the call and emit tool_use;
    # otherwise stay on max_tokens so the client surfaces a real error.
    # Mirrors the streaming-side logic in anthropic_streaming_wrapper.
    if finish_reason == "length" and has_tool_use:
        args_complete = all(
            _is_complete_tool_json(b)
            for b in content_blocks
            if b.get("type") == "tool_use"
        )
        if args_complete:
            stop_reason = "tool_use"

    # Build usage
    # Note: Google's promptTokenCount INCLUDES cached tokens, but Anthropic's
    # input_tokens EXCLUDES cached tokens. We need to subtract cached tokens.
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached_tokens = 0

    # Extract cached tokens if present
    if usage.get("prompt_tokens_details"):
        details = usage["prompt_tokens_details"]
        cached_tokens = details.get("cached_tokens", 0)

    anthropic_usage = {
        "input_tokens": prompt_tokens - cached_tokens,  # Subtract cached tokens
        "output_tokens": usage.get("completion_tokens", 0),
    }

    # Add cache tokens if present
    if cached_tokens > 0:
        anthropic_usage["cache_read_input_tokens"] = cached_tokens
        anthropic_usage["cache_creation_input_tokens"] = 0

    return {
        "id": openai_response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": original_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
    }


def translate_anthropic_request(request: AnthropicMessagesRequest) -> Dict[str, Any]:
    """
    Translate a complete Anthropic Messages API request to OpenAI format.

    This is a high-level function that handles all aspects of request translation,
    including messages, tools, tool_choice, and thinking configuration.

    Args:
        request: An AnthropicMessagesRequest object

    Returns:
        Dictionary containing the OpenAI-compatible request parameters
    """
    anthropic_request = request.model_dump(exclude_none=True)

    messages = anthropic_request.get("messages", [])
    tool_name_mapping: Dict[str, str] = {}
    openai_messages = anthropic_to_openai_messages(
        messages,
        anthropic_request.get("system"),
        tool_name_mapping=tool_name_mapping,
    )

    openai_tools = anthropic_to_openai_tools(
        anthropic_request.get("tools"),
        tool_name_mapping=tool_name_mapping,
    )
    openai_tool_choice = anthropic_to_openai_tool_choice(
        anthropic_request.get("tool_choice"),
        tool_name_mapping=tool_name_mapping,
    )

    # Build OpenAI-compatible request
    openai_request = {
        "model": request.model,
        "messages": openai_messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream or False,
    }

    if request.temperature is not None:
        openai_request["temperature"] = request.temperature
    if request.top_p is not None:
        openai_request["top_p"] = request.top_p
    if request.top_k is not None:
        openai_request["top_k"] = request.top_k
    if request.stop_sequences:
        openai_request["stop"] = request.stop_sequences
    if openai_tools:
        openai_request["tools"] = openai_tools
    if openai_tool_choice:
        openai_request["tool_choice"] = openai_tool_choice
    if tool_name_mapping:
        openai_request["_anthropic_tool_name_mapping"] = tool_name_mapping

    # Note: request.metadata is intentionally not mapped.
    # OpenAI's API doesn't have an equivalent field for client-side metadata.
    # The metadata is typically used by Anthropic clients for tracking purposes
    # and doesn't affect the model's behavior.

    # Handle Anthropic thinking config -> reasoning_effort translation
    # Only set reasoning_effort if thinking is explicitly configured
    if request.thinking:
        if request.thinking.type == "enabled":
            # Only set reasoning_effort if budget_tokens was specified
            if request.thinking.budget_tokens is not None:
                openai_request["reasoning_effort"] = _budget_to_reasoning_effort(
                    request.thinking.budget_tokens, request.model
                )
            # If thinking enabled but no budget specified, don't set anything
            # Let the provider decide the default
        elif request.thinking.type == "disabled":
            openai_request["reasoning_effort"] = "disable"

    if openai_tools and _force_tool_fallback_enabled():
        _attach_mandatory_tool_fallback(openai_request, openai_tools)

    provider = request.model.split("/", 1)[0].lower() if "/" in request.model else ""
    if provider in {"hosted_vllm", "vllm", "lm_studio", "ollama"}:
        _sanitize_openai_request_for_vllm(openai_request)

    return openai_request
