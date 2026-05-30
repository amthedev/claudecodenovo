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
from typing import Any, Dict, List, Optional, Union

from .models import AnthropicMessagesRequest

MIN_THINKING_SIGNATURE_LENGTH = 100

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

# Hosted vLLM defaults are intentionally conservative because Claude Desktop
# and Claude Code assume much larger Claude-native limits than local/vLLM
# deployments usually have.
VLLM_MAX_OUTPUT_TOKENS = 4096
VLLM_MAX_INPUT_CHARS = 36000
VLLM_MAX_MESSAGE_CHARS = 12000
VLLM_MAX_TOOL_RESULT_CHARS = 6000
VLLM_MAX_RESPONSE_TEXT_CHARS = 5000
VLLM_MAX_TOOLS = 16
VLLM_TOOL_USE_SYSTEM_PROMPT = (
    "You are running inside Claude Code. When the user asks to create, edit, "
    "inspect, or run project files or commands, call the available tools "
    "instead of only explaining or pasting code. If the request is in Portuguese, "
    "such as 'faca uma calculadora em python', create or edit the file directly. "
    "For file creation or edits, "
    "prefer Create, Update, Write, Edit, or MultiEdit over shell heredocs like "
    "`cat > file` or `touch file`. If Create/Update requires reading the file "
    "first, read it and then retry the file edit tool. Do not abandon the edit "
    "and do not run an older existing file instead. Use Bash for running "
    "commands only after files are written. Never run programs that wait for "
    "interactive input unless you pipe input, pass arguments, or use a short "
    "timeout; avoid commands that can print endlessly. For Python scripts with "
    "input(), test with piped input such as `printf '1\\n2\\n3\\n' | python3 "
    "file.py`, not plain `python3 file.py`. Do not tell the user to copy code "
    "when a file operation is needed. Answer concisely. Do not repeat the same "
    "question, instruction, status, or conclusion in different words. Never "
    "reveal hidden reasoning, chain-of-thought, policy checks, or planning "
    "notes. Return only the final answer or tool call. Once the answer is "
    "complete, stop generating."
)
VLLM_TEXTUAL_TOOL_PROMPT = (
    "The upstream vLLM server may not support native OpenAI tool calling. "
    "When you need a tool, output exactly one tool call using this format and "
    "no extra prose:\n"
    "<tool_call><function=ToolName><parameter=param_name>value</parameter></function></tool_call>\n"
    "Use only tool names from the available tools list."
)
VLLM_MANDATORY_TOOL_MARKER = "CURRENT REQUEST REQUIRES TOOL USE"
VLLM_AGENT_FLOW_PROMPT = (
    "Agent workflow contract:\n"
    "1. Keep working until the user's task is actually complete; do not stop "
    "after a partial edit or first command result.\n"
    "2. For multi-step work, call TodoWrite when available, then keep statuses "
    "current as you inspect, edit, run, and verify.\n"
    "3. Before using a file path, verify it from the current workspace with LS, "
    "Glob, Grep, Read, pwd, or rg --files when uncertain. Never invent paths.\n"
    "4. If a command fails because of path, cwd, missing file, syntax, or usage, "
    "inspect the error and retry with a corrected command instead of giving up.\n"
    "5. After creating or editing code, run a relevant verification command "
    "such as py_compile, unit tests, lint, or a small smoke test. If verification "
    "cannot run, state the concrete blocker in the final answer.\n"
    "6. Do not give a final answer until edits and verification are complete. "
    "The final answer must briefly list files changed and verification results."
)
VLLM_MANDATORY_TOOL_PROMPT = (
    f"{VLLM_MANDATORY_TOOL_MARKER}: Do not answer with prose, examples, code "
    "blocks, or permission questions. Your next output must be exactly one "
    "tool call in the textual tool-call format. You already have permission "
    "to inspect, edit, create, and run project files when the user asks for it. "
    "After tool results come back, continue with more tool calls when needed; "
    "only give the final answer after the task is actually done.\n"
    f"{VLLM_AGENT_FLOW_PROMPT}"
)
VLLM_CREATE_FILE_TOOL_PROMPT = (
    "This is a create/edit request. Use a file editing tool such as Write, "
    "Create, Edit, Update, or MultiEdit. If the user asks for a Python "
    "calculator and no path is specified, create or update calculadora.py. "
    "Then run it or compile it before finalizing."
)
VLLM_INSPECT_PROJECT_TOOL_PROMPT = (
    "This is a project inspection request. Start by inspecting the current "
    "directory with LS, Glob, Read, Grep, or Bash; do not ask the user to "
    "provide the project structure. Read the main docs/config files and finish "
    "with a concise project report."
)
VLLM_RUN_COMMAND_TOOL_PROMPT = (
    "This is a run/test request. Use Bash to execute the relevant command "
    "instead of explaining how the user can run it. If the command fails, "
    "inspect and retry when the fix is obvious."
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


_TEXTUAL_TOOL_CALL_RE = re.compile(
    r"(?:<tool_call>\s*)?<function=([A-Za-z0-9_.:-]+)>(.*?</function>)(?:\s*</tool_call>)?",
    re.DOTALL,
)
_TEXTUAL_TOOL_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.:-]+)>\s*(.*?)(?=(?:</parameter>\s*)?<parameter=|</function>)",
    re.DOTALL,
)
_THINK_TAG_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_REASONING_PREAMBLE_MARKERS = (
    "i need to ",
    "i should ",
    "i'll ",
    "i will ",
    "let me ",
    "since the user ",
    "since they ",
    "the user ",
    "system reminders",
    "simple response",
    "that should ",
    "allowed scope",
    "no need for that",
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
    "edite",
    "editar",
    "altere",
    "modifique",
    "corrija",
    "conserte",
    "salve",
    "write ",
    "create ",
    "edit ",
    "modify ",
    "fix ",
)
_PROJECT_INSPECTION_MARKERS = (
    "analise",
    "analisa",
    "analisar",
    "veja oq",
    "veja o que",
    "o que e",
    "oq e",
    "projeto",
    "repo",
    "repositorio",
    "repository",
    "estrutura",
    "codebase",
)
_RUN_COMMAND_MARKERS = (
    "rode",
    "rodar",
    "execute",
    "executar",
    "testa",
    "teste",
    "run ",
)


def _parse_textual_tool_calls(text: str) -> tuple[str, List[dict]]:
    """
    Convert common text-emitted tool call markup into Anthropic tool_use blocks.

    Some OpenAI-compatible coding models emit Claude-style tool calls as text,
    for example:
        <tool_call><function=Bash><parameter=command>ls</parameter></function></tool_call>
    Claude Code only executes real Anthropic tool_use blocks, so normalize here.
    """
    if not text or "<function=" not in text:
        return text, []

    tool_blocks: List[dict] = []

    def replace_tool_call(match: re.Match) -> str:
        tool_name = match.group(1).strip()
        body = match.group(2)
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

        if tool_name and tool_input:
            tool_blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
        return ""

    cleaned_text = _TEXTUAL_TOOL_CALL_RE.sub(replace_tool_call, text).strip()
    return cleaned_text, tool_blocks


def _vllm_response_language_instruction() -> str:
    language = os.getenv("PROXY_RESPONSE_LANGUAGE", "pt-BR").strip()
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
    anthropic_messages: List[dict], system: Optional[Union[str, List[dict]]] = None
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
                        openai_content.append(
                            {"type": "text", "text": block.get("text", "")}
                        )
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
                        tool_calls.append(
                            {
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
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
    for tool in anthropic_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return openai_tools


def anthropic_to_openai_tool_choice(
    anthropic_tool_choice: Optional[dict],
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
        return {
            "type": "function",
            "function": {"name": anthropic_tool_choice.get("name", "")},
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
    return _env_int(
        ["HOSTED_VLLM_MAX_INPUT_CHARS", "VLLM_MAX_INPUT_CHARS"],
        VLLM_MAX_INPUT_CHARS,
        minimum=1000,
    )


def _message_char_size(message: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(message, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(message))


def _compact_text_middle(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    head_chars = max(1, max_chars // 2)
    tail_chars = max(1, max_chars - head_chars)
    omitted = len(text) - head_chars - tail_chars
    return (
        f"{text[:head_chars]}\n\n"
        f"[... {omitted} characters omitted from {label} to fit hosted vLLM context ...]\n\n"
        f"{text[-tail_chars:]}"
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
    compacted = []
    for message in messages:
        message_copy = dict(message)
        role = message_copy.get("role")
        if role == "tool":
            message_copy["content"] = _compact_content_for_vllm(
                message_copy.get("content", ""),
                VLLM_MAX_TOOL_RESULT_CHARS,
                "tool output",
            )
        elif _message_char_size(message_copy) > VLLM_MAX_MESSAGE_CHARS:
            message_copy["content"] = _compact_content_for_vllm(
                message_copy.get("content", ""),
                VLLM_MAX_MESSAGE_CHARS,
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


def _truncate_messages_for_vllm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = _compact_large_messages_for_vllm(messages)
    max_chars = _vllm_max_input_chars()
    if sum(_message_char_size(message) for message in messages) <= max_chars:
        return messages

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

    return [*system_messages, *kept]


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


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


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
        existing = messages[0].get("content") or ""
        if VLLM_MANDATORY_TOOL_MARKER not in existing:
            messages[0]["content"] = f"{existing}\n\n{instruction}".strip()
        return
    messages.insert(0, {"role": "system", "content": instruction})


def _inject_vllm_mandatory_tool_instruction(
    openai_request: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
) -> None:
    if not tools:
        return

    intent = _classify_tool_intent(
        _latest_user_text(openai_request.get("messages", []))
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


def _calculator_source() -> str:
    return (
        "#!/usr/bin/env python3\n"
        "\"\"\"Calculadora simples de terminal.\"\"\"\n\n"
        "def calcular(a, operador, b):\n"
        "    if operador == '+':\n"
        "        return a + b\n"
        "    if operador == '-':\n"
        "        return a - b\n"
        "    if operador == '*':\n"
        "        return a * b\n"
        "    if operador == '/':\n"
        "        if b == 0:\n"
        "            raise ZeroDivisionError('divisao por zero')\n"
        "        return a / b\n"
        "    raise ValueError('operador invalido')\n\n"
        "def main():\n"
        "    print('Calculadora Python')\n"
        "    print('Exemplo: 10 + 5')\n"
        "    entrada = input('Digite a conta: ').strip().split()\n"
        "    if len(entrada) != 3:\n"
        "        print('Formato invalido')\n"
        "        return\n"
        "    a = float(entrada[0])\n"
        "    operador = entrada[1]\n"
        "    b = float(entrada[2])\n"
        "    print(calcular(a, operador, b))\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def _build_vllm_forced_tool_call(
    intent: str,
    tools: List[Dict[str, Any]],
    openai_request: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    previous_count = int(openai_request.get("_vllm_previous_tool_count") or 0)
    user_text = _latest_user_text(openai_request.get("messages", []))
    lowered = user_text.lower()

    bash = _available_tool_name(tools, ("Bash", "bash"))
    ls_tool = _available_tool_name(tools, ("LS", "ls", "List"))
    write = _available_tool_name(tools, ("Write", "Create", "Update", "write"))

    if intent == "create":
        is_calculator = "calculadora" in lowered or "calculator" in lowered
        if is_calculator and previous_count == 0:
            if write:
                return _tool_call(
                    write,
                    {"file_path": "calculadora.py", "content": _calculator_source()},
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
        if previous_count == 0 and ls_tool:
            return _tool_call(ls_tool, {"path": "."})

    if intent == "inspect":
        if previous_count == 0 and ls_tool:
            return _tool_call(ls_tool, {"path": "."})
        if previous_count <= 1 and bash:
            return _tool_call(
                bash,
                {
                    "command": (
                        "pwd && find . -maxdepth 3 -type f | sort | head -120 && "
                        "test -f README.md && sed -n '1,220p' README.md || true"
                    ),
                    "description": "Inspect project structure and README",
                },
            )

    if intent == "run" and bash:
        if "calculadora" in lowered or "calculator" in lowered:
            return _tool_call(
                bash,
                {
                    "command": "printf '10 + 5\\n' | python3 calculadora.py",
                    "description": "Run the calculator",
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
        existing = messages[0].get("content") or ""
        if VLLM_TOOL_USE_SYSTEM_PROMPT not in existing:
            messages[0]["content"] = f"{existing}\n\n{prompt}".strip()
        return
    messages.insert(0, {"role": "system", "content": prompt})


def _sanitize_openai_request_for_vllm(openai_request: Dict[str, Any]) -> None:
    # OpenAI-compatible local/vLLM servers commonly reject Anthropic-only
    # sampling/thinking fields. Keep the request strict for Claude Code.
    openai_request.pop("top_k", None)
    # vLLM não suporta reasoning_effort — remover sempre independente do valor
    openai_request.pop("reasoning_effort", None)
    openai_request.setdefault("frequency_penalty", 0.2)
    max_output_tokens = _vllm_max_output_tokens()
    requested_max_tokens = openai_request.get("max_tokens")
    if requested_max_tokens and requested_max_tokens > max_output_tokens:
        openai_request["max_tokens"] = max_output_tokens

    openai_request["messages"] = _strip_vllm_rejected_fields(
        openai_request.get("messages", [])
    )

    native_tools_enabled = _vllm_native_tools_enabled()
    tools = openai_request.get("tools")
    if tools:
        tools = _strip_vllm_rejected_fields(tools)
        tools = _compact_tools_for_vllm(tools)
        if not tools:
            openai_request.pop("tools", None)
            openai_request.pop("tool_choice", None)
        elif native_tools_enabled:
            openai_request["tools"] = tools
            _inject_vllm_tool_use_prompt(openai_request, tools)
            _inject_vllm_mandatory_tool_instruction(openai_request, tools)
            if openai_request.get("tool_choice") not in (None, "auto"):
                # vLLM's OpenAI server is picky about forced/required tool choice,
                # while Claude Code still works with auto tool selection.
                openai_request["tool_choice"] = "auto"
        else:
            _inject_vllm_tool_use_prompt(openai_request, tools)
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


def openai_to_anthropic_response(openai_response: dict, original_model: str) -> dict:
    """
    Convert OpenAI chat completion response to Anthropic Messages format.

    Args:
        openai_response: Response from OpenAI-compatible API
        original_model: The model name requested by the client

    Returns:
        Response in Anthropic Messages format
    """
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = openai_response.get("usage", {})

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

        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": func.get("name", ""),
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
    if textual_tool_blocks:
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
    openai_messages = anthropic_to_openai_messages(
        messages, anthropic_request.get("system")
    )

    openai_tools = anthropic_to_openai_tools(anthropic_request.get("tools"))
    openai_tool_choice = anthropic_to_openai_tool_choice(
        anthropic_request.get("tool_choice")
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

    provider = request.model.split("/", 1)[0].lower() if "/" in request.model else ""
    if provider in {"hosted_vllm", "vllm", "lm_studio", "ollama"}:
        _sanitize_openai_request_for_vllm(openai_request)

    return openai_request
