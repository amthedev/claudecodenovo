# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Streaming wrapper for converting OpenAI streaming format to Anthropic streaming format.

This module provides a framework-agnostic streaming wrapper that converts
OpenAI SSE (Server-Sent Events) format to Anthropic's streaming format.
"""

import json
import logging
import re
import uuid
from typing import AsyncGenerator, Callable, Optional, Awaitable, Any, TYPE_CHECKING

from .translator import (
    _normalize_tool_arguments_for_anthropic,
    _normalize_tool_input_for_anthropic,
    _parse_textual_tool_calls,
    _restore_tool_name,
)

if TYPE_CHECKING:
    from ..transaction_logger import TransactionLogger

logger = logging.getLogger("rotator_library.anthropic_compat")

_FENCED_CODE_RE = re.compile(
    r"```(?:[A-Za-z0-9_+.-]+)?\s*\n(.*?)```",
    re.DOTALL,
)
_FILE_PATH_RE = re.compile(
    r"([A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|jsx|html|css|go|rs))"
)
_CODE_START_RE = re.compile(
    r"(?m)^(?:#![^\n]*\n)?\s*(?:import |from |class |def |async def |if __name__|print\(|[A-Za-z_][A-Za-z0-9_]*\s*=)"
)


def _extract_file_path_from_text(text: str, fallback: str) -> str:
    match = _FILE_PATH_RE.search(text or "")
    return match.group(1) if match else fallback


def _strip_leaked_reasoning(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text or "")
    text = re.sub(
        r"(?is)^\s*okay,.*?(?=\n\s*(?:vou|i will|i'll|import |from |class |def |```))",
        "",
        text,
    )
    return text.strip()


def _extract_code_from_model_text(text: str) -> str:
    text = _strip_leaked_reasoning(text)
    fenced = _FENCED_CODE_RE.findall(text)
    if fenced:
        return max((block.strip() for block in fenced), key=len, default="")

    match = _CODE_START_RE.search(text)
    if not match:
        return text.strip()

    code = text[match.start() :].strip()
    trailing_markers = (
        "\n\nO arquivo ",
        "\n\nA calculadora ",
        "\n\nO cron",
        "\n\nO jogo ",
        "\n\nVocê pode ",
        "\n\nPara executar",
        "\n\nSe quiser",
        "\n\nThe file ",
    )
    cut_at = len(code)
    for marker in trailing_markers:
        pos = code.find(marker)
        if pos >= 0:
            cut_at = min(cut_at, pos)
    return code[:cut_at].strip()


def _model_text_to_write_tool_call(
    forced_tool_call: dict,
    model_text: str,
) -> dict:
    file_path = _extract_file_path_from_text(
        model_text, forced_tool_call.get("_proxy_file_path_hint") or "script.py"
    )
    content = _extract_code_from_model_text(model_text)
    return {
        "id": forced_tool_call.get("id", f"toolu_proxy_{uuid.uuid4().hex[:12]}"),
        "name": forced_tool_call.get("name", ""),
        "input": {"file_path": file_path, "content": content},
    }


async def anthropic_response_to_streaming_events(
    anthropic_response: dict,
) -> AsyncGenerator[str, None]:
    """
    Emit Anthropic SSE events from an already-complete Anthropic response.

    This is useful for OpenAI-compatible backends that reject streaming requests
    but work normally with non-streaming chat completions.
    """
    message = dict(anthropic_response)
    content_blocks = message.get("content") or []
    usage = message.get("usage") or {"input_tokens": 0, "output_tokens": 0}

    start_message = dict(message)
    start_message["content"] = []
    start_message["stop_reason"] = None
    start_message["stop_sequence"] = None
    start_message["usage"] = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": 0,
    }
    if usage.get("cache_read_input_tokens"):
        start_message["usage"]["cache_read_input_tokens"] = usage.get(
            "cache_read_input_tokens", 0
        )
        start_message["usage"]["cache_creation_input_tokens"] = usage.get(
            "cache_creation_input_tokens", 0
        )

    yield (
        "event: message_start\n"
        f"data: {json.dumps({'type': 'message_start', 'message': start_message})}\n\n"
    )

    for index, block in enumerate(content_blocks):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            yield (
                "event: content_block_start\n"
                f"data: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            )
            text = block.get("text", "")
            if text:
                yield (
                    "event: content_block_delta\n"
                    f"data: {json.dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                )
            yield (
                "event: content_block_stop\n"
                f"data: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"
            )
        elif block_type == "tool_use":
            tool_block = {
                "type": "tool_use",
                "id": block.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": block.get("name", ""),
                "input": {},
            }
            yield (
                "event: content_block_start\n"
                f"data: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': tool_block})}\n\n"
            )
            tool_input = block.get("input") or {}
            if tool_input:
                yield (
                    "event: content_block_delta\n"
                    f"data: {json.dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(tool_input)}})}\n\n"
                )
            yield (
                "event: content_block_stop\n"
                f"data: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"
            )

    final_usage = {"output_tokens": usage.get("output_tokens", 0)}
    if usage.get("cache_read_input_tokens"):
        final_usage["cache_read_input_tokens"] = usage.get("cache_read_input_tokens", 0)
        final_usage["cache_creation_input_tokens"] = usage.get(
            "cache_creation_input_tokens", 0
        )
    message_delta = {
        "type": "message_delta",
        "delta": {
            "stop_reason": message.get("stop_reason") or "end_turn",
            "stop_sequence": message.get("stop_sequence"),
        },
        "usage": final_usage,
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
    yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


async def anthropic_streaming_wrapper(
    openai_stream: AsyncGenerator[str, None],
    original_model: str,
    request_id: Optional[str] = None,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    transaction_logger: Optional["TransactionLogger"] = None,
    forced_tool_call: Optional[dict] = None,
    tool_name_mapping: Optional[dict] = None,
    allowed_tool_names: Optional[set[str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert OpenAI streaming format to Anthropic streaming format.

    This is a framework-agnostic wrapper that can be used with any async web framework.
    Instead of taking a FastAPI Request object, it accepts an optional callback function
    to check for client disconnection.

    Anthropic SSE events:
    - message_start: Initial message metadata
    - content_block_start: Start of a content block
    - content_block_delta: Content chunk
    - content_block_stop: End of a content block
    - message_delta: Final message metadata (stop_reason, usage)
    - message_stop: End of message

    Args:
        openai_stream: AsyncGenerator yielding OpenAI SSE format strings
        original_model: The model name to include in responses
        request_id: Optional request ID (auto-generated if not provided)
        is_disconnected: Optional async callback that returns True if client disconnected
        transaction_logger: Optional TransactionLogger for logging the final Anthropic response
        forced_tool_call: Optional fallback tool call to emit if the model tries
            to end a mandatory agent step without using a tool

    Yields:
        SSE format strings in Anthropic's streaming format
    """
    if request_id is None:
        request_id = f"msg_{uuid.uuid4().hex[:24]}"

    message_started = False
    content_block_started = False
    thinking_block_started = False
    current_block_index = 0
    tool_calls_by_index = {}  # Track tool calls by their index
    tool_block_indices = {}  # Track which block index each tool call uses
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0  # Track cached tokens for proper Anthropic format
    accumulated_text = ""  # Track accumulated text for logging
    accumulated_thinking = ""  # Track accumulated thinking for logging
    textual_tool_buffer = ""  # Buffer text-emitted tool calls until complete
    delayed_model_text_buffer = ""
    stop_reason_final = "end_turn"  # Track final stop reason for logging
    provider_finish_reason = None
    write_from_model_text = bool(
        forced_tool_call
        and forced_tool_call.get("_proxy_strategy") == "write_from_model_text"
    )

    try:
        async for chunk_str in openai_stream:
            # Check for client disconnection if callback provided
            if is_disconnected is not None and await is_disconnected():
                break

            if not chunk_str.strip() or not chunk_str.startswith("data:"):
                continue

            data_content = chunk_str[len("data:") :].strip()
            if data_content == "[DONE]":
                # CRITICAL: Send message_start if we haven't yet (e.g., empty response)
                # Claude Code and other clients require message_start before message_stop
                if not message_started:
                    # Build usage with cached tokens properly handled
                    usage_dict = {
                        "input_tokens": input_tokens - cached_tokens,
                        "output_tokens": 0,
                    }
                    if cached_tokens > 0:
                        usage_dict["cache_read_input_tokens"] = cached_tokens
                        usage_dict["cache_creation_input_tokens"] = 0

                    message_start = {
                        "type": "message_start",
                        "message": {
                            "id": request_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": original_model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": usage_dict,
                        },
                    }
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                    message_started = True

                # Close any open thinking block
                if thinking_block_started:
                    yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                    current_block_index += 1
                    thinking_block_started = False

                # Close any open text block
                if content_block_started:
                    yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                    current_block_index += 1
                    content_block_started = False

                model_text_buffer = delayed_model_text_buffer + textual_tool_buffer
                if model_text_buffer:
                    cleaned_text, textual_tool_blocks = _parse_textual_tool_calls(
                        model_text_buffer
                    )
                    delayed_model_text_buffer = ""
                    textual_tool_buffer = ""

                    if write_from_model_text and not textual_tool_blocks:
                        forced_tool_call = _model_text_to_write_tool_call(
                            forced_tool_call or {}, cleaned_text
                        )
                        cleaned_text = ""

                    if cleaned_text and not write_from_model_text:
                        block_start = {
                            "type": "content_block_start",
                            "index": current_block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                        block_delta = {
                            "type": "content_block_delta",
                            "index": current_block_index,
                            "delta": {"type": "text_delta", "text": cleaned_text},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"
                        yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                        accumulated_text += cleaned_text
                        current_block_index += 1

                    for block in textual_tool_blocks:
                        tc_index = len(tool_calls_by_index)
                        block_idx = current_block_index
                        tool_name = _restore_tool_name(
                            str(block.get("name", "")),
                            tool_name_mapping,
                        )
                        tool_input = _normalize_tool_input_for_anthropic(
                            tool_name,
                            block.get("input") or {},
                        )
                        arguments = json.dumps(tool_input)
                        if (
                            allowed_tool_names is not None
                            and tool_name not in allowed_tool_names
                        ):
                            logger.warning(
                                "Ignoring unavailable textual tool emitted by model: %s",
                                tool_name,
                            )
                            continue
                        tool_calls_by_index[tc_index] = {
                            "id": block.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": tool_name,
                            "arguments": arguments,
                        }
                        tool_block_indices[tc_index] = block_idx

                        block_start = {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_calls_by_index[tc_index]["id"],
                                "name": tool_calls_by_index[tc_index]["name"],
                                "input": {},
                            },
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                        if arguments:
                            block_delta = {
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": arguments,
                                },
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"
                        current_block_index += 1

                if forced_tool_call and not tool_block_indices:
                    tc_index = 0
                    block_idx = current_block_index
                    arguments = json.dumps(forced_tool_call.get("input") or {})
                    tool_calls_by_index[tc_index] = {
                        "id": forced_tool_call.get(
                            "id", f"toolu_proxy_{uuid.uuid4().hex[:12]}"
                        ),
                        "name": forced_tool_call.get("name", ""),
                        "arguments": arguments,
                    }
                    tool_block_indices[tc_index] = block_idx
                    block_start = {
                        "type": "content_block_start",
                        "index": block_idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_calls_by_index[tc_index]["id"],
                            "name": tool_calls_by_index[tc_index]["name"],
                            "input": {},
                        },
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    if arguments:
                        block_delta = {
                            "type": "content_block_delta",
                            "index": block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments,
                            },
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"
                    current_block_index += 1

                for tc_index in sorted(tool_block_indices.keys()):
                    tc = tool_calls_by_index.get(tc_index) or {}
                    if not tc.get("native"):
                        continue
                    arguments = _normalize_tool_arguments_for_anthropic(
                        str(tc.get("name") or ""),
                        str(tc.get("arguments") or ""),
                    )
                    if not arguments:
                        continue
                    try:
                        json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    block_delta = {
                        "type": "content_block_delta",
                        "index": tool_block_indices[tc_index],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        },
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"

                # Close all open tool_use blocks
                for tc_index in sorted(tool_block_indices.keys()):
                    block_idx = tool_block_indices[tc_index]
                    yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {block_idx}}}\n\n'

                # Determine stop_reason based on emitted blocks first, then the
                # provider finish_reason. This mirrors Anthropic's contract while
                # avoiding heuristic "tool use" stops for plain text responses.
                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                    "content_filter": "end_turn",
                    "function_call": "tool_use",
                }
                # If the provider truncated the response (finish_reason == "length"),
                # the tool_call arguments are very likely incomplete JSON (the vLLM
                # hermes parser raises "Unterminated string"). Emitting stop_reason
                # "tool_use" with broken input makes Claude Code retry the same call
                # forever (the edit-the-same-file loop). Signal max_tokens instead so
                # the client surfaces a truncation error instead of looping.
                # Also treat any tool call whose accumulated arguments are not valid
                # JSON as truncated — covers cases where the provider didn't set
                # finish_reason=length but the JSON still came out incomplete.
                def _args_incomplete() -> bool:
                    for tc in tool_calls_by_index.values():
                        args = tc.get("arguments", "")
                        if not args:
                            continue
                        try:
                            json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            return True
                    return False

                truncated = provider_finish_reason == "length" or _args_incomplete()
                if tool_block_indices and not truncated:
                    stop_reason = "tool_use"
                elif tool_block_indices and truncated:
                    stop_reason = "max_tokens"
                else:
                    stop_reason = stop_reason_map.get(provider_finish_reason, "end_turn")
                    if stop_reason == "tool_use":
                        stop_reason = "end_turn"
                stop_reason_final = stop_reason

                # Build final usage dict with cached tokens
                final_usage = {"output_tokens": output_tokens}
                if cached_tokens > 0:
                    final_usage["cache_read_input_tokens"] = cached_tokens
                    final_usage["cache_creation_input_tokens"] = 0

                # Send message_delta with final info
                yield f'event: message_delta\ndata: {{"type": "message_delta", "delta": {{"stop_reason": "{stop_reason}", "stop_sequence": null}}, "usage": {json.dumps(final_usage)}}}\n\n'

                # Send message_stop
                yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

                # Log final Anthropic response if logger provided
                if transaction_logger:
                    # Build content blocks for logging
                    content_blocks = []
                    if accumulated_thinking:
                        content_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": accumulated_thinking,
                            }
                        )
                    if accumulated_text:
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": accumulated_text,
                            }
                        )
                    # Add tool use blocks
                    for tc_index in sorted(tool_block_indices.keys()):
                        tc = tool_calls_by_index[tc_index]
                        # Parse arguments JSON string to dict
                        try:
                            input_data = json.loads(tc.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            input_data = {}
                        input_data = _normalize_tool_input_for_anthropic(
                            tc.get("name", ""),
                            input_data,
                        )
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("name", ""),
                                "input": input_data,
                            }
                        )

                    # Build usage for logging
                    log_usage = {
                        "input_tokens": input_tokens - cached_tokens,
                        "output_tokens": output_tokens,
                    }
                    if cached_tokens > 0:
                        log_usage["cache_read_input_tokens"] = cached_tokens
                        log_usage["cache_creation_input_tokens"] = 0

                    anthropic_response = {
                        "id": request_id,
                        "type": "message",
                        "role": "assistant",
                        "content": content_blocks,
                        "model": original_model,
                        "stop_reason": stop_reason_final,
                        "stop_sequence": None,
                        "usage": log_usage,
                    }
                    transaction_logger.log_response(
                        anthropic_response,
                        filename="anthropic_response.json",
                    )

                break

            try:
                chunk = json.loads(data_content)
            except json.JSONDecodeError:
                continue

            # Extract usage if present
            # Note: Google's promptTokenCount INCLUDES cached tokens, but Anthropic's
            # input_tokens EXCLUDES cached tokens. We extract cached tokens and subtract.
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)
                # Extract cached tokens from prompt_tokens_details
                if usage.get("prompt_tokens_details"):
                    cached_tokens = usage["prompt_tokens_details"].get(
                        "cached_tokens", cached_tokens
                    )

            # Send message_start on first chunk
            if not message_started:
                # Build usage with cached tokens properly handled for Anthropic format
                usage_dict = {
                    "input_tokens": input_tokens - cached_tokens,
                    "output_tokens": 0,
                }
                if cached_tokens > 0:
                    usage_dict["cache_read_input_tokens"] = cached_tokens
                    usage_dict["cache_creation_input_tokens"] = 0

                message_start = {
                    "type": "message_start",
                    "message": {
                        "id": request_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": original_model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage_dict,
                    },
                }
                yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                message_started = True

            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0] or {}
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                if finish_reason == "tool_calls" or provider_finish_reason != "tool_calls":
                    provider_finish_reason = finish_reason

            delta = choice.get("delta", {})

            # Handle reasoning/thinking content (from OpenAI-style reasoning_content)
            reasoning_content = delta.get("reasoning_content")
            if reasoning_content:
                if not thinking_block_started:
                    # Start a thinking content block
                    block_start = {
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    thinking_block_started = True

                # Send thinking delta
                block_delta = {
                    "type": "content_block_delta",
                    "index": current_block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning_content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"
                # Accumulate thinking for logging
                accumulated_thinking += reasoning_content

            # Handle text content
            content = delta.get("content")
            if content:
                if write_from_model_text and not tool_calls_by_index:
                    delayed_model_text_buffer += content
                    continue

                if textual_tool_buffer:
                    textual_tool_buffer += content
                    continue

                marker_positions = [
                    pos
                    for pos in (
                        content.find("<tool_call"),
                        content.find("<function="),
                        content.find("<execute"),
                    )
                    if pos >= 0
                ]
                marker_pos = min(marker_positions) if marker_positions else -1
                if marker_pos >= 0:
                    textual_tool_buffer = content[marker_pos:]
                    content = content[:marker_pos]
                    if not content:
                        continue
                else:
                    stripped = content.lstrip()
                    if (
                        stripped.startswith("<tool")
                        or stripped.startswith("<fun")
                        or stripped.startswith("<function")
                        or stripped.startswith("<exe")
                    ):
                        textual_tool_buffer = content
                        continue

                # If we were in a thinking block, close it first
                if thinking_block_started and not content_block_started:
                    yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                    current_block_index += 1
                    thinking_block_started = False

                if not content_block_started:
                    # Start a text content block
                    block_start = {
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    content_block_started = True

                # Send content delta
                block_delta = {
                    "type": "content_block_delta",
                    "index": current_block_index,
                    "delta": {"type": "text_delta", "text": content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"
                # Accumulate text for logging
                accumulated_text += content

            # Handle tool calls
            # Use `or []` to handle providers that send "tool_calls": null
            tool_calls = delta.get("tool_calls") or []
            for tc in tool_calls:
                tc_index = tc.get("index", 0)

                if tc_index not in tool_calls_by_index:
                    tool_calls_by_index[tc_index] = {
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": "",
                        "emitted_arguments_length": 0,
                        "started": False,
                        "native": True,
                    }
                elif tc.get("id"):
                    tool_calls_by_index[tc_index]["id"] = tc["id"]

                # Accumulate arguments
                func = tc.get("function", {})
                if func.get("name"):
                    restored_name = _restore_tool_name(
                        str(func["name"]),
                        tool_name_mapping,
                    )
                    if (
                        allowed_tool_names is not None
                        and restored_name not in allowed_tool_names
                    ):
                        logger.warning(
                            "Ignoring unavailable native tool emitted by model: %s",
                            restored_name,
                        )
                        tool_calls_by_index[tc_index]["rejected"] = True
                    else:
                        tool_calls_by_index[tc_index]["name"] = restored_name
                        tool_calls_by_index[tc_index]["rejected"] = False
                if func.get("arguments"):
                    tool_calls_by_index[tc_index]["arguments"] += func["arguments"]

                if (
                    tool_calls_by_index[tc_index]["name"]
                    and not tool_calls_by_index[tc_index].get("rejected")
                    and not tool_calls_by_index[tc_index]["started"]
                ):
                    # Anthropic requires the tool name in content_block_start.
                    # Some OpenAI-compatible streams send id/type first and name
                    # in a later delta, so wait until the reducer has a name.
                    if thinking_block_started:
                        yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                        current_block_index += 1
                        thinking_block_started = False

                    if content_block_started:
                        yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                        current_block_index += 1
                        content_block_started = False

                    tool_block_indices[tc_index] = current_block_index
                    tool_calls_by_index[tc_index]["started"] = True

                    block_start = {
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_calls_by_index[tc_index]["id"],
                            "name": tool_calls_by_index[tc_index]["name"],
                            "input": {},
                        },
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    current_block_index += 1

                # Native OpenAI tool arguments are emitted after [DONE], once the
                # full JSON can be normalized for Anthropic's tool schemas.

            # Note: We intentionally ignore finish_reason here.
            # Block closing is handled when we receive [DONE] to avoid
            # premature closes with providers that send finish_reason on each chunk.

    except Exception as e:
        logger.error(f"Error in Anthropic streaming wrapper: {e}")

        # If we haven't sent message_start yet, send it now so the client can display the error
        # Claude Code and other clients may ignore events that come before message_start
        if not message_started:
            # Build usage with cached tokens properly handled
            usage_dict = {
                "input_tokens": input_tokens - cached_tokens,
                "output_tokens": 0,
            }
            if cached_tokens > 0:
                usage_dict["cache_read_input_tokens"] = cached_tokens
                usage_dict["cache_creation_input_tokens"] = 0

            message_start = {
                "type": "message_start",
                "message": {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": original_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage_dict,
                },
            }
            yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Send the error as a text content block so it's visible to the user
        error_message = f"Error: {str(e)}"
        error_block_start = {
            "type": "content_block_start",
            "index": current_block_index,
            "content_block": {"type": "text", "text": ""},
        }
        yield f"event: content_block_start\ndata: {json.dumps(error_block_start)}\n\n"

        error_block_delta = {
            "type": "content_block_delta",
            "index": current_block_index,
            "delta": {"type": "text_delta", "text": error_message},
        }
        yield f"event: content_block_delta\ndata: {json.dumps(error_block_delta)}\n\n"

        yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'

        # Build final usage with cached tokens
        final_usage = {"output_tokens": 0}
        if cached_tokens > 0:
            final_usage["cache_read_input_tokens"] = cached_tokens
            final_usage["cache_creation_input_tokens"] = 0

        # Send message_delta and message_stop to properly close the stream
        yield f'event: message_delta\ndata: {{"type": "message_delta", "delta": {{"stop_reason": "end_turn", "stop_sequence": null}}, "usage": {json.dumps(final_usage)}}}\n\n'
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        # Also send the formal error event for clients that handle it
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
