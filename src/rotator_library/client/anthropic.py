# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Anthropic API compatibility handler for RotatingClient.

This module provides Anthropic SDK compatibility methods that allow using
Anthropic's Messages API format with the credential rotation system.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from ..anthropic_compat import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
    translate_anthropic_request,
    openai_to_anthropic_response,
    anthropic_streaming_wrapper,
    anthropic_response_to_streaming_events,
    anthropic_to_openai_messages,
    anthropic_to_openai_tools,
)
from ..anthropic_compat.streaming import _model_text_to_write_tool_call
from ..anthropic_compat.image_captioning import caption_images_in_request
from ..anthropic_compat.context_compaction import compact_context_if_needed
from ..anthropic_compat.self_critique import maybe_critique_response
from ..anthropic_compat import web_search as _ws
from ..transaction_logger import TransactionLogger

if TYPE_CHECKING:
    from .rotating_client import RotatingClient

lib_logger = logging.getLogger("rotator_library")


# Per-process semaphore to cap concurrent requests to text-only backends
# (vLLM/Qwen/Ollama). When Claude Code/Cursor prefetch (count_tokens + main +
# opus + sonnet simultaneous) all hit the SAME vLLM single-GPU model, prefill
# saturates and a tiny request (e.g. "oi") can wait 2 minutes behind 7 others.
# Cap = how many parallel requests the backend can serve well. For a single
# Qwen3-32B on A40, 4 is a sane default (prefill scales linearly with batch).
# Env: VLLM_BACKEND_CONCURRENCY (set 0 to disable the cap).
_BACKEND_SEMAPHORE: Optional[asyncio.Semaphore] = None
_BACKEND_SEMAPHORE_LOCK = asyncio.Lock()


async def _get_backend_semaphore() -> Optional[asyncio.Semaphore]:
    global _BACKEND_SEMAPHORE
    if _BACKEND_SEMAPHORE is not None:
        return _BACKEND_SEMAPHORE
    async with _BACKEND_SEMAPHORE_LOCK:
        if _BACKEND_SEMAPHORE is not None:
            return _BACKEND_SEMAPHORE
        try:
            cap = int(os.environ.get("VLLM_BACKEND_CONCURRENCY", "4"))
        except (TypeError, ValueError):
            cap = 4
        if cap <= 0:
            # Sentinel "no cap" — use a sema with very high value rather than
            # branching everywhere on None.
            _BACKEND_SEMAPHORE = asyncio.Semaphore(10_000)
        else:
            _BACKEND_SEMAPHORE = asyncio.Semaphore(cap)
        return _BACKEND_SEMAPHORE


def _json_dump_str(s: str) -> str:
    """JSON-encode a string with surrounding quotes (used to build tool_call
    arguments inline). e.g. _json_dump_str('hi"x') == '"hi\\"x"'."""
    return json.dumps(s, ensure_ascii=False)


def _raise_if_error_response(response: Any) -> None:
    if not isinstance(response, dict) or "error" not in response:
        return
    error = response.get("error") or {}
    message = error.get("message") if isinstance(error, dict) else str(error)
    if not message:
        message = str(response)
    raise ValueError(message)


def _has_tool_use(anthropic_response: dict) -> bool:
    return any(
        isinstance(block, dict) and block.get("type") == "tool_use"
        for block in anthropic_response.get("content") or []
    )


def _response_tool_allowlist(
    openai_request: dict,
    tool_name_mapping: Optional[dict],
) -> set[str]:
    names = openai_request.pop("_vllm_allowed_tool_names", None)
    if names is None:
        names = [
            (tool.get("function") or {}).get("name")
            for tool in openai_request.get("tools") or []
        ]
    return {
        tool_name_mapping.get(str(name), str(name))
        if tool_name_mapping
        else str(name)
        for name in names
        if name
    }


def _force_tool_use_response(
    anthropic_response: dict,
    forced_tool_call: Optional[dict],
) -> dict:
    if not forced_tool_call or _has_tool_use(anthropic_response):
        return anthropic_response

    response = dict(anthropic_response)
    content = list(response.get("content") or [])
    if forced_tool_call.get("_proxy_strategy") == "write_from_model_text":
        model_text = "\n".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        forced_tool_call = _model_text_to_write_tool_call(
            forced_tool_call, model_text
        )
        content = []
    content.append(
        {
            "type": "tool_use",
            "id": forced_tool_call.get("id", f"toolu_proxy_{uuid.uuid4().hex[:12]}"),
            "name": forced_tool_call.get("name", ""),
            "input": forced_tool_call.get("input") or {},
        }
    )
    response["content"] = content
    response["stop_reason"] = "tool_use"
    return response


class AnthropicHandler:
    """
    Handler for Anthropic API compatibility methods.

    This class provides methods to handle Anthropic Messages API requests
    by translating them to OpenAI format, processing through the client's
    acompletion method, and converting responses back to Anthropic format.

    Example:
        handler = AnthropicHandler(client)
        response = await handler.messages(request, raw_request)
    """

    def __init__(self, client: "RotatingClient"):
        """
        Initialize the Anthropic handler.

        Args:
            client: The RotatingClient instance to use for completions
        """
        self._client = client

    async def messages(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
    ) -> Any:
        """
        Handle Anthropic Messages API requests.

        Wraps the actual work in a per-process semaphore for text-only backends,
        because a single vLLM/GPU model can't truly parallelize 8 requests
        without ruinous prefill latency. Without this cap, Claude Code's
        prefetch (count_tokens + messages + opus + sonnet simultaneous) made
        a tiny "oi" request wait 2 minutes behind 7 large ones.
        """
        provider_check = (request.model.split("/")[0] if "/" in (request.model or "") else "unknown")
        if provider_check not in {"hosted_vllm", "vllm", "lm_studio", "ollama"}:
            return await self._messages_impl(request, raw_request, pre_request_callback)

        sem = await _get_backend_semaphore()
        # Acquire BEFORE building the request so prefill is what's gated. For a
        # streaming request _messages_impl returns an async generator that hasn't
        # produced any tokens yet — the expensive prefill happens while the
        # generator is consumed downstream. So we must hold the semaphore until
        # the stream is fully drained, not just until the generator is created.
        await sem.acquire()
        try:
            result = await self._messages_impl(request, raw_request, pre_request_callback)
        except BaseException:
            sem.release()
            raise

        if not hasattr(result, "__aiter__"):
            # Non-streaming: work is done, release now.
            sem.release()
            return result

        async def _release_after_stream():
            try:
                async for chunk in result:
                    yield chunk
            finally:
                sem.release()

        return _release_after_stream()

    async def _messages_impl(
        self,
        request: AnthropicMessagesRequest,
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
    ) -> Any:
        """
        Handle Anthropic Messages API requests.

        This method accepts requests in Anthropic's format, translates them to
        OpenAI format internally, processes them through the existing acompletion
        method, and returns responses in Anthropic's format.

        Args:
            request: An AnthropicMessagesRequest object
            raw_request: Optional raw request object for disconnect checks
            pre_request_callback: Optional async callback before each API request

        Returns:
            For non-streaming: dict in Anthropic Messages format
            For streaming: AsyncGenerator yielding Anthropic SSE format strings
        """
        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        original_model = request.model

        # Extract provider from model for logging
        provider = original_model.split("/")[0] if "/" in original_model else "unknown"

        # Create Anthropic transaction logger if request logging is enabled
        anthropic_logger = None
        if self._client.enable_request_logging:
            anthropic_logger = TransactionLogger(
                provider,
                original_model,
                enabled=True,
                api_format="ant",
            )
            # Log original Anthropic request
            anthropic_logger.log_request(
                request.model_dump(exclude_none=True),
                filename="anthropic_request.json",
            )

        # Text-only backends (vLLM/Qwen, etc.) can't see images. Replace image
        # blocks with contextual text descriptions from an external vision model
        # before translating, so the request reaches the backend as plain text.
        if provider in {"hosted_vllm", "vllm", "lm_studio", "ollama"}:
            request = await caption_images_in_request(
                request, self._client, log=lib_logger
            )
            # Long conversations can fill the model's context ceiling, leaving no
            # room for the output — the model reasons (<think>) and stops before
            # acting. Summarize the old middle (keeping system + recent tail) so
            # there's always room to reason AND execute.
            request = await compact_context_if_needed(
                request, self._client, log=lib_logger
            )
            # Inject WebSearch tool when configured. The client (Claude Code/
            # Cursor) doesn't ship it to non-Claude models, and our /web chat
            # benefits from having search. We execute the search ourselves
            # below; client never sees WebSearch is being run server-side.
            if _ws.is_enabled():
                new_tools = _ws.inject_tool(request.tools)
                if new_tools is not request.tools:
                    request = request.model_copy(update={"tools": new_tools})

        # Translate Anthropic request to OpenAI format
        openai_request = translate_anthropic_request(request)
        forced_tool_call = openai_request.pop("_vllm_forced_tool_call", None)
        tool_name_mapping = openai_request.pop("_anthropic_tool_name_mapping", None)
        allowed_tool_names = _response_tool_allowlist(
            openai_request, tool_name_mapping
        )
        openai_request.pop("_vllm_tool_intent", None)
        openai_request.pop("_vllm_previous_tool_count", None)

        # Pass parent log directory to acompletion for nested logging
        if anthropic_logger and anthropic_logger.log_dir:
            openai_request["_parent_log_dir"] = anthropic_logger.log_dir

        # Force non-streaming when (a) operator explicitly opted in, OR (b)
        # WebSearch is enabled and there are tools — the server-side WebSearch
        # loop needs to inspect the full response before deciding to re-invoke,
        # which can't be done mid-stream. Without this, a streaming client
        # would get a tool_use WebSearch block it doesn't know how to execute
        # and would stall waiting for nothing.
        websearch_active = _ws.is_enabled() and bool(openai_request.get("tools"))
        force_nonstream_for_stream = (
            provider in {"hosted_vllm", "vllm", "lm_studio", "ollama"}
            and (
                os.getenv("ANTHROPIC_STREAM_FALLBACK_NONSTREAM", "false").lower()
                not in {"false", "0", "no"}
                or websearch_active
            )
        )
        if request.stream and force_nonstream_for_stream:
            lib_logger.info(
                "Using non-stream fallback for Anthropic stream request on %s",
                provider,
            )
            fallback_request = dict(openai_request)
            fallback_request["stream"] = False
            response = await self._client.acompletion(
                request=raw_request,
                pre_request_callback=pre_request_callback,
                **fallback_request,
            )
            openai_response = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else dict(response)
            )
            _raise_if_error_response(openai_response)
            anthropic_response = openai_to_anthropic_response(
                openai_response,
                original_model,
                tool_name_mapping=tool_name_mapping,
                allowed_tool_names=allowed_tool_names,
            )
            anthropic_response["id"] = request_id
            anthropic_response = _force_tool_use_response(
                anthropic_response, forced_tool_call
            )
            # WebSearch loop also runs in the force-nonstream fallback path —
            # otherwise a streaming client with WebSearch enabled would get a
            # tool_use the client can't execute and stall.
            anthropic_response = await self._run_web_search_loop(
                anthropic_response,
                request,
                openai_request,
                raw_request,
                pre_request_callback,
                original_model,
                tool_name_mapping,
                allowed_tool_names,
                forced_tool_call,
                request_id,
            )
            if anthropic_logger:
                anthropic_logger.log_response(
                    anthropic_response,
                    filename="anthropic_response.json",
                )
            return anthropic_response_to_streaming_events(anthropic_response)

        if request.stream:
            # Streaming response
            response_generator = await self._client.acompletion(
                request=raw_request,
                pre_request_callback=pre_request_callback,
                **openai_request,
            )

            # Create disconnect checker if raw_request provided
            is_disconnected = None
            if raw_request is not None and hasattr(raw_request, "is_disconnected"):
                is_disconnected = raw_request.is_disconnected

            # Return the streaming wrapper
            # Note: For streaming, the anthropic response logging happens in the wrapper
            return anthropic_streaming_wrapper(
                openai_stream=response_generator,
                original_model=original_model,
                request_id=request_id,
                is_disconnected=is_disconnected,
                transaction_logger=anthropic_logger,
                forced_tool_call=forced_tool_call,
                tool_name_mapping=tool_name_mapping,
                allowed_tool_names=allowed_tool_names,
            )
        else:
            # Non-streaming response
            response = await self._client.acompletion(
                request=raw_request,
                pre_request_callback=pre_request_callback,
                **openai_request,
            )

            # Convert OpenAI response to Anthropic format
            openai_response = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else dict(response)
            )
            _raise_if_error_response(openai_response)
            anthropic_response = openai_to_anthropic_response(
                openai_response,
                original_model,
                tool_name_mapping=tool_name_mapping,
                allowed_tool_names=allowed_tool_names,
            )

            # Override the ID with our request ID
            anthropic_response["id"] = request_id
            anthropic_response = _force_tool_use_response(
                anthropic_response, forced_tool_call
            )

            # Server-side WebSearch loop: when the model emitted ONLY WebSearch
            # tool_use blocks, run the searches here and re-invoke until the
            # model produces a final answer (or we hit a sanity cap). Client
            # never sees WebSearch — it just gets the final synthesized reply.
            anthropic_response = await self._run_web_search_loop(
                anthropic_response,
                request,
                openai_request,
                raw_request,
                pre_request_callback,
                original_model,
                tool_name_mapping,
                allowed_tool_names,
                forced_tool_call,
                request_id,
            )

            # Optional self-critique pass (VLLM_SELF_CRITIQUE=on). Doubles cost
            # and latency; off by default. Skips automatically if the response
            # contains tool_use blocks or is empty.
            anthropic_response = await maybe_critique_response(
                anthropic_response, request, self._client, log=lib_logger
            )

            # Log Anthropic response
            if anthropic_logger:
                anthropic_logger.log_response(
                    anthropic_response,
                    filename="anthropic_response.json",
                )

            return anthropic_response

    async def _run_web_search_loop(
        self,
        anthropic_response: dict,
        request: AnthropicMessagesRequest,
        openai_request: dict,
        raw_request,
        pre_request_callback,
        original_model: str,
        tool_name_mapping,
        allowed_tool_names,
        forced_tool_call,
        request_id: str,
    ) -> dict:
        """When the model emits WebSearch tool_use, run the search server-side
        and feed the result back in. Loop until model gives a final answer or
        we hit max_iters. The client never sees WebSearch tool calls."""
        if not _ws.is_enabled():
            return anthropic_response

        import os, asyncio  # noqa: E401
        max_iters = max(1, int(os.getenv("WEB_SEARCH_MAX_ITERS", "3")))
        iters = 0
        # We'll mutate this list of OpenAI-format messages across iterations.
        msgs = list(openai_request.get("messages") or [])

        while (
            iters < max_iters
            and _ws.has_only_websearch_tool_calls(anthropic_response)
        ):
            iters += 1
            calls = _ws.extract_search_calls(anthropic_response)
            if not calls:
                break

            # Run searches in parallel (usually 1, occasionally 2-3).
            search_outputs = await asyncio.gather(
                *(_ws.search(c["query"]) for c in calls),
                return_exceptions=True,
            )
            results = []
            for call, out in zip(calls, search_outputs):
                text = out if isinstance(out, str) else f"[web search] error: {out!r}"
                results.append({"id": call["id"], "result": text})

            # Append assistant-with-tool_calls and tool results to the OpenAI
            # message history, then call the model again.
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": _ws.WEB_SEARCH_TOOL_NAME,
                            "arguments": '{"query": ' + _json_dump_str(c["query"]) + "}",
                        },
                    }
                    for c in calls
                ],
            }
            msgs.append(assistant_msg)
            for r in results:
                msgs.append({"role": "tool", "tool_call_id": r["id"], "content": r["result"]})

            next_req = dict(openai_request)
            next_req["messages"] = msgs
            next_req["stream"] = False
            response = await self._client.acompletion(
                request=raw_request,
                pre_request_callback=pre_request_callback,
                **next_req,
            )
            openai_response = (
                response.model_dump() if hasattr(response, "model_dump") else dict(response)
            )
            _raise_if_error_response(openai_response)
            anthropic_response = openai_to_anthropic_response(
                openai_response,
                original_model,
                tool_name_mapping=tool_name_mapping,
                allowed_tool_names=allowed_tool_names,
            )
            anthropic_response["id"] = request_id
            anthropic_response = _force_tool_use_response(
                anthropic_response, forced_tool_call
            )

        return anthropic_response

    async def count_tokens(
        self,
        request: AnthropicCountTokensRequest,
    ) -> dict:
        """
        Handle Anthropic count_tokens API requests.

        Counts the number of tokens that would be used by a Messages API request.
        This is useful for estimating costs and managing context windows.

        Args:
            request: An AnthropicCountTokensRequest object

        Returns:
            Dict with input_tokens count in Anthropic format
        """
        anthropic_request = request.model_dump(exclude_none=True)

        openai_messages = anthropic_to_openai_messages(
            anthropic_request.get("messages", []), anthropic_request.get("system")
        )

        # Count tokens for messages
        message_tokens = self._client.token_count(
            model=request.model,
            messages=openai_messages,
        )

        # Count tokens for tools if present
        tool_tokens = 0
        if request.tools:
            # Tools add tokens based on their definitions
            # Convert to JSON string and count tokens for tool definitions
            openai_tools = anthropic_to_openai_tools(
                [tool.model_dump() for tool in request.tools]
            )
            if openai_tools:
                # Serialize tools to count their token contribution
                tools_text = json.dumps(openai_tools)
                tool_tokens = self._client.token_count(
                    model=request.model,
                    text=tools_text,
                )

        total_tokens = message_tokens + tool_tokens

        return {"input_tokens": total_tokens}
