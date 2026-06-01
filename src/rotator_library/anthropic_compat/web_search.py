# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Web search tool exposed to text-only backends (vLLM/Qwen).

Claude Code's native WebSearch is executed CLIENT-SIDE by the Claude Code app
itself — the proxy never sees the tool call. So for our hosted Qwen, the proxy
has to be both the tool emitter (we inject the schema into the request) AND the
tool executor (we run the search and feed the result back). The model just sees
a regular tool called `WebSearch` with one argument `query`.

Provider: Tavily (https://tavily.com) — free tier 1000 req/month, returns
clean LLM-ready results (snippet + URL + title).

Config:
  TAVILY_API_KEY            — required to enable. Without it, the tool is
                              never injected (silently disabled).
  WEB_SEARCH_INJECT         — on/off, default on. Turn off if you want clients
                              to bring their own WebSearch tool instead.
  WEB_SEARCH_MAX_RESULTS    — int, default 5.
  WEB_SEARCH_TIMEOUT        — int seconds, default 10.

Tool schema is Anthropic-compatible (input_schema) so it can be added directly
to AnthropicMessagesRequest.tools. The translator already converts that to
OpenAI tool format downstream.
"""
import asyncio
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import httpx


_LOG = logging.getLogger("rotator_library.web_search")

WEB_SEARCH_TOOL_NAME = "WebSearch"

# Anthropic-compatible tool schema. The model sees a tool called WebSearch
# with one required string argument. We picked the simplest possible shape
# because Qwen3 handles minimal schemas better than nested ones.
WEB_SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "name": WEB_SEARCH_TOOL_NAME,
    "description": (
        "Search the public web for current information. Use this when the user "
        "asks about recent events, current prices, dates, versions, news, or "
        "anything you might not know reliably from training data. Returns a "
        "list of result snippets with URLs. Do not use for questions you can "
        "answer confidently from your own knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, in natural language.",
            },
        },
        "required": ["query"],
    },
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def is_enabled() -> bool:
    """Web search is enabled iff Tavily key is configured AND inject flag is on."""
    if os.getenv("WEB_SEARCH_INJECT", "on").lower() in {"off", "0", "false", "no"}:
        return False
    return bool(os.getenv("TAVILY_API_KEY"))


# --- Tiny TTL cache so the same query within a session doesn't burn quota ---
_CACHE: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
_CACHE_MAX = 64
_CACHE_TTL = 300  # 5 min


def _cache_get(query: str) -> Optional[str]:
    key = query.strip().lower()
    item = _CACHE.get(key)
    if not item:
        return None
    expires_at, value = item
    if expires_at < time.monotonic():
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return value


def _cache_put(query: str, value: str) -> None:
    key = query.strip().lower()
    _CACHE[key] = (time.monotonic() + _CACHE_TTL, value)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


async def search(query: str) -> str:
    """Run a search and return a formatted string ready to put in tool_result.

    Returns a human-readable plain-text block (the model reads it as content);
    on failure returns an error message instead of raising — we never want to
    break the agent loop because of a search hiccup."""
    query = (query or "").strip()
    if not query:
        return "[web search] empty query — nothing to search for."

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "[web search] not configured on this proxy (TAVILY_API_KEY missing)."

    cached = _cache_get(query)
    if cached is not None:
        return cached

    max_results = max(1, _env_int("WEB_SEARCH_MAX_RESULTS", 5))
    timeout = max(2, _env_int("WEB_SEARCH_TIMEOUT", 10))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,  # Tavily's pre-summarized answer
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except asyncio.TimeoutError:
        msg = f"[web search] timed out after {timeout}s. Try again or rephrase."
        _LOG.warning(msg)
        return msg
    except httpx.HTTPStatusError as e:
        msg = f"[web search] provider returned {e.response.status_code}. Try again later."
        _LOG.warning("Tavily HTTP error: %s", e.response.status_code)
        return msg
    except Exception as e:
        _LOG.warning("Web search error: %r", e)
        return f"[web search] failed: {e!s}"

    formatted = _format_results(query, data)
    _cache_put(query, formatted)
    return formatted


def _format_results(query: str, data: Dict[str, Any]) -> str:
    answer = (data.get("answer") or "").strip()
    results = data.get("results") or []
    lines = [f"Web search results for: {query}\n"]
    if answer:
        lines.append(f"Quick answer (from Tavily): {answer}\n")
    if not results:
        lines.append("No results found.")
        return "\n".join(lines)
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        if not (title or url):
            continue
        if len(content) > 600:
            content = content[:600] + "..."
        lines.append(f"{i}. {title}\n   {url}\n   {content}")
    return "\n".join(lines)


def extract_search_calls(anthropic_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return [{id, query}] for each WebSearch tool_use block in the response."""
    out: List[Dict[str, Any]] = []
    for block in anthropic_response.get("content") or []:
        if not (isinstance(block, dict) and block.get("type") == "tool_use"):
            continue
        if block.get("name") != WEB_SEARCH_TOOL_NAME:
            continue
        inp = block.get("input") or {}
        query = inp.get("query") if isinstance(inp, dict) else ""
        out.append({"id": block.get("id"), "query": str(query or "")})
    return out


def build_tool_results_message(results: List[Dict[str, str]]) -> Dict[str, Any]:
    """Build the user-role tool_result message that feeds search results back
    to the model in the same Anthropic format the client would use."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": r["id"],
                "content": r["result"],
            }
            for r in results
        ],
    }


def has_only_websearch_tool_calls(anthropic_response: Dict[str, Any]) -> bool:
    """True if the response's tool_use blocks are ALL WebSearch (server-side
    executable) — meaning we can resolve them transparently and re-invoke the
    model, without leaking a tool_use the client doesn't know how to handle."""
    blocks = anthropic_response.get("content") or []
    tool_uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
    if not tool_uses:
        return False
    return all(b.get("name") == WEB_SEARCH_TOOL_NAME for b in tool_uses)


def inject_tool(request_tools: Optional[List[Any]]) -> List[Any]:
    """Add WebSearch to the request's tools list (if enabled and not already there).

    Tools come in many shapes depending on caller (Anthropic dict or pydantic).
    We append a plain Anthropic-shaped dict; the translator handles both."""
    tools = list(request_tools or [])
    if not is_enabled():
        return tools
    # Avoid duplicate injection.
    for t in tools:
        name = None
        if isinstance(t, dict):
            name = t.get("name")
        else:
            name = getattr(t, "name", None)
        if name == WEB_SEARCH_TOOL_NAME:
            return tools
    tools.append(WEB_SEARCH_TOOL_SCHEMA)
    return tools
