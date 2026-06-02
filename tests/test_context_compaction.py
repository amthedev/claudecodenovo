import asyncio
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "rotator_library"

rotator_package = types.ModuleType("rotator_library")
rotator_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("rotator_library", rotator_package)

anthropic_package = types.ModuleType("rotator_library.anthropic_compat")
anthropic_package.__path__ = [str(PACKAGE_ROOT / "anthropic_compat")]
sys.modules.setdefault("rotator_library.anthropic_compat", anthropic_package)

from rotator_library.anthropic_compat.context_compaction import (
    _BACKGROUND_SUMMARY_TASKS,
    _SUMMARY_CACHE,
    _SUMMARY_EVOLVE_HEADER,
    _SUMMARY_FAILURE_CACHE,
    _SUMMARY_MARKER,
    _estimate_tools_tokens,
    _request_input_tokens,
    compact_context_if_needed,
)
from rotator_library.anthropic_compat.models import AnthropicMessagesRequest


def _request(messages):
    return AnthropicMessagesRequest(
        model="hosted_vllm/qwen",
        max_tokens=4096,
        messages=messages,
    )


def _content(message):
    return message.get("content", "") if isinstance(message, dict) else message.content


class _SummaryClient:
    def __init__(self, content="resumo atualizado"):
        self.calls = []
        self.content = content

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.content}}]}


class ContextCompactionTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the summary PRODUCTION mechanics (budget, cache reuse, cooldown,
    timeout). These run in synchronous mode — the summary is computed inline and
    its effect is observable right after the await. Background scheduling (the
    default in production) is covered by BackgroundSummaryTests below."""

    def setUp(self):
        _SUMMARY_CACHE.clear()
        _SUMMARY_FAILURE_CACHE.clear()
        # These assert on the inline call happening during the await. Force the
        # synchronous path; the off-critical-path scheduling is tested separately.
        self._env = mock.patch.dict(
            os.environ, {"VLLM_CONTEXT_SUMMARY_BACKGROUND": "off"}, clear=False
        )
        self._env.start()

    def tearDown(self):
        self._env.stop()
        _SUMMARY_CACHE.clear()
        _SUMMARY_FAILURE_CACHE.clear()

    async def test_first_compaction_uses_short_summary_budget(self):
        client = _SummaryClient()
        request = _request(
            [
                {"role": "user", "content": "antigo " * 1800},
                {"role": "assistant", "content": "feito " * 1800},
                {"role": "user", "content": "tarefa atual " * 500},
            ]
        )

        with mock.patch.dict(
            os.environ,
            {
                "VLLM_MODEL_CONTEXT": "4000",
                "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
                "VLLM_CONTEXT_KEEP_TAIL_TOKENS": "1000",
            },
            clear=False,
        ):
            compacted = await compact_context_if_needed(request, client)

        self.assertEqual(len(client.calls), 1)
        # 1500: lighter summary so the (background) call completes within the
        # timeout on a busy GPU instead of timing out and never caching.
        self.assertEqual(client.calls[0]["max_tokens"], 1500)
        self.assertEqual(len(compacted.messages), 2)
        self.assertIn(_SUMMARY_MARKER, _content(compacted.messages[0]))

    async def test_reuses_previous_summary_without_another_model_call(self):
        client = _SummaryClient()
        request = _request(
            [
                {"role": "user", "content": f"{_SUMMARY_MARKER}\nresumo anterior " * 120},
                {"role": "assistant", "content": "passo intermediario " * 160},
                {"role": "user", "content": "tarefa atual " * 500},
            ]
        )

        with mock.patch.dict(
            os.environ,
            {
                "VLLM_MODEL_CONTEXT": "4000",
                "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
                "VLLM_CONTEXT_KEEP_TAIL_TOKENS": "1000",
                "VLLM_CONTEXT_REFRESH_MIN_TOKENS": "2500",
            },
            clear=False,
        ):
            compacted = await compact_context_if_needed(request, client)

        self.assertEqual(client.calls, [])
        self.assertEqual(len(compacted.messages), 2)
        self.assertIn(_SUMMARY_MARKER, _content(compacted.messages[0]))
        self.assertIn("tarefa atual", _content(compacted.messages[1]))

    async def test_reuses_cached_summary_when_client_resends_raw_history(self):
        client = _SummaryClient()
        request = _request(
            [
                {"role": "user", "content": "antigo " * 1800},
                {"role": "assistant", "content": "feito " * 1800},
                {"role": "user", "content": "tarefa atual " * 500},
            ]
        )

        with mock.patch.dict(
            os.environ,
            {
                "VLLM_MODEL_CONTEXT": "4000",
                "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
                "VLLM_CONTEXT_KEEP_TAIL_TOKENS": "1000",
            },
            clear=False,
        ):
            first = await compact_context_if_needed(request, client)
            second = await compact_context_if_needed(request, client)

        self.assertEqual(len(client.calls), 1)
        self.assertIn(_SUMMARY_MARKER, _content(first.messages[0]))
        self.assertIn(_SUMMARY_MARKER, _content(second.messages[0]))

    async def test_summary_timeout_falls_back_to_bounded_recent_context(self):
        class _SlowClient:
            async def acompletion(self, **kwargs):
                await asyncio.sleep(10)

        request = _request(
            [
                {"role": "user", "content": "antigo " * 1800},
                {"role": "assistant", "content": "feito " * 1800},
                {"role": "user", "content": "tarefa atual " * 500},
            ]
        )

        with mock.patch.dict(
            os.environ,
            {
                "VLLM_MODEL_CONTEXT": "4000",
                "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
                "VLLM_CONTEXT_KEEP_TAIL_TOKENS": "1000",
                "VLLM_CONTEXT_SUMMARY_TIMEOUT_SECONDS": "1",
            },
            clear=False,
        ):
            compacted = await compact_context_if_needed(request, _SlowClient())

        self.assertIsNot(compacted, request)
        self.assertIn(_SUMMARY_MARKER, _content(compacted.messages[0]))
        self.assertIn("tarefa atual", _content(compacted.messages[-1]))
        self.assertLessEqual(_request_input_tokens(compacted), 2000)

    async def test_single_oversized_message_is_trimmed_to_budget(self):
        client = _SummaryClient()
        request = _request(
            [{"role": "user", "content": "pedido recente " * 3000}]
        )

        with mock.patch.dict(
            os.environ,
            {
                "VLLM_MODEL_CONTEXT": "4000",
                "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
            },
            clear=False,
        ):
            compacted = await compact_context_if_needed(request, client)

        # A SINGLE oversized message has no "middle" to summarize and no other
        # message to drop — summarizing it would also block the response for no
        # benefit. So we truncate the message's HEAD in place, preserving its
        # TAIL (the recent request the user actually typed), with no model call.
        self.assertEqual(client.calls, [])
        self.assertIn("recente", _content(compacted.messages[-1]))
        self.assertLessEqual(_request_input_tokens(compacted), 2100)

    async def test_summary_failure_cooldown_skips_repeated_hidden_call(self):
        class _FailingClient:
            def __init__(self):
                self.calls = 0

            async def acompletion(self, **kwargs):
                self.calls += 1
                raise RuntimeError("temporary failure")

        client = _FailingClient()
        request = _request(
            [
                {"role": "user", "content": "antigo " * 1800},
                {"role": "assistant", "content": "feito " * 1800},
                {"role": "user", "content": "tarefa atual " * 500},
            ]
        )

        with mock.patch.dict(
            os.environ,
            {
                "VLLM_MODEL_CONTEXT": "4000",
                "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
                "VLLM_CONTEXT_KEEP_TAIL_TOKENS": "1000",
                "VLLM_CONTEXT_FAILURE_CACHE_SECONDS": "300",
            },
            clear=False,
        ):
            first = await compact_context_if_needed(request, client)
            second = await compact_context_if_needed(request, client)

        self.assertEqual(client.calls, 1)
        self.assertLessEqual(_request_input_tokens(first), 2000)
        self.assertLessEqual(_request_input_tokens(second), 2000)


class ToolsReserveEstimateTests(unittest.TestCase):
    """The tools reserve must reflect what's ACTUALLY sent to vLLM (compacted),
    not the raw verbose Anthropic schemas. Over-reserving stole ~26k tokens from
    the conversation, collapsing the tail to ~3 messages."""

    def _verbose_tool(self, name, n_params=8):
        props = {
            f"p{i}": {
                "type": "string",
                "description": "D" * 300,
                "examples": ["e" * 150],
                "default": "x" * 100,
            }
            for i in range(n_params)
        }
        return {
            "name": name,
            "description": "X" * 1500,
            "input_schema": {
                "type": "object",
                "properties": props,
                "$defs": {"big": {"type": "string", "description": "Z" * 500}},
                "required": list(props.keys()),
            },
        }

    def test_estimate_uses_compacted_tools_not_raw(self):
        import json

        tools = [self._verbose_tool(n) for n in
                 ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task",
                  "TodoWrite", "WebFetch", "WebSearch", "LS", "create"]]
        request = AnthropicMessagesRequest(
            model="hosted_vllm/qwen",
            max_tokens=4096,
            messages=[{"role": "user", "content": "oi"}],
            tools=tools,
        )
        estimate = _estimate_tools_tokens(request)
        raw_serialized = json.dumps(
            [t.model_dump(exclude_none=True) for t in request.tools]
        )
        raw_estimate = int(len(raw_serialized) / 2.8)
        # The compacted estimate must be much smaller than the raw schema size.
        self.assertLess(estimate, raw_estimate * 0.6)
        self.assertGreater(raw_estimate - estimate, 10000)


class BackgroundSummaryTests(unittest.IsolatedAsyncioTestCase):
    """The default production behaviour: the summary must NOT block the response.
    The first over-budget turn returns immediately via the tail fallback and the
    summary is produced in the background, warming the cache for the next turn."""

    def setUp(self):
        _SUMMARY_CACHE.clear()
        _SUMMARY_FAILURE_CACHE.clear()

    def tearDown(self):
        _SUMMARY_CACHE.clear()
        _SUMMARY_FAILURE_CACHE.clear()

    async def _drain_background(self):
        # Wait for any spawned background summary tasks to finish.
        for _ in range(200):
            if not _BACKGROUND_SUMMARY_TASKS:
                return
            await asyncio.sleep(0.01)

    async def test_first_turn_does_not_block_on_summary(self):
        class _SlowSummaryClient:
            def __init__(self):
                self.calls = 0

            async def acompletion(self, **kwargs):
                self.calls += 1
                await asyncio.sleep(0.3)
                return {"choices": [{"message": {"content": "## Objetivo\nresumo ok"}}]}

        client = _SlowSummaryClient()
        request = _request(
            [
                {"role": "user", "content": "antigo " * 1800},
                {"role": "assistant", "content": "feito " * 1800},
                {"role": "user", "content": "tarefa atual " * 500},
            ]
        )
        env = {
            "VLLM_MODEL_CONTEXT": "4000",
            "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
            "VLLM_CONTEXT_KEEP_TAIL_TOKENS": "1000",
            "VLLM_CONTEXT_SUMMARY_BACKGROUND": "on",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            compacted = await compact_context_if_needed(request, client)
            elapsed = loop.time() - t0
            # Returned BEFORE the 0.3s summary call completed.
            self.assertLess(elapsed, 0.2)
            self.assertEqual(client.calls, 0)
            # Recent task survived in the fast fallback.
            self.assertIn("tarefa atual", _content(compacted.messages[-1]))
            # Background summary eventually runs.
            await self._drain_background()
            self.assertEqual(client.calls, 1)

    async def test_second_turn_reuses_background_cached_summary(self):
        class _SummaryClientCounting:
            def __init__(self):
                self.calls = 0

            async def acompletion(self, **kwargs):
                self.calls += 1
                return {"choices": [{"message": {"content": "## Objetivo\nresumo cacheado"}}]}

        client = _SummaryClientCounting()
        history = [
            {"role": "user", "content": "antigo " * 1800},
            {"role": "assistant", "content": "feito " * 1800},
            {"role": "user", "content": "tarefa atual " * 500},
        ]
        env = {
            "VLLM_MODEL_CONTEXT": "4000",
            "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
            "VLLM_CONTEXT_KEEP_TAIL_TOKENS": "1000",
            "VLLM_CONTEXT_SUMMARY_BACKGROUND": "on",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            await compact_context_if_needed(_request(list(history)), client)
            await self._drain_background()
            self.assertEqual(client.calls, 1)  # background produced + cached it

            # Next turn: same history (+ a new short turn). Must reuse the cache,
            # no new model call on the critical path.
            turn2 = history + [{"role": "user", "content": "proxima pergunta " * 100}]
            compacted = await compact_context_if_needed(_request(turn2), client)
            self.assertEqual(client.calls, 1)  # unchanged — cache hit
            self.assertIn(_SUMMARY_MARKER, _content(compacted.messages[0]))


class ProgressiveCompactionTests(unittest.IsolatedAsyncioTestCase):
    """The progressive compaction features: proactive (yellow-zone) warming,
    evolving summaries, and active forgetting."""

    def setUp(self):
        _SUMMARY_CACHE.clear()
        _SUMMARY_FAILURE_CACHE.clear()

    def tearDown(self):
        _SUMMARY_CACHE.clear()
        _SUMMARY_FAILURE_CACHE.clear()

    async def _drain_background(self):
        for _ in range(200):
            if not _BACKGROUND_SUMMARY_TASKS:
                return
            await asyncio.sleep(0.01)

    async def test_green_zone_does_nothing(self):
        """Well under budget: no model call, request returned unchanged."""
        client = _SummaryClient()
        request = _request([{"role": "user", "content": "oi"}])
        env = {
            "VLLM_MODEL_CONTEXT": "4000",
            "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
            "VLLM_CONTEXT_PROACTIVE_MARGIN": "500",
            "VLLM_CONTEXT_SUMMARY_BACKGROUND": "on",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            out = await compact_context_if_needed(request, client)
            await self._drain_background()
        self.assertIs(out, request)  # unchanged
        self.assertEqual(client.calls, [])  # no summary attempted

    async def test_yellow_zone_warms_cache_without_mutating_request(self):
        """Fits today but within the proactive margin: request returned
        UNCHANGED (lossless), but a summary is produced in the background."""
        client = _SummaryClient(content="## Objetivo\nresumo proativo")
        # Build a conversation that FITS but lands in the yellow zone.
        # budget = 8000 - 2000 - 1024 = 4976. margin 1500 => yellow at 3476.
        # Keep total between 3476 and 4976 tokens (~2.8 chars/tok).
        history = [
            {"role": "user", "content": "alfa " * 800},       # ~1430 tok
            {"role": "assistant", "content": "beta " * 800},   # ~1430 tok
            {"role": "user", "content": "gama agora " * 350},  # ~1250 tok
        ]
        request = _request(history)
        env = {
            "VLLM_MODEL_CONTEXT": "8000",
            "VLLM_CONTEXT_OUTPUT_RESERVE": "2000",
            "VLLM_CONTEXT_KEEP_TAIL_TOKENS": "1200",
            "VLLM_CONTEXT_PROACTIVE_MARGIN": "1500",
            "VLLM_CONTEXT_REFRESH_MIN_TOKENS": "100",
            "VLLM_CONTEXT_SUMMARY_BACKGROUND": "on",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            out = await compact_context_if_needed(request, client)
            # CRITICAL: request not mutated in the yellow zone (lossless).
            self.assertIs(out, request)
            # But the summary was warmed in the background.
            await self._drain_background()
        self.assertEqual(len(client.calls), 1)
        # And it's cached now (next overflow will hit it).
        self.assertTrue(len(_SUMMARY_CACHE) >= 1)

    async def test_evolving_summary_uses_evolve_prompt_on_second_pass(self):
        """When a prior (cached) summary exists, the second summarization must
        use the EVOLVING prompt (fold + distill), not re-summarize from zero."""
        captured_prompts = []

        class _CapturingClient:
            async def acompletion(self, **kwargs):
                captured_prompts.append(kwargs["messages"][0]["content"])
                return {"choices": [{"message": {"content": "## Objetivo\nevoluído"}}]}

        client = _CapturingClient()
        base = [
            {"role": "user", "content": "antigo " * 1800},
            {"role": "assistant", "content": "feito " * 1800},
            {"role": "user", "content": "tarefa atual " * 500},
        ]
        env = {
            "VLLM_MODEL_CONTEXT": "4000",
            "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
            "VLLM_CONTEXT_KEEP_TAIL_TOKENS": "1000",
            "VLLM_CONTEXT_REFRESH_MIN_TOKENS": "100",
            "VLLM_CONTEXT_PROACTIVE_MARGIN": "0",  # isolate: only red-zone path
            "VLLM_CONTEXT_SUMMARY_BACKGROUND": "on",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            # First pass: builds & caches the initial summary.
            await compact_context_if_needed(_request(list(base)), client)
            await self._drain_background()
            self.assertEqual(len(captured_prompts), 1)
            self.assertNotIn("RESUMO ATUAL", captured_prompts[0])  # first-pass prompt

            # Second pass: more messages → cache hit → evolving summary.
            turn2 = base + [
                {"role": "assistant", "content": "passo novo " * 400},
                {"role": "user", "content": "mais uma coisa " * 400},
            ]
            await compact_context_if_needed(_request(turn2), client)
            await self._drain_background()

        self.assertEqual(len(captured_prompts), 2)
        # The 2nd prompt is the EVOLVING one: contains the prior summary section.
        self.assertIn("RESUMO ATUAL", captured_prompts[1])
        self.assertIn("MENSAGENS NOVAS", captured_prompts[1])

    async def test_proactive_margin_zero_disables_yellow_zone(self):
        """With margin 0, a request that fits triggers NO background summary."""
        client = _SummaryClient()
        history = [
            {"role": "user", "content": "alfa " * 200},
            {"role": "assistant", "content": "beta " * 200},
            {"role": "user", "content": "gama " * 100},
        ]
        env = {
            "VLLM_MODEL_CONTEXT": "4000",
            "VLLM_CONTEXT_OUTPUT_RESERVE": "1000",
            "VLLM_CONTEXT_PROACTIVE_MARGIN": "0",
            "VLLM_CONTEXT_SUMMARY_BACKGROUND": "on",
        }
        request = _request(history)
        with mock.patch.dict(os.environ, env, clear=False):
            out = await compact_context_if_needed(request, client)
            await self._drain_background()
        self.assertIs(out, request)  # unchanged
        self.assertEqual(client.calls, [])  # nothing warmed (margin disabled)


if __name__ == "__main__":
    unittest.main()
