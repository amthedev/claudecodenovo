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
    _SUMMARY_CACHE,
    _SUMMARY_MARKER,
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
    def setUp(self):
        _SUMMARY_CACHE.clear()

    def tearDown(self):
        _SUMMARY_CACHE.clear()

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
        self.assertEqual(client.calls[0]["max_tokens"], 1000)
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

    async def test_summary_timeout_falls_back_without_breaking_request(self):
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

        self.assertIs(compacted, request)


if __name__ == "__main__":
    unittest.main()
