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

from rotator_library.anthropic_compat.image_captioning import (
    _DEFAULT_VISION_MODEL,
    _DESCRIPTION_CACHE,
    _FAILURE_CACHE,
    _FAILURE_PLACEHOLDER,
    caption_images_in_request,
    _describe_image,
)
from rotator_library.anthropic_compat.models import AnthropicMessagesRequest


class _VisionClient:
    def __init__(self, content="interface descrita"):
        self.calls = []
        self.content = content

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.content}}]}


class ImageCaptioningTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _DESCRIPTION_CACHE.clear()
        _FAILURE_CACHE.clear()

    def tearDown(self):
        _DESCRIPTION_CACHE.clear()
        _FAILURE_CACHE.clear()

    def test_default_vision_model_is_current_openrouter_id(self):
        self.assertEqual(
            _DEFAULT_VISION_MODEL,
            "openrouter/qwen/qwen3-vl-8b-instruct",
        )

    async def test_successful_description_is_reused_from_cache(self):
        client = _VisionClient()

        first = await _describe_image(
            "data:image/png;base64,AAAA",
            "descreva a tela",
            client,
            _DEFAULT_VISION_MODEL,
            256,
            mock.Mock(),
        )
        second = await _describe_image(
            "data:image/png;base64,AAAA",
            "descreva a tela",
            client,
            _DEFAULT_VISION_MODEL,
            256,
            mock.Mock(),
        )

        self.assertEqual(first, second)
        self.assertEqual(len(client.calls), 1)

    async def test_timeout_returns_placeholder(self):
        class _SlowVisionClient:
            async def acompletion(self, **kwargs):
                await asyncio.sleep(10)

        with mock.patch.dict(
            os.environ, {"VISION_TIMEOUT_SECONDS": "1"}, clear=False
        ):
            result = await _describe_image(
                "data:image/png;base64,BBBB",
                "",
                _SlowVisionClient(),
                _DEFAULT_VISION_MODEL,
                256,
                mock.Mock(),
            )

        self.assertEqual(result, _FAILURE_PLACEHOLDER)

    async def test_failure_placeholder_is_reused_briefly(self):
        class _FailingVisionClient:
            def __init__(self):
                self.calls = 0

            async def acompletion(self, **kwargs):
                self.calls += 1
                raise RuntimeError("temporary failure")

        client = _FailingVisionClient()
        first = await _describe_image(
            "data:image/png;base64,CCCC",
            "",
            client,
            _DEFAULT_VISION_MODEL,
            256,
            mock.Mock(),
        )
        second = await _describe_image(
            "data:image/png;base64,CCCC",
            "",
            client,
            _DEFAULT_VISION_MODEL,
            256,
            mock.Mock(),
        )

        self.assertEqual(first, _FAILURE_PLACEHOLDER)
        self.assertEqual(second, _FAILURE_PLACEHOLDER)
        self.assertEqual(client.calls, 1)

    async def test_legacy_configured_model_is_mapped_to_current_id(self):
        client = _VisionClient()
        request = AnthropicMessagesRequest(
            model="hosted_vllm/qwen",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AAAA",
                            },
                        }
                    ],
                }
            ],
        )

        with mock.patch.dict(
            os.environ,
            {"VISION_MODEL": "openrouter/qwen/qwen-2.5-vl-7b-instruct"},
            clear=False,
        ):
            await caption_images_in_request(request, client)

        self.assertEqual(client.calls[0]["model"], _DEFAULT_VISION_MODEL)


if __name__ == "__main__":
    unittest.main()
