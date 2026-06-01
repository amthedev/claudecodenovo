import asyncio
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "rotator_library"

litellm_package = types.ModuleType("litellm")
litellm_package.get_model_info = lambda model: {}
sys.modules.setdefault("litellm", litellm_package)

litellm_exceptions = types.ModuleType("litellm.exceptions")
for exception_name in (
    "APIConnectionError",
    "RateLimitError",
    "ServiceUnavailableError",
    "AuthenticationError",
    "InvalidRequestError",
    "BadRequestError",
    "OpenAIError",
    "InternalServerError",
    "Timeout",
    "ContextWindowExceededError",
):
    setattr(litellm_exceptions, exception_name, type(exception_name, (Exception,), {}))
sys.modules.setdefault("litellm.exceptions", litellm_exceptions)

rotator_package = types.ModuleType("rotator_library")
rotator_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("rotator_library", rotator_package)

client_package = types.ModuleType("rotator_library.client")
client_package.__path__ = [str(PACKAGE_ROOT / "client")]
sys.modules.setdefault("rotator_library.client", client_package)

from rotator_library.client.streaming import StreamingHandler
from rotator_library.core.errors import StreamedAPIError


class StreamingTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_total_timeout_closes_stream_that_keeps_request_open(self):
        async def stalled_stream():
            yield {"choices": [{"delta": {"content": "inicio"}}]}
            await asyncio.sleep(10)

        chunks = []
        with mock.patch.dict(
            os.environ, {"TIMEOUT_TOTAL_STREAMING": "0.05"}, clear=False
        ):
            with self.assertRaisesRegex(
                StreamedAPIError, "Stream exceeded total timeout"
            ):
                async for chunk in StreamingHandler().wrap_stream(
                    stalled_stream(),
                    "secret",
                    "hosted_vllm/qwen",
                ):
                    chunks.append(chunk)

        self.assertEqual(len(chunks), 1)
        self.assertIn("inicio", chunks[0])


if __name__ == "__main__":
    unittest.main()
