import sys
import types
import unittest
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "rotator_library"

if "litellm.exceptions" not in sys.modules:
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
        setattr(
            litellm_exceptions,
            exception_name,
            type(exception_name, (Exception,), {}),
        )
    sys.modules.setdefault("litellm.exceptions", litellm_exceptions)

rotator_package = types.ModuleType("rotator_library")
rotator_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("rotator_library", rotator_package)

from rotator_library.error_handler import classify_error


class ErrorHandlerTests(unittest.TestCase):
    def test_cache_access_denied_string_is_upstream_connection_error(self):
        classified = classify_error(
            Exception("API Error: ERROR: Cache Access Denied"),
            provider="hosted_vllm",
        )

        self.assertEqual(classified.error_type, "api_connection")
        self.assertEqual(classified.status_code, 503)

    def test_cache_access_denied_http_403_is_not_treated_as_forbidden_key(self):
        request = httpx.Request("POST", "https://example.test/v1/messages")
        response = httpx.Response(
            403,
            request=request,
            text="ERROR: Cache Access Denied",
        )
        error = httpx.HTTPStatusError(
            "403 Forbidden",
            request=request,
            response=response,
        )

        classified = classify_error(error, provider="hosted_vllm")

        self.assertEqual(classified.error_type, "api_connection")
        self.assertEqual(classified.status_code, 503)


if __name__ == "__main__":
    unittest.main()
