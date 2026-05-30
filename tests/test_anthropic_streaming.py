import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "rotator_library"

rotator_package = types.ModuleType("rotator_library")
rotator_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("rotator_library", rotator_package)

anthropic_package = types.ModuleType("rotator_library.anthropic_compat")
anthropic_package.__path__ = [str(PACKAGE_ROOT / "anthropic_compat")]
sys.modules.setdefault("rotator_library.anthropic_compat", anthropic_package)

from rotator_library.anthropic_compat.streaming import anthropic_streaming_wrapper
from rotator_library.anthropic_compat.translator import (
    _build_vllm_forced_tool_call,
    _parse_textual_tool_calls,
)


async def _stream(chunks):
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _collect_events(chunks, forced_tool_call=None):
    events = []
    async for event in anthropic_streaming_wrapper(
        _stream(chunks), "claude-test", forced_tool_call=forced_tool_call
    ):
        if not event.startswith("event: "):
            continue
        event_type = event.split("\n", 1)[0].removeprefix("event: ")
        data_line = next(line for line in event.splitlines() if line.startswith("data: "))
        events.append((event_type, json.loads(data_line.removeprefix("data: "))))
    return events


class AnthropicStreamingToolUseTests(unittest.IsolatedAsyncioTestCase):
    def test_generic_create_intent_uses_write_from_model_text_fallback(self):
        fallback = _build_vllm_forced_tool_call(
            "create",
            [{"function": {"name": "Write"}}],
            {
                "messages": [
                    {"role": "user", "content": "fassa um jogo da cobrinha m oython"}
                ],
                "_vllm_previous_tool_count": 0,
            },
        )

        self.assertIsNotNone(fallback)
        self.assertEqual(fallback["name"], "Write")
        self.assertEqual(fallback["_proxy_strategy"], "write_from_model_text")
        self.assertEqual(fallback["input"]["file_path"], "snake_game.py")

    def test_parses_malformed_textual_tool_call_without_closing_tags(self):
        text = (
            "Vou escrever o arquivo agora.\n"
            "<function=Write><parameter=file_path>cronometro_avancado.py"
            "<parameter=content>print('ok')\n"
        )

        cleaned_text, tool_blocks = _parse_textual_tool_calls(text)

        self.assertEqual(cleaned_text, "Vou escrever o arquivo agora.")
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["name"], "Write")
        self.assertEqual(
            tool_blocks[0]["input"],
            {"file_path": "cronometro_avancado.py", "content": "print('ok')"},
        )

    async def test_converts_textual_tool_call_after_prose_to_tool_use(self):
        events = await _collect_events(
            [
                {"choices": [{"delta": {"content": "Vou escrever o arquivo agora.\n"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    "<function=Write>"
                                    "<parameter=file_path>cronometro_avancado.py"
                                    "<parameter=content>print('ok')\n"
                                )
                            }
                        }
                    ]
                },
            ]
        )

        text_delta = "".join(
            data["delta"]["text"]
            for event_type, data in events
            if event_type == "content_block_delta"
            and data["delta"]["type"] == "text_delta"
        )
        self.assertNotIn("<function=", text_delta)

        tool_starts = [
            data
            for event_type, data in events
            if event_type == "content_block_start"
            and data["content_block"]["type"] == "tool_use"
        ]
        self.assertEqual(len(tool_starts), 1)
        self.assertEqual(tool_starts[0]["content_block"]["name"], "Write")

        partial_json = "".join(
            data["delta"]["partial_json"]
            for event_type, data in events
            if event_type == "content_block_delta"
            and data["delta"]["type"] == "input_json_delta"
        )
        self.assertEqual(
            json.loads(partial_json),
            {"file_path": "cronometro_avancado.py", "content": "print('ok')"},
        )

        message_delta = next(
            data for event_type, data in events if event_type == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    async def test_converts_plain_code_response_to_write_for_mandatory_create(self):
        forced_tool_call = {
            "id": "toolu_proxy_create",
            "name": "Write",
            "input": {"file_path": "snake_game.py", "content": ""},
            "_proxy_strategy": "write_from_model_text",
            "_proxy_file_path_hint": "snake_game.py",
        }
        events = await _collect_events(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    "Vou criar um jogo da cobrinha em Python.\n"
                                    "Vamos salvar em snake_game.py.\n\n"
                                    "import curses\n"
                                    "import random\n\n"
                                    "def main(stdscr):\n"
                                    "    stdscr.addstr('snake')\n\n"
                                    "if __name__ == '__main__':\n"
                                    "    curses.wrapper(main)\n\n"
                                    "Para executar: python3 snake_game.py\n"
                                )
                            }
                        }
                    ]
                },
            ],
            forced_tool_call=forced_tool_call,
        )

        text_delta = "".join(
            data["delta"]["text"]
            for event_type, data in events
            if event_type == "content_block_delta"
            and data["delta"]["type"] == "text_delta"
        )
        self.assertEqual(text_delta, "")

        partial_json = "".join(
            data["delta"]["partial_json"]
            for event_type, data in events
            if event_type == "content_block_delta"
            and data["delta"]["type"] == "input_json_delta"
        )
        tool_input = json.loads(partial_json)
        self.assertEqual(tool_input["file_path"], "snake_game.py")
        self.assertIn("import curses", tool_input["content"])
        self.assertIn("curses.wrapper(main)", tool_input["content"])
        self.assertNotIn("Para executar", tool_input["content"])

        message_delta = next(
            data for event_type, data in events if event_type == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    async def test_delays_tool_block_until_name_is_known(self):
        events = await _collect_events(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_123",
                                        "type": "function",
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "Write",
                                            "arguments": '{"file_path":"calculadora.py"',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": ',"content":"print(1)"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
            ]
        )

        starts = [
            data
            for event_type, data in events
            if event_type == "content_block_start"
            and data["content_block"]["type"] == "tool_use"
        ]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["content_block"]["id"], "call_123")
        self.assertEqual(starts[0]["content_block"]["name"], "Write")

        partial_json = "".join(
            data["delta"]["partial_json"]
            for event_type, data in events
            if event_type == "content_block_delta"
            and data["delta"]["type"] == "input_json_delta"
        )
        self.assertEqual(
            json.loads(partial_json),
            {"file_path": "calculadora.py", "content": "print(1)"},
        )

        message_delta = next(
            data for event_type, data in events if event_type == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    async def test_buffers_arguments_that_arrive_before_name(self):
        events = await _collect_events(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_123",
                                        "function": {"arguments": '{"command":"pwd"}'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"name": "Bash"},
                                    }
                                ]
                            }
                        }
                    ]
                },
            ]
        )

        partial_json = "".join(
            data["delta"]["partial_json"]
            for event_type, data in events
            if event_type == "content_block_delta"
            and data["delta"]["type"] == "input_json_delta"
        )
        self.assertEqual(json.loads(partial_json), {"command": "pwd"})


if __name__ == "__main__":
    unittest.main()
