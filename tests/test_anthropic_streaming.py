import json
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

from rotator_library.anthropic_compat.streaming import anthropic_streaming_wrapper
from rotator_library.anthropic_compat.models import AnthropicMessagesRequest
from rotator_library.anthropic_compat.translator import (
    _build_vllm_forced_tool_call,
    _parse_textual_tool_calls,
    anthropic_to_openai_messages,
    openai_to_anthropic_response,
    translate_anthropic_request,
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
    def test_server_side_tools_without_input_schema_are_ignored(self):
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "oi"}],
            tools=[
                {
                    "type": "advisor_20260301",
                    "name": "advisor",
                    "model": "claude-sonnet-4-6",
                },
                {
                    "name": "Bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            ],
        )

        openai_request = translate_anthropic_request(request)

        self.assertEqual(len(openai_request["tools"]), 1)
        self.assertEqual(openai_request["tools"][0]["function"]["name"], "Bash")

    def test_providerless_model_uses_native_tool_contract_by_default(self):
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            stream=True,
            messages=[
                {"role": "user", "content": "crie o jogo da cobrinha em python"}
            ],
            tools=[
                {
                    "name": "Write",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["file_path", "content"],
                    },
                }
            ],
        )

        openai_request = translate_anthropic_request(request)

        self.assertIn("tools", openai_request)
        self.assertNotIn("_vllm_forced_tool_call", openai_request)
        self.assertFalse(
            any(
                "CURRENT REQUEST REQUIRES TOOL USE" in (message.get("content") or "")
                for message in openai_request["messages"]
                if message.get("role") == "system"
            )
        )

    def test_providerless_fallback_is_explicit_opt_in(self):
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            stream=True,
            messages=[
                {"role": "user", "content": "crie o jogo da cobrinha em python"}
            ],
            tools=[
                {
                    "name": "Write",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["file_path", "content"],
                    },
                }
            ],
        )

        with mock.patch.dict(
            "os.environ", {"ANTHROPIC_COMPAT_FORCE_TOOL_FALLBACK": "true"}
        ):
            openai_request = translate_anthropic_request(request)

        fallback = openai_request.get("_vllm_forced_tool_call")
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback["name"], "Write")
        self.assertEqual(fallback["_proxy_strategy"], "write_from_model_text")
        self.assertEqual(fallback["input"]["file_path"], "snake_game.py")

    def test_long_tool_names_round_trip_through_openai_limit(self):
        long_name = (
            "mcp__project_management__create_issue_with_extended_context"
            "__and_repository_metadata"
        )
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            stream=False,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": long_name,
                            "input": {"title": "Fix"},
                        }
                    ],
                }
            ],
            tools=[
                {
                    "name": long_name,
                    "input_schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                    },
                }
            ],
            tool_choice={"type": "tool", "name": long_name},
        )

        openai_request = translate_anthropic_request(request)
        mapping = openai_request["_anthropic_tool_name_mapping"]
        mapped_name = openai_request["tools"][0]["function"]["name"]

        self.assertLessEqual(len(mapped_name), 64)
        self.assertEqual(mapping[mapped_name], long_name)
        self.assertEqual(
            openai_request["messages"][0]["tool_calls"][0]["function"]["name"],
            mapped_name,
        )
        self.assertEqual(
            openai_request["tool_choice"]["function"]["name"],
            mapped_name,
        )

        anthropic_response = openai_to_anthropic_response(
            {
                "id": "chatcmpl_1",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": mapped_name,
                                        "arguments": '{"title":"Fix"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            "claude-sonnet-4-5",
            tool_name_mapping=mapping,
        )

        self.assertEqual(anthropic_response["content"][0]["name"], long_name)
        self.assertEqual(anthropic_response["stop_reason"], "tool_use")

        textual_response = openai_to_anthropic_response(
            {
                "id": "chatcmpl_2",
                "choices": [
                    {
                        "message": {
                            "content": (
                                f"<function={mapped_name}>"
                                "<parameter=title>Fix</parameter>"
                                "</function>"
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            "claude-sonnet-4-5",
            tool_name_mapping=mapping,
        )
        self.assertEqual(textual_response["content"][0]["name"], long_name)
        self.assertEqual(textual_response["stop_reason"], "tool_use")

    def test_empty_text_blocks_are_not_forwarded_to_openai(self):
        openai_messages = anthropic_to_openai_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Bash",
                            "input": {"command": "pwd"},
                        },
                    ],
                }
            ]
        )

        self.assertIsNone(openai_messages[0]["content"])
        self.assertEqual(openai_messages[0]["tool_calls"][0]["function"]["name"], "Bash")

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

    def test_project_inspection_fallback_excludes_sensitive_files(self):
        fallback = _build_vllm_forced_tool_call(
            "inspect",
            [{"function": {"name": "Bash"}}],
            {
                "messages": [
                    {"role": "user", "content": "analise todo o projeto"}
                ],
                "_vllm_previous_tool_count": 0,
            },
        )

        self.assertIsNotNone(fallback)
        command = fallback["input"]["command"]
        self.assertIn("-not -iname '.env'", command)
        self.assertIn("-not -iname '*secret*'", command)
        self.assertIn("-not -iname '*credential*'", command)
        self.assertIn("-not -iname '*api_key*'", command)
        self.assertIn("-not -iname '*.pem'", command)

    def test_mandatory_prompt_warns_not_to_hunt_for_credentials(self):
        request = AnthropicMessagesRequest(
            model="hosted_vllm/qwen",
            max_tokens=1024,
            stream=True,
            messages=[
                {"role": "user", "content": "analise todo o projeto"}
            ],
            tools=[
                {
                    "name": "Bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
        )

        openai_request = translate_anthropic_request(request)
        self.assertNotIn("tools", openai_request)
        self.assertNotIn("_vllm_forced_tool_call", openai_request)
        system_text = "\n".join(
            message.get("content", "")
            for message in openai_request["messages"]
            if message.get("role") == "system"
        )
        self.assertIn("Do not inspect, grep, read", system_text)
        self.assertIn(".env files, secrets, tokens, API keys", system_text)

    def test_hosted_vllm_native_tools_are_passed_cleanly(self):
        """Native mode must pass tools through WITHOUT injecting textual tool-call
        instructions. Injecting the <tool_call> textual format alongside native
        tool calling makes the model flip-flop between formats — the root cause of
        the intermittent "doesn't edit / acts as chat / hallucinates" behavior."""
        request = AnthropicMessagesRequest(
            model="hosted_vllm/qwen",
            max_tokens=1024,
            messages=[{"role": "user", "content": "crie agent_probe.txt"}],
            tools=[
                {
                    "name": "Write",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                }
            ],
        )

        with mock.patch.dict("os.environ", {"HOSTED_VLLM_NATIVE_TOOLS": "true"}):
            openai_request = translate_anthropic_request(request)

        # Tools must be present in the native field.
        self.assertIn("tools", openai_request)
        self.assertEqual(openai_request["tools"][0]["function"]["name"], "Write")

        system_text = "\n".join(
            message.get("content", "")
            for message in openai_request["messages"]
            if message.get("role") == "system"
        )
        # The textual tool-call format must NOT be injected in native mode.
        self.assertNotIn("<tool_call><function=ToolName>", system_text)
        self.assertNotIn("CURRENT REQUEST REQUIRES TOOL USE", system_text)
        # No forced-tool fallback unless explicitly opted in.
        self.assertNotIn("_vllm_forced_tool_call", openai_request)

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

    def test_parses_execute_json_tool_call(self):
        text = (
            "Vou criar o arquivo agora.\n"
            "<execute>\n"
            '{"tool":"Write","input":{"path":"agent_probe.txt","content":"agent-ok"}}\n'
            "</execute>"
        )

        cleaned_text, tool_blocks = _parse_textual_tool_calls(text)

        self.assertEqual(cleaned_text, "Vou criar o arquivo agora.")
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["name"], "Write")
        self.assertEqual(
            tool_blocks[0]["input"],
            {"file_path": "agent_probe.txt", "content": "agent-ok"},
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

    async def test_converts_execute_json_stream_to_tool_use(self):
        events = await _collect_events(
            [
                {"choices": [{"delta": {"content": "Vou criar o arquivo agora.\n"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    "<execute>\n"
                                    '{"tool":"Write","input":{"path":"agent_probe.txt",'
                                    '"content":"agent-ok"}}\n'
                                    "</execute>"
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
        self.assertNotIn("<execute>", text_delta)

        tool_start = next(
            data
            for event_type, data in events
            if event_type == "content_block_start"
            and data["content_block"]["type"] == "tool_use"
        )
        self.assertEqual(tool_start["content_block"]["name"], "Write")

        partial_json = "".join(
            data["delta"]["partial_json"]
            for event_type, data in events
            if event_type == "content_block_delta"
            and data["delta"]["type"] == "input_json_delta"
        )
        self.assertEqual(
            json.loads(partial_json),
            {"file_path": "agent_probe.txt", "content": "agent-ok"},
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

    async def test_stream_maps_length_finish_reason_to_max_tokens(self):
        events = await _collect_events(
            [
                {"choices": [{"delta": {"content": "texto"}}]},
                {"choices": [{"delta": {}, "finish_reason": "length"}]},
            ]
        )

        message_delta = next(
            data for event_type, data in events if event_type == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "max_tokens")


if __name__ == "__main__":
    unittest.main()
