"""Tests for Step 9: vLLM/Qwen LLM adapter and prompt builder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    DEFAULT_QWEN_MODEL,
    ANTI_FABRICATION_PROMPT_RULES,
    LLMClientError,
    ToolRegistry,
    VLLMConfig,
    VLLMOpenAIClient,
    build_chat_messages,
    build_gridmind_prompt,
    extract_context_hints,
    parse_chat_completion_response,
    parse_completion_response,
    parse_openai_function_call,
    parse_tool_calls_from_text,
    remove_tool_call_blocks,
    render_qwen_chatml,
    is_qwen35_family_model,
    strip_qwen_thinking,
    tool_result_message,
)


class FakeVLLMClient(VLLMOpenAIClient):
    def __init__(
        self,
        response: Optional[Dict[str, Any]] = None,
        *,
        model: str = "qwen-test",
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            VLLMConfig(
                base_url="http://gpu-node:8000/v1",
                model=model,
                extra_body=extra_body or {},
            )
        )
        self.response = response or {"choices": [{"message": {"content": "ok"}}]}
        self.calls: list[tuple[str, str, Optional[Mapping[str, Any]]]] = []

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append((method, path, payload))
        return dict(self.response)


class RoutedFakeVLLMClient(VLLMOpenAIClient):
    def __init__(self, served_model_id: str = "my-local-qwen") -> None:
        super().__init__(VLLMConfig(base_url="http://127.0.0.1:9000/v1", model=DEFAULT_LOCAL_MODEL))
        self.served_model_id = served_model_id
        self.calls: list[tuple[str, str, Optional[Mapping[str, Any]]]] = []

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.calls.append((method, path, payload))
        if path == "/models":
            return {"data": [{"id": self.served_model_id}]}
        return {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}


class LLMAdapterTest(unittest.TestCase):
    def test_chat_posts_to_vllm_chat_endpoint_and_parses_native_tool_call(self) -> None:
        client = FakeVLLMClient(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "run_powerflow",
                                        "arguments": "{\"case_path\":\"ieee14\"}",
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )

        response = client.chat(
            [{"role": "user", "content": "Run a power flow on ieee14"}],
            tools=[{"type": "function", "function": {"name": "run_powerflow"}}],
        )

        self.assertEqual(client.calls[0][0:2], ("POST", "/chat/completions"))
        self.assertEqual(client.calls[0][2]["model"], "qwen-test")
        self.assertEqual(client.calls[0][2]["tools"][0]["function"]["name"], "run_powerflow")
        self.assertEqual(response.tool_calls[0].name, "run_powerflow")
        self.assertEqual(response.tool_calls[0].arguments, {"case_path": "ieee14"})
        self.assertEqual(response.finish_reason, "tool_calls")

    def test_completion_posts_to_vllm_completion_endpoint(self) -> None:
        client = FakeVLLMClient({"choices": [{"text": "hello", "finish_reason": "stop"}]})

        response = client.complete("<|im_start|>user\nhi<|im_end|>")

        self.assertEqual(client.calls[0][0:2], ("POST", "/completions"))
        self.assertEqual(client.calls[0][2]["prompt"], "<|im_start|>user\nhi<|im_end|>")
        self.assertEqual(response.text, "hello")

    def test_list_models_calls_models_endpoint(self) -> None:
        client = FakeVLLMClient({"data": [{"id": DEFAULT_QWEN_MODEL}]})

        models = client.list_models()

        self.assertEqual(client.calls[0][0:2], ("GET", "/models"))
        self.assertEqual(models["data"][0]["id"], DEFAULT_QWEN_MODEL)

    def test_model_auto_resolves_first_local_served_model(self) -> None:
        client = RoutedFakeVLLMClient()

        response = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(response.content, "done")
        self.assertEqual(client.calls[0][0:2], ("GET", "/models"))
        self.assertEqual(client.calls[1][0:2], ("POST", "/chat/completions"))
        self.assertEqual(client.calls[1][2]["model"], "my-local-qwen")

    def test_auto_resolved_qwen35_served_model_uses_nothink_template(self) -> None:
        client = RoutedFakeVLLMClient("Qwen/Qwen3.5-27B-FP8")

        client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(client.calls[1][2]["model"], "Qwen/Qwen3.5-27B-FP8")
        self.assertEqual(
            client.calls[1][2]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_qwen35_detector_matches_only_qwen35_family(self) -> None:
        self.assertTrue(is_qwen35_family_model("Qwen/Qwen3.5-35B-A3B"))
        self.assertTrue(is_qwen35_family_model("qwen3_5-27b"))
        self.assertTrue(is_qwen35_family_model("qwen35-27b"))
        self.assertFalse(is_qwen35_family_model("Qwen/Qwen3-32B"))
        self.assertFalse(is_qwen35_family_model("qwen3-5b"))
        self.assertFalse(is_qwen35_family_model("Qwen2.5-32B"))

    def test_qwen35_chat_request_disables_thinking_by_default(self) -> None:
        client = FakeVLLMClient(model="Qwen/Qwen3.5-35B-A3B")

        client.chat([{"role": "user", "content": "hello"}])

        body = client.calls[0][2]
        self.assertEqual(
            body["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_plain_qwen3_chat_request_does_not_change_thinking_template(self) -> None:
        client = FakeVLLMClient(model="Qwen/Qwen3-32B")

        client.chat([{"role": "user", "content": "hello"}])

        self.assertNotIn("chat_template_kwargs", client.calls[0][2])

    def test_explicit_extra_body_can_override_qwen35_auto_nothink(self) -> None:
        client = FakeVLLMClient(
            model="qwen35-27b",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

        client.chat(
            [{"role": "user", "content": "hello"}],
            extra_body={"chat_template_kwargs": {"custom": "value"}},
        )

        self.assertEqual(
            client.calls[0][2]["chat_template_kwargs"],
            {"enable_thinking": True, "custom": "value"},
        )

    def test_parse_qwen_text_tool_call_and_thinking_block(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<think>I should use a solver.</think>\n"
                            "<tool_call>{\"name\":\"inspect_violations\","
                            "\"arguments\":{\"case_path\":\"ieee118\"}}</tool_call>"
                        )
                    }
                }
            ]
        }

        response = parse_chat_completion_response(payload)

        self.assertEqual(response.reasoning_content, "I should use a solver.")
        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls[0].name, "inspect_violations")
        self.assertEqual(response.tool_calls[0].arguments, {"case_path": "ieee118"})

    def test_parse_legacy_openai_function_call(self) -> None:
        response = parse_chat_completion_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "function_call": {
                                "name": "run_powerflow",
                                "arguments": "{\"case_path\":\"ieee14\"}",
                            },
                        }
                    }
                ]
            }
        )

        self.assertEqual(response.tool_calls[0].source, "legacy_function_call")
        self.assertEqual(response.tool_calls[0].name, "run_powerflow")
        self.assertEqual(response.tool_calls[0].arguments, {"case_path": "ieee14"})

    def test_parse_legacy_function_call_rejects_invalid_shape(self) -> None:
        with self.assertRaises(LLMClientError):
            parse_openai_function_call("run_powerflow")

    def test_parse_fenced_json_tool_call_fallback(self) -> None:
        calls = parse_tool_calls_from_text(
            "```json\n"
            "{\"tool_calls\":[{\"function\":{\"name\":\"run_contingency\","
            "\"arguments\":\"{\\\"case_path\\\":\\\"ieee14\\\"}\"}}]}"
            "\n```"
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "run_contingency")
        self.assertEqual(calls[0].arguments, {"case_path": "ieee14"})

    def test_remove_tool_call_blocks_removes_parsed_blocks(self) -> None:
        cleaned = remove_tool_call_blocks(
            "before <tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call> after"
        )

        self.assertEqual(cleaned, "before  after")

    def test_strip_qwen_thinking_removes_chatml_wrappers(self) -> None:
        visible, reasoning = strip_qwen_thinking(
            "<|im_start|>assistant\n<think>plan</think>\nFinal answer.<|im_end|>"
        )

        self.assertEqual(visible, "Final answer.")
        self.assertEqual(reasoning, "plan")

    def test_render_qwen_chatml_for_completion_endpoint(self) -> None:
        prompt = render_qwen_chatml(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ]
        )

        self.assertIn("<|im_start|>system\nsys<|im_end|>", prompt)
        self.assertIn("<|im_start|>user\nhello<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_tool_result_message_serializes_and_truncates_result(self) -> None:
        call = parse_tool_calls_from_text(
            "{\"name\":\"run_powerflow\",\"arguments\":{\"case_path\":\"ieee14\"}}"
        )[0]

        message = tool_result_message(call, {"ok": True, "rows": "x" * 100}, max_chars=60)

        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["name"], "run_powerflow")
        self.assertIn("<truncated tool result>", message["content"])

    def test_invalid_payload_without_choices_raises(self) -> None:
        with self.assertRaises(LLMClientError):
            parse_completion_response({"not_choices": []})


class PromptBuilderTest(unittest.TestCase):
    def test_context_hints_extract_case_bus_mw_type_and_ibr(self) -> None:
        hints = extract_context_hints(
            "Run CIA for a 25 MW solar project at bus 10 on IEEE 118 with N-1."
        )

        self.assertEqual(hints.case_path, "ieee118")
        self.assertEqual(hints.bus, 10)
        self.assertEqual(hints.mw, 25.0)
        self.assertEqual(hints.connection_type, "solar")
        self.assertTrue(hints.is_ibr)
        self.assertTrue(hints.enable_contingency)

    def test_prompt_includes_gridmind_rules_tools_and_context_hints(self) -> None:
        prompt = build_gridmind_prompt(
            ToolRegistry(),
            history=[
                {
                    "role": "user",
                    "content": "Check a 10 MW data center at bus 5 on ieee14.",
                }
            ],
            lessons=["Do not call OPF before a basic violation scan."],
        )

        text = prompt.system_prompt
        self.assertIn("Mini Grid-Mind", text)
        self.assertIn(ANTI_FABRICATION_PROMPT_RULES, text)
        self.assertIn("find_max_capacity", text)
        self.assertIn("run_cia_with_mitigation [roadmap", text)
        self.assertIn("connection_type: load", text)
        self.assertIn("is_ibr: False", text)
        self.assertIn("Do not call OPF", text)

    def test_prompt_accepts_single_lesson_string(self) -> None:
        prompt = build_gridmind_prompt(
            ToolRegistry(),
            history=[{"role": "user", "content": "Run power flow on ieee14."}],
            lessons="Prefer run_powerflow for direct power-flow requests.",
        )

        self.assertIn("- Prefer run_powerflow for direct power-flow requests.", prompt.system_prompt)
        self.assertNotIn("- P\n- r\n- e\n- f", prompt.system_prompt)

    def test_build_chat_messages_appends_current_user_message(self) -> None:
        prompt = build_chat_messages(
            ToolRegistry(),
            "Run power flow on ieee14.",
            history=[{"role": "assistant", "content": "Ready."}],
        )

        messages = prompt.messages

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertEqual(prompt.context_hints.case_path, "ieee14")


if __name__ == "__main__":
    unittest.main()
