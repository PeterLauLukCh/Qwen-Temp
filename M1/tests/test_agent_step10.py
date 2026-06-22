"""Tests for Step 10: Mini Grid-Mind agent loop."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (  # noqa: E402
    AgentConfig,
    ChatCompletion,
    GridMindAgent,
    GROUNDING_WARNING,
    ToolCall,
    ToolRegistry,
    ToolRegistryError,
)


class FakeRegistry(ToolRegistry):
    def __init__(self) -> None:
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def list_tools(self, include_unimplemented: bool = True) -> Dict[str, Any]:
        return {
            "active_backend": "fake",
            "tool_count": 2,
            "tools": [
                {
                    "name": "run_powerflow",
                    "description": "Fake power-flow tool.",
                    "group": "analysis",
                    "implemented": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"case_path": {"type": "string"}},
                        "required": ["case_path"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "find_max_capacity",
                    "description": "Fake capacity search tool.",
                    "group": "capacity_search",
                    "implemented": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "case_path": {"type": "string"},
                            "bus": {"type": "integer"},
                            "connection_type": {"type": "string"},
                        },
                        "required": ["case_path", "bus", "connection_type"],
                        "additionalProperties": False,
                    },
                },
            ],
        }

    def openai_tool_specs(self, include_unimplemented: bool = False) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self.list_tools()["tools"]
        ]

    def call_tool(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        args = dict(arguments or {})
        self.calls.append((name, args))
        if name == "run_powerflow":
            if "case_path" not in args:
                raise ToolRegistryError("Missing required argument 'run_powerflow.case_path'")
            return {
                "ok": True,
                "tool": "run_powerflow",
                "case_info": {"case_name": args["case_path"]},
                "powerflow_summary": {
                    "converged": True,
                    "bus_summary": {"min_vm_pu": 0.99, "max_vm_pu": 1.02},
                    "branch_summary": {"max_line_loading_percent": 55.0},
                },
                "violation_report": {
                    "status": "pass",
                    "passed": True,
                    "total_violations": 0,
                    "hard_count": 0,
                    "borderline_count": 0,
                },
            }
        if name == "find_max_capacity":
            return {
                "ok": True,
                "tool": "find_max_capacity",
                "case_path": args["case_path"],
                "request": {
                    "bus": args["bus"],
                    "connection_type": args["connection_type"],
                    "is_ibr": False,
                },
                "status": "max_bound_approved",
                "max_approved_mw": 20.0,
                "lower_bound_mw": 20.0,
                "upper_bound_mw": None,
                "tolerance_mw": 1.0,
                "rejection_explanation": None,
            }
        raise ToolRegistryError(f"Unknown fake tool {name}")


class FakeLLMClient:
    def __init__(self, responses: List[ChatCompletion]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages: Any, **kwargs: Any) -> ChatCompletion:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("FakeLLMClient received more calls than expected")
        return self.responses.pop(0)


def tool_response(name: str, arguments: Dict[str, Any], *, call_id: str = "call_1") -> ChatCompletion:
    return ChatCompletion(
        content="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                source="native",
            )
        ],
        reasoning_content="",
        finish_reason="tool_calls",
        raw={},
    )


def final_response(text: str) -> ChatCompletion:
    return ChatCompletion(
        content=text,
        tool_calls=[],
        reasoning_content="",
        finish_reason="stop",
        raw={},
    )


class AgentLoopTest(unittest.TestCase):
    def test_agent_executes_tool_and_returns_grounded_final_answer(self) -> None:
        registry = FakeRegistry()
        llm = FakeLLMClient(
            [
                tool_response("run_powerflow", {"case_path": "ieee14"}),
                final_response("Power flow converged for ieee14 with 0 violations."),
            ]
        )
        agent = GridMindAgent(registry=registry, llm_client=llm)

        result = agent.run_turn("Run a power flow on ieee14.")

        self.assertEqual(result.status, "completed")
        self.assertEqual(registry.calls, [("run_powerflow", {"case_path": "ieee14"})])
        self.assertEqual(result.invoked_tools, ["run_powerflow"])
        self.assertFalse(result.grounding.warning_appended)
        self.assertIn("Power flow converged", result.output_text)
        self.assertEqual(llm.calls[1]["messages"][-1]["role"], "tool")
        tool_payload = json.loads(llm.calls[1]["messages"][-1]["content"])
        self.assertIn("observation", tool_payload)
        self.assertIn("raw_result", tool_payload)
        self.assertEqual(tool_payload["observation"]["tool"], "run_powerflow")
        self.assertEqual(result.tool_records[0].observation["tool"], "run_powerflow")
        self.assertIsNotNone(result.deterministic_report)
        self.assertEqual(result.deterministic_report.source_tool, "run_powerflow")
        self.assertIn("Power-flow report for ieee14", result.deterministic_report.summary_text)
        self.assertEqual(
            result.to_dict(include_messages=False)["deterministic_report"]["source_tool"],
            "run_powerflow",
        )

    def test_agent_can_send_observation_without_raw_tool_result(self) -> None:
        registry = FakeRegistry()
        llm = FakeLLMClient(
            [
                tool_response("run_powerflow", {"case_path": "ieee14"}),
                final_response("Power flow converged for ieee14."),
            ]
        )
        agent = GridMindAgent(
            registry=registry,
            llm_client=llm,
            config=AgentConfig(include_raw_tool_result_in_message=False),
        )

        result = agent.run_turn("Run a power flow on ieee14.")

        self.assertEqual(result.status, "completed")
        tool_payload = json.loads(llm.calls[1]["messages"][-1]["content"])
        self.assertIn("observation", tool_payload)
        self.assertNotIn("raw_result", tool_payload)

    def test_agent_uses_deterministic_report_when_final_text_is_empty(self) -> None:
        registry = FakeRegistry()
        llm = FakeLLMClient(
            [
                tool_response("run_powerflow", {"case_path": "ieee14"}),
                final_response(""),
            ]
        )
        agent = GridMindAgent(registry=registry, llm_client=llm)

        result = agent.run_turn("Run a power flow on ieee14.")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.raw_output_text, "")
        self.assertIsNotNone(result.deterministic_report)
        self.assertEqual(result.output_text, result.deterministic_report.summary_text)
        self.assertIn("Violation status: status=pass", result.output_text)

    def test_agent_appends_grounding_warning_without_tool_credit(self) -> None:
        agent = GridMindAgent(
            registry=FakeRegistry(),
            llm_client=FakeLLMClient([final_response("The capacity is 127 MW.")]),
            config=AgentConfig(enable_forced_capacity_routing=False),
        )

        result = agent.run_turn("Guess the capacity.")

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.grounding.warning_appended)
        self.assertIn(GROUNDING_WARNING, result.output_text)
        self.assertEqual(result.invoked_tools, [])

    def test_forced_capacity_route_bypasses_llm_when_ready(self) -> None:
        registry = FakeRegistry()
        llm = FakeLLMClient([])
        agent = GridMindAgent(registry=registry, llm_client=llm)

        result = agent.run_turn("What is the max load capacity at bus 10 on ieee14?")

        self.assertEqual(result.status, "forced_capacity_executed")
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(registry.calls[0][0], "find_max_capacity")
        self.assertEqual(result.invoked_tools, ["find_max_capacity"])
        self.assertIn("Maximum approved capacity: 20 MW", result.output_text)
        self.assertFalse(result.grounding.warning_appended)
        self.assertIsNotNone(result.deterministic_report)
        self.assertEqual(result.deterministic_report.source_tool, "find_max_capacity")

    def test_forced_capacity_route_returns_clarification_when_missing_type(self) -> None:
        agent = GridMindAgent(registry=FakeRegistry(), llm_client=FakeLLMClient([]))

        result = agent.run_turn("What is the max capacity at bus 10 on ieee14?")

        self.assertEqual(result.status, "clarification_required")
        self.assertIn("resource type", result.output_text)
        self.assertEqual(result.tool_records, [])

    def test_cia_readiness_gate_returns_clarification_without_llm(self) -> None:
        llm = FakeLLMClient([])
        agent = GridMindAgent(registry=FakeRegistry(), llm_client=llm)

        result = agent.run_turn("Run CIA for a solar project at bus 10 on ieee14.")

        self.assertEqual(result.status, "clarification_required")
        self.assertIn("project size", result.output_text)
        self.assertEqual(len(llm.calls), 0)
        self.assertEqual(result.invoked_tools, [])
        self.assertIsNotNone(result.readiness_check)
        self.assertEqual(
            result.readiness_check["decision"]["missing_inputs"],
            ["p_mw"],
        )

    def test_complete_cia_request_passes_to_llm_planner(self) -> None:
        llm = FakeLLMClient([final_response("I can now call run_cia with the provided inputs.")])
        agent = GridMindAgent(registry=FakeRegistry(), llm_client=llm)

        result = agent.run_turn(
            "Run CIA for a 25 MW solar project at bus 10 on ieee14 with N-1."
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(llm.calls), 1)
        self.assertIsNone(result.readiness_check)
        self.assertIn("run_cia", result.output_text)

    def test_specific_sized_host_project_is_not_forced_to_capacity_search(self) -> None:
        registry = FakeRegistry()
        llm = FakeLLMClient([final_response("I should evaluate this as a specific CIA request.")])
        agent = GridMindAgent(registry=registry, llm_client=llm)

        result = agent.run_turn("Can bus 10 host a 25 MW solar project on ieee118?")

        self.assertEqual(result.status, "completed")
        self.assertEqual(registry.calls, [])
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("CIA", result.output_text)

    def test_context_sized_host_project_is_not_forced_to_capacity_search(self) -> None:
        registry = FakeRegistry()
        llm = FakeLLMClient([final_response("I should evaluate the provided 25 MW project as CIA.")])
        agent = GridMindAgent(registry=registry, llm_client=llm)

        result = agent.run_turn(
            "Can bus 10 host a solar project on ieee118?",
            context={"mw": 25.0},
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(registry.calls, [])
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("CIA", result.output_text)

    def test_tool_policy_guard_blocks_wrong_capacity_tool_for_cia_request(self) -> None:
        registry = FakeRegistry()
        llm = FakeLLMClient(
            [
                tool_response(
                    "find_max_capacity",
                    {"case_path": "ieee118", "bus": 10, "connection_type": "solar"},
                ),
                final_response("I should use run_cia for the specified 25 MW project."),
            ]
        )
        agent = GridMindAgent(registry=registry, llm_client=llm)

        result = agent.run_turn("Can bus 10 host a 25 MW solar project on ieee118?")

        self.assertEqual(result.status, "completed")
        self.assertEqual(registry.calls, [])
        self.assertEqual(len(result.tool_records), 1)
        self.assertFalse(result.tool_records[0].ok)
        self.assertEqual(result.tool_records[0].source, "tool_call_policy_guard")
        self.assertEqual(result.tool_records[0].result["error_type"], "tool_policy_violation")
        self.assertEqual(result.tool_records[0].result["recommended_tool"], "run_cia")
        self.assertEqual(llm.calls[1]["messages"][-1]["role"], "tool")

    def test_tool_policy_guard_allows_explicit_capacity_tool_call(self) -> None:
        registry = FakeRegistry()
        llm = FakeLLMClient(
            [
                tool_response(
                    "find_max_capacity",
                    {"case_path": "ieee118", "bus": 10, "connection_type": "solar"},
                ),
                final_response("The capacity search completed through the tool."),
            ]
        )
        agent = GridMindAgent(
            registry=registry,
            llm_client=llm,
            config=AgentConfig(enable_forced_capacity_routing=False),
        )

        result = agent.run_turn("What is the max solar hosting capacity at bus 10 on ieee118?")

        self.assertEqual(result.status, "completed")
        self.assertEqual(registry.calls[0][0], "find_max_capacity")
        self.assertEqual(result.invoked_tools, ["find_max_capacity"])

    def test_tool_error_is_returned_to_model_for_repair(self) -> None:
        registry = FakeRegistry()
        agent = GridMindAgent(
            registry=registry,
            llm_client=FakeLLMClient(
                [
                    tool_response("run_powerflow", {}),
                    final_response("I need the case name before I can run the power flow."),
                ]
            ),
            config=AgentConfig(enable_forced_capacity_routing=False),
        )

        result = agent.run_turn("Run a power flow.")

        self.assertEqual(result.status, "completed")
        self.assertFalse(result.tool_records[0].ok)
        self.assertIn("case_path", result.tool_records[0].error or "")
        self.assertEqual(result.invoked_tools, [])
        self.assertIn("case name", result.output_text)

    def test_agent_stops_after_max_tool_rounds(self) -> None:
        registry = FakeRegistry()
        agent = GridMindAgent(
            registry=registry,
            llm_client=FakeLLMClient(
                [
                    tool_response("run_powerflow", {"case_path": "ieee14"}, call_id="call_1"),
                    tool_response("run_powerflow", {"case_path": "ieee14"}, call_id="call_2"),
                ]
            ),
            config=AgentConfig(max_tool_rounds=2, enable_forced_capacity_routing=False),
        )

        result = agent.run_turn("Keep running tools.")

        self.assertEqual(result.status, "max_tool_rounds_exceeded")
        self.assertEqual(result.llm_rounds, 2)
        self.assertEqual(len(result.tool_records), 2)
        self.assertEqual(len(registry.calls), 2)
        self.assertIsNotNone(result.deterministic_report)
        self.assertIn("Deterministic tool report", result.output_text)


if __name__ == "__main__":
    unittest.main()
