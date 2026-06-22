"""Tests for the Step 8 anti-hallucination guardrails."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from gridmind_mini import (  # noqa: E402
    ANTI_FABRICATION_PROMPT_RULES,
    GROUNDING_WARNING,
    detect_capacity_route,
    detect_cia_readiness,
    find_numeric_claims,
    handle_forced_capacity_routing,
    has_grounding_credit,
    validate_tool_call_policy,
    validate_grounding,
)


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        args = dict(arguments)
        self.calls.append((name, args))
        return {
            "ok": True,
            "tool": name,
            "case_path": args["case_path"],
            "request": {
                "bus": args["bus"],
                "connection_type": args["connection_type"],
                "is_ibr": args["connection_type"] in {"solar", "wind", "bess", "hybrid"},
            },
            "status": "max_bound_approved",
            "max_approved_mw": args.get("max_mw", 500.0),
        }


class AntiHallucinationTest(unittest.TestCase):
    def test_specific_bus_capacity_query_routes_with_required_args(self) -> None:
        decision = detect_capacity_route(
            "What is the maximum load capacity at bus 14 on IEEE 118?",
        )

        self.assertTrue(decision.should_route)
        self.assertTrue(decision.ready)
        self.assertEqual(decision.route_type, "specific_bus_capacity")
        self.assertEqual(decision.tool_name, "find_max_capacity")
        self.assertEqual(
            decision.tool_args,
            {"case_path": "ieee118", "bus": 14, "connection_type": "load"},
        )

    def test_data_center_capacity_maps_to_load(self) -> None:
        decision = detect_capacity_route(
            "How much data center can bus 10 host?",
            context={"case_path": "ieee14"},
        )

        self.assertTrue(decision.ready)
        self.assertEqual(decision.tool_args["connection_type"], "load")
        self.assertEqual(decision.tool_args["case_path"], "ieee14")

    def test_missing_capacity_inputs_returns_clarification(self) -> None:
        decision = detect_capacity_route(
            "How much capacity can bus 14 accept?",
            context={"case_path": "ieee118"},
        )

        self.assertTrue(decision.should_route)
        self.assertFalse(decision.ready)
        self.assertEqual(decision.missing_inputs, ["connection_type"])
        self.assertIn("resource type", decision.clarification_prompt or "")

    def test_best_bus_capacity_query_is_caught_but_requires_bus_for_current_tool(self) -> None:
        decision = detect_capacity_route(
            "Which bus has the best maximum solar capacity on IEEE 14?",
        )

        self.assertTrue(decision.should_route)
        self.assertFalse(decision.ready)
        self.assertEqual(decision.route_type, "best_bus_capacity")
        self.assertIn("bus", decision.missing_inputs)
        self.assertEqual(decision.extracted["connection_type"], "solar")

    def test_non_capacity_query_does_not_route(self) -> None:
        decision = detect_capacity_route("Run a normal power flow on IEEE 14.")

        self.assertFalse(decision.should_route)
        self.assertFalse(decision.ready)
        self.assertEqual(decision.tool_args, {})

    def test_specific_sized_host_project_does_not_force_capacity_search(self) -> None:
        decision = detect_capacity_route(
            "Can bus 10 host a 25 MW solar project on IEEE 118?"
        )

        self.assertFalse(decision.should_route)
        self.assertEqual(decision.tool_args, {})

    def test_context_sized_host_project_does_not_force_capacity_search(self) -> None:
        decision = detect_capacity_route(
            "Can bus 10 host a solar project on IEEE 118?",
            context={"mw": 25.0},
        )

        self.assertFalse(decision.should_route)
        self.assertEqual(decision.tool_args, {})

    def test_explicit_capacity_question_with_mw_can_still_force_capacity_search(self) -> None:
        decision = detect_capacity_route(
            "What is the max solar hosting capacity at bus 10 on IEEE 118 if I am considering 25 MW?"
        )

        self.assertTrue(decision.should_route)
        self.assertTrue(decision.ready)
        self.assertEqual(
            decision.tool_args,
            {"case_path": "ieee118", "bus": 10, "connection_type": "solar"},
        )

    def test_complete_cia_request_is_ready_with_tool_args(self) -> None:
        decision = detect_cia_readiness(
            "Run CIA for a 25 MW solar project at bus 10 on IEEE 118 with N-1."
        )

        self.assertTrue(decision.should_check)
        self.assertTrue(decision.ready)
        self.assertEqual(decision.tool_name, "run_cia")
        self.assertEqual(
            decision.tool_args,
            {
                "case_path": "ieee118",
                "connection": {
                    "bus": 10,
                    "p_mw": 25.0,
                    "connection_type": "solar",
                    "is_ibr": True,
                },
                "enable_contingency": True,
            },
        )

    def test_cia_readiness_uses_context_to_fill_missing_fields(self) -> None:
        decision = detect_cia_readiness(
            "Please run an interconnection impact study.",
            context={
                "case_path": "ieee14",
                "bus": 7,
                "mw": 12.5,
                "connection_type": "load",
                "enable_contingency": True,
            },
        )

        self.assertTrue(decision.ready)
        self.assertEqual(decision.tool_args["case_path"], "ieee14")
        self.assertEqual(
            decision.tool_args["connection"],
            {
                "bus": 7,
                "p_mw": 12.5,
                "connection_type": "load",
                "is_ibr": False,
            },
        )
        self.assertTrue(decision.tool_args["enable_contingency"])

    def test_cia_readiness_returns_clarification_for_missing_inputs(self) -> None:
        decision = detect_cia_readiness(
            "Run CIA for a solar project at bus 10 on IEEE 118."
        )

        self.assertTrue(decision.should_check)
        self.assertFalse(decision.ready)
        self.assertEqual(decision.missing_inputs, ["p_mw"])
        self.assertEqual(decision.tool_args, {})
        self.assertIn("project size", decision.clarification_prompt or "")

    def test_cia_readiness_does_not_assume_unspecified_resource_is_load(self) -> None:
        decision = detect_cia_readiness("Run CIA for a 25 MW project at bus 10 on ieee118.")

        self.assertFalse(decision.ready)
        self.assertIn("connection_type", decision.missing_inputs)
        self.assertIn("resource type", decision.clarification_prompt or "")

    def test_non_cia_request_does_not_trigger_cia_readiness_gate(self) -> None:
        decision = detect_cia_readiness("Run a normal power flow on IEEE 14.")

        self.assertFalse(decision.should_check)
        self.assertFalse(decision.ready)
        self.assertEqual(decision.tool_args, {})

    def test_capacity_route_rejects_non_mapping_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "context"):
            detect_capacity_route(
                "max load capacity at bus 10",
                context=[("case_path", "ieee14")],  # type: ignore[arg-type]
            )

    def test_cia_readiness_rejects_non_mapping_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "context"):
            detect_cia_readiness(
                "Run CIA for a 25 MW solar project at bus 10.",
                context=[("case_path", "ieee14")],  # type: ignore[arg-type]
            )

    def test_tool_policy_blocks_capacity_tool_for_specific_cia_project(self) -> None:
        decision = validate_tool_call_policy(
            tool_name="find_max_capacity",
            arguments={"case_path": "ieee118", "bus": 10, "connection_type": "solar"},
            user_message="Can bus 10 host a 25 MW solar project on IEEE 118?",
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.recommended_tool, "run_cia")
        self.assertIn("specific_sized_cia_request", decision.reason_codes)
        self.assertEqual(decision.to_tool_result()["error_type"], "tool_policy_violation")

    def test_tool_policy_blocks_capacity_tool_for_context_sized_cia_project(self) -> None:
        decision = validate_tool_call_policy(
            tool_name="find_max_capacity",
            arguments={"case_path": "ieee118", "bus": 10, "connection_type": "solar"},
            user_message="Can bus 10 host a solar project on IEEE 118?",
            context={"mw": 25.0},
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.recommended_tool, "run_cia")

    def test_tool_policy_allows_explicit_capacity_search(self) -> None:
        decision = validate_tool_call_policy(
            tool_name="find_max_capacity",
            arguments={"case_path": "ieee118", "bus": 10, "connection_type": "solar"},
            user_message="What is the max solar hosting capacity at bus 10 on IEEE 118?",
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason_codes, ["tool_call_policy_allowed"])

    def test_tool_policy_allows_non_capacity_tools(self) -> None:
        decision = validate_tool_call_policy(
            tool_name="run_cia",
            arguments={
                "case_path": "ieee118",
                "connection": {
                    "bus": 10,
                    "p_mw": 25.0,
                    "connection_type": "solar",
                    "is_ibr": True,
                },
            },
            user_message="Can bus 10 host a 25 MW solar project on IEEE 118?",
        )

        self.assertTrue(decision.allowed)

    def test_forced_capacity_handler_uses_consistent_not_routed_shape(self) -> None:
        output = handle_forced_capacity_routing(
            "Run a normal power flow on IEEE 14.",
            FakeRegistry(),
        )

        self.assertFalse(output["routed"])
        self.assertFalse(output["executed"])
        self.assertIsNone(output["result"])

    def test_forced_capacity_handler_executes_ready_route(self) -> None:
        registry = FakeRegistry()

        output = handle_forced_capacity_routing(
            "Max solar capacity at bus 10 on ieee14?",
            registry,
            context={"max_mw": 20.0, "tolerance_mw": 5.0},
        )

        self.assertTrue(output["routed"])
        self.assertTrue(output["executed"])
        self.assertEqual(registry.calls[0][0], "find_max_capacity")
        self.assertEqual(
            registry.calls[0][1],
            {
                "case_path": "ieee14",
                "bus": 10,
                "connection_type": "solar",
                "max_mw": 20.0,
                "tolerance_mw": 5.0,
            },
        )
        self.assertEqual(output["result"]["max_approved_mw"], 20.0)

    def test_forced_capacity_handler_returns_clarification_when_missing(self) -> None:
        output = handle_forced_capacity_routing(
            "Max capacity at bus 10?",
            FakeRegistry(),
            context={"case_path": "ieee14"},
        )

        self.assertTrue(output["routed"])
        self.assertFalse(output["executed"])
        self.assertIn("resource type", output["clarification"])

    def test_numeric_claim_scanner_detects_grid_units_and_capacity_value(self) -> None:
        claims = find_numeric_claims("The limit is 127 MW and capacity is 44.")

        self.assertEqual([claim.unit for claim in claims], ["mw", "capacity_value"])
        self.assertEqual([claim.value for claim in claims], [127.0, 44.0])

    def test_numeric_claim_scanner_rejects_invalid_context_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_window"):
            find_numeric_claims("127 MW", context_window=-1)

    def test_grounding_validator_warns_without_analytical_tool(self) -> None:
        result = validate_grounding(
            "The maximum capacity at bus 14 is approximately 127 MW.",
            invoked_tools=[],
        )

        self.assertTrue(result.warning_appended)
        self.assertEqual(len(result.ungrounded_claims), 1)
        self.assertIn(GROUNDING_WARNING, result.output_text)

    def test_grounding_validator_does_not_credit_metadata_tools(self) -> None:
        result = validate_grounding(
            "The capacity is 127 MW.",
            invoked_tools=["list_cases", "set_backend"],
        )

        self.assertTrue(result.warning_appended)
        self.assertFalse(has_grounding_credit(["list_cases", "set_backend"]))

    def test_grounding_validator_does_not_credit_unimplemented_roadmap_tools(self) -> None:
        result = validate_grounding(
            "The capacity is 127 MW.",
            invoked_tools=["run_opf", "run_cia_with_mitigation"],
        )

        self.assertTrue(result.warning_appended)
        self.assertFalse(has_grounding_credit(["run_opf", "run_cia_with_mitigation"]))

    def test_grounding_credit_rejects_single_string_argument(self) -> None:
        with self.assertRaisesRegex(ValueError, "iterable of tool names"):
            has_grounding_credit("find_max_capacity")  # type: ignore[arg-type]

    def test_grounding_validator_accepts_analytical_tool_credit(self) -> None:
        result = validate_grounding(
            "The maximum approved capacity is 40 MW.",
            invoked_tools=["find_max_capacity"],
        )

        self.assertTrue(result.tool_grounded)
        self.assertFalse(result.warning_appended)

    def test_grounding_validator_allows_safe_standard_context(self) -> None:
        result = validate_grounding(
            "The NERC-informed normal profile uses a voltage range near 0.95 p.u. to 1.05 p.u.",
            invoked_tools=[],
        )

        self.assertEqual(len(result.claims), 2)
        self.assertTrue(all(claim.safe for claim in result.claims))
        self.assertFalse(result.warning_appended)

    def test_generic_standard_word_does_not_silence_warning(self) -> None:
        result = validate_grounding(
            "The standard answer is that this bus can take 127 MW.",
            invoked_tools=[],
        )

        self.assertTrue(result.warning_appended)

    def test_nerc_context_does_not_silence_mw_capacity_claim(self) -> None:
        result = validate_grounding(
            "NERC standards are important. The bus 14 capacity is 127 MW.",
            invoked_tools=[],
        )

        self.assertTrue(result.warning_appended)

    def test_prompt_hardening_rule_is_available(self) -> None:
        self.assertIn("Never state specific MW", ANTI_FABRICATION_PROMPT_RULES)
        self.assertIn("Session memory", ANTI_FABRICATION_PROMPT_RULES)


if __name__ == "__main__":
    unittest.main()
