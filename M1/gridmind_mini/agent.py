"""LLM-first agent loop for the Mini Grid-Mind reproduction.

Step 10 connects the model-facing layer from Step 9 to the deterministic
Grid-Mind tool registry. The agent is intentionally thin: it handles safety
guardrails, prompt construction, multi-round tool execution, and final response
grounding while leaving grid calculations inside the existing tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .anti_hallucination import (
    GroundingValidation,
    detect_cia_readiness,
    handle_forced_capacity_routing,
    validate_tool_call_policy,
    validate_grounding,
)
from .llm import ChatCompletion, ToolCall, VLLMOpenAIClient, tool_result_message
from .memory import StudyMemoryRecord, StudyMemoryStore
from .observations import build_tool_observation, tool_observation_payload
from .prompting import PromptContextHints, build_chat_messages
from .reporting import (
    DeterministicReport,
    build_deterministic_report,
    report_text_or_original,
)
from .tools import ToolRegistry, ToolRegistryError


class AgentLoopError(RuntimeError):
    """Raised when the agent loop is configured or invoked incorrectly."""


@dataclass(frozen=True)
class AgentConfig:
    """Runtime controls for one Mini Grid-Mind agent."""

    max_tool_rounds: int = 5
    tool_result_max_chars: int = 12000
    memory_limit: int = 5
    enable_forced_capacity_routing: bool = True
    enable_cia_readiness_gate: bool = True
    enable_tool_call_policy_guard: bool = True
    enable_tool_observation_summary: bool = True
    include_raw_tool_result_in_message: bool = True
    enable_deterministic_report: bool = True
    use_deterministic_report_when_final_empty: bool = True
    use_deterministic_report_on_max_rounds: bool = True
    tool_choice: str | Mapping[str, Any] = "auto"
    parallel_tool_calls: Optional[bool] = None
    chat_extra_body: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecutionRecord:
    """Audit record for one model-requested or guardrail-requested tool call."""

    call_id: str
    name: str
    arguments: Dict[str, Any]
    source: str
    ok: bool
    result: Dict[str, Any]
    error: Optional[str] = None
    observation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "source": self.source,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "observation": self.observation,
        }


@dataclass(frozen=True)
class AgentTurnResult:
    """Complete result for one user turn."""

    status: str
    output_text: str
    raw_output_text: str
    messages: List[Dict[str, Any]]
    tool_records: List[ToolExecutionRecord]
    invoked_tools: List[str]
    grounding: GroundingValidation
    prompt_context_hints: Optional[PromptContextHints] = None
    memory_records: List[StudyMemoryRecord] = field(default_factory=list)
    forced_route: Optional[Dict[str, Any]] = None
    readiness_check: Optional[Dict[str, Any]] = None
    llm_rounds: int = 0
    final_response: Optional[ChatCompletion] = None
    deterministic_report: Optional[DeterministicReport] = None

    def to_dict(self, *, include_messages: bool = True) -> Dict[str, Any]:
        result = {
            "status": self.status,
            "output_text": self.output_text,
            "raw_output_text": self.raw_output_text,
            "tool_records": [record.to_dict() for record in self.tool_records],
            "invoked_tools": list(self.invoked_tools),
            "grounding": self.grounding.to_dict(),
            "prompt_context_hints": None
            if self.prompt_context_hints is None
            else self.prompt_context_hints.to_dict(),
            "memory_records": [record.to_reference() for record in self.memory_records],
            "forced_route": self.forced_route,
            "readiness_check": self.readiness_check,
            "llm_rounds": self.llm_rounds,
            "final_response": None
            if self.final_response is None
            else {
                "content": self.final_response.content,
                "reasoning_content": self.final_response.reasoning_content,
                "finish_reason": self.final_response.finish_reason,
                "tool_calls": [
                    call.to_openai_dict() for call in self.final_response.tool_calls
                ],
            },
            "deterministic_report": None
            if self.deterministic_report is None
            else self.deterministic_report.to_dict(),
        }
        if include_messages:
            result["messages"] = self.messages
        return result


class GridMindAgent:
    """Minimal Grid-Mind LLM agent with multi-round tool execution."""

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        llm_client: Optional[Any] = None,
        memory_store: Optional[StudyMemoryStore] = None,
        lessons: Iterable[str] = (),
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.registry = registry or ToolRegistry(memory_store=memory_store)
        self.llm_client = llm_client or VLLMOpenAIClient()
        self.memory_store = memory_store
        self.lessons = lessons
        self.config = _validate_config(config or AgentConfig())

    def run_turn(
        self,
        user_message: str,
        *,
        history: Sequence[Mapping[str, Any]] = (),
        context: Optional[Mapping[str, Any]] = None,
        lessons: Optional[Iterable[str]] = None,
    ) -> AgentTurnResult:
        """Run one user turn through guardrails, LLM planning, and tools."""

        if not isinstance(user_message, str) or not user_message.strip():
            raise AgentLoopError("user_message must be a non-empty string")
        if context is not None and not isinstance(context, Mapping):
            raise AgentLoopError("context must be a mapping when provided")

        if self.config.enable_forced_capacity_routing:
            forced = self._handle_forced_capacity(user_message, context=context)
            if forced is not None:
                return forced
        if self.config.enable_cia_readiness_gate:
            readiness = self._handle_cia_readiness(user_message, context=context)
            if readiness is not None:
                return readiness

        prompt = build_chat_messages(
            self.registry,
            user_message.strip(),
            history=history,
            memory_store=self.memory_store,
            lessons=self.lessons if lessons is None else lessons,
            context=context,
            memory_limit=self.config.memory_limit,
        )
        messages = prompt.messages
        tool_specs = self.registry.openai_tool_specs()
        tool_records: List[ToolExecutionRecord] = []
        invoked_tools: List[str] = []
        final_response: Optional[ChatCompletion] = None

        for round_index in range(1, self.config.max_tool_rounds + 1):
            response = self.llm_client.chat(
                messages,
                tools=tool_specs,
                tool_choice=self.config.tool_choice,
                parallel_tool_calls=self.config.parallel_tool_calls,
                extra_body=self.config.chat_extra_body,
            )
            final_response = response
            if not response.tool_calls:
                raw_text = response.content.strip()
                deterministic_report = self._deterministic_report(tool_records)
                final_text = report_text_or_original(
                    raw_text,
                    deterministic_report,
                    use_when_empty=self.config.use_deterministic_report_when_final_empty,
                )
                grounding = validate_grounding(final_text, invoked_tools=invoked_tools)
                return AgentTurnResult(
                    status="completed",
                    output_text=grounding.output_text,
                    raw_output_text=raw_text,
                    messages=messages + [_assistant_message(response)],
                    tool_records=tool_records,
                    invoked_tools=invoked_tools,
                    grounding=grounding,
                    prompt_context_hints=prompt.context_hints,
                    memory_records=prompt.memory_records,
                    llm_rounds=round_index,
                    final_response=response,
                    deterministic_report=deterministic_report,
                )

            messages.append(_assistant_message(response))
            for tool_call in response.tool_calls:
                record = self._execute_tool_call(
                    tool_call,
                    user_message=user_message,
                    context=context,
                )
                tool_records.append(record)
                if record.ok:
                    invoked_tools.append(record.name)
                messages.append(
                    tool_result_message(
                        tool_call,
                        self._tool_message_payload(record),
                        max_chars=self.config.tool_result_max_chars,
                    )
                )

        deterministic_report = self._deterministic_report(tool_records)
        raw_text = (
            "I reached the maximum tool-call rounds before producing a final answer. "
            "Please narrow the request or increase max_tool_rounds."
        )
        if (
            self.config.use_deterministic_report_on_max_rounds
            and deterministic_report is not None
            and deterministic_report.available
        ):
            raw_text += "\n\nDeterministic tool report: " + deterministic_report.summary_text
        grounding = validate_grounding(raw_text, invoked_tools=invoked_tools)
        return AgentTurnResult(
            status="max_tool_rounds_exceeded",
            output_text=grounding.output_text,
            raw_output_text=raw_text,
            messages=messages,
            tool_records=tool_records,
            invoked_tools=invoked_tools,
            grounding=grounding,
            prompt_context_hints=prompt.context_hints,
            memory_records=prompt.memory_records,
            llm_rounds=self.config.max_tool_rounds,
            final_response=final_response,
            deterministic_report=deterministic_report,
        )

    def _handle_forced_capacity(
        self,
        user_message: str,
        *,
        context: Optional[Mapping[str, Any]],
    ) -> Optional[AgentTurnResult]:
        try:
            route = handle_forced_capacity_routing(
                user_message,
                self.registry,
                context=context,
            )
        except (ToolRegistryError, ValueError) as exc:
            route = {
                "routed": True,
                "executed": False,
                "decision": None,
                "result": {
                    "ok": False,
                    "tool": "find_max_capacity",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                "clarification": None,
            }

        if not route.get("routed", False):
            return None

        if not route.get("executed", False):
            raw_text = str(route.get("clarification") or _tool_error_text(route.get("result")))
            grounding = validate_grounding(raw_text, invoked_tools=[])
            return AgentTurnResult(
                status="clarification_required"
                if route.get("clarification")
                else "forced_capacity_error",
                output_text=grounding.output_text,
                raw_output_text=raw_text,
                messages=[{"role": "user", "content": user_message}],
                tool_records=[],
                invoked_tools=[],
                grounding=grounding,
                forced_route=route,
            )

        result = route.get("result")
        if not isinstance(result, Mapping):
            result = {
                "ok": False,
                "tool": "find_max_capacity",
                "error_type": "invalid_route_result",
                "error": "Forced capacity route returned a non-object result.",
            }
        result_dict = dict(result)
        tool_name = str(result_dict.get("tool", "find_max_capacity"))
        record = ToolExecutionRecord(
            call_id="forced_capacity",
            name=tool_name,
            arguments=dict(route.get("decision", {}).get("tool_args", {}))
            if isinstance(route.get("decision"), Mapping)
            else {},
            source="forced_capacity_routing",
            ok=bool(result_dict.get("ok", False)),
            result=result_dict,
            error=None if result_dict.get("ok", False) else _tool_error_text(result_dict),
            observation=_safe_observation(result_dict),
        )
        invoked_tools = [tool_name] if record.ok else []
        deterministic_report = self._deterministic_report([record])
        raw_text = summarize_capacity_result(result_dict)
        raw_text = report_text_or_original(
            raw_text,
            deterministic_report,
            use_when_empty=self.config.use_deterministic_report_when_final_empty,
        )
        grounding = validate_grounding(raw_text, invoked_tools=invoked_tools)
        return AgentTurnResult(
            status="forced_capacity_executed" if record.ok else "forced_capacity_error",
            output_text=grounding.output_text,
            raw_output_text=raw_text,
            messages=[{"role": "user", "content": user_message}],
            tool_records=[record],
            invoked_tools=invoked_tools,
            grounding=grounding,
            forced_route=route,
            deterministic_report=deterministic_report,
        )

    def _handle_cia_readiness(
        self,
        user_message: str,
        *,
        context: Optional[Mapping[str, Any]],
    ) -> Optional[AgentTurnResult]:
        decision = detect_cia_readiness(user_message, context=context)
        if not decision.should_check or decision.ready:
            return None

        raw_text = str(decision.clarification_prompt or "CIA required inputs are missing.")
        grounding = validate_grounding(raw_text, invoked_tools=[])
        readiness_check = {
            "type": "cia_readiness",
            "decision": decision.to_dict(),
        }
        return AgentTurnResult(
            status="clarification_required",
            output_text=grounding.output_text,
            raw_output_text=raw_text,
            messages=[{"role": "user", "content": user_message}],
            tool_records=[],
            invoked_tools=[],
            grounding=grounding,
            readiness_check=readiness_check,
        )

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        *,
        user_message: str,
        context: Optional[Mapping[str, Any]],
    ) -> ToolExecutionRecord:
        if self.config.enable_tool_call_policy_guard:
            try:
                policy = validate_tool_call_policy(
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    user_message=user_message,
                    context=context,
                )
            except ValueError as exc:
                policy_result = {
                    "ok": False,
                    "tool": tool_call.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                return ToolExecutionRecord(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    arguments=dict(tool_call.arguments),
                    source="tool_call_policy_guard",
                    ok=False,
                    result=policy_result,
                    error=str(exc),
                    observation=_safe_observation(policy_result),
                )
            if not policy.allowed:
                policy_result = policy.to_tool_result()
                return ToolExecutionRecord(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    arguments=dict(tool_call.arguments),
                    source="tool_call_policy_guard",
                    ok=False,
                    result=policy_result,
                    error=str(policy_result.get("message", "tool policy violation")),
                    observation=_safe_observation(policy_result),
                )

        try:
            result = self.registry.call_tool(tool_call.name, tool_call.arguments)
            ok = bool(result.get("ok", False))
            return ToolExecutionRecord(
                call_id=tool_call.id,
                name=tool_call.name,
                arguments=dict(tool_call.arguments),
                source=tool_call.source,
                ok=ok,
                result=result,
                error=None if ok else _tool_error_text(result),
                observation=_safe_observation(result),
            )
        except (ToolRegistryError, ValueError, TypeError) as exc:
            result = {
                "ok": False,
                "tool": tool_call.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return ToolExecutionRecord(
                call_id=tool_call.id,
                name=tool_call.name,
                arguments=dict(tool_call.arguments),
                source=tool_call.source,
                ok=False,
                result=result,
                error=str(exc),
                observation=_safe_observation(result),
            )

    def _tool_message_payload(self, record: ToolExecutionRecord) -> Dict[str, Any]:
        if not self.config.enable_tool_observation_summary:
            return record.result
        return tool_observation_payload(
            record.result,
            include_raw_result=self.config.include_raw_tool_result_in_message,
        )

    def _deterministic_report(
        self,
        tool_records: Sequence[ToolExecutionRecord],
    ) -> Optional[DeterministicReport]:
        if not self.config.enable_deterministic_report or not tool_records:
            return None
        return build_deterministic_report([record.result for record in tool_records])


def summarize_capacity_result(result: Mapping[str, Any]) -> str:
    """Create a deterministic summary for forced capacity-routing results."""

    if not isinstance(result, Mapping):
        raise ValueError("result must be a mapping")
    if not result.get("ok", False):
        return "The forced capacity search did not complete: " + _tool_error_text(result)

    request = result.get("request", {})
    request = request if isinstance(request, Mapping) else {}
    case_path = result.get("case_path", "the requested case")
    bus = request.get("bus", "the requested bus")
    connection_type = request.get("connection_type", "resource")
    status = result.get("status", "unknown")
    max_mw = result.get("max_approved_mw")
    tolerance = result.get("tolerance_mw")
    parts = [
        f"Forced capacity search completed for {connection_type} at bus {bus} on {case_path}.",
        f"Search status: {status}.",
    ]
    if isinstance(max_mw, (int, float)) and not isinstance(max_mw, bool):
        parts.append(f"Maximum approved capacity: {float(max_mw):.6g} MW.")
    else:
        parts.append("No approved capacity was found within the requested bounds.")
    lower = result.get("lower_bound_mw")
    upper = result.get("upper_bound_mw")
    bound_bits = []
    if isinstance(lower, (int, float)) and not isinstance(lower, bool):
        bound_bits.append(f"lower bound {float(lower):.6g} MW")
    if isinstance(upper, (int, float)) and not isinstance(upper, bool):
        bound_bits.append(f"upper bound {float(upper):.6g} MW")
    if bound_bits:
        parts.append("Boundary: " + ", ".join(bound_bits) + ".")
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
        parts.append(f"Search tolerance: {float(tolerance):.6g} MW.")
    rejection = result.get("rejection_explanation")
    if isinstance(rejection, Mapping) and rejection:
        limiting_stage = rejection.get("limiting_stage")
        rejection_status = rejection.get("status")
        details = []
        if limiting_stage:
            details.append(f"limiting stage {limiting_stage}")
        if rejection_status:
            details.append(f"status {rejection_status}")
        if details:
            parts.append("First rejection summary: " + ", ".join(details) + ".")
    return " ".join(parts)


def _assistant_message(response: ChatCompletion) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
    }
    if response.tool_calls:
        message["tool_calls"] = [call.to_openai_dict() for call in response.tool_calls]
    return message


def _tool_error_text(result: Any) -> str:
    if isinstance(result, Mapping):
        message = result.get("message") or result.get("error")
        if message:
            return str(message)
        error_type = result.get("error_type")
        if error_type:
            return str(error_type)
    return "unknown tool error"


def _validate_config(config: AgentConfig) -> AgentConfig:
    if not isinstance(config.max_tool_rounds, int) or config.max_tool_rounds <= 0:
        raise ValueError("max_tool_rounds must be a positive integer")
    if not isinstance(config.tool_result_max_chars, int) or config.tool_result_max_chars <= 0:
        raise ValueError("tool_result_max_chars must be a positive integer")
    if not isinstance(config.memory_limit, int) or config.memory_limit <= 0:
        raise ValueError("memory_limit must be a positive integer")
    if not isinstance(config.enable_forced_capacity_routing, bool):
        raise ValueError("enable_forced_capacity_routing must be a boolean")
    if not isinstance(config.enable_cia_readiness_gate, bool):
        raise ValueError("enable_cia_readiness_gate must be a boolean")
    if not isinstance(config.enable_tool_call_policy_guard, bool):
        raise ValueError("enable_tool_call_policy_guard must be a boolean")
    if not isinstance(config.enable_tool_observation_summary, bool):
        raise ValueError("enable_tool_observation_summary must be a boolean")
    if not isinstance(config.include_raw_tool_result_in_message, bool):
        raise ValueError("include_raw_tool_result_in_message must be a boolean")
    if not isinstance(config.enable_deterministic_report, bool):
        raise ValueError("enable_deterministic_report must be a boolean")
    if not isinstance(config.use_deterministic_report_when_final_empty, bool):
        raise ValueError("use_deterministic_report_when_final_empty must be a boolean")
    if not isinstance(config.use_deterministic_report_on_max_rounds, bool):
        raise ValueError("use_deterministic_report_on_max_rounds must be a boolean")
    if not isinstance(config.chat_extra_body, dict):
        raise ValueError("chat_extra_body must be a dict")
    return config


def _safe_observation(result: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        return build_tool_observation(result)
    except (TypeError, ValueError) as exc:
        return {
            "tool": str(result.get("tool", "unknown")),
            "status": "observation_error",
            "message": str(exc),
        }
