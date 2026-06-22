"""vLLM/OpenAI-compatible LLM adapter for the Mini Grid-Mind agent.

This module is intentionally dependency-free. The GPU runtime is expected to
host a Qwen-family model behind vLLM's OpenAI-compatible endpoints:

- GET  /v1/models
- POST /v1/chat/completions
- POST /v1/completions

The parser accepts native OpenAI-style tool calls first and then falls back to
Qwen-style text tool-call blocks when a local model emits calls as text.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib import error, request


DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_LOCAL_MODEL = "auto"
DEFAULT_QWEN_MODEL = DEFAULT_LOCAL_MODEL


class LLMClientError(RuntimeError):
    """Raised when an LLM endpoint request or response is invalid."""


@dataclass(frozen=True)
class VLLMConfig:
    """Connection and generation defaults for a vLLM OpenAI-compatible server."""

    base_url: str = DEFAULT_VLLM_BASE_URL
    model: Optional[str] = DEFAULT_LOCAL_MODEL
    api_key: Optional[str] = None
    timeout_s: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: Optional[float] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    auto_qwen35_nothink: bool = True


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call extracted from a model response."""

    id: str
    name: str
    arguments: Dict[str, Any]
    source: str
    raw_arguments: Any = None

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, sort_keys=True),
            },
        }


@dataclass(frozen=True)
class ChatCompletion:
    """Normalized chat completion returned by the LLM client."""

    content: str
    tool_calls: List[ToolCall]
    reasoning_content: str
    finish_reason: Optional[str]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class Completion:
    """Normalized text completion returned by the LLM client."""

    text: str
    finish_reason: Optional[str]
    raw: Dict[str, Any]


class VLLMOpenAIClient:
    """Small vLLM client for OpenAI-compatible Qwen inference servers."""

    def __init__(self, config: Optional[VLLMConfig] = None) -> None:
        self.config = _validate_config(config or VLLMConfig())

    def list_models(self) -> Dict[str, Any]:
        """Call GET /v1/models and return the raw JSON payload."""

        return self._request_json("GET", "/models")

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        tool_choice: str | Mapping[str, Any] = "auto",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        parallel_tool_calls: Optional[bool] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> ChatCompletion:
        """Call POST /v1/chat/completions and parse the first choice."""

        resolved_model = self._resolve_model(model)
        body: Dict[str, Any] = {
            "model": resolved_model,
            "messages": [_message_dict(message) for message in messages],
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        effective_top_p = self.config.top_p if top_p is None else top_p
        if effective_top_p is not None:
            body["top_p"] = effective_top_p
        if tools is not None:
            body["tools"] = [dict(tool) for tool in tools]
            body["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            body["parallel_tool_calls"] = parallel_tool_calls
        if self.config.auto_qwen35_nothink and is_qwen35_family_model(resolved_model):
            _merge_request_body(
                body,
                {"chat_template_kwargs": {"enable_thinking": False}},
            )
        _merge_request_body(body, self.config.extra_body)
        _merge_request_body(body, dict(extra_body or {}))

        payload = self._request_json("POST", "/chat/completions", body)
        return parse_chat_completion_response(payload)

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> Completion:
        """Call POST /v1/completions and parse the first choice."""

        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")

        body: Dict[str, Any] = {
            "model": self._resolve_model(model),
            "prompt": prompt,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        effective_top_p = self.config.top_p if top_p is None else top_p
        if effective_top_p is not None:
            body["top_p"] = effective_top_p
        body.update(self.config.extra_body)
        body.update(dict(extra_body or {}))

        payload = self._request_json("POST", "/completions", body)
        return parse_completion_response(payload)

    def _resolve_model(self, override: Optional[str] = None) -> str:
        model = override if override is not None else self.config.model
        if isinstance(model, str) and model.strip() and model.strip().lower() != "auto":
            return model.strip()
        return _first_model_id(self.list_models())

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = _join_url(self.config.base_url, path)
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.config.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"{method} {url} failed with HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise LLMClientError(f"{method} {url} failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMClientError(f"{method} {url} timed out") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"{method} {url} returned non-JSON response") from exc
        if not isinstance(parsed, dict):
            raise LLMClientError(f"{method} {url} returned a non-object JSON payload")
        return parsed


def parse_chat_completion_response(payload: Mapping[str, Any]) -> ChatCompletion:
    """Normalize an OpenAI-compatible chat completion payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    choice = _first_choice(payload)
    message = choice.get("message", {})
    if not isinstance(message, Mapping):
        raise LLMClientError("chat completion choice is missing a message object")

    native_tool_calls = parse_openai_tool_calls(message.get("tool_calls"))
    if not native_tool_calls:
        native_tool_calls = parse_openai_function_call(message.get("function_call"))
    content = _content_to_text(message.get("content", ""))
    reasoning = _reasoning_to_text(message)
    visible_content, embedded_reasoning = strip_qwen_thinking(content)
    if not reasoning:
        reasoning = embedded_reasoning

    text_tool_calls: List[ToolCall] = []
    if not native_tool_calls:
        text_tool_calls = parse_tool_calls_from_text(visible_content)
        if text_tool_calls:
            visible_content = remove_tool_call_blocks(visible_content).strip()

    return ChatCompletion(
        content=visible_content,
        tool_calls=native_tool_calls or text_tool_calls,
        reasoning_content=reasoning,
        finish_reason=_optional_str(choice.get("finish_reason")),
        raw=dict(payload),
    )


def parse_completion_response(payload: Mapping[str, Any]) -> Completion:
    """Normalize an OpenAI-compatible text completion payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    choice = _first_choice(payload)
    return Completion(
        text=_content_to_text(choice.get("text", "")),
        finish_reason=_optional_str(choice.get("finish_reason")),
        raw=dict(payload),
    )


def parse_openai_tool_calls(raw_tool_calls: Any) -> List[ToolCall]:
    """Parse native OpenAI-style tool calls from a message object."""

    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, list):
        raise LLMClientError("message.tool_calls must be a list when present")

    parsed: List[ToolCall] = []
    for index, item in enumerate(raw_tool_calls):
        if not isinstance(item, Mapping):
            raise LLMClientError("each tool call must be a JSON object")
        function = item.get("function", {})
        if not isinstance(function, Mapping):
            raise LLMClientError("tool call function must be a JSON object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise LLMClientError("tool call function name is required")
        raw_arguments = function.get("arguments", {})
        parsed.append(
            ToolCall(
                id=_optional_str(item.get("id")) or f"call_{index}",
                name=name.strip(),
                arguments=_arguments_to_dict(raw_arguments),
                raw_arguments=raw_arguments,
                source="native",
            )
        )
    return parsed


def parse_openai_function_call(raw_function_call: Any) -> List[ToolCall]:
    """Parse legacy OpenAI-style message.function_call payloads."""

    if raw_function_call is None:
        return []
    if not isinstance(raw_function_call, Mapping):
        raise LLMClientError("message.function_call must be a JSON object when present")
    name = raw_function_call.get("name")
    if not isinstance(name, str) or not name.strip():
        raise LLMClientError("function_call name is required")
    raw_arguments = raw_function_call.get("arguments", {})
    return [
        ToolCall(
            id="function_call",
            name=name.strip(),
            arguments=_arguments_to_dict(raw_arguments),
            raw_arguments=raw_arguments,
            source="legacy_function_call",
        )
    ]


def parse_tool_calls_from_text(text: str) -> List[ToolCall]:
    """Extract Qwen-style JSON tool calls emitted as assistant text.

    Supported fallback forms include:

    - ``<tool_call>{...}</tool_call>``
    - ``<|tool_call|>{...}<|/tool_call|>``
    - fenced JSON containing ``tool_calls`` or a single ``name``/``arguments``
      object
    - a whole-message JSON object/list with the same shapes
    """

    if not isinstance(text, str):
        raise ValueError("text must be a string")

    calls: List[ToolCall] = []
    for candidate in _tool_call_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        for call in _coerce_tool_calls(payload, source="text"):
            if not _duplicate_call(call, calls):
                calls.append(call)
    return calls


def strip_qwen_thinking(text: str) -> tuple[str, str]:
    """Remove Qwen-style thinking blocks from visible assistant content."""

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    cleaned = _strip_chatml_wrappers(text)
    reasoning_parts = [match.group(1).strip() for match in _THINK_RE.finditer(cleaned)]
    cleaned = _THINK_RE.sub("", cleaned)
    if cleaned.lstrip().startswith("<think>"):
        before, sep, after = cleaned.partition("</think>")
        if sep:
            reasoning_parts.append(before.replace("<think>", "", 1).strip())
            cleaned = after
    return cleaned.strip(), "\n\n".join(part for part in reasoning_parts if part)


def remove_tool_call_blocks(text: str) -> str:
    """Remove text tool-call blocks after they have been parsed."""

    if not isinstance(text, str):
        raise ValueError("text must be a string")
    result = _TOOL_CALL_XML_RE.sub("", text)
    result = _TOOL_CALL_CHATML_RE.sub("", result)
    result = _FENCED_JSON_RE.sub("", result)
    stripped = result.strip()
    if _looks_like_json(stripped):
        return ""
    return stripped


def render_qwen_chatml(
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render role messages as Qwen ChatML for /v1/completions fallback use."""

    chunks: List[str] = []
    for message in messages:
        msg = _message_dict(message)
        role = msg["role"]
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported ChatML role: {role}")
        content = _content_to_text(msg.get("content", ""))
        if role == "tool":
            name = msg.get("name")
            if isinstance(name, str) and name.strip():
                content = f"{name.strip()}\n{content}"
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")
    return "\n".join(chunks)


def tool_result_message(
    tool_call: ToolCall,
    result: Mapping[str, Any],
    *,
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible tool-result message for the agent loop."""

    if not isinstance(result, Mapping):
        raise ValueError("result must be a mapping")
    if not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    content = json.dumps(result, allow_nan=False, sort_keys=True)
    if len(content) > max_chars:
        marker = "...<truncated tool result>"
        if max_chars <= len(marker):
            content = marker[:max_chars]
        else:
            content = content[: max_chars - len(marker)] + marker
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "content": content,
    }


def is_qwen35_family_model(model_id: str) -> bool:
    """Return whether a served model id looks like the Qwen3.5 family.

    This intentionally does not match plain Qwen3 names such as ``Qwen3-32B``.
    It accepts common served-name spellings such as ``Qwen/Qwen3.5-35B-A3B``,
    ``qwen3_5-27b``, ``qwen3-5-27b``, and ``qwen35-27b``.
    """

    if not isinstance(model_id, str):
        return False
    lower = model_id.strip().lower()
    if not lower:
        return False
    if re.search(r"qwen\s*[/_-]?\s*3\.5(?:[^a-z0-9]|$)", lower):
        return True
    if re.search(r"qwen\s*[/_-]?\s*3_5(?:[^a-z0-9]|$)", lower):
        return True
    if re.search(r"qwen\s*[/_-]?\s*3-5(?:[^a-z0-9]|$)", lower):
        return True
    if re.search(r"qwen35(?:[^a-z0-9]|$)", lower):
        return True
    return False


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.I | re.S)
_TOOL_CALL_XML_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.I | re.S)
_TOOL_CALL_CHATML_RE = re.compile(
    r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|<\|im_end\|>|$)",
    re.I | re.S,
)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.I | re.S)


def _validate_config(config: VLLMConfig) -> VLLMConfig:
    if not isinstance(config.base_url, str) or not config.base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if config.model is not None and (not isinstance(config.model, str) or not config.model.strip()):
        raise ValueError("model must be a non-empty string, 'auto', or None")
    if config.api_key is not None and not isinstance(config.api_key, str):
        raise ValueError("api_key must be a string when provided")
    if not isinstance(config.timeout_s, (int, float)) or config.timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if not isinstance(config.max_tokens, int) or config.max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(config.extra_body, dict):
        raise ValueError("extra_body must be a dict")
    if not isinstance(config.auto_qwen35_nothink, bool):
        raise ValueError("auto_qwen35_nothink must be a boolean")
    return config


def _merge_request_body(target: Dict[str, Any], extra: Mapping[str, Any]) -> None:
    """Merge request-body extras while preserving nested chat-template kwargs."""

    if not extra:
        return
    for key, value in extra.items():
        if (
            key == "chat_template_kwargs"
            and isinstance(value, Mapping)
            and isinstance(target.get(key), Mapping)
        ):
            merged = dict(target[key])
            merged.update(dict(value))
            target[key] = merged
        else:
            target[key] = value


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _message_dict(message: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(message, Mapping):
        raise ValueError("messages must contain mappings")
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("each message must include a non-empty role")
    result = dict(message)
    result["role"] = role.strip()
    return result


def _first_choice(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMClientError("completion payload is missing choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise LLMClientError("completion choice must be a JSON object")
    return choice


def _first_model_id(payload: Mapping[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise LLMClientError("/models response did not include any served models")
    first = data[0]
    if not isinstance(first, Mapping):
        raise LLMClientError("/models response model entry must be a JSON object")
    model_id = first.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise LLMClientError("/models response first entry is missing a model id")
    return model_id.strip()


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                value = item.get("text", item.get("content", ""))
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _reasoning_to_text(message: Mapping[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        value = message.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def _optional_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _arguments_to_dict(raw_arguments: Any) -> Dict[str, Any]:
    if raw_arguments is None or raw_arguments == "":
        return {}
    if isinstance(raw_arguments, Mapping):
        return dict(raw_arguments)
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"tool arguments are not valid JSON: {raw_arguments}") from exc
        if not isinstance(parsed, Mapping):
            raise LLMClientError("tool arguments must decode to a JSON object")
        return dict(parsed)
    raise LLMClientError("tool arguments must be a JSON object or JSON string")


def _tool_call_candidates(text: str) -> List[str]:
    candidates = []
    candidates.extend(match.group(1).strip() for match in _TOOL_CALL_XML_RE.finditer(text))
    candidates.extend(match.group(1).strip() for match in _TOOL_CALL_CHATML_RE.finditer(text))
    candidates.extend(match.group(1).strip() for match in _FENCED_JSON_RE.finditer(text))
    stripped = remove_tool_call_blocks(_strip_chatml_wrappers(text)).strip()
    if stripped:
        candidates.append(stripped)
    full = _strip_chatml_wrappers(text).strip()
    if full and full not in candidates:
        candidates.append(full)
    return [candidate for candidate in candidates if _looks_like_json(candidate)]


def _coerce_tool_calls(payload: Any, *, source: str) -> List[ToolCall]:
    if isinstance(payload, list):
        calls: List[ToolCall] = []
        for item in payload:
            calls.extend(_coerce_tool_calls(item, source=source))
        return calls
    if not isinstance(payload, Mapping):
        return []

    if "tool_calls" in payload:
        return _coerce_tool_calls(payload["tool_calls"], source=source)
    if "function_call" in payload:
        return _coerce_tool_calls(payload["function_call"], source=source)
    if "calls" in payload:
        return _coerce_tool_calls(payload["calls"], source=source)

    function = payload.get("function")
    function_map = function if isinstance(function, Mapping) else {}
    name = (
        payload.get("name")
        or payload.get("tool")
        or payload.get("tool_name")
        or function_map.get("name")
    )
    if not isinstance(name, str) or not name.strip():
        return []

    raw_arguments = (
        payload.get("arguments")
        if "arguments" in payload
        else payload.get("args", function_map.get("arguments", {}))
    )
    return [
        ToolCall(
            id=_optional_str(payload.get("id")) or f"call_{uuid.uuid4().hex[:12]}",
            name=name.strip(),
            arguments=_arguments_to_dict(raw_arguments),
            raw_arguments=raw_arguments,
            source=source,
        )
    ]


def _duplicate_call(candidate: ToolCall, existing: Iterable[ToolCall]) -> bool:
    return any(
        call.name == candidate.name and call.arguments == candidate.arguments
        for call in existing
    )


def _strip_chatml_wrappers(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^<\|im_start\|>assistant\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"<\|im_end\|>\s*$", "", stripped, flags=re.I)
    return stripped.strip()


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    )
