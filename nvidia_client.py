"""Minimal OpenAI-compatible NVIDIA Nemotron client for RAMS.

The client intentionally uses the standard library so the existing Gemini client
and tool registry remain untouched. Tool calls are executed through RAMS's existing
``execute_tool`` dispatcher, never through arbitrary shell evaluation.
"""
from __future__ import annotations

import inspect
import json
import os
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-nano-30b-a3b"


def nvidia_config(api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None) -> Dict[str, str]:
    key = api_key or os.getenv("NVIDIA_API_KEY", "")
    endpoint = (base_url or os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL)).strip().rstrip("/")
    target_model = model or os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
    if not key:
        raise ValueError("NVIDIA_API_KEY is not configured.")
    if not endpoint.startswith(("https://", "http://")) or "/v1" not in endpoint:
        raise ValueError("NVIDIA_BASE_URL must be an HTTP(S) OpenAI-compatible /v1 endpoint.")
    if not target_model.strip():
        raise ValueError("NVIDIA_MODEL cannot be empty.")
    return {"api_key": key, "base_url": endpoint, "model": target_model.strip()}


def _openai_tool_definitions() -> List[Dict[str, Any]]:
    from tools import TOOLS_REGISTRY
    definitions = []
    for name, fn in TOOLS_REGISTRY.items():
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param_name, param in inspect.signature(fn).parameters.items():
            if param_name == "db_path":
                continue
            annotation = str(param.annotation).lower()
            typ = "number" if any(t in annotation for t in ("int", "float")) else "boolean" if "bool" in annotation else "string"
            properties[param_name] = {"type": typ}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        definitions.append({"type": "function", "function": {"name": name, "description": (fn.__doc__ or "").strip()[:500], "parameters": {"type": "object", "properties": properties, "required": required}}})
    return definitions


def _request(payload: Dict[str, Any], config: Dict[str, str]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        config["base_url"] + "/chat/completions",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"NVIDIA inference HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"NVIDIA inference request failed: {exc}") from exc


def generate_nvidia_response(messages: List[Dict[str, str]], model: Optional[str] = None, api_key: Optional[str] = None, system_instruction: Optional[str] = None, enable_tools: bool = True, base_url: Optional[str] = None, max_tool_rounds: int = 4) -> Dict[str, Any]:
    config = nvidia_config(api_key=api_key, base_url=base_url, model=model)
    api_messages: List[Dict[str, Any]] = []
    if system_instruction:
        api_messages.append({"role": "system", "content": system_instruction})
    api_messages.extend({"role": m["role"], "content": m.get("content", "")} for m in messages)
    payload: Dict[str, Any] = {"model": config["model"], "messages": api_messages, "temperature": 0.2}
    if enable_tools:
        payload["tools"] = _openai_tool_definitions()
        payload["tool_choice"] = "auto"
    for _ in range(max_tool_rounds + 1):
        response = _request(payload, config)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("NVIDIA response contained no choices.")
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            content = message.get("content") or ""
            return {"text": content, "model": config["model"], "provider": "nvidia", "connected": True}
        api_messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})
        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            from tools import execute_tool
            result = execute_tool(name, args)
            api_messages.append({"role": "tool", "tool_call_id": call.get("id", name), "name": name, "content": json.dumps(result, default=str)})
        payload["messages"] = api_messages
    raise RuntimeError("NVIDIA model exceeded the safe tool-call round limit.")


def stream_nvidia_chat_response(*args, **kwargs):
    result = generate_nvidia_response(*args, **kwargs)
    text = result.get("text", "") or "✅ Task operation completed successfully."
    for index in range(0, len(text), 20):
        yield text[index:index + 20]


def nvidia_status() -> Dict[str, Any]:
    configured = bool(os.getenv("NVIDIA_API_KEY"))
    return {"provider": "nvidia", "model": os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL), "base_url": os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL), "configured": configured, "connected": False, "openshell_available": False}
