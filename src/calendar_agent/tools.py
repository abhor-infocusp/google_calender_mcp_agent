"""Shared tool conversion utilities and constants.

Centralizes Vertex AI -> OpenAI tool format conversion and the return_final_answer
tool definition used across training and eval scripts.
"""

import json

from calendar_agent.core import TOOL_DECLARATIONS


# ── Vertex AI -> OpenAI type mapping ──────────────────────

VERTEX_TO_OPENAI_TYPES = {
    "STRING": "string",
    "OBJECT": "object",
    "ARRAY": "array",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
}


def _convert_params(params: dict) -> dict:
    """Recursively convert Vertex AI parameter schema to OpenAI JSON Schema."""
    result = {}
    for key, value in params.items():
        if key == "property_ordering":
            continue
        if key in ("type", "type_") and isinstance(value, str):
            result["type"] = VERTEX_TO_OPENAI_TYPES.get(value, value.lower())
        elif key == "items" and isinstance(value, dict):
            result["items"] = _convert_params(value)
        elif key == "properties" and isinstance(value, dict):
            result["properties"] = {k: _convert_params(v) for k, v in value.items()}
        else:
            result[key] = value
    return result


# ── return_final_answer tool ──────────────────────────────

RETURN_FINAL_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "return_final_answer",
        "description": "Return the final answer or confirmation after completing the calendar task.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final response to the user's query, summarizing what was done or the requested information.",
                },
            },
            "required": ["answer"],
        },
    },
}


def get_openai_tools(include_final_answer: bool = False) -> list[dict]:
    """Convert Vertex AI TOOL_DECLARATIONS to OpenAI tool format.

    Args:
        include_final_answer: If True, append the return_final_answer tool.
    """
    tools = []
    for fd in TOOL_DECLARATIONS:
        d = fd.to_dict()
        tools.append({
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": _convert_params(d.get("parameters", {})),
            },
        })
    if include_final_answer:
        tools.append(RETURN_FINAL_ANSWER_TOOL)
    return tools


def _strip_descriptions(params: dict) -> dict:
    """Strip descriptions from parameter schema, keeping only structure."""
    result = {}
    for key, value in params.items():
        if key == "description":
            continue
        if key == "properties" and isinstance(value, dict):
            result["properties"] = {k: _strip_descriptions(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result["items"] = _strip_descriptions(value)
        else:
            result[key] = value
    return result


RETURN_FINAL_ANSWER_TOOL_MINIMAL = {
    "type": "function",
    "function": {
        "name": "return_final_answer",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    },
}


def get_openai_tools_minimal(include_final_answer: bool = False) -> list[dict]:
    """Minimal tool definitions: names + parameter names/types only, no descriptions.

    Saves ~950 tokens per trajectory. The model learns tool semantics from SFT
    trajectories, not from runtime definitions. vLLM's hermes parser only needs
    tools present to activate the parsing path.
    """
    tools = []
    for fd in TOOL_DECLARATIONS:
        d = fd.to_dict()
        params = _convert_params(d.get("parameters", {}))
        tools.append({
            "type": "function",
            "function": {
                "name": d["name"],
                "parameters": _strip_descriptions(params),
            },
        })
    if include_final_answer:
        tools.append(RETURN_FINAL_ANSWER_TOOL_MINIMAL)
    return tools
