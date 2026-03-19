"""Shared tool conversion utilities and constants.

Centralizes Vertex AI -> OpenAI tool format conversion, the return_final_answer
tool definition, and compact tool result formatting used across training and eval scripts.
"""

import json
import re

from calendar_agent.core import TOOL_DECLARATIONS


def serialize_tool_result(result: dict) -> dict:
    """Serialize tool result, converting datetimes to strings."""
    return json.loads(json.dumps(result, default=str))


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


# ── Compact tool results ─────────────────────────────────


def _extract_emails(attendees: list) -> list[str]:
    """Extract email addresses from attendee objects or repr strings."""
    emails = []
    for a in attendees:
        if isinstance(a, dict) and "user" in a:
            emails.append(a["user"]["email"])
        elif isinstance(a, str):
            m = re.search(r"email='([^']+)'", a)
            if m:
                emails.append(m.group(1))
    return emails


def _compact_event(evt_raw) -> dict:
    """Strip redundant fields from an event dict, flatten attendees to emails."""
    evt = json.loads(evt_raw) if isinstance(evt_raw, str) else evt_raw
    start = str(evt["start"]).replace(" ", "T")
    end = str(evt["end"]).replace(" ", "T")
    ce = {"id": evt["id"], "summary": evt["summary"], "start": start, "end": end}
    desc = evt.get("description")
    if desc:
        ce["description"] = desc
    attendees = evt.get("attendees", [])
    if attendees:
        emails = _extract_emails(attendees)
        if emails:
            ce["attendees"] = emails
    return ce


def compact_tool_result(name: str, result):
    """Compact a tool call result to reduce token count for training.

    list_events  -> flat list of compact event dicts
    get_event    -> single compact event dict
    create/update/delete -> {"message": ..., **compact_event_fields}
    """
    if name == "list_events":
        events = result.get("events", result) if isinstance(result, dict) else result
        if isinstance(events, list):
            return [_compact_event(e) for e in events]
        return result
    if name == "get_event":
        evt = result.get("event", result) if isinstance(result, dict) else result
        return _compact_event(evt)
    if name in ("create_event", "update_event", "delete_event"):
        ce = {"message": result.get("message", "")}
        evt = result.get("event")
        if evt:
            ce.update(_compact_event(evt))
        else:
            ce.update(_compact_event(result))
        return ce
    return result
