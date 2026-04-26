import json

from calendar_agent.core import TOOL_DECLARATIONS
from calendar_agent.tools import (
    RETURN_FINAL_ANSWER_TOOL,
    RETURN_FINAL_ANSWER_TOOL_MINIMAL,
    get_openai_tools,
    get_openai_tools_minimal,
)


EXPECTED_NAMES = [fd.to_dict()["name"] for fd in TOOL_DECLARATIONS]


def _walk_types(node):
    """Yield every value at a 'type' key, recursively."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                yield v
            else:
                yield from _walk_types(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_types(item)


def _has_string_description(node):
    """True if any key 'description' maps to a string anywhere in the tree.

    Note: 'description' can also appear as a *property name* on create_event
    (the event's description field). We only care about schema-level
    descriptions, which are always string-valued.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "description" and isinstance(v, str):
                return True
            if _has_string_description(v):
                return True
    elif isinstance(node, list):
        return any(_has_string_description(x) for x in node)
    return False


def test_get_openai_tools_count_and_shape():
    tools = get_openai_tools()
    assert len(tools) == 7
    for t in tools:
        assert t["type"] == "function"
        assert set(t["function"].keys()) >= {"name", "description", "parameters"}
    assert [t["function"]["name"] for t in tools] == EXPECTED_NAMES


def test_get_openai_tools_types_are_lowercase_openai():
    tools = get_openai_tools()
    valid = {"string", "object", "array", "integer", "number", "boolean"}
    for t in tools:
        for typ in _walk_types(t["function"]["parameters"]):
            assert typ in valid, f"unexpected type {typ!r}"


def test_get_openai_tools_with_final_answer():
    tools = get_openai_tools(include_final_answer=True)
    assert len(tools) == 8
    assert tools[-1] == RETURN_FINAL_ANSWER_TOOL


def test_get_openai_tools_minimal_no_descriptions():
    tools = get_openai_tools_minimal()
    assert len(tools) == 7
    for t in tools:
        # function-level: name only, no description key
        assert "description" not in t["function"]
        # parameters: no schema description (string-valued) anywhere
        assert not _has_string_description(t["function"]["parameters"])


def test_get_openai_tools_minimal_is_smaller():
    full = json.dumps(get_openai_tools())
    minimal = json.dumps(get_openai_tools_minimal())
    assert len(minimal) < len(full)


def test_get_openai_tools_minimal_with_final_answer():
    tools = get_openai_tools_minimal(include_final_answer=True)
    assert len(tools) == 8
    assert tools[-1] == RETURN_FINAL_ANSWER_TOOL_MINIMAL


def test_return_final_answer_tool_structure():
    fn = RETURN_FINAL_ANSWER_TOOL["function"]
    assert fn["name"] == "return_final_answer"
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["answer"]["type"] == "string"
    assert params["required"] == ["answer"]
