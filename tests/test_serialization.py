#!/usr/bin/env python3
"""Regression test: trajectory_to_messages must always serialize tool calls.

Each assistant turn must contain at most ONE tool_call. This ensures the model
learns to read each tool result before deciding the next call (sequential
execution), rather than batching them as parallel calls.
"""

import json
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'training'))

from sft_train_100ep import trajectory_to_messages, load_trajectories


def test_all_tool_calls_are_serialized():
    """Every assistant message must have at most 1 tool_call."""
    trajs = load_trajectories()
    assert len(trajs) > 0, "No trajectories loaded"

    violations = []
    for idx, traj in enumerate(trajs):
        messages = trajectory_to_messages(traj)
        for mi, msg in enumerate(messages):
            if msg["role"] == "assistant" and "tool_calls" in msg:
                n = len(msg["tool_calls"])
                if n > 1:
                    violations.append(
                        f"traj={idx} msg={mi}: {n} tool_calls "
                        f"(names={[tc['function']['name'] for tc in msg['tool_calls']]})"
                    )

    assert not violations, (
        f"Found {len(violations)} assistant messages with >1 tool_call "
        f"(parallel). All tool calls must be serialized (1 per turn).\n"
        + "\n".join(violations[:10])
    )


def test_tool_call_followed_by_tool_response():
    """Each assistant tool_call message must be immediately followed by a tool response."""
    trajs = load_trajectories()

    for idx, traj in enumerate(trajs):
        messages = trajectory_to_messages(traj)
        for mi, msg in enumerate(messages):
            if msg["role"] == "assistant" and "tool_calls" in msg:
                assert mi + 1 < len(messages), (
                    f"traj={idx}: assistant tool_call at end with no response"
                )
                next_msg = messages[mi + 1]
                assert next_msg["role"] == "tool", (
                    f"traj={idx} msg={mi}: assistant tool_call not followed by "
                    f"tool response (got role={next_msg['role']})"
                )
                assert next_msg["tool_call_id"] == msg["tool_calls"][0]["id"], (
                    f"traj={idx} msg={mi}: tool_call_id mismatch"
                )


if __name__ == "__main__":
    test_all_tool_calls_are_serialized()
    print("PASS: All tool calls are serialized (1 per assistant turn)")
    test_tool_call_followed_by_tool_response()
    print("PASS: All tool calls followed by matching tool response")
    print("All serialization tests passed.")
