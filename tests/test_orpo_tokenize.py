"""Unit tests for src/calendar_agent/orpo/tokenize.py.

The most important test is `test_rendered_prompt_contains_tool_schema` —
this is the explicit guard against the prior DPO bug where `tools=` wasn't
threaded into apply_chat_template. If that test ever fails, treat it as
a critical regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from calendar_agent.orpo.tokenize import (
    IGNORE_INDEX,
    TokenizedTrajectory,
    tokenize_pair,
    tokenize_trajectory,
    trajectory_to_messages,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tokenizer():
    """Real Qwen3 tokenizer. Marked module-scope to amortize the load.
    Skipped automatically if HF cache lacks the model."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-14B", trust_remote_code=True
        )
    except Exception as e:
        pytest.skip(f"Qwen3-14B tokenizer unavailable: {e}")


@pytest.fixture
def calendar_tools():
    """Real OPENAI_TOOLS — same schema the rollout uses."""
    from calendar_agent.tools import get_openai_tools
    return get_openai_tools()


# ── Fake trajectory builders ───────────────────────────────────────────


@dataclass
class FakeTraj:
    messages_and_choices: list


def _choice(content: str = "", tool_calls: list[dict] | None = None):
    """Build a fake openai Choice object that quacks enough for tokenize.py."""
    msg_kwargs = {"content": content}
    if tool_calls:
        msg_kwargs["tool_calls"] = [
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls
        ]
    else:
        msg_kwargs["tool_calls"] = None
    return SimpleNamespace(message=SimpleNamespace(**msg_kwargs))


def _simple_traj() -> FakeTraj:
    """A short two-turn trajectory: user asks, asst responds."""
    return FakeTraj(messages_and_choices=[
        {"role": "system", "content": "/no_think\nYou are a calendar assistant."},
        {"role": "user", "content": "What's on Monday?"},
        _choice(content="Nothing scheduled."),
    ])


def _toolcall_traj() -> FakeTraj:
    """A multi-turn trajectory with a tool call + response + final answer."""
    return FakeTraj(messages_and_choices=[
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "What time is it?"},
        _choice(tool_calls=[{
            "id": "tc1",
            "function": {"name": "get_current_time", "arguments": "{}"},
        }]),
        {"role": "tool", "tool_call_id": "tc1", "name": "get_current_time",
         "content": "2026-05-01 12:00"},
        _choice(content="It's noon."),
    ])


# ── trajectory_to_messages ────────────────────────────────────────────


def test_trajectory_to_messages_simple():
    msgs = trajectory_to_messages(_simple_traj())
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2]["content"] == "Nothing scheduled."


def test_trajectory_to_messages_with_tool_calls():
    msgs = trajectory_to_messages(_toolcall_traj())
    assert [m["role"] for m in msgs] == [
        "system", "user", "assistant", "tool", "assistant"
    ]
    asst1 = msgs[2]
    assert asst1.get("tool_calls"), "tool_calls should be carried through"
    assert asst1["tool_calls"][0]["function"]["name"] == "get_current_time"
    # Tool message preserves tool_call_id + name
    tool = msgs[3]
    assert tool["tool_call_id"] == "tc1"
    assert tool["name"] == "get_current_time"


# ── THE bug guard ─────────────────────────────────────────────────────


def test_rendered_prompt_contains_tool_schema(tokenizer, calendar_tools):
    """Critical regression guard for the prior DPO bug.

    If this test fails, it means tokenize_trajectory rendered the prompt
    *without* the calendar tools schema. Inference and SFT both render
    *with* the schema, so any drift would mean the model is being trained
    on a token distribution it never sees in production.

    See auto-memory: feedback_dpo_skipped — `tools=` was missing from the
    chat template for the first day of the 2026-04-23 DPO run.
    """
    out = tokenize_trajectory(_simple_traj(), tokenizer, calendar_tools)
    # The Qwen3 chat template inserts a "# Tools" header + <tools>...</tools> block.
    assert "# Tools" in out.rendered_text, out.rendered_text[:500]
    assert "<tools>" in out.rendered_text, "missing <tools> XML block"
    # All 7 calendar tools by name should appear in the rendered prompt.
    for tool in calendar_tools:
        name = tool["function"]["name"]
        assert name in out.rendered_text, f"missing tool {name}"


# ── Tokenization correctness ──────────────────────────────────────────


def test_tokenize_returns_three_aligned_tensors(tokenizer, calendar_tools):
    out = tokenize_trajectory(_simple_traj(), tokenizer, calendar_tools)
    assert out.input_ids.shape == out.attention_mask.shape
    assert out.input_ids.shape == out.labels.shape
    assert (out.attention_mask == 1).all()


def test_labels_mask_system_and_user(tokenizer, calendar_tools):
    """No system/user tokens should be in labels (all -100 in those ranges)."""
    out = tokenize_trajectory(_simple_traj(), tokenizer, calendar_tools)
    # First few tokens are definitely system. Find the first non-IGNORE label
    # → it should be after the system+user prefix, inside the assistant turn.
    first_label_idx = next(
        (i for i, lbl in enumerate(out.labels.tolist()) if lbl != IGNORE_INDEX),
        -1,
    )
    assert first_label_idx > 0, "no labels found at all"
    # The token just before the first label should be the assistant opener
    # (`<|im_start|>assistant\n` ended by a newline, since opener is masked).
    # We only check that there's a substantial system+user prefix masked.
    assert first_label_idx >= 10, (
        f"first label at idx {first_label_idx} is suspiciously early — "
        "system + user should account for many tokens"
    )


def test_labels_unmask_only_assistant_content(tokenizer, calendar_tools):
    """Labels.n_assistant_tokens should match what we'd predict for the
    response content."""
    out = tokenize_trajectory(_simple_traj(), tokenizer, calendar_tools)
    n_unmasked = (out.labels != IGNORE_INDEX).sum().item()
    assert n_unmasked == out.n_assistant_tokens
    # "Nothing scheduled." is short — should be < 20 tokens incl. <|im_end|>.
    assert 1 <= n_unmasked <= 30, n_unmasked


def test_tool_response_not_in_labels(tokenizer, calendar_tools):
    """Tool message content ('2026-05-01 12:00') is environment output —
    must NOT be unmasked. Counts of unmasked tokens should reflect *only*
    the two assistant turns, not the tool reply."""
    out = tokenize_trajectory(_toolcall_traj(), tokenizer, calendar_tools)
    decoded_unmasked = tokenizer.decode([
        tok for tok, lbl in zip(out.input_ids.tolist(), out.labels.tolist())
        if lbl != IGNORE_INDEX
    ])
    # The tool reply text should NOT appear in the unmasked stream.
    assert "2026-05-01 12:00" not in decoded_unmasked, (
        f"tool response leaked into labels: {decoded_unmasked!r}"
    )
    # The final answer should appear.
    assert "noon" in decoded_unmasked.lower(), decoded_unmasked


def test_max_length_truncation(tokenizer, calendar_tools):
    out = tokenize_trajectory(
        _toolcall_traj(), tokenizer, calendar_tools, max_length=20
    )
    assert out.input_ids.shape[0] == 20
    assert out.labels.shape[0] == 20


def test_empty_trajectory_raises(tokenizer, calendar_tools):
    with pytest.raises(ValueError):
        tokenize_trajectory(
            FakeTraj(messages_and_choices=[]), tokenizer, calendar_tools
        )


# ── Pair tokenization ─────────────────────────────────────────────────


def test_tokenize_pair_returns_both(tokenizer, calendar_tools):
    chosen = _simple_traj()
    rejected = _toolcall_traj()
    pair = tokenize_pair(chosen, rejected, tokenizer, calendar_tools)
    assert isinstance(pair.chosen, TokenizedTrajectory)
    assert isinstance(pair.rejected, TokenizedTrajectory)
    # Both should have tools schema in rendered text (chosen and rejected
    # rendered separately — both must pass the bug guard).
    assert "<tools>" in pair.chosen.rendered_text
    assert "<tools>" in pair.rejected.rendered_text


def test_tokenize_pair_metadata_passes_through(tokenizer, calendar_tools):
    pair = tokenize_pair(
        _simple_traj(), _simple_traj(), tokenizer, calendar_tools,
        metadata={"scenario_id": "cal_0_q_3", "from_reuse_buffer": True},
    )
    assert pair.metadata["scenario_id"] == "cal_0_q_3"
    assert pair.metadata["from_reuse_buffer"] is True
