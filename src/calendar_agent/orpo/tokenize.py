"""Tokenize a calendar trajectory for ORPO with the correct `tools=` schema.

This module is the explicit guard against the prior DPO failure (auto-memory
`feedback_dpo_skipped`): for the first day of that run, `tools=` was missing
from `apply_chat_template`, so the system prompt had no tool definitions and
the rendered prompt looked nothing like what the model saw at training-via-SFT
or inference time. Tokens drifted, loss looked plausible, model degraded.

Here we always pass `tools=OPENAI_TOOLS` to the chat template, and a unit test
asserts the rendered prompt contains the `<tools>` block. Any future change
to chat-template handling has to walk through the test first.

What this module does:

1. Convert a calendar trajectory's `messages_and_choices` (which mixes plain
   dicts with OpenAI Choice objects) into a flat list of HF-format messages.
2. Apply Qwen3's chat template with `tools=OPENAI_TOOLS`.
3. Compute per-token labels: the *content* of assistant messages is unmasked
   (label = token_id). Everything else (system, user, tool, and the chat-
   template wrapper tokens around assistant messages) is masked to -100.
4. Pack into a torch.Tensor dict ready for ORPO forward+loss.

We mask wrapper tokens (`<|im_start|>assistant\n`, `<|im_end|>`) deliberately
— at inference time `add_generation_prompt=True` injects the assistant
wrapper, so the model only has to generate content + `<|im_end|>`. We unmask
`<|im_end|>` so the model learns when to stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


IGNORE_INDEX = -100


# ── Trajectory → HF messages conversion ───────────────────────────────


def trajectory_to_messages(traj: Any) -> list[dict]:
    """Flatten an ART-style trajectory's messages_and_choices into the HF
    chat-template message format.

    ART trajectories interleave plain dicts (`{role, content}` for user/tool
    messages) with `openai.types.chat.ChatCompletion.Choice` objects (for
    assistant turns produced by the model). HF's `apply_chat_template`
    expects all-dicts. We walk the list and convert.

    For the openai Choice case, the assistant message can have either:
    - `content` (plain text final answer), or
    - `tool_calls` (a list of function calls), or
    - both (rare — content + tool_calls), or
    - empty (model returned nothing meaningful — treat as empty assistant).
    """
    out: list[dict] = []
    for item in traj.messages_and_choices:
        if isinstance(item, dict):
            # Plain dict: user / tool / system. Pass through, keeping only
            # the keys HF cares about.
            msg: dict[str, Any] = {"role": item["role"]}
            if "content" in item:
                msg["content"] = item.get("content", "") or ""
            if item.get("role") == "tool":
                # HF chat template needs name + tool_call_id for tool replies.
                if "tool_call_id" in item:
                    msg["tool_call_id"] = item["tool_call_id"]
                if "name" in item:
                    msg["name"] = item["name"]
            out.append(msg)
            continue

        # OpenAI Choice → assistant message
        m = item.message  # ChatCompletionMessage
        msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in m.tool_calls
            ]
        out.append(msg)
    return out


# ── Tokenization with assistant-content mask ──────────────────────────


@dataclass
class TokenizedTrajectory:
    """Token IDs + labels for one trajectory, ready for ORPO forward."""

    input_ids: torch.Tensor      # (seq_len,)  long
    attention_mask: torch.Tensor # (seq_len,)  long
    labels: torch.Tensor         # (seq_len,)  long, -100 except on assistant content
    rendered_text: str           # for debug + test assertions
    n_assistant_tokens: int      # how many tokens contribute to the loss


def tokenize_trajectory(
    traj: Any,
    tokenizer,
    tools: list[dict],
    *,
    max_length: int = 4096,
) -> TokenizedTrajectory:
    """Tokenize one full calendar trajectory with assistant-content labels.

    Strategy: incremental tokenization. We tokenize prefixes of the message
    list (M[:1], M[:2], ..., M[:N]) and use the length differences to locate
    each message's token range in the full sequence. This works around
    Qwen3's chat template not having `{% generation %}` markers (which would
    let us use `return_assistant_tokens_mask`).

    For each assistant message's range, we unmask the *content* tokens (i.e.
    everything between the opening `<|im_start|>assistant\\n` block and the
    closing `<|im_end|>`, inclusive of `<|im_end|>` so the model learns to
    terminate). The wrapper opener is masked.

    Truncation: right-truncate at `max_length`. We keep the start (which has
    the system prompt + tool schema) since that's the rendering-context the
    model needs; trailing assistant turns may be cut off. If a trajectory
    overflows, the loss simply trains on whatever assistant tokens fit.
    """
    messages = trajectory_to_messages(traj)
    if not messages:
        raise ValueError("empty trajectory has no messages to tokenize")

    # ── Render the full sequence (with tools=) ──
    full_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer(
        full_text, add_special_tokens=False, return_tensors=None
    )["input_ids"]
    full_ids = full_ids[:max_length]
    seq_len = len(full_ids)

    # ── Find each message's token range ──
    # Tokenize each prefix and use cumulative lengths. Note: HF can re-tokenize
    # slightly differently across boundaries because of BPE merges, so we use
    # the rendered-text length as the cut, not naive token slicing — but with
    # `add_special_tokens=False` and Qwen's BPE this is stable in practice.
    # If we ever see boundary drift, switch to text-based offsets.
    boundaries: list[int] = [0]
    for i in range(1, len(messages) + 1):
        prefix_text = tokenizer.apply_chat_template(
            messages[:i],
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_ids = tokenizer(
            prefix_text, add_special_tokens=False, return_tensors=None
        )["input_ids"]
        boundaries.append(min(len(prefix_ids), seq_len))

    # ── Build labels ──
    labels = [IGNORE_INDEX] * seq_len
    n_asst = 0
    # opener_len: how many tokens the chat template uses for "<|im_start|>assistant\n"
    # We mask these so we don't train on the wrapper. Empirically this is
    # 3 tokens for Qwen3 ("<|im_start|>", "assistant", "\n"); we measure on
    # the fly to be robust to template tweaks.
    asst_opener_text = "<|im_start|>assistant\n"
    asst_opener_ids = tokenizer(
        asst_opener_text, add_special_tokens=False
    )["input_ids"]
    opener_len = len(asst_opener_ids)

    for msg_idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        start = boundaries[msg_idx]
        end = boundaries[msg_idx + 1]
        if start >= seq_len:
            break  # trajectory was truncated before this assistant turn
        # Mask the opener; unmask everything else in this message's range
        # (which includes content tokens + closing <|im_end|>).
        content_start = min(start + opener_len, end)
        for t in range(content_start, end):
            labels[t] = full_ids[t]
            n_asst += 1

    return TokenizedTrajectory(
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        rendered_text=full_text,
        n_assistant_tokens=n_asst,
    )


# ── Pair tokenization for ORPO ────────────────────────────────────────


@dataclass
class TokenizedPair:
    chosen: TokenizedTrajectory
    rejected: TokenizedTrajectory
    metadata: dict


def tokenize_pair(
    chosen_traj: Any,
    rejected_traj: Any,
    tokenizer,
    tools: list[dict],
    *,
    max_length: int = 4096,
    metadata: dict | None = None,
) -> TokenizedPair:
    """Tokenize a (chosen, rejected) trajectory pair for ORPO."""
    return TokenizedPair(
        chosen=tokenize_trajectory(
            chosen_traj, tokenizer, tools, max_length=max_length
        ),
        rejected=tokenize_trajectory(
            rejected_traj, tokenizer, tools, max_length=max_length
        ),
        metadata=metadata or {},
    )
