"""End-to-end smoke test for the ORPO data pipeline.

This stitches together: difficulty tracker → sampling → fake rollouts →
pair builder → tokenizer → ORPO loss. Verifies the full data flow without
needing GPU / vLLM / ART. If this passes, the trainer's per-step loop is
validated up to the point where `orpo_train_step` would be called.

Catches integration bugs that the per-module unit tests miss — e.g. shape
mismatches between trajectory metadata, sampler output, pair_builder
expectations, and tokenize_pair's fake-trajectory format.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from calendar_agent.orpo.difficulty_tracker import DifficultyTracker
from calendar_agent.orpo.reuse_buffer import ReuseBuffer
from calendar_agent.orpo.pair_builder import build_pairs_for_step
from calendar_agent.orpo.tokenize import tokenize_pair
from calendar_agent.orpo.orpo_loss import orpo_loss


@dataclass
class FakeScenario:
    id: str
    category: str = "Schedule"


@dataclass
class FakeTraj:
    reward: float
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    messages_and_choices: list = field(default_factory=list)


def _trajectory(sid: str, reward: float, content: str = "OK"):
    """A minimal trajectory with system+user+assistant — enough for tokenize."""
    return FakeTraj(
        reward=reward,
        metadata={"scenario_id": sid, "category": "Schedule"},
        messages_and_choices=[
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "Schedule something."},
            SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None)),
        ],
    )


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-14B", trust_remote_code=True
        )
    except Exception as e:
        pytest.skip(f"Qwen3-14B tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def calendar_tools():
    from calendar_agent.tools import get_openai_tools
    return get_openai_tools()


class _ToyLM(nn.Module):
    def __init__(self, vocab_size: int = 152064, hidden: int = 32):
        super().__init__()
        # vocab matches Qwen3 to avoid out-of-range gather indices.
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        out = SimpleNamespace()
        out.logits = self.head(x)
        return out


def test_pipeline_end_to_end(tokenizer, calendar_tools):
    """Full pipeline: register scenarios → sample → fake rollouts (mixed
    correct/incorrect) → build pairs → tokenize → ORPO forward+loss.

    Asserts that the tokens reach the loss correctly and gradients flow."""
    scenarios = [FakeScenario(f"s{i}") for i in range(5)]

    tracker = DifficultyTracker(cold_start_n=1)
    tracker.register_many(scenarios)
    # Bootstrap pass-rate: scenario 0..2 are productive (mid),
    # scenario 3 is saturated easy, scenario 4 is hard.
    tracker.update("s0", [True, False], step=0)
    tracker.update("s1", [True, False], step=0)
    tracker.update("s2", [True, False], step=0)
    tracker.update("s3", [True, True], step=0)
    tracker.update("s4", [False, False], step=0)

    buffer = ReuseBuffer(per_scenario_cap=2)

    # Sample 4 scenarios
    rng = random.Random(0)
    sampled = tracker.sample_without_replacement(4, rng=rng)
    assert len(sampled) == 4

    # Fake rollouts: each sampled scenario gets a mix of correct/incorrect
    rollouts_by_scenario = {}
    for sid in sampled:
        # Always supply both correct and incorrect to guarantee pairs
        rollouts_by_scenario[sid] = [
            _trajectory(sid, 1.0, content="Done"),
            _trajectory(sid, 0.0, content="Failed"),
            _trajectory(sid, 1.0, content="Also done"),
        ]

    # Update tracker + buffer
    for sid, trajs in rollouts_by_scenario.items():
        tracker.update(sid, [t.reward == 1.0 for t in trajs], step=1)
    buffer.add_correct(t for trajs in rollouts_by_scenario.values() for t in trajs)
    assert buffer.total_size() > 0

    # Build pairs
    per_scenario, flat_pairs = build_pairs_for_step(
        rollouts_by_scenario, reuse_buffer=buffer, rng=rng,
    )
    assert flat_pairs, "no pairs generated — pipeline broken"
    assert all(sp.skip_reason is None for sp in per_scenario), (
        [sp.skip_reason for sp in per_scenario]
    )

    # Tokenize
    tokenized = []
    for sp in per_scenario:
        for chosen, rejected in sp.pairs:
            tokenized.append(tokenize_pair(
                chosen, rejected, tokenizer, calendar_tools, max_length=2048,
            ))
    # Sanity: every tokenized pair must have at least 1 unmasked label on both
    # sides. If max_length is too small, the assistant content gets truncated
    # away and we silently get zero gradient — which is what would have hit
    # the "no grad" assert below. Catch it earlier with a clear message.
    for tp in tokenized:
        assert tp.chosen.n_assistant_tokens > 0, (
            "chosen has no assistant tokens after tokenization — "
            "likely max_length too small for the rendered prompt"
        )
        assert tp.rejected.n_assistant_tokens > 0, "rejected has no assistant tokens"

    # Pad each side and stack into a single forward pass
    B = len(tokenized)
    T_c = max(t.chosen.input_ids.shape[0] for t in tokenized)
    T_r = max(t.rejected.input_ids.shape[0] for t in tokenized)

    def _stack(side: str, T: int):
        ids = torch.zeros(B, T, dtype=torch.long)
        attn = torch.zeros(B, T, dtype=torch.long)
        lbl = torch.full((B, T), -100, dtype=torch.long)
        for i, p in enumerate(tokenized):
            t = getattr(p, side)
            L = t.input_ids.shape[0]
            ids[i, :L] = t.input_ids
            attn[i, :L] = t.attention_mask
            lbl[i, :L] = t.labels
        return ids, attn, lbl

    c_ids, c_attn, c_lbl = _stack("chosen", T_c)
    r_ids, r_attn, r_lbl = _stack("rejected", T_r)

    # ORPO forward + loss
    model = _ToyLM(vocab_size=tokenizer.vocab_size + 1024)  # cushion for special tokens
    out = orpo_loss(model, c_ids, c_attn, c_lbl, r_ids, r_attn, r_lbl)

    assert torch.isfinite(out.loss)
    assert out.loss.requires_grad
    out.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
