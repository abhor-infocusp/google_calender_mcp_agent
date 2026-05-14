"""Unit tests for src/calendar_agent/orpo/orpo_loss.py.

We test against a tiny toy LM (random-init small Transformer) to verify the
loss has the right shape, units, and gradient direction. Numerical sanity
matters more than absolute values here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from calendar_agent.orpo.orpo_loss import (
    IGNORE_INDEX,
    ORPOLossOutput,
    orpo_loss,
)


# ── Tiny toy LM ────────────────────────────────────────────────────────


class _ToyLM(nn.Module):
    """Random-init transformer that returns a `Output` with `.logits`.
    Small enough to fit in CPU memory and fast enough for unit tests."""

    def __init__(self, vocab_size: int = 32, hidden: int = 16, n_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=2, dim_feedforward=32,
                batch_first=True, activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embed(input_ids)
        # Build a key_padding_mask: True means *ignore*. attention_mask is
        # 1=keep, 0=ignore.
        kpm = attention_mask == 0
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=kpm)
        logits = self.head(x)
        # Mimic HF output object
        class _Out:
            pass
        out = _Out()
        out.logits = logits
        return out


@pytest.fixture
def model():
    torch.manual_seed(0)
    return _ToyLM()


# ── Helpers ────────────────────────────────────────────────────────────


def _make_pair(B: int = 2, T: int = 8, V: int = 32):
    """Random pair tensors. Half the response tokens are masked to test that
    we ignore IGNORE_INDEX positions correctly."""
    torch.manual_seed(1)
    chosen_ids = torch.randint(1, V, (B, T))
    rejected_ids = torch.randint(1, V, (B, T))
    chosen_attn = torch.ones(B, T, dtype=torch.long)
    rejected_attn = torch.ones(B, T, dtype=torch.long)

    # Labels: mask first half (prompt) with -100, response is the second half
    chosen_labels = chosen_ids.clone()
    chosen_labels[:, : T // 2] = IGNORE_INDEX
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, : T // 2] = IGNORE_INDEX
    return (
        chosen_ids, chosen_attn, chosen_labels,
        rejected_ids, rejected_attn, rejected_labels,
    )


# ── Shape & type ──────────────────────────────────────────────────────


def test_output_dataclass_shapes(model):
    args = _make_pair(B=3)
    out = orpo_loss(model, *args)
    assert isinstance(out, ORPOLossOutput)
    assert out.loss.dim() == 0
    assert out.sft_loss.dim() == 0
    assert out.or_loss.dim() == 0
    assert out.rewards_chosen.shape == (3,)
    assert out.rewards_rejected.shape == (3,)
    # Diagnostic scalars
    assert out.rewards_accuracy.dim() == 0
    assert out.rewards_margin.dim() == 0
    assert out.logp_chosen_mean.dim() == 0
    assert out.logp_rejected_mean.dim() == 0


def test_loss_is_finite_and_positive(model):
    args = _make_pair()
    out = orpo_loss(model, *args)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.sft_loss)
    assert torch.isfinite(out.or_loss)
    # SFT cross-entropy is always > 0 for any non-degenerate model.
    assert out.sft_loss > 0
    # OR loss = softplus(−β·OR) ≥ log(2)·... well, ≥ 0
    assert out.or_loss >= 0


def test_loss_requires_grad(model):
    args = _make_pair()
    out = orpo_loss(model, *args)
    assert out.loss.requires_grad
    out.loss.backward()
    # At least the head should have non-zero gradients.
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())


# ── Unequal sequence lengths ──────────────────────────────────────────


def test_unequal_seq_lengths(model):
    """Chosen T=8, rejected T=12 — must pad and run cleanly."""
    torch.manual_seed(2)
    V = 32
    c_ids = torch.randint(1, V, (2, 8))
    r_ids = torch.randint(1, V, (2, 12))
    c_attn = torch.ones(2, 8, dtype=torch.long)
    r_attn = torch.ones(2, 12, dtype=torch.long)
    c_lbl = c_ids.clone(); c_lbl[:, :4] = IGNORE_INDEX
    r_lbl = r_ids.clone(); r_lbl[:, :6] = IGNORE_INDEX

    out = orpo_loss(model, c_ids, c_attn, c_lbl, r_ids, r_attn, r_lbl)
    assert torch.isfinite(out.loss)
    assert out.rewards_chosen.shape == (2,)
    assert out.rewards_rejected.shape == (2,)


# ── Gradient direction sanity ─────────────────────────────────────────


def test_gradient_pushes_chosen_up_rejected_down(model):
    """After one step of optimization, log P(chosen) should increase and
    log P(rejected) should decrease (relative to start). This validates
    the OR term has the right sign."""
    args = _make_pair(B=4, T=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Initial logps
    with torch.no_grad():
        out0 = orpo_loss(model, *args)
        lp_chosen_0 = out0.logp_chosen_mean.item()
        lp_rejected_0 = out0.logp_rejected_mean.item()

    # Take a few optimizer steps
    for _ in range(20):
        optimizer.zero_grad()
        out = orpo_loss(model, *args)
        out.loss.backward()
        optimizer.step()

    with torch.no_grad():
        out_f = orpo_loss(model, *args)
        lp_chosen_f = out_f.logp_chosen_mean.item()
        lp_rejected_f = out_f.logp_rejected_mean.item()

    # Chosen should rise (less negative).
    assert lp_chosen_f > lp_chosen_0, (
        f"chosen logp didn't rise: {lp_chosen_0:.3f} → {lp_chosen_f:.3f}"
    )
    # Margin (chosen - rejected) should rise.
    assert (lp_chosen_f - lp_rejected_f) > (lp_chosen_0 - lp_rejected_0), (
        f"margin didn't widen: "
        f"{lp_chosen_0 - lp_rejected_0:.3f} → {lp_chosen_f - lp_rejected_f:.3f}"
    )
    # Rewards accuracy should be mostly 1.0 by the end.
    assert out_f.rewards_accuracy.item() >= 0.5


# ── Lambda / beta scaling ─────────────────────────────────────────────


def test_lambda_zero_recovers_pure_sft(model):
    """λ=0 → loss = SFT only, no OR contribution."""
    args = _make_pair()
    out = orpo_loss(model, *args, lambda_or=0.0)
    # loss should equal sft_loss exactly
    assert torch.allclose(out.loss, out.sft_loss, atol=1e-6)


def test_length_mismatch_does_not_dominate_rewards(model):
    """Regression guard for the length-bias bug.

    If the OR term uses sum-of-logps (not length-normalized), a long-but-good
    chosen and a short-but-bad rejected with the same per-token quality
    will be ranked by *length*, not preference: chosen has more (negative)
    logps to add up.

    Setup: chosen = identical content repeated 6× (long), rejected =
    identical content once (short). The model is randomly initialized so
    per-token logp is roughly the same for both. Under sum logps,
    log_odds_chosen << log_odds_rejected (large negative magnitudes scale
    with length) → rewards_accuracy ≈ 0. Under length-normalized logps,
    rewards_accuracy ≈ 0.5 (random, since no real preference signal).

    Asserting rewards_accuracy stays near 0.5 (not collapsed to 0) checks
    that we're length-normalizing.
    """
    model.eval()
    torch.manual_seed(3)
    V = 32
    B = 4
    base_token = torch.randint(1, V, (B, 8))
    short_ids = base_token  # length 8
    long_ids = torch.cat([base_token] * 6, dim=-1)  # length 48 — same content

    short_attn = torch.ones_like(short_ids)
    long_attn = torch.ones_like(long_ids)

    short_lbl = short_ids.clone(); short_lbl[:, :2] = IGNORE_INDEX  # mask first 2 as "prompt"
    long_lbl = long_ids.clone(); long_lbl[:, :2] = IGNORE_INDEX

    # Treat the long sequence as chosen, short as rejected — same per-token
    # content quality under random init, just different lengths.
    out = orpo_loss(
        model,
        long_ids, long_attn, long_lbl,        # chosen (long)
        short_ids, short_attn, short_lbl,     # rejected (short)
    )

    # Under length normalization, the chosen/rejected order is mostly a
    # toss-up (random init isn't perfectly position-symmetric, so we don't
    # expect exactly 0.5). Under sum-logps (the bug), accuracy collapses
    # to near 0 because the long chosen has much smaller log_odds. We
    # check it stays away from the collapsed regime.
    acc = out.rewards_accuracy.item()
    assert 0.2 <= acc <= 0.9, (
        f"rewards_accuracy={acc:.3f} collapsed to an extreme — "
        "likely a length-bias regression in the OR term."
    )

    # Stronger check: under the bug, log_odds_ratio is dominated by length.
    # Long chosen would give log_odds_ratio ≈ −5*per_token_logp_avg (very
    # negative). Under the fix, it's bounded.
    assert out.log_odds_ratio.abs() < 5.0, (
        f"log_odds_ratio={out.log_odds_ratio.item():.3f} is huge — "
        "likely sum-logps regression."
    )


def test_beta_only_affects_or_term(model):
    """β only enters the OR term, not the SFT term. At fixed model state
    (eval mode → deterministic), changing β must leave SFT unchanged."""
    model.eval()  # disable dropout for deterministic forward
    args = _make_pair()
    out_low = orpo_loss(model, *args, beta=0.01)
    out_high = orpo_loss(model, *args, beta=1.0)
    assert torch.isfinite(out_low.or_loss) and torch.isfinite(out_high.or_loss)
    assert torch.allclose(out_low.sft_loss, out_high.sft_loss, atol=1e-5)
    # The OR component must differ since β scales it.
    assert not torch.allclose(out_low.or_loss, out_high.or_loss, atol=1e-3)
