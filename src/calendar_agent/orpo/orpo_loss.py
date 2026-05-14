"""ORPO loss: SFT cross-entropy on chosen + odds-ratio preference term.

Reference: Hong et al. 2024, "ORPO: Monolithic Preference Optimization without
Reference Model" (arxiv 2403.07691).

L_ORPO  =  L_SFT(chosen)  −  λ · log σ( β · log_odds_ratio )

where
    log P(y|x)        = sum of per-token log-probs over response tokens
    log_odds(y|x)     = log P(y|x) − log(1 − P(y|x))
    log_odds_ratio    = log_odds(chosen) − log_odds(rejected)
    L_SFT(chosen)     = standard cross-entropy on chosen response tokens

We do one concatenated forward pass for memory efficiency: stack the chosen
and rejected sequences along the batch dim, run the model once, then split
the outputs to compute the two log P values and the SFT loss.

The β here is ORPO's β (controls preference scaling, default 0.1 from TRL).
The λ is the relative weight of the OR term vs. SFT term (TRL default: 1.0,
i.e. equal weight). We use TRL defaults but expose both as args.

This is a self-contained loss function — no torch.nn.Module wrapper, no
Trainer state. Caller supplies the model, the batch tensors, and the hyper-
parameters; we return the loss + a dict of training-side metrics for logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


@dataclass
class ORPOLossOutput:
    loss: torch.Tensor                    # scalar, requires_grad
    sft_loss: torch.Tensor                # scalar, the L_SFT(chosen) component
    or_loss: torch.Tensor                 # scalar, the −λ·log σ(β·OR) component
    log_odds_ratio: torch.Tensor          # scalar, mean log_odds(c) − log_odds(r) across batch
    rewards_chosen: torch.Tensor          # (B,) per-pair log_odds(chosen)
    rewards_rejected: torch.Tensor        # (B,) per-pair log_odds(rejected)
    rewards_accuracy: torch.Tensor        # scalar, fraction where rewards_chosen > rewards_rejected
    rewards_margin: torch.Tensor          # scalar, mean (rewards_chosen − rewards_rejected)
    logp_chosen_mean: torch.Tensor        # scalar, mean log P(chosen)
    logp_rejected_mean: torch.Tensor      # scalar, mean log P(rejected)


def _token_logps(
    logits: torch.Tensor,           # (B, T, V)
    labels: torch.Tensor,           # (B, T)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence sum-of-logp (over un-masked positions) + token count.

    Memory-aware computation: NEVER materialize log_softmax over the full
    (B, T-1, V) tensor — that's ~20 GiB in fp32 for our (8, 4095, 152064)
    case and OOMs the slice. Instead use the identity

        log P(y_t) = logits[y_t] − logsumexp(logits)

    which keeps memory at (B, T) for both the gather output and the
    logsumexp output. logsumexp is computed in fp32 internally for
    numerical stability without ever creating a (B, T, V) fp32 tensor.

    Standard causal-LM shift: logits at position t predict token at t+1, so
    we predict labels[:, 1:] from logits[:, :-1].

    Returns
    -------
    logps  : (B,) sum of per-token log-probs over un-masked positions
    n_toks : (B,) count of un-masked positions per sequence
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, Tm1, V = shift_logits.shape

    mask = shift_labels != IGNORE_INDEX
    safe_labels = shift_labels.masked_fill(~mask, 0)

    # gather: (B, T-1, 1) → (B, T-1). Tiny; same dtype as logits (bf16 OK).
    label_logits = shift_logits.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    # logsumexp in chunks along T to avoid materializing (B, T-1, V) fp32
    # all at once. (B=8, T-1≈4095, V=152064) → 19.9 GiB if done in one shot.
    # Chunk size 256 keeps each step at (B, 256, V) fp32 ≈ 1.24 GiB.
    LSE_CHUNK = 256
    lse = torch.empty(B, Tm1, dtype=torch.float32, device=shift_logits.device)
    for start in range(0, Tm1, LSE_CHUNK):
        end = min(start + LSE_CHUNK, Tm1)
        # `.float()` here only materializes a (B, chunk, V) tensor.
        lse[:, start:end] = torch.logsumexp(
            shift_logits[:, start:end, :].float(), dim=-1
        )
    per_token_logps = (label_logits.float() - lse) * mask

    seq_logps = per_token_logps.sum(dim=-1)
    n_toks = mask.sum(dim=-1).clamp(min=1)
    return seq_logps, n_toks


def orpo_loss(
    model,
    chosen_input_ids: torch.Tensor,       # (B, T_c) long
    chosen_attention_mask: torch.Tensor,  # (B, T_c) long
    chosen_labels: torch.Tensor,          # (B, T_c) long
    rejected_input_ids: torch.Tensor,     # (B, T_r) long
    rejected_attention_mask: torch.Tensor,# (B, T_r) long
    rejected_labels: torch.Tensor,        # (B, T_r) long
    *,
    beta: float = 0.1,
    lambda_or: float = 1.0,
) -> ORPOLossOutput:
    """One concatenated forward pass + ORPO loss computation.

    `model` is expected to be a HF causal-LM-style module returning an object
    with `.logits`. Both PEFT-wrapped and raw models satisfy this.

    chosen_* and rejected_* may have different sequence lengths (T_c ≠ T_r).
    We pad both to max(T_c, T_r) before stacking into the concatenated batch,
    using the model's pad token id from labels (-100 padding is fine for
    labels; for input_ids we left-pad with the pad_token_id implicit in the
    attention mask — well, actually right-pad with 0s and rely on
    attention_mask to ignore them).
    """
    B, T_c = chosen_input_ids.shape
    _, T_r = rejected_input_ids.shape
    T_max = max(T_c, T_r)
    device = chosen_input_ids.device

    def _pad(x: torch.Tensor, T: int, fill: int) -> torch.Tensor:
        if x.shape[-1] == T:
            return x
        pad = torch.full((x.shape[0], T - x.shape[-1]), fill,
                         dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)

    # Right-pad to T_max. attention_mask=0 on padding makes the model ignore
    # them; labels=IGNORE_INDEX makes them not contribute to loss.
    cat_input_ids = torch.cat([
        _pad(chosen_input_ids, T_max, 0),
        _pad(rejected_input_ids, T_max, 0),
    ], dim=0)
    cat_attention_mask = torch.cat([
        _pad(chosen_attention_mask, T_max, 0),
        _pad(rejected_attention_mask, T_max, 0),
    ], dim=0)
    cat_labels = torch.cat([
        _pad(chosen_labels, T_max, IGNORE_INDEX),
        _pad(rejected_labels, T_max, IGNORE_INDEX),
    ], dim=0)

    # ── Forward pass ──
    outputs = model(
        input_ids=cat_input_ids,
        attention_mask=cat_attention_mask,
    )
    logits = outputs.logits  # (2B, T_max, V)

    seq_logps, n_toks = _token_logps(logits, cat_labels)
    logp_chosen, logp_rejected = seq_logps[:B], seq_logps[B:]
    n_chosen, n_rejected = n_toks[:B], n_toks[B:]

    # ── Per-token average log-prob (length-normalized) ──
    # Used for BOTH the SFT loss term and the OR term. Without normalization,
    # log P(y|x) is dominated by sequence length (a 500-token chosen with
    # logp_avg=−2 and a 50-token rejected with logp_avg=−1 would yield
    # logp_chosen=−1000, logp_rejected=−50 → log_odds_ratio ≈ −950 driven
    # entirely by length, not preference). TRL's ORPOTrainer length-normalizes
    # before the OR term for the same reason. We do this once here and use
    # the avg-logp for both terms.
    avg_logp_chosen = logp_chosen / n_chosen      # (B,)
    avg_logp_rejected = logp_rejected / n_rejected

    # ── SFT loss on chosen ── per-sequence avg-CE → batch mean.
    sft_loss = -avg_logp_chosen.mean()

    # ── Odds-ratio loss ──
    # log_odds(y|x) = log P_avg − log(1 − P_avg)
    # where P_avg = exp(avg_logp). With length-normalized logps in the
    # range roughly [−10, 0], log(1 − exp(logp)) is well-conditioned;
    # we still use the stable two-branch formulation for safety.
    def _log1mexp(x: torch.Tensor) -> torch.Tensor:
        # Stable log(1 − exp(x)) for x ≤ 0. Two-branch: use log(-expm1(x))
        # when x > -log(2) ≈ -0.693, else log1p(-exp(x)).
        return torch.where(
            x > -0.6931,
            torch.log(-torch.expm1(x.clamp(max=-1e-30))),
            torch.log1p(-torch.exp(x)),
        )

    log_odds_chosen = avg_logp_chosen - _log1mexp(avg_logp_chosen)
    log_odds_rejected = avg_logp_rejected - _log1mexp(avg_logp_rejected)
    log_odds_ratio = log_odds_chosen - log_odds_rejected  # (B,)

    # OR loss: −log σ(β · OR). Equivalent to softplus(−β · OR).
    or_loss_per = F.softplus(-beta * log_odds_ratio)
    or_loss = or_loss_per.mean()

    loss = sft_loss + lambda_or * or_loss

    # ── Diagnostic reward signals ──
    rewards_chosen = log_odds_chosen.detach()
    rewards_rejected = log_odds_rejected.detach()
    rewards_margin = (rewards_chosen - rewards_rejected).mean()
    rewards_accuracy = (rewards_chosen > rewards_rejected).float().mean()

    return ORPOLossOutput(
        loss=loss,
        sft_loss=sft_loss.detach(),
        or_loss=or_loss.detach(),
        log_odds_ratio=log_odds_ratio.mean().detach(),
        rewards_chosen=rewards_chosen,
        rewards_rejected=rewards_rejected,
        rewards_accuracy=rewards_accuracy.detach(),
        rewards_margin=rewards_margin.detach(),
        # Report the *avg* per-token logp (length-normalized) for diagnostics —
        # consistent with what the OR term sees. Sum-logps are misleading
        # across variable-length pairs.
        logp_chosen_mean=avg_logp_chosen.mean().detach(),
        logp_rejected_mean=avg_logp_rejected.mean().detach(),
    )
