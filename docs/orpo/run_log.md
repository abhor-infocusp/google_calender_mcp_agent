# ORPO run log

> Per-run summary table. Append, don't edit prior entries.
> See `docs/orpo/design.md` for the algorithm and hyperparameter rationale.

| Run dir | Started | Initial LoRA | N | k_easy/k_hard | β | λ | LR | Stopped at step | Result |
|---|---|---|---|---|---|---|---|---|---|
| `runs/rl_orpo_qwen3_14b_20260508_0625` | 2026-05-08 | SFT v6 ckpt-4659 merged-base (fresh r=8) | 20 | 4/8 | 0.1 | 1.0 | 5e-6 | 621 (full 20 epochs) | **+4.1pp** held-out: SFT 80.1 → ORPO ckpt-600 **84.25%**. ckpt-427 tied at 84.2%. |

## 2026-05-08 → 2026-05-14: first end-to-end ORPO run

**Result table (held-out `test_data/`, Gemini-structured judge):**

| ckpt | overall | category gainers/losers vs SFT |
|---|---|---|
|  50 | 79.9% | ≈ baseline |
| 100 | 80.6% | |
| 150 | 80.5% | |
| 200 | **83.1%** | first break-through |
| 250 | 82.7% | |
| 300 | 82.1% | |
| 350 | 82.5% | |
| 400 | 82.8% | |
| **427** | **84.2%** | tied peak (pre-stop) |
| 450 | 82.2% | post-resume regress |
| 475 | 83.4% | |
| 500 | 83.8% | |
| 525 | 82.95% | |
| 550 | (failed eval; retry returned 0%) | adapter-load issue, skipped |
| 575 | 82.8% | |
| **600** | **84.25%** | tied peak — new best by 1 question |
| 620 | 83.1% | |
| 621 | (last; pending eval) | |

**Per-category at ckpt-600 vs ckpt-50 (≈ SFT baseline):**

| category | ckpt-50 | ckpt-600 | Δ |
|---|---|---|---|
| Vague & Contextual | 74.5 | 85.7 | **+11.2** |
| Schedule | 79.6 | 87.8 | +8.2 |
| Complex | 59.2 | 66.3 | +7.1 |
| Human Chaos | 81.6 | 84.7 | +3.1 |
| Relative Time | 83.7 | 86.5 | +2.8 |
| Information Retrieval | 93.9 | 93.9 | 0 |
| Modifier & Correction | 86.7 | 84.7 | −2.0 |

**Incidents during the run:**

1. **Slurm 72h timeout (step 427).** Cleanly stopped at the time limit, not a hang.
2. **Optimizer-device crash on resume.** `orpo_optimizer.pt` was saved with Adam moments on CPU (model is offloaded between steps); on first `optimizer.step()` after resume, params were on GPU → `RuntimeError: tensors on different devices`. Patched: after `reload_to_gpu()` in `orpo_train_step`, walk `optimizer.state` and move any CPU tensor to the param's device. See `rl_orpo.py:594-606`.
3. **Buffer reset on resume.** Pre-stop buffer 2,386 trajectories / 603 scenarios → post-resume 0. Wasn't on the persistence list. Patched: added `ReuseBuffer.save()`/`load()` (pickle, atomic) in `src/calendar_agent/orpo/reuse_buffer.py`; `rl_orpo.py` now loads `reuse_buffer.pkl` at startup and persists in the `FLUSH_EVERY` block + final flush. Patch took effect for *future* runs; this run's resume paid a ~150-step cost re-warming the buffer (hit rate 88% → 27% → 58%).
4. **Held-out eval bug for ckpt-550.** Two separate eval runs (initial parallel batch + retry) produced empty results / 0% accuracy. Adapter dir is intact on disk; likely a vLLM LoRA-load edge case specific to that step. Not investigated — left as a gap in the trajectory.
5. **Parallel eval orchestration accident.** First "serial" orchestrator submitted all 9 ckpts as separate sbatch jobs because the squeue wait-loop was flaky; cancelled and replaced with a JSON-existence-based controller (`/tmp/orpo_serial_v3.sh`). Single-slice serial ran clean for the rest.

**Algorithm internals (full run):**

- Difficulty sampler `1 − p^G − (1−p)^G` *worked at the margins* — mid scenarios were over-sampled by 29× vs their pool share at peak — but **couldn't escape easies** as the pool composition skewed. By step 621: 594 easy / 1 mid / 27 hard / 0 cold. **Late-window sampled distribution: 88% easy / 7% mid / 5% hard.**
- Adaptive-k saved compute: k=8 share fell 67% → 5%. Step time 819s → 275s (−66%).
- Buffer rescue hit rate climbed to 88% peak by step ~150, dropped to ~27% post-resume re-warm, recovered to 58% by end. `reuse_pairs_share` climbed 3% → 15% late as the buffer carried more of the gradient load when mixed-reward groups dried up.
- **Compute waste growth:** `skip_easy/step` 8.7 → 16+/20. Pair-producing scenarios collapsed 272 → 34 (top-50 share of pairs: 38% → 100% by end).
- **Training signal kept improving:** loss 0.86 → 0.66 (best), margin +0.14 → +1.19, rew_acc 0.59 → 0.71. No likelihood displacement.

**Comparison to GRPO (`rl_qwen3_14b_20260420`):**

- GRPO: ~3.5 days active training → +5pp (rl-sft-4952) / ~40k rollouts.
- ORPO: ~3 days active training → +4.1pp (ckpt-427/600) / ~51k rollouts.
- Wall-clock parity. ORPO trained on ~30k preference pairs vs GRPO's 4,952 advantage updates → **~6× more signal-dense per rollout**, but per-step training cost is ~20× higher (concatenated-forward, many optimizer steps), so the net was a wash.

**Promotion decision:** ckpt-600 (84.25%) is the candidate. Update `runs/analysis/test_eval_summary.md` after final ckpt-621 eval lands.

**Postmortem and v2 plan:** see `docs/orpo/design.md` "Postmortem (run-1) / v2 plan" section.

## Template for new runs

```
| `runs/rl_orpo_<...>` | YYYY-MM-DD | <ckpt path> | N | 4/8 | 0.1 | 1.0 | 5e-6 | <step> | <one-line outcome> |
```

After each run, also update PROGRESS.md with the headline result and any
notable observations (skip rates, buffer hit rates, reward_accuracy curve,
catastrophic-forgetting alerts).

## What to look at first when reading a run

The headline metric for ORPO is `training.rewards_accuracy` from
`runs/<run>/orpo_diagnostic.jsonl` — fraction of pairs where the current
model prefers chosen over rejected. Should rise from ~0.5 (random) toward
1.0 over training.

Other key signals:

- `pairs.skipped.all_correct`: should drift down as the sampler avoids
  saturated easy scenarios.
- `pairs.skipped.all_fail_no_buffer`: should drift down as the buffer warms
  up. After ~50 steps the buffer should cover most regularly-sampled
  hard scenarios.
- `buffer.rescue_hits / rescue_attempts`: ratio of all-fail groups we
  successfully rescued via buffer splice. Climbs from ~0 to ~0.7+ over
  the first ~100 steps if the scenario distribution is sane.
- `training.rewards_margin`: log-odds gap between chosen and rejected.
  Mirrors `rewards_accuracy` but on a continuous scale.
- `training.loss_orpo_mean / loss_sft_mean / loss_or_mean`: split into
  components so you can see which term is doing the work. Expect SFT to
  dominate early, OR to grow in importance after warmup.

## Common diagnostic queries

```bash
# Reward accuracy progression
jq '.training.rewards_accuracy' runs/<run>/orpo_diagnostic.jsonl

# Skip-reason breakdown
jq '.pairs.skipped' runs/<run>/orpo_diagnostic.jsonl | sort | uniq -c | sort -rn

# Buffer warmup curve
jq '{step, buf_size: .buffer.size_total, scen: .buffer.scenarios_covered, rescues: .buffer.rescue_hits}' \
   runs/<run>/orpo_diagnostic.jsonl

# Wall-clock budget breakdown
jq '.phase_timing_s' runs/<run>/orpo_diagnostic.jsonl
```
