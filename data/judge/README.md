# `data/judge/` — judge-distillation datasets

## Canonical: `v2_20260502/`

The current dataset for distilling a small student judge. Built 2026-05-02 from:
- 285 manual-oracle gold,
- 224 fresh Claude-agent labels (Phase 1 + Phase 3 disagreement adjudication),
- 373 silver labels (Qwen-v2 ∧ Gemini-v2 agreement, gated by per-cat `P(both wrong | agree)` ≤ 5%).

Total: **744 train / 110 eval / 85 disagreements (kept separately for analysis)**.

See `v2_20260502/README.md` for the full schema, per-source counts, quality gates, and known limitations.

## Historical: `train.jsonl` + `val.jsonl` (v1)

- **Status**: superseded.
- **Built**: ~2026-04-25.
- **Source**: Gemini-2.0-flash labels (canonical truth at the time was Gemini, before the 285-traj manual oracle existed). The 86.7% accuracy of Gemini propagates ~13% label noise into v1.
- **Why kept**: reproducing earlier 7B-judge SFT runs and ablations. Do **not** use for new distillation work.
- **Documented as superseded** in `docs/judge/plan.md:14-77`.

## Choosing between v1 and v2

| Use case | Use |
|---|---|
| Distilling a new student judge | **v2** |
| Re-running 2026-04-25 SFT for comparison | v1 |
| Fresh evaluation against gold | `v2_20260502/eval.jsonl` only — `v1/val.jsonl` was Gemini-labeled. |
