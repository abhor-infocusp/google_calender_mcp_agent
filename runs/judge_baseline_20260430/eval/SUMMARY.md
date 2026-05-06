# Judge Baseline Eval — Run Artifacts

> Run-dir summary. **Canonical knowledge moved to
> [`docs/judge/baseline.md`](../../../docs/judge/baseline.md) on 2026-05-01.**

This run produced the **manual oracle** that's used as ground truth for
every judge measurement going forward:

- `manual_verdicts.jsonl` — 285 hand-labeled trajectories (185 C / 100 I after 2026-05-01 relabel)
- `manual_verdicts_v1.jsonl` — pre-2026-05-01 snapshot (187 C / 98 I)
- `manual_review_input.jsonl` — compact view of the 285 trajectories
- `art_holdout_qwen3_8b_base.json`, `art_holdout_qwen3_14b_base.json`,
  `art_holdout_qwen3_32b_base.json` — per-trajectory predictions of the
  three base-model judges
- `manual_judge_comparison.json` — per-judge per-category accuracy table

For interpretation (which judges win where, error correlation, label-noise
analysis), read [`docs/judge/baseline.md`](../../../docs/judge/baseline.md).
