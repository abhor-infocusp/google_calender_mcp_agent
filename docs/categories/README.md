# Training Categories

Seven categories the model is trained on and evaluated against. Each has its
own doc with current performance, failure modes, and improvement targets.
Numbers here are pulled from `runs/analysis/test_eval_summary.md` (held-out
`test_data/`, SFT v6 ckpt-4659 unless noted) and should be re-pulled when
a new ckpt wins.

| Category | Acc (test_data) | Status | Doc |
|---|---|---|---|
| RelTime — relative time references | 95% | saturated | [reltime.md](reltime.md) |
| IR — information retrieval | 90% | saturated | [ir.md](ir.md) |
| Modifier — correct/modify last action | 89% | saturated | [modifier.md](modifier.md) |
| Chaos — edge cases, ambiguity, mess | 83% | strong | [chaos.md](chaos.md) |
| Vague — under-specified, contextual | 80% | room to improve | [vague.md](vague.md) |
| Schedule — schedule a single event | 70% | room to improve | [schedule.md](schedule.md) |
| Complex — multi-step / conflict | 59% | weakest | [complex.md](complex.md) |

## Conventions

Each category doc has these sections (keep them tight; bullets, not prose):

1. **What it tests** — one paragraph + 1-2 representative example queries.
2. **Current best** — model, ckpt, accuracy on `test_data/`. One line.
3. **History** — short bullet list of how this category has moved across
   experiments (SFT v5 → v6, RL1-3, DPO, etc). Cross-link to PROGRESS.md
   timeline entries; do not duplicate full tables.
4. **Failure modes** — recurring patterns from rollout/eval analysis.
5. **Next experiments** — open hypotheses. When one is tried, move it to
   History with the result.

Don't put per-checkpoint tables here — those live in
`runs/<run>/eval/summary.csv` and `runs/analysis/test_eval_summary.md`.
This doc explains *why* a category is hard, what's been tried, and what's next.
