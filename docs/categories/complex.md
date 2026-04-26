# Complex — Multi-step Logic & Conflict

> **Currently the weakest category.** Drives most of the gap from 80% → 90%.

## What it tests

Queries that require chaining multiple tool calls with intermediate reasoning:
detect a conflict, resolve it under constraints, or compute a multi-step
inference over event metadata.

Example queries: *(fill in 1-2 representative real queries from `test_data/`)*

## Current best

SFT v6 ckpt-4659 (ep 3) — **59%** on test_data.

## History

- 1.5B SFT v3: 12.5% on RL data
- 1.5B SFT v5 (ep 4): 32.5%
- 1.5B RL1 Modifier-focused: 22.5% (positive transfer)
- 14B SFT v6 ep3: 62% on RL data; ep4: 57% (regresses) — model trades Complex
  for gains in other categories during longer SFT
- 14B DPO-from-SFT: 52% on test_data (−7 pp vs SFT v6) — clearly hurt

## Failure modes

- **Multi-step inference**: e.g. "what's after my networking event" — model
  fails to retrieve, then filter, then answer.
- **Reasoning over tool results**: long `list_events` results (>1500 chars)
  drop accuracy from ~80% to ~33% (1.5B-era trajectory analysis).
- **Capacity ceiling on 1.5B**: 79% of v5 failures were comprehension, not
  procedure. 14B helped but did not close the gap.

## Next experiments

- RFT/expert iteration on Complex-only correct rollouts from SFT v6.
- Re-augment Complex training data with longer multi-step trajectories.
- Local judge with reasoning trace may give denser RL signal here than
  binary Gemini verdict.
