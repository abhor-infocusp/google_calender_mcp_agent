# Schedule — Schedule a Single Event

## What it tests

Create a single event with the right time, attendees, and metadata from a
direct user request. Often requires reading existing events first to avoid
conflicts.

Example queries: *(fill in)*

## Current best

SFT v6 ckpt-4659 (ep 3) — **70%** on test_data. DPO-from-SFT actually wins
here at **79%** (+9 pp), the only category where DPO helped.

## History

- 1.5B SFT v3: 52.5% on RL data
- 1.5B SFT v5 (ep 4): 87.5% on RL data
- 1.5B RL1 Modifier: 67.5% (regression vs v5; suggests RL on other categories
  hurts Schedule)
- 14B SFT v6: 80% (RL data) / 70% (test_data) — different ckpt wins per benchmark
- 14B DPO-from-SFT: **79%** on test_data — wins this category

## Failure modes

- **Action vs query confusion** (1.5B-era): 16% of failures were creating an
  event when a read was expected, or vice versa.
- Conflict-detection: model schedules over an existing event without
  acknowledging it.
- Time-zone / relative-time edge cases (overlap with RelTime).

## Next experiments

- Investigate why DPO-from-SFT wins here while losing Complex — pair
  composition may be skewed toward Schedule rejections.
- Re-eval after RL adaptive run completes.
