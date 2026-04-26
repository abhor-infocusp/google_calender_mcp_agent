# Chaos — Edge Cases & Human Mess

## What it tests

Messy real-world inputs: typos, contradictions, mid-sentence corrections,
incomplete sentences, multiple conflicting requests in one query.

Example queries: *(fill in)*

## Current best

SFT v6 ckpt-4659 (ep 3) — **83%** on test_data.

## History

- 1.5B SFT v3: 15% on RL data — by far the worst category at the time
- Originally only 5 trajectories (3.1% of SFT data); augmentation took it to 65 (6.3%)
- 1.5B SFT v5 (ep 4): 77.5% on RL data — augmentation worked
- 1.5B RL1 Modifier: 12.5% (regression)
- 14B SFT v6 ep3: 75% on RL data; ep4: 88%; on test_data: 83%

## Failure modes

- Picking up the wrong intent when the user contradicts themselves mid-query.
- Over-literally executing a typo'd request instead of interpreting it.
- Ignoring corrections (overlap with Modifier).

## Next experiments

- Already strong; not a priority. Watch for regression during RL.
