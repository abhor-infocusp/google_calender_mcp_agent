# RelTime — Relative Time References

## What it tests

Resolving relative time expressions ("tomorrow", "next Tuesday", "in 3 hours",
"the day after my flight") into absolute timestamps before querying or
creating events.

Example queries: *(fill in)*

## Current best

SFT v6 ckpt-4659 (ep 3) — **95%** on test_data. Saturated.

## History

- 1.5B SFT v3: 72.5% on RL data
- 1.5B SFT v5 (ep 4): 92.5%
- 1.5B RL runs: stable 72-80%
- 14B SFT v6 ep3: 98% on RL data; ep4: **100%**; on test_data: 95%

## Failure modes

- Edge weeks (year boundary, DST shifts) — rare in test_data.
- Implicit timezone assumptions when user doesn't specify.

## Next experiments

- Saturated. Any future loss here flags a regression.
