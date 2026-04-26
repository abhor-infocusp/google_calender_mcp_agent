# IR — Information Retrieval

## What it tests

Fetch and summarize event information: list events in a window, find an
event by attribute, report attendees / location / description.

Example queries: *(fill in)*

## Current best

SFT v6 ckpt-4659 (ep 3) — **90%** on test_data. Effectively saturated.

## History

- 1.5B SFT v3: 62.5% on RL data
- 1.5B SFT v5 (ep 4): 95%
- 1.5B RL2 IR-focused: 75% (regression vs v5; small gain over RL1's 72.5%)
- 14B SFT v6 ep4: **100%** on RL data; ep3: 95%; on test_data: 90%

## Failure modes

- Long `list_events` results (>1500 chars) — accuracy drops sharply with
  result length (1.5B-era trajectory analysis).
- Missing time-range filter on `list_events` (correlates strongly with
  failure: 72.7% acc with filter vs 38.7% without).

## Next experiments

- Saturated. Any future loss here flags a regression in another change.
