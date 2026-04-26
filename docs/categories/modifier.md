# Modifier — Correction / Modify Last Action

## What it tests

User asks the assistant to modify, correct, or undo a recent action. Requires
either tracking previous turns or re-querying calendar state.

Example queries: *(fill in)*

## Current best

SFT v6 ckpt-4659 (ep 3) — **89%** on test_data. Saturated.

## History

- 1.5B SFT v3: 32.5% on RL data — among the worst
- 1.5B RL1 Modifier-focused: 32.5% → 45% (+12.5 pp), best RL gain in the project
- 1.5B RL2 IR (continued from RL1): retained at 50%
- 14B SFT v6 ep3: 85% on RL data; ep4: 82%; on test_data: 89%

## Failure modes

- Re-running a tool with the same args instead of modifying.
- Confusing "the meeting" reference resolution when there are multiple.

## Next experiments

- Saturated; not a priority. Watch for regression during RL.
