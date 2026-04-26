# Vague — Under-specified, Contextual

## What it tests

Queries where the user gives an incomplete or ambiguous request and the model
must infer intent from calendar context.

Example queries: *(fill in)*

## Current best

SFT v6 ckpt-4659 (ep 3) — **80%** on test_data.

## History

- 1.5B SFT v3: 50% on RL data
- 1.5B SFT v5 (ep 4): 62.5%
- 1.5B RL3 Vague-focused: 65–69% across 5 epochs, **flat — no improvement**.
  Run deleted, no archive.
- 1.5B IR/Modifier RL runs caused steady regression here: 50% → 42% → 35%
  (catastrophic forgetting under β=0)
- 14B SFT v6 ep3: 60% on RL data; ep4: 70%; on test_data: 80%

## Failure modes

- Semantic reasoning over event content ("for fun", "with clients", "related
  to real estate") — 1.5B couldn't do this; 14B mostly can.
- Over-asking for clarification when the answer is inferable.
- Under-asking when the answer genuinely needs clarification.

## Next experiments

- Local judge with reasoning may help RL signal here (binary correctness is a
  poor proxy for "did the model handle ambiguity sensibly").
- Re-test with non-zero KL β on RL to prevent the v5-era regression.
