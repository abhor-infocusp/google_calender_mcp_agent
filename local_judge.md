# Local Judge

The judge documentation moved to [`docs/judge/`](docs/judge/) on 2026-05-01.

| Doc | Topic |
|---|---|
| [`docs/judge/README.md`](docs/judge/README.md) | Current state + entry point |
| [`docs/judge/plan.md`](docs/judge/plan.md) | Strategy + phases |
| [`docs/judge/baseline.md`](docs/judge/baseline.md) | Manual oracle, base-model judge agreement |
| [`docs/judge/prompt_tuning.md`](docs/judge/prompt_tuning.md) | 19 prompt variants, failure modes |
| [`docs/judge/latency.md`](docs/judge/latency.md) | vLLM optimization sweep, Gemini comparison |
| [`docs/judge/rl_integration.md`](docs/judge/rl_integration.md) | How RL calls the judge |

Top-line (updated 2026-05-02):
- 14B fp8 + router-v1 with `/no_think` (live deployment): **91.93%** on the 285-oracle. Held-out CV ~92.4%. p50 latency 11s.
- The historical 95.44% number was real on 2026-04-30 but the deployed prompt drifted (FN 6→14). Three later router runs and today's live measurement all sit at 93.3-93.7%.
- Phase 2 v2 work added re-tuned per-judge per-cat maps — `ROUTER_MAP_QWEN_V2` and `ROUTER_MAP_GEMINI_V2` in `src/calendar_agent/judge/prompts.py`, selectable via `JUDGE_ROUTER` env var. Complex stays at ~80% on both judges (real category difficulty).
- Curated v2 distillation dataset: `data/judge/v2_20260502/{train.jsonl (744), eval.jsonl (110), disagreements.jsonl (85)}`.

See `runs/judge_prompt_tune_20260430/` and `runs/judge_tune_per_judge_20260502_1628/` for run artifacts.
