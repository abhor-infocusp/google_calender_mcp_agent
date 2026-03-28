# SFT Data Enhancement Plan

> **Created:** 2026-03-28
> **Status:** Phase 1 complete (2.5-pro + v11 prompt + 3-attempt retry = 84.3%), Phase 2 next

---

## Motivation

Current SFT baseline generalizes poorly: 81.4% on SFT data vs 42.5% on RL data (unseen calendars). Root causes:

| Gap | Detail |
|---|---|
| Calendar diversity | Only 20 SFT calendars vs 48 RL calendars |
| Human Chaos data | Only 5 original trajectories (3.1%), augmented to 65 (6.3%) |
| Shallow trajectories | 96% use only 2-3 tool calls; complex scenarios need 4-5 |
| Teacher ceiling | Gemini 2.0 Flash solves ~66%; **2.5-pro with tuned prompt solves ~73%** |

**Goal:** Scale to ~100 calendars, improve trajectory quality, retrain SFT, and validate on RL data per epoch.

---

## Phase 1: Teacher Model Experiment — COMPLETE

### Phase 1a: Initial Comparison (default SYSTEM_PROMPT)

Compared gemini-2.0-flash-001 vs gemini-2.5-pro on 70 queries (calendars 0-5, 10 per category) using the current SYSTEM_PROMPT. Eval judge: gemini-2.0-flash-001.

| Category | 2.0-flash | 2.5-pro |
|---|---|---|
| Schedule a Single Event | 9/10 (90%) | 6/10 (60%) |
| Vague & Contextual | 8/10 (80%) | 8/10 (80%) |
| Modifier & Correction | 8/10 (80%) | 7/10 (70%) |
| Information Retrieval | 7/10 (70%) | 7/10 (70%) |
| Complex Logic & Conflict | 6/10 (60%) | 6/10 (60%) |
| Human Chaos | 0/10 (0%) | 2/10 (20%) |
| Relative Time References | 8/10 (80%) | 7/10 (70%) |
| **Overall** | **46/70 (65.7%)** | **43/70 (61.4%)** |

### Phase 1b: Prompt Tuning for 2.5-pro

Iterated through 5 prompt variants (baseline, v1_structured, v2_chaos, v3_concise, v4_chaos_refined, v5_refined). Tested on category subsets first, then full 70-query validation.

**Key prompt improvements in winning variant (v4_chaos_refined):**
- Explicit workflow steps (get_current_time → list_events → reason → act → confirm)
- Detailed datetime rules (relative dates, default 1-hour duration, day name resolution)
- Event creation/modification guidance (loose name matching, duration preservation)
- Fragment interpretation rules for Human Chaos (search vs clarify vs action patterns)
- Conciseness instruction ("Be concise. Call tools, read results, act.")

**Final comparison (all 70 queries):**

| Category | Flash (orig) | Flash (v4) | Pro (orig) | **Pro (v4)** |
|---|---|---|---|---|
| Schedule a Single Event | 9/10 | 9/10 | 6/10 | **8/10** |
| Vague & Contextual | 8/10 | 9/10 | 8/10 | **10/10** |
| Modifier & Correction | 8/10 | 8/10 | 7/10 | **7/10** |
| Information Retrieval | 7/10 | 7/10 | 7/10 | **8/10** |
| Complex Logic & Conflict | 6/10 | 4/10 | 6/10 | **6/10** |
| Human Chaos | 0/10 | 3/10 | 2/10 | **5/10** |
| Relative Time References | 8/10 | 7/10 | 7/10 | **7/10** |
| **Overall** | **46/70 (65.7%)** | **47/70 (67.1%)** | **43/70 (61.4%)** | **51/70 (72.9%)** |

**Observations:**
- v4 prompt + 2.5-pro: **51/70 (72.9%)** — +11.5pp over pro baseline, +7.2pp over flash
- Vague & Contextual hit **100%** (10/10) — 2.5-pro's reasoning strength fully unlocked
- Human Chaos: 2/10 → **5/10** — fragment handling rules work well
- Schedule: 6/10 → **8/10** — datetime rules closed most of the gap
- v4 prompt on flash: only +1/70 (67.1%), and Complex Logic regressed (6→4). Verbose prompt hurts flash.
- v5 (even longer prompt) regressed to 44/70 — prompt length has diminishing/negative returns
- Remaining failures are model-level limits: cal 4 absolute dates (July 16/19), "yesterday" queries, multi-step Complex Logic

### Phase 1c: Gemini 2.5-Flash Comparison

Tested gemini-2.5-flash with baseline, v1 (structured), and v4 (chaos) prompts. Gemini 3 Flash (`gemini-3.0-flash`) not available on this Vertex AI project.

| Category | 2.0-flash (orig) | 2.5-flash (base) | 2.5-flash (v1) | 2.5-flash (v4) | **2.5-pro (v4)** |
|---|---|---|---|---|---|
| Schedule | 9/10 | 8/10 | 8/10 | 7/10 | **8/10** |
| Vague & Contextual | 8/10 | 6/10 | 9/10 | 7/10 | **10/10** |
| Modifier | 8/10 | 8/10 | 8/10 | 7/10 | 7/10 |
| Info Retrieval | 7/10 | 6/10 | 7/10 | 7/10 | **8/10** |
| Complex Logic | 6/10 | 4/10 | 5/10 | 6/10 | **6/10** |
| Human Chaos | 0/10 | 1/10 | 1/10 | 3/10 | **5/10** |
| Relative Time | 8/10 | 7/10 | 7/10 | 7/10 | 7/10 |
| **Overall** | **46/70 (65.7%)** | **40/70 (57.1%)** | **45/70 (64.3%)** | **44/70 (62.9%)** | **51/70 (72.9%)** |

**Findings:**
- 2.5-flash baseline (40/70) is worse than 2.0-flash (46/70)
- v1 (shorter, structured) is best for 2.5-flash at 45/70 — roughly ties 2.0-flash
- v4 (longer, chaos rules) slightly hurts 2.5-flash (44/70 vs 45/70 with v1)
- Flash models prefer shorter prompts; pro benefits from longer structured guidance
- 2.5-flash not competitive as teacher model

### Phase 1d: Compact Tool Returns + v11 Prompt + Retry

Refactored tool returns from JSON dicts to human-readable strings. list_events returns only id+summary lines; get_event returns full detail block with RSVP. This saves tokens and forces the model to call get_event for details.

Prompt v11 ("plan once, then execute") replaced v4 ("step-by-step workflow"). Key changes:
- Removed verbose workflow steps; simplified to "write out your plan, then execute"
- Added "search YOUR calendar" instruction (fixes "can't access other calendars" failures)
- Added tool return format examples so model knows what to expect
- Removed fragment handling rules (over-engineering for Human Chaos)

Added 3-attempt retry: each query gets up to 3 tries, counted as Correct if any attempt passes. This absorbs Gemini's stochasticity and judge inconsistency.

**v11 + retry results (70 queries):**

| Category | v4 (single) | v11 + retry |
|---|---|---|
| Schedule a Single Event | 8/10 | **10/10 (100%)** |
| Vague & Contextual | 10/10 | **10/10 (100%)** |
| Modifier & Correction | 7/10 | **10/10 (100%)** |
| Information Retrieval | 8/10 | **9/10 (90%)** |
| Complex Logic & Conflict | 6/10 | **8/10 (80%)** |
| Relative Time References | 7/10 | **7/10 (70%)** |
| Human Chaos | 5/10 | **5/10 (50%)** |
| **Overall** | **51/70 (72.9%)** | **59/70 (84.3%)** |

### Decision

**gemini-2.5-pro with v11 prompt (`prompts/v11_reason_act.txt`) + 3-attempt retry is the teacher config for trajectory generation.**
- 84.3% solve rate — best across all configurations tested
- 3 categories at 100% (Schedule, Vague, Modifier)
- Remaining failures concentrated in Human Chaos (50%) and Relative Time (70%)
- Retry absorbs ~30% of judge errors and model stochasticity

### Artifacts

- Tuning script: `scripts/data_generation/tune_prompt.py`
- Trajectory generation: `scripts/data_generation/generate_trajectories.py`
- Winning prompt: `prompts/v11_reason_act.txt`
- Legacy prompt: `prompts/best_25pro.txt` (v4, 72.9%)
- Results log: `sft_data/prompt_tuning_log.jsonl`
- Best run: `sft_data/tuning_runs/v11_retry3/`

---

## Phase 2: Generate 100 New Calendars

**Goal:** Create 100 fresh calendars with personas, text calendars, JSON calendars, and queries.

### Steps

1. **Expand profession list** in `generate_data.py`
   - Current: 20 professions
   - Target: 50+ (add: chef, lawyer, artist, musician, pilot, firefighter, pharmacist, veterinarian, architect, journalist, librarian, plumber, electrician, therapist, dentist, professor, photographer, farmer, mechanic, politician, etc.)
   - Each profession → 2 personas → 100 calendars

2. **Run generation pipeline** (`generate_data.py` with `DATA_SIZE=100`)
   - Personas → text calendars → JSON calendars → queries
   - Expected: 100 calendars × 14 queries = 1,400 queries
   - Model: gemini-2.0-flash-001 (calendar/query generation, not trajectories)

3. **Validate outputs**
   - All 100 calendars have valid JSON with 7 days
   - All queries have proper category/complexity fields

**Files:** `scripts/data_generation/generate_data.py`

---

## Phase 3: Generate Trajectories

**Goal:** Run the winning teacher model on all 1,400 queries.

### Token Budget Constraint

Qwen training uses `MAX_SEQ_LENGTH=3076`. Trajectories exceeding this are dropped.

Tool returns are now human-readable strings (not JSON dicts), which significantly reduces token count:
- `list_events`: one line per event (`id: evt_x | Summary — Day HH:MM-HH:MM`)
- `get_event`: compact multi-line block (ID, Time, Description, Attendees, RSVP)
- Other tools return similar compact strings

**Post-processing may still be needed** for calendars with 30+ events where unfiltered `list_events` produces long output. Solution: replace unfiltered `list_events` results with filtered versions containing only events on `addressed_days`.

### Steps

1. **Run trajectory generation** (`generate_trajectories.py`)
   - Model: **gemini-2.5-pro** with v11 prompt (`prompts/v11_reason_act.txt`)
   - 3-attempt retry per query (correct on any attempt = saved)
   - 1,400 queries, rate limited at 0.5s between queries
   - Expected: ~1,180 correct trajectories (at ~84% solve rate)
   - Output: `sft_data/trajectories/`

2. **Post-process: filter list_events results (if needed)**
   - For trajectories exceeding 3076 tokens, replace unfiltered `list_events` results with filtered versions
   - Tool returns are already compact strings, so this may not be needed for most trajectories

3. **Analyze solve rates** per category
   - Human Chaos (~50%) and Relative Time (~70%) are expected weak spots

**Files:** `scripts/data_generation/generate_trajectories.py`

---

## Phase 4: Augmentation

**Goal:** Expand trajectories via entity substitution + paraphrasing.

### Steps

1. **Run augmentation** (`augment_trajectories.py`)
   - Model: gemini-2.0-flash-001 (proven and stable for paraphrasing)
   - Entity substitution: 2 variants per trajectory
   - Paraphrasing: category-weighted (Human Chaos: 10x, Complex Logic: 5x, others: 3x)
   - Expected: ~1,020 originals × ~5x = ~5,000 augmented trajectories

2. **Validate**
   - Token lengths fit within 3076 context
   - Category distribution is balanced
   - Final dataset stats reported

**Output:** `sft_data/trajectories_augmented/`

**Files:** `scripts/data_generation/augment_trajectories.py`

---

## Phase 5: SFT Training with RL Validation

**Goal:** Train on the larger dataset. Pick best checkpoint by RL data accuracy, not SFT eval loss.

### Steps

1. **Train SFT** (`sft_train_100ep.py`)
   - Model: Qwen/Qwen2.5-1.5B-Instruct
   - LoRA rank 64, same config as original
   - NUM_EPOCHS = 10
   - Expected: ~5,000 trajectories → ~1,250 steps/epoch
   - Save checkpoint every epoch
   - Output: `sft_output/`

2. **Evaluate all checkpoints on RL data**
   - Use `eval_all_checkpoints.py` (modified for new checkpoint steps)
   - For each checkpoint: merge → serve vLLM → run 280-query eval → record per-category accuracy
   - Pick best checkpoint based on RL data accuracy

**Files:** `scripts/training/sft_train_100ep.py`, `scripts/eval/eval_all_checkpoints.py`

---

## Current SFT Data Snapshot (for reference)

| Metric | Value |
|---|---|
| Original trajectories | 161 (from 252 queries, 63.9% solve rate) |
| Augmented trajectories | 1,039 (6.5x expansion) |
| Calendars | 20 |
| Categories | 7 (unbalanced: Human Chaos only 6.3%) |
| Avg tool calls/trajectory | 2.28 |
| Token budget | 3076 max |

### Category Distribution (augmented)

| Category | Count | % |
|---|---|---|
| Information Retrieval | 174 | 16.7% |
| Schedule a Single Event | 162 | 15.6% |
| Modifier & Correction | 162 | 15.6% |
| Relative Time References | 162 | 15.6% |
| Vague & Contextual | 162 | 15.6% |
| Complex Logic & Conflict | 152 | 14.6% |
| Human Chaos | 65 | 6.3% |
