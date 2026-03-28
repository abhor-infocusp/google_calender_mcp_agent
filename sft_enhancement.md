# SFT Data Enhancement Plan

> **Created:** 2026-03-28
> **Status:** Phase 1 complete (2.5-pro + tuned prompt = 72.9%), Phase 2 next

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

### Decision

**gemini-2.5-pro with v4 prompt (`prompts/best_25pro.txt`) is the teacher model for trajectory generation.**
- 72.9% solve rate — best across all model+prompt combinations tested
- Dominates on Vague & Contextual (100%), Human Chaos (50%), Info Retrieval (80%)
- Competitive on Schedule (80% vs 90%) and Modifier (70% vs 80%)
- Use the v4 prompt as system_instruction for trajectory generation (NOT the default SYSTEM_PROMPT)

### Artifacts

- Tuning script: `scripts/data_generation/tune_prompt.py`
- Comparison script: `scripts/data_generation/compare_teachers.py`
- Winning prompt: `prompts/best_25pro.txt` (= `prompts/v4_chaos_refined.txt`)
- All prompts: `prompts/baseline.txt`, `v1_structured.txt`, `v2_chaos.txt`, `v3_concise.txt`, `v4_chaos_refined.txt`, `v5_refined.txt`
- Results log: `sft_data/prompt_tuning_log.jsonl`
- Original comparison: `sft_data/teacher_comparison.json`

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

**Problem discovered:** 2.5-pro with v4 prompt calls `list_events` unfiltered, returning all 20-45 events (~3,700-6,700 chars). This causes **71% of trajectories to exceed 3076 tokens** (avg 2,658 vs original 1,389).

**Solution — post-process trajectories after generation:**
1. Generate with the v4/best prompt (unfiltered `list_events` → 73% accuracy)
2. After saving, replace unfiltered `list_events` results with filtered versions containing only events on `addressed_days`
3. Filtering a single day: ~600 chars vs ~3,900 chars (6x reduction)
4. Also add "Keep your final response to 1-2 sentences" to the prompt to control response length
5. This preserves the model's correct decisions while making trajectories fit the token budget

**Alternative tested and rejected:** Forcing filtered `list_events` in the prompt (v6_compact) — drops accuracy from 72.9% to 62.9% because the model can't see the full calendar for keyword searches and conflict checks.

### Steps

1. **Run trajectory generation** (`generate_trajectories.py`)
   - Model: **gemini-2.5-pro** with custom prompt (`prompts/best_25pro.txt`)
   - 1,400 queries, rate limited at 0.5s between queries
   - Expected: ~1,020 correct trajectories (at ~73% solve rate)
   - Output: `sft_data/trajectories/`

2. **Post-process: filter list_events results**
   - For each trajectory, find `list_events` tool_call steps with no `time_min`/`time_max` args
   - Replace the result with a re-executed filtered call using the query's `addressed_days`
   - Verify token count fits within 3076 after filtering
   - If still over, truncate the assistant's final response

3. **Analyze solve rates** per category
   - For categories below 40%: retry failed queries with temperature variation
   - For Human Chaos specifically: may still have ~50% solve rate — retry failures with varied temperature

**Files:** `scripts/data_generation/generate_trajectories.py` (needs update to accept custom prompt file + post-processing)

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
