"""Generate a PDF deck mirroring make_results_ppt.py.

Run: /home/abhor/miniconda3/envs/agentic/bin/python scripts/utils/make_results_pdf.py
Output: runs/analysis/experiments_summary.pdf
"""
from pathlib import Path
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

OUT = Path("runs/analysis/experiments_summary.pdf")
PAGESIZE = (13.333 * inch, 7.5 * inch)
W, H = PAGESIZE

NAVY = HexColor("#0B2E4F")
LIGHT_BLUE = HexColor("#CFE2F3")
GREY = HexColor("#555555")
HIGHLIGHT = HexColor("#FFF2CC")

c = canvas.Canvas(str(OUT), pagesize=PAGESIZE)


def title_slide(title, subtitle):
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 40)
    c.drawString(0.6 * inch, H - 3.4 * inch, title)
    c.setFillColor(LIGHT_BLUE)
    c.setFont("Helvetica", 20)
    c.drawString(0.6 * inch, H - 4.1 * inch, subtitle)
    c.showPage()


def section_slide(title):
    c.setFillColor(white)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(NAVY)
    c.rect(0, H / 2 - 1.0 * inch, W, 2.0 * inch, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 36)
    c.drawString(0.6 * inch, H / 2 - 0.15 * inch, title)
    c.showPage()


def header(title):
    c.setFillColor(white)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(NAVY)
    c.rect(0, H - 0.75 * inch, W, 0.75 * inch, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(0.4 * inch, H - 0.5 * inch, title)


def draw_bullets(bullets, x=0.5, y=None, size=15):
    if y is None:
        y = H - 1.1 * inch
    line_h = size * 1.55
    c.setFillColor(NAVY)
    for b in bullets:
        text, level = (b if isinstance(b, tuple) else (b, 0))
        bullet = "• " if level == 0 else "    – "
        c.setFont("Helvetica", size if level == 0 else size - 2)
        c.setFillColor(NAVY if level == 0 else GREY)
        # naive wrap
        max_chars = 130 if level == 0 else 120
        words = text.split()
        line = bullet
        first = True
        for w in words:
            trial = (line + (" " if line and not line.endswith(("• ", "– ")) else "") + w).strip()
            if len(trial) > max_chars:
                c.drawString(x * inch, y, line)
                y -= line_h
                line = "    " + w if first else "    " + w
                first = False
            else:
                line = trial
        if line.strip():
            c.drawString(x * inch, y, line)
            y -= line_h
        y -= 2  # space between bullets
    return y


def draw_table(headers, rows, x=0.4, y_top=None, col_widths=None, highlight_rows=None, font_size=10):
    highlight_rows = highlight_rows or set()
    data = [headers] + [list(r) for r in rows]
    if col_widths is None:
        col_widths = [W / len(headers) - 0.1 * inch] * len(headers)
    t = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, 0), white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), font_size + 1),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), font_size),
        ("TEXTCOLOR",  (0, 1), (-1, -1), NAVY),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r + 1), (-1, r + 1), HIGHLIGHT))
    t.setStyle(TableStyle(style))
    w, h = t.wrapOn(c, W - 0.8 * inch, H)
    if y_top is None:
        y_top = H - 1.1 * inch
    t.drawOn(c, x * inch, y_top - h)
    return y_top - h - 0.15 * inch


# 1. Title
title_slide("Google Calendar MCP Agent",
            "All experiments & results — 2026-03 to 2026-04   |   Qwen2.5-1.5B → Qwen3-14B")

# 2. Problem
header("Problem & Approach")
draw_bullets([
    "Goal: tool-calling agent for Google Calendar — 7 tools, multi-turn, calendar-aware",
    "Stack: SFT (Unsloth + TRL, LoRA) → optional RL (GRPO via ART 0.5.17 + vLLM) / DPO",
    "Eval: Gemini-2.0-flash-as-judge compares calendar state before/after vs expected behavior",
    "7 task categories: RelTime, IR, Modifier, Schedule, Vague, Chaos, Complex Logic",
    "Datasets:",
    ("SFT: 6,947 augmented trajectories from 114 calendars (gemini-2.5-pro teacher)", 1),
    ("RL:  622 scenarios across 50 calendars (binary correct/incorrect rewards)", 1),
    ("Test (held-out, canonical): 692 queries × 49 fresh calendars — zero overlap", 1),
    "Hardware: TITAN X 12 GiB (1.5B era) → Blackwell 96 GiB MIG-partitioned 4×24 GiB",
])
c.showPage()

# 3. Pipeline
header("Pipeline Overview")
draw_bullets([
    "Data generation",
    ("114 calendars + 7-category prompt templates → gemini-2.5-pro trajectories (compact tool-result format)", 1),
    ("Augmentation: entity substitution (2x) + paraphrasing (2-5x) → 6,947 balanced trajectories", 1),
    "SFT (Qwen3-14B + 4-bit bnb + LoRA r=64, bf16, /no_think system prompt)",
    ("5 epochs, cosine LR 2e-4, max_seq_len=4096, loss-masked on assistant + tool_call tokens", 1),
    "RL — GRPO (openpipe-art 0.5.17)",
    ("rollouts_per_group=8, num_generations=4, lr=5e-6, beta=0, binary reward from Gemini judge", 1),
    ("Multi-tenant: MIG slice 0/1/2/3, auto_restart wrapper, Patch G deadlock recovery", 1),
    "DPO — pairs mined from RL rollouts (chosen=correct, rejected=incorrect) on same scenario",
    "Eval — vLLM serve (hermes parser) → batched judge across 692 queries × 7 categories",
])
c.showPage()

# 4. Section
section_slide("Phase 1 — Qwen2.5-1.5B Era  (TITAN X 12 GiB)")

# 5. SFT v3/v5
header("SFT v3 → v5 (1.5B) — overall acc on RL data, 280 queries")
y = draw_table(
    ["Run", "Best ckpt", "RL-data acc", "Notes"],
    [
        ["SFT v3 (compact)",   "ckpt-933 (ep 6)",  "42.5%", "Baseline; trained on 1,164 trajectories"],
        ["SFT v5 (augmented)", "ckpt-6152 (ep 4)", "74.6%", "Augmented to 6,947; narration-merge fix"],
    ],
    col_widths=[2.5*inch, 2.2*inch, 1.8*inch, 6.0*inch],
)
draw_bullets([
    "v5 fixes that mattered:",
    ("Phase-4 augmentation bug: 29.4% of training data was corrupted (get_current_time results swapped with random event names) — first run 18-21%; fix lifted +50pp", 1),
    ("Narration-merge in trajectory_to_messages — assistant narration merged into tool_call message", 1),
    ("Cosine-with-restarts produced even/odd LR oscillation; epoch 4,6,8 > 3,5,7,9", 1),
    "Best v5 per-category (epoch 4): IR 95%, RelTime 92.5%, Schedule 87.5%; Complex 32.5%, Vague 62.5%",
], y=y - 0.1 * inch, size=14)
c.showPage()

# 6. RL on 1.5B
header("RL Runs 1-3 on 1.5B  (GRPO, binary reward, beta=0)")
y = draw_table(
    ["Run", "Target cat.", "Steps", "Overall (RL data)", "Δ vs prior", "Verdict"],
    [
        ["RL1 Modifier",        "Modifier",  "395 (5 ep)", "42.5% → 47.9%", "+5.4 pp", "Target +12.5pp"],
        ["RL2 Information Retr","IR",        "234",         "47.9% → 47.5%", "−0.4 pp", "Flat overall"],
        ["RL3 Vague",           "Vague",     "440 (5 ep)", "Flat 66-69%",   "—",       "No improvement"],
        ["SFT-on-RL recovery",  "All",       "1 epoch",    "47.9% → 39.3%", "−8.6 pp", "Overwrote RL gains"],
    ],
    col_widths=[2.2*inch, 1.7*inch, 1.4*inch, 2.6*inch, 1.4*inch, 3.2*inch],
)
draw_bullets([
    "Failure mode (RL3 trajectory analysis, 2,464 rollouts, 806 incorrect):",
    ("79% incomplete/wrong answer — right tools, wrong interpretation", 1),
    ("16% action instead of query (created/updated when should have read)", 1),
    ("→ Comprehension ceiling, not procedure. 1.5B can't replicate teacher's semantic reasoning.", 1),
    "Other 1.5B failure modes: bimodal skip rate (50-65%), catastrophic forgetting (Vague 50→35%)",
], y=y - 0.1 * inch, size=13)
c.showPage()

# 7. Section
section_slide("Phase 2 — Switch to Qwen3-14B  (Blackwell, 4×24 GiB MIG)")

# 8. Stack
header("Stack upgrade & key config changes  (2026-04-15)")
y = draw_table(
    ["Component", "Old (1.5B / TITAN X)", "New (14B / Blackwell)"],
    [
        ["Model",            "Qwen2.5-1.5B",              "Qwen3-14B + /no_think"],
        ["Quantization",     "fp16",                       "4-bit bnb + bf16 + FA2"],
        ["LoRA rank",        "8 (RL OOM @ 64)",            "64 (SFT) / 64 (RL with sleep_mode)"],
        ["max_model_len",    "3076",                       "4096"],
        ["num_generations",  "2",                          "4"],
        ["torch / vLLM",     "2.x / 0.7.3 (sm_61 build)",  "2.10.0+cu128 / 0.19.0"],
        ["unsloth / TRL",    "older",                      "2026.4.4 / 0.24.0"],
        ["openpipe-art",     "0.5.4",                      "0.5.17 (with patches D, E, G, H, I)"],
    ],
    col_widths=[3.0*inch, 4.5*inch, 5.0*inch],
)
draw_bullets([
    "vLLM tool-call parser MUST be hermes (not qwen3_xml) — Qwen3 outputs <tool_call>...</tool_call>",
    "RL adapters must be served via --enable-lora; merging RL LoRA → fp16 produces 0% acc",
], y=y - 0.1 * inch, size=13)
c.showPage()

# 9. SFT v6 full table
header("SFT v6 (Qwen3-14B) — full per-checkpoint eval on held-out test_data (692 queries)")
y = draw_table(
    ["Epoch", "Ckpt", "Overall", "Complex", "Chaos", "IR", "Modif", "RelTime", "Schedule", "Vague"],
    [
        ["1", "1553", "72.4%", "44%", "67%", "88%", "70%", "88%", "78%", "70%"],
        ["2", "3106", "78.3%", "54%", "77%", "89%", "87%", "93%", "68%", "80%"],
        ["3", "4659", "80.1%", "55%", "77%", "92%", "85%", "93%", "77%", "82%"],
        ["4", "6212", "78.9%", "59%", "79%", "89%", "89%", "90%", "67%", "79%"],
        ["5", "7765", "79.2%", "52%", "83%", "90%", "87%", "92%", "70%", "80%"],
    ],
    highlight_rows={2},
    col_widths=[0.7*inch]*2 + [1.0*inch] + [1.1*inch]*7,
)
draw_bullets([
    "BEST: ckpt-4659 (epoch 3) at 80.1% on held-out test_data — current production ckpt",
    "Cross-set: ckpt-6212 wins on RL-data benchmark (82.5%); ckpt-4659 wins on test_data",
    "Eval loss is misleading for ckpt selection — track full eval, not loss minimum",
    "Versus 1.5B SFT v5 best (74.6% RL-data): +7.9 pp overall after switch to 14B",
], y=y - 0.1 * inch, size=13)
c.showPage()

# 10. SFT v6 vs Instruct
header("SFT v6 vs Qwen3-14B-Instruct baseline — per-category gains")
y = draw_table(
    ["Category", "Instruct base", "SFT v6 best", "Δ", "Comment"],
    [
        ["RelTime",  "95%", "93%",  "−2 pp",  "Saturated by base model"],
        ["IR",       "60%", "92%",  "+32 pp", "Largest absolute lift"],
        ["Modifier", "70%", "85%",  "+14 pp", "Strong"],
        ["Chaos",    "30%", "77%",  "+47 pp", "Calendar-specific knowledge"],
        ["Vague",    "70%", "82%",  "+11 pp", "Solid"],
        ["Schedule", "72%", "77%",  "+5 pp",  "Marginal"],
        ["Complex",  "41%", "55%",  "+14 pp", "Weakest absolute (multi-step reasoning)"],
        ["Overall",  "63.0%", "80.1%", "+17.1 pp", "—"],
    ],
    highlight_rows={7},
    col_widths=[2.0*inch, 2.0*inch, 2.0*inch, 1.5*inch, 5.0*inch],
)
draw_bullets([
    "SFT lifts the hardest, knowledge-bound categories most (Chaos +47, IR +32)",
    "Categories already near ceiling (RelTime) cannot be moved further by domain SFT",
], y=y - 0.1 * inch, size=13)
c.showPage()

# 11. DPO
header("DPO experiments (paused 2026-04-25)")
y = draw_table(
    ["Model", "Best epoch", "Test-set acc", "vs starting point", "Verdict"],
    [
        ["DPO-from-SFT v6",   "ep 1 / ep 2", "79.3%", "−0.8 pp vs SFT 80.1%",   "Within noise; no convincing win"],
        ["DPO-from-Instruct", "ep 2",        "61.3%", "−1.7 pp vs Instruct 63%", "Actively regresses (likelihood displacement)"],
    ],
    col_widths=[3.0*inch, 1.6*inch, 1.5*inch, 2.7*inch, 3.7*inch],
)
draw_bullets([
    "Pair source: 1,913 mined pairs from 5,506 ART rollouts on RL-data scenarios",
    "Why DPO-from-Instruct failed (predicted ex-ante):",
    ("π_ref assigns near-zero prob to <tool_call> XML for un-SFT'd Instruct → likelihood displacement (Pal et al. 2024). Chaos 30→26%, Vague 70→58% confirms.", 1),
    "Why DPO-from-SFT washed out:",
    ("Pair saturation — chosen trajectories already near SFT v6 ceiling", 1),
    ("Per-category: Schedule +9 pp gain offset by Complex −7 pp regression; net ~ 0", 1),
    "Decision: pause DPO; explore RFT / iterative-DPO / Complex-targeted SFT instead",
], y=y - 0.1 * inch, size=13)
c.showPage()

# 12. RL on 14B
header("RL on Qwen3-14B  (GRPO, ART 0.5.17)")
draw_bullets([
    "Setup: all 7 categories joint, 622 scenarios, rollouts_per_group=8, num_gens=4, lr=5e-6, beta=0",
    "Plan: 622 × 20 epochs = 12,440 steps. Currently paused at step 9220.",
    "Mid-run snapshot (step 2325, 2026-04-17): mean reward 0.70, recent-500 ≈ 0.76",
    ("Per-category recent: RelTime 93%, Schedule 75%, IR 74%, Vague 71%, Modifier 68%, Chaos 61%, Complex 47%", 1),
    "Incidents:",
    ("2026-04-17 — 57h hang inside model.train() at step 2325; root cause unresolved, restarted 2026-04-20", 1),
    ("2026-04-25 — reward cliff; led to multi-tenant hardening + Patch K v2 (milestone preservation)", 1),
    "Currently paused: real-RL adapter at .art/calendar-agent/.../checkpoints/9220/",
    "Active follow-ups: rl_adaptive variant, rl_grpo from base + from sft4659 (parallel 2026-04-26 runs)",
], size=14)
c.showPage()

# 13. ART patches
header("ART 0.5.17 runtime patches  (src/calendar_agent/art_patches.py)")
y = draw_table(
    ["Patch", "Target", "Purpose"],
    [
        ["D", "_calculate_logprobs",          "entropy detach from autograd; remove chunk_size assertion"],
        ["E", "_prepare_backend_for_training","guard done_callback against error/cancel (prevents OOM on health-check timeout)"],
        ["G", "inputs_queue.get",             "300s timeout + exit(42) on asyncio queue deadlock; auto_restart relaunches"],
        ["H", "tokenize_trajectory",          "REPAIR Qwen3 empty <think></think>-only finals → patch content=' ', retain for negative gradient"],
        ["I", "(asyncio nest)",               "nest_asyncio shim — replaces older A/B/C patch trio"],
        ["F", "(optional)",                   "LoRA injection helper for resuming RL from arbitrary adapter"],
    ],
    col_widths=[1.0*inch, 3.5*inch, 8.0*inch],
)
draw_bullets([
    "Patches A, B, C removed (fixed upstream / replaced by nest_asyncio approach)",
    "Patch H upgrade (2026-04-24): drop-only → repair-then-train. Was losing ~7% of rollouts (~1 per 15 steps).",
    "Import calendar_agent.art_patches BEFORE import art in rl_train.py",
], y=y - 0.1 * inch, size=13)
c.showPage()

# 14. Headline
header("Headline — Held-out test_data benchmark  (49 cals × 692 queries)")
y = draw_table(
    ["Model", "Best", "Accuracy", "Notes"],
    [
        ["Qwen3-14B-Instruct (no calendar training)", "—",       "63.0%",                "Format-strong, knowledge-weak"],
        ["DPO-from-Instruct",                          "ep 2",   "61.3% (−1.7 pp)",      "Likelihood displacement"],
        ["DPO-from-SFT",                               "ep 1/2", "79.3% (−0.8 pp)",      "Within noise of SFT"],
        ["SFT v6 ckpt-4659 (epoch 3)",                 "—",      "80.1%   <-- BEST",     "Production checkpoint"],
        ["SFT v6 ckpt-6212 (epoch 4)",                 "—",      "78.9%",                "Best on RL-data set (82.5%)"],
    ],
    highlight_rows={3},
    col_widths=[4.5*inch, 1.5*inch, 2.5*inch, 4.0*inch],
)
draw_bullets([
    "test_data created 2026-04-24 because rl_data was contaminated for DPO eval (74% overlap with mined pairs)",
    "Canonical benchmark going forward; old rl_data numbers retained for SFT-vs-SFT historical comparison",
    "SE ≈ 1.5 pp on 692 queries — differences below ~3 pp are within noise",
], y=y - 0.1 * inch, size=13)
c.showPage()

# 15. Per-category status
header("Per-category status  (SFT v6 ckpt-4659 on test_data)")
y = draw_table(
    ["Category", "Acc", "Status", "Notes"],
    [
        ["RelTime",  "95%", "saturated", "Improved by base model alone"],
        ["IR",       "92%", "saturated", "Largest SFT lift over baseline"],
        ["Modifier", "85%", "saturated", "Strong"],
        ["Chaos",    "77%", "strong",    "Edge-case calendars; 47-pp SFT lift"],
        ["Vague",    "82%", "room",      "Contextual / underspecified queries"],
        ["Schedule", "77%", "room",      "Event creation format precision"],
        ["Complex",  "55%", "weakest",   "Multi-step reasoning + tool chaining"],
    ],
    col_widths=[2.0*inch, 1.2*inch, 1.8*inch, 7.5*inch],
)
draw_bullets([
    "Top improvement target: Complex Logic (multi-step reasoning + conflict resolution)",
    "Secondary: Schedule + Vague (room above 75-80% with focused data)",
], y=y - 0.1 * inch, size=14)
c.showPage()

# 16. Open threads
header("Open threads & next experiments")
draw_bullets([
    "Local judge (Qwen3-7B SFT) — replaces ~99k Gemini API calls per RL run; baseline eval in progress",
    "ART asyncio deadlock — workaround deployed (Patch G/I); upstream issue not yet filed",
    "RL beyond GRPO+binary rewards:",
    ("RFT / Expert iteration — filter correct rollouts → SFT (avoids DPO ratio failure modes)", 1),
    ("Iterative DPO — regenerate pairs from current SFT v6 ckpt-4659 (on-policy)", 1),
    ("Dr. GRPO / non-zero beta — stop catastrophic forgetting; better advantage for binary reward", 1),
    ("Complex-category focused SFT — directly attack the 55% bottleneck", 1),
    "Multi-tenant hardening (2026-04-26): centralized auto_restart, slice_map, telemetry, /rl-status & /rl-stop skills",
], size=15)
c.showPage()

# 17. Summary
header("Summary")
draw_bullets([
    "End-to-end pipeline (data gen → SFT → RL/DPO → judge eval) running on 4× MIG Blackwell partitions",
    "1.5B era hit a comprehension ceiling at 47.9% (RL data); 14B switch lifted SFT to 80.1% on held-out",
    "Current best model: SFT v6 Qwen3-14B ckpt-4659 (epoch 3) — 80.1% on 692-query held-out benchmark",
    "DPO experiments paused: neither variant convincingly beats its starting point",
    "RL on 14B paused at step 9220 after reward cliff; multi-tenant safeguards now in place for resume",
    "Next priorities: Local judge, RFT / iterative DPO, Complex-category targeted SFT",
], size=18)
c.showPage()

c.save()
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KiB)")
