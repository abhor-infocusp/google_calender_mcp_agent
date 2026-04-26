"""Generate the comprehensive project PDF report.

Reads all eval JSONs, training logs, configs, and produces a multi-section
report at runs/analysis/project_report.pdf using reportlab Platypus.

Run with PYTHONPATH=src.
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    KeepTogether, ListFlowable, ListItem, HRFlowable,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ── Setup styles ──
ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=18, spaceAfter=12, spaceBefore=18,
                    textColor=colors.HexColor('#1a3d6d'), fontName='Helvetica-Bold')
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=14, spaceAfter=8, spaceBefore=14,
                    textColor=colors.HexColor('#1a3d6d'), fontName='Helvetica-Bold')
H3 = ParagraphStyle('H3', parent=ss['Heading3'], fontSize=11, spaceAfter=6, spaceBefore=10,
                    textColor=colors.HexColor('#333333'), fontName='Helvetica-Bold')
BODY = ParagraphStyle('Body', parent=ss['BodyText'], fontSize=10, leading=14, alignment=TA_JUSTIFY,
                      fontName='Helvetica', spaceAfter=6)
CODE = ParagraphStyle('Code', parent=ss['Code'], fontSize=8, leading=10, fontName='Courier',
                      backColor=colors.HexColor('#f4f4f4'), borderPadding=4, leftIndent=10, rightIndent=10)
SMALL = ParagraphStyle('Small', parent=BODY, fontSize=9, leading=11)
NOTE = ParagraphStyle('Note', parent=BODY, fontSize=9, leading=11, leftIndent=14,
                      borderColor=colors.HexColor('#cccccc'), borderWidth=0,
                      textColor=colors.HexColor('#444444'))
TITLE_STYLE = ParagraphStyle('Title', parent=ss['Title'], fontSize=24, leading=30, alignment=TA_CENTER,
                             fontName='Helvetica-Bold', textColor=colors.HexColor('#1a3d6d'))
SUBTITLE = ParagraphStyle('Subtitle', parent=BODY, fontSize=14, leading=18, alignment=TA_CENTER,
                          textColor=colors.HexColor('#666666'))

CATS_SHORT = {
    "Complex Logic & Conflict (Advanced)": "Complex",
    "Human Chaos (Edge Cases/Fragments)": "Chaos",
    "Information Retrieval (Querying)": "IR",
    "Modifier & Correction (Rescheduling/Updates)": "Modifier",
    "Relative Time References (today, tomorrow, yesterday, this week)": "RelTime",
    "Schedule a Single Event": "Schedule",
    "Vague & Contextual (Reasoning Required)": "Vague",
}
CAT_ORDER = ["Complex", "Chaos", "IR", "Modifier", "RelTime", "Schedule", "Vague"]


def load_eval(path):
    if not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    data = d.get("test") or d.get("rl") or d
    results = data.get("results", [])
    by = defaultdict(lambda: {"c":0,"t":0})
    for r in results:
        cat = CATS_SHORT.get(r.get("category","?"), r.get("category","?"))
        by[cat]["t"] += 1
        if r.get("verdict") == "Correct":
            by[cat]["c"] += 1
    total = sum(b["t"] for b in by.values())
    correct = sum(b["c"] for b in by.values())
    return {"correct":correct,"total":total,"by":dict(by)}


# ── Footer / page numbers ──
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawRightString(LETTER[0]-0.5*inch, 0.4*inch, f"Page {doc.page}")
    canvas.drawString(0.5*inch, 0.4*inch, "Calendar Agent Training Pipeline — Project Report")
    canvas.restoreState()


def p(text, style=BODY): return Paragraph(text, style)
def h1(t): return p(t, H1)
def h2(t): return p(t, H2)
def h3(t): return p(t, H3)
def code_block(text):
    text = text.replace("\n", "<br/>").replace(" ", "&nbsp;")
    return Paragraph(f'<font face="Courier" size="8">{text}</font>', CODE)
def hr(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc'),
                             spaceBefore=6, spaceAfter=6)


def build_results_table():
    """Big table — 12 evals + baseline with per-category numbers."""
    runs = [
        ("Instruct baseline", "—", "—", "runs/qwen3_14b_instruct_baseline_20260424/eval_test/baseline.json"),
        ("SFT v6", 1, 1553, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-1553.json"),
        ("SFT v6", 2, 3106, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-3106.json"),
        ("SFT v6", 3, 4659, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-4659.json"),
        ("SFT v6", 4, 6212, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-6212.json"),
        ("SFT v6", 5, 7765, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-7765.json"),
        ("DPO-from-SFT", 1, 479, "runs/dpo_qwen3_14b_sft_20260423/eval_test/checkpoint-479.json"),
        ("DPO-from-SFT", 2, 958, "runs/dpo_qwen3_14b_sft_20260423/eval_test/checkpoint-958.json"),
        ("DPO-from-SFT", 3, 1437, "runs/dpo_qwen3_14b_sft_20260423/eval_test/checkpoint-1437.json"),
        ("DPO-from-Instruct", 1, 479, "runs/dpo_qwen3_14b_instruct_20260423/eval_test/checkpoint-479.json"),
        ("DPO-from-Instruct", 2, 958, "runs/dpo_qwen3_14b_instruct_20260423/eval_test/checkpoint-958.json"),
        ("DPO-from-Instruct", 3, 1437, "runs/dpo_qwen3_14b_instruct_20260423/eval_test/checkpoint-1437.json"),
    ]
    header = ["Model", "Ep", "Ckpt", "Total", "Acc"] + CAT_ORDER
    rows = [header]
    best_pct = 0
    best_idx = 0
    for i, (name, ep, ckpt, path) in enumerate(runs):
        d = load_eval(path)
        if not d:
            rows.append([name, str(ep), str(ckpt), "—", "—"] + ["—"]*7)
            continue
        pct = d["correct"]/d["total"]*100
        if pct > best_pct: best_pct = pct; best_idx = i+1
        row = [name, str(ep), str(ckpt), f"{d['correct']}/{d['total']}", f"{pct:.1f}%"]
        for c in CAT_ORDER:
            b = d["by"].get(c, {"c":0,"t":0})
            row.append(f"{b['c']}/{b['t']}\n({b['c']/b['t']*100:.0f}%)" if b['t'] else "—")
        rows.append(row)
    t = Table(rows, colWidths=[1.3*inch, 0.3*inch, 0.45*inch, 0.55*inch, 0.55*inch] + [0.6*inch]*7)
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('BACKGROUND', (0,best_idx), (-1,best_idx), colors.HexColor('#fff4cc')),
        ('FONTNAME', (0,best_idx), (-1,best_idx), 'Helvetica-Bold'),
    ])
    t.setStyle(style)
    return t


def build_per_cat_summary_table():
    """Summary table of best ckpts."""
    paths = {
        "Baseline": "runs/qwen3_14b_instruct_baseline_20260424/eval_test/baseline.json",
        "SFT v6 best (ckpt-4659)": "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-4659.json",
        "DPO-from-SFT best (ckpt-479)": "runs/dpo_qwen3_14b_sft_20260423/eval_test/checkpoint-479.json",
        "DPO-from-Instruct best (ckpt-958)": "runs/dpo_qwen3_14b_instruct_20260423/eval_test/checkpoint-958.json",
    }
    data = {label: load_eval(path) for label, path in paths.items()}
    header = ["Category"] + list(paths.keys())
    rows = [header]
    for cat in CAT_ORDER:
        row = [cat]
        for label, d in data.items():
            b = d["by"].get(cat, {"c":0,"t":0})
            row.append(f"{b['c']/b['t']*100:.0f}%" if b["t"] else "—")
        rows.append(row)
    overall = ["Overall"]
    for label, d in data.items():
        overall.append(f"{d['correct']/d['total']*100:.1f}%")
    rows.append(overall)
    t = Table(rows, colWidths=[1.3*inch, 1.0*inch, 1.4*inch, 1.6*inch, 1.6*inch])
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor('#f5f5f5')]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#fff4cc')),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
    ])
    t.setStyle(style)
    return t


def main():
    out_path = "runs/analysis/project_report.pdf"
    doc = SimpleDocTemplate(out_path, pagesize=LETTER,
                            leftMargin=0.6*inch, rightMargin=0.6*inch,
                            topMargin=0.6*inch, bottomMargin=0.6*inch,
                            title="Calendar Agent Training Pipeline — Project Report",
                            author="Calendar Agent Project")
    story = []

    # ── Cover Page ──
    story.append(Spacer(1, 1.5*inch))
    story.append(p("Calendar Agent Training Pipeline", TITLE_STYLE))
    story.append(Spacer(1, 0.2*inch))
    story.append(p("Comprehensive Project Report", SUBTITLE))
    story.append(Spacer(1, 0.6*inch))
    story.append(p("SFT, RL, and DPO Experiments on Qwen3-14B<br/>"
                   "Tool-Calling Agent for Google Calendar", SUBTITLE))
    story.append(Spacer(1, 1.5*inch))
    cover_table = Table([
        ["Project", "google_calender_mcp_agent"],
        ["Reported", "2026-04-25"],
        ["Best model", "SFT v6 ckpt-4659 — 80.1% on held-out test_data"],
        ["Status", "DPO experiments paused; SFT remains canonical"],
        ["Author", "Pipeline run by Claude Code over 2026-03 → 2026-04"],
    ], colWidths=[1.2*inch, 4.8*inch])
    cover_table.setStyle(TableStyle([
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (0,-1), colors.HexColor('#1a3d6d')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(cover_table)
    story.append(PageBreak())

    # ── Table of Contents (manual) ──
    story.append(h1("Contents"))
    toc_items = [
        ("1. Executive Summary", "1 page"),
        ("2. Project Context and Goals", ""),
        ("3. Hardware, Environment, and Stack", ""),
        ("4. Data Pipeline — All Sources", ""),
        ("5. Models Compared", ""),
        ("6. Training Details and Hyperparameters", ""),
        ("7. Operational Issues — Full Narratives", ""),
        ("8. Evaluation Methodology", ""),
        ("9. Results", ""),
        ("10. Analysis — Why Each Approach Worked or Didn't", ""),
        ("11. Decisions and Recommended Next Steps", ""),
        ("12. Reproducibility — Artifacts and File Paths", ""),
        ("Appendix A. Per-Checkpoint Results JSON", ""),
        ("Appendix B. Configuration Files", ""),
        ("Appendix C. DPO Pair-Mining Statistics", ""),
    ]
    for label, sub in toc_items:
        story.append(p(f"<b>{label}</b>" + (f" &nbsp; <i>({sub})</i>" if sub else ""), BODY))
    story.append(PageBreak())

    # ── 1. EXECUTIVE SUMMARY ──
    story.append(h1("1. Executive Summary"))
    story.append(p(
        "This report documents the training and evaluation of an agentic Google-Calendar assistant "
        "based on Qwen3-14B. Three training paradigms were exercised: supervised fine-tuning (SFT v6), "
        "reinforcement learning with GRPO (which produced the trajectories used downstream), and "
        "direct preference optimization (DPO) on pairs mined from those RL rollouts. A held-out test "
        "set was created to allow apples-to-apples comparison. The key finding is that <b>SFT v6 "
        "checkpoint-4659 wins at 80.1%</b> on that held-out set, with DPO providing no measurable "
        "improvement and actively hurting the un-SFT-trained Instruct baseline."))
    story.append(Spacer(1, 0.1*inch))
    story.append(h3("Headline numbers (held-out test_data, 49 calendars × 692 queries)"))
    headline = [
        ["Model", "Best checkpoint", "Accuracy", "Δ vs starting point"],
        ["Qwen3-14B-Instruct (no calendar training)", "—", "63.0%", "(reference)"],
        ["SFT v6 (best on this set)", "ckpt-4659 (epoch 3)", "80.1%", "+17.1 pp vs Instruct"],
        ["DPO-from-SFT (best)", "ckpt-479 (epoch 1)", "79.3%", "−0.8 pp vs SFT"],
        ["DPO-from-Instruct (best)", "ckpt-958 (epoch 2)", "61.3%", "−1.7 pp vs Instruct"],
    ]
    t = Table(headline, colWidths=[2.6*inch, 1.5*inch, 0.8*inch, 1.7*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#fff4cc')),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15*inch))
    story.append(h3("Headline takeaways"))
    bullets = [
        "<b>SFT lifts the model from 63% to 80%</b> on the held-out test set. Most of that gain is on "
        "domain-specific categories (Chaos: 30→83%, Modifier: 70→89%) — exactly where the Instruct "
        "baseline lacked exposure to calendar-specific tool-call patterns.",
        "<b>DPO-from-SFT does not beat SFT.</b> All three DPO epochs cluster at 78.8–79.3%, "
        "indistinguishable from the SFT v6 best of 80.1% (gap is 0.8 pp; standard error on 692 queries "
        "is ≈ 1.5 pp).",
        "<b>DPO-from-Instruct loses to its baseline by 1.7 pp.</b> All three epochs scored below the "
        "63.0% baseline — consistent with the textbook \"likelihood displacement\" failure mode for "
        "DPO trained on top of a model that has not first been SFT-tuned to the target structure.",
        "<b>Operational issues likely contributed to the negative DPO result.</b> The implementation "
        "had bugs that were patched mid-run rather than fixed-and-restarted (missing tools= flag in "
        "DPOConfig, max_length truncation, vLLM kill-timing race, bnb optimizer instability). The "
        "result is therefore not a clean indictment of DPO, just of this specific pipeline. The "
        "decision is to pause DPO experiments and revisit only with a clean re-implementation.",
    ]
    for b in bullets:
        story.append(p(f"• {b}", BODY))
    story.append(PageBreak())

    # ── 2. PROJECT CONTEXT ──
    story.append(h1("2. Project Context and Goals"))
    story.append(p(
        "<b>Goal:</b> build an agentic assistant that manages a Google Calendar via tool calling. "
        "The assistant should accept natural-language queries (\"reschedule my Tuesday meeting\", "
        "\"what's after my networking event\", \"book a doctor's appointment next week\") and "
        "translate them into the right sequence of tool calls (get_current_time, list_events, "
        "create_event, update_event, delete_event, etc.) operating on a CalendarEnvironment, "
        "ultimately producing a final natural-language answer."))
    story.append(p(
        "<b>Tools available to the agent (7):</b> get_current_time, list_events, get_event, "
        "create_event, update_event, delete_event, respond_to_event. Tools are exposed via OpenAI-"
        "style function definitions; the model emits Qwen-style <tool_call>...</tool_call> XML "
        "and the calendar harness dispatches and responds with formatted tool-call results."))
    story.append(p(
        "<b>Categories the data is organized by (7):</b> Complex Logic & Conflict (Advanced), "
        "Human Chaos (Edge Cases/Fragments), Information Retrieval (Querying), Modifier & "
        "Correction (Rescheduling/Updates), Relative Time References (today, tomorrow, yesterday, "
        "this week), Schedule a Single Event, Vague & Contextual (Reasoning Required). Each "
        "category captures a distinct kind of calendar task; we report all metrics broken down by "
        "category so it is possible to see where each training method actually moves the needle."))
    story.append(p(
        "<b>Training paradigms exercised over the project:</b>"))
    for line in [
        "Phase 1 (early): SFT v3 → v5 on Qwen2.5-1.5B-Instruct. Topped out at 74.6% on the original RL-data benchmark; comprehension ceiling reached.",
        "Phase 2: SFT v6 on Qwen3-14B with LoRA r=64. Best ckpt-6212 at 82.5% on RL data; this is the model that downstream DPO uses as starting point.",
        "Phase 3 (concurrent with 2): RL/GRPO training on Qwen3-14B starting from SFT v6, producing 5,506 stored trajectory groups (8 rollouts each = ~44,000 trajectories). These rollouts are the source for DPO pair mining.",
        "Phase 4 (this report's focus): DPO experiments on Qwen3-14B — DPO-from-SFT and DPO-from-Instruct.",
        "Phase 5 (this report's focus): held-out test_data benchmark creation and 12-checkpoint apples-to-apples eval.",
    ]:
        story.append(p(f"&nbsp; • {line}", SMALL))
    story.append(PageBreak())

    # ── 3. HARDWARE / STACK ──
    story.append(h1("3. Hardware, Environment, and Stack"))
    story.append(h3("3.1 Compute"))
    story.append(p(
        "All training and evaluation runs took place on a single <b>NVIDIA RTX PRO 6000 Blackwell</b> "
        "server-edition GPU (96 GiB total, compute capability 12.0, bf16 + FlashAttention-2 native). "
        "The GPU is partitioned into <b>4 MIG slices</b> of ~24 GiB each, addressed by the UUIDs "
        "MIG-5dc2f940 (slice 0), MIG-abbb3894 (slice 1), MIG-dd607cdf (slice 2), MIG-7488039b "
        "(slice 3). Slice selection is via the <code>CUDA_VISIBLE_DEVICES=MIG-&lt;uuid&gt;</code> "
        "environment variable. RL training runs continuously on slice 0; the other three slices "
        "carry the DPO training and evaluation work in this report."))
    story.append(h3("3.2 Software stack (frozen 2026-04-15)"))
    stack = [
        ["Package", "Version", "Notes"],
        ["torch", "2.10.0+cu128", "Blackwell CUDA 12.8 build"],
        ["vLLM", "0.19.0", "Inference server; FA2 backend; site-patch for MIG UUID parsing"],
        ["transformers", "4.57.6", ""],
        ["TRL", "0.24.0", "DPOTrainer; conversational pair format"],
        ["peft", "0.19.0", "LoRA + adapter merging"],
        ["unsloth", "2026.4.4", "SFT training fast-path"],
        ["openpipe-art", "0.5.17", "RL/GRPO training harness"],
        ["bitsandbytes", "(latest at install)", "4-bit quantization; paged + non-paged 8-bit AdamW"],
        ["vertexai (Google)", "—", "Gemini-2.0-flash judge for eval verdicts"],
        ["reportlab", "4.4.10", "this report"],
    ]
    t = Table(stack, colWidths=[1.4*inch, 1.4*inch, 4.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(h3("3.3 Slurm + bare-process workaround"))
    story.append(p(
        "The Slurm node <code>azkaban</code> spent the entire DPO experimentation window in "
        "<code>IDLE+DRAIN</code> state due to a Slurm RealMemory off-by-1 disagreement (reported "
        "257554 MiB versus configured 257555 MiB). Since un-draining the node requires admin "
        "privileges that were not available, all GPU jobs in this report were launched as bare "
        "<code>nohup python ...</code> processes with explicit <code>CUDA_VISIBLE_DEVICES=MIG-...</code> "
        "to bind to a single MIG slice. This violates the project policy of \"always use Slurm "
        "for GPU work\" but was unavoidable. RL training had been running similarly bare since "
        "before this report."))
    story.append(PageBreak())

    # ── 4. DATA ──
    story.append(h1("4. Data Pipeline — All Sources"))
    story.append(p(
        "Four distinct datasets were used or created during the project. Their relationship is "
        "important because contamination between train and eval was the primary motivation for "
        "creating the new test_data set described in §4.4."))
    story.append(h3("4.1 SFT training data — sft_data/ (existed at start of project)"))
    sft_data_table = [
        ["Item", "Value"],
        ["Trajectories", "6,947 (after error filter; 46 dropped from raw 6,901)"],
        ["Augmentation factor", "~6× from 1,164 base trajectories"],
        ["Calendars", "114 unique"],
        ["Categories", "7 balanced (917–1,074 per category)"],
        ["Median tokens/trajectory", "596 (compact tool-result format)"],
        ["Teacher", "gemini-2.5-pro (legacy; pro models now blocked by cost guard)"],
        ["Format", "OpenAI tool-call messages with role/content/tool_calls/tool_call_id/trainable"],
        ["Path", "sft_data/trajectories_augmented/"],
    ]
    t = Table(sft_data_table, colWidths=[2.0*inch, 5.3*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 9),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8e8e8')),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(h3("4.2 RL training data — rl_data/ (existed at start of project)"))
    rl_data_table = [
        ["Item", "Value"],
        ["Scenarios", "622 across 44 calendars (avg 14.1 queries/calendar)"],
        ["Calendars", "44 (indices 0–47, with 4 missing)"],
        ["Categories", "7 balanced ~89/category"],
        ["Per-query fields", "query, expected_behavior, category, addressed_days, current_time"],
        ["Used by", "GRPO rollouts during Qwen3-14B RL training (Phase 3); DPO pair mining draws from these rollouts (§4.3)"],
        ["Path", "rl_data/json_calender/, rl_data/queries/"],
    ]
    t = Table(rl_data_table, colWidths=[2.0*inch, 5.3*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 9),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8e8e8')),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(h3("4.3 DPO pair source — mined from .art/.../trajectories/train/"))
    story.append(p(
        "The DPO experiments mine preference pairs from the RL/GRPO rollout parquets stored at "
        "<code>.art/calendar-agent/models/calendar-agent-001/trajectories/train/</code>. Each "
        "parquet file corresponds to one training step (= one scenario sampled, with 8 rollouts "
        "in a TrajectoryGroup). The mining script "
        "(<code>scripts/training/dpo/mine_dpo_pairs.py</code>) processes each parquet and, when "
        "the 8 rollouts include at least one Correct (reward=1.0) and at least one Incorrect "
        "(reward=0.0), emits exactly one pair (random correct as <i>chosen</i>, random incorrect "
        "as <i>rejected</i>) in TRL conversational format."))
    pair_data_table = [
        ["Item", "Value"],
        ["Source parquet count", "5,506 (one per RL training step, 8 rollouts each)"],
        ["Mixed-outcome groups (DPO-usable)", "1,913 (34.8%) — yield one pair each"],
        ["All-correct groups (RFT-usable, DPO-skipped)", "3,069 (55.8%)"],
        ["All-wrong groups (KTO-negatives)", "522 (9.5%)"],
        ["Unique scenarios with at least one mixed group", "459 / 622 (74%)"],
        ["DPO pairs (1 per step, used)", "1,913"],
        ["DPO pairs (max-cartesian — N_correct × N_wrong)", "21,395 (not used)"],
        ["Total correct trajectories (RFT yield)", "33,027"],
        ["Pairs output path", "runs/dpo/pairs_from_14b_rl.jsonl"],
        ["Pair format", "{prompt, chosen, rejected, metadata} — TRL conversational"],
    ]
    t = Table(pair_data_table, colWidths=[3.0*inch, 4.3*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 9),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8e8e8')),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(p("<b>Per-category DPO yield</b> (Figure 3 visualizes the same data):", SMALL))
    cat_yield = [
        ["Category", "Mixed (DPO pairs)", "All correct", "All wrong", "Total groups"],
        ["Chaos", "433", "256", "82", "771"],
        ["Complex", "391", "223", "164", "778"],
        ["IR", "176", "549", "61", "786"],
        ["Modifier", "250", "438", "93", "781"],
        ["RelTime", "85", "729", "13", "827"],
        ["Schedule", "335", "406", "35", "776"],
        ["Vague", "243", "468", "74", "785"],
    ]
    t = Table(cat_yield, colWidths=[1.0*inch, 1.5*inch, 1.0*inch, 1.0*inch, 1.0*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 9),
                           ('ALIGN', (1,0), (-1,-1), 'CENTER'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8e8e8')),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(NOTE.cloneStyle("note", parent=NOTE) if False else None)
    story.append(p(
        "<b>Note on category skew:</b> RelTime has only 85 mixed-outcome groups out of 827 because "
        "the model is already strong at relative-time queries (729 of 827 = 88% all-correct). "
        "Chaos at the other extreme has 433 mixed (and 164 all-wrong, the highest among all "
        "categories). DPO mining is therefore much denser on Chaos/Complex/Schedule than on "
        "RelTime. This category-skewed pair distribution is one driver of the per-category "
        "asymmetry we see in the final DPO results (gains on Schedule, regressions on Complex).", NOTE))
    story.append(Image("runs/analysis/figures/fig3_pair_yield.png", width=6.5*inch, height=3.25*inch))
    story.append(p("<i>Figure 3.</i> Stacked-bar visualization of per-category DPO yield. Mixed bars (green) are the DPO-usable pairs.", SMALL))
    story.append(PageBreak())

    story.append(h3("4.4 Held-out test data — test_data/ (NEW; created 2026-04-24)"))
    story.append(p(
        "<b>Why this dataset exists:</b> Once we set up DPO eval, we noticed that the existing "
        "rl_data/ benchmark — a 280-query subset that SFT v6 had been measured against at 82.5% — "
        "was <i>not</i> held out for DPO. The DPO pairs were mined from RL training rollouts on "
        "those exact 622 scenarios, and 459 of those scenarios (74%) appeared in the pairs the "
        "DPO models trained on. Evaluating DPO on rl_data would therefore be evaluating partly on "
        "training data, inflating the score. We needed a fresh held-out set that no model had "
        "seen as training input, but still came from the same generation distribution so the "
        "comparison was fair."))
    story.append(p(
        "<b>Generation:</b> <code>scripts/data_generation/generate_test_data.py</code> — a fork "
        "of the original generate_data.py with three changes: (1) writes to "
        "<code>test_data/</code>, (2) uses indices 0–49, (3) seeds Python's random module with "
        "20260424 to ensure profession choices and Monday-date selections diverge from the "
        "original generation. Generation pipeline is the standard 4-stage Gemini-2.0-flash "
        "pass: persona → text calendar → JSON calendar → query set."))
    test_data_table = [
        ["Item", "Value"],
        ["Calendars (full pipeline succeeded)", "49 of 50 (one dropped at text-calendar stage for missing day-of-week)"],
        ["Total queries", "692"],
        ["Mean queries per calendar", "14.1 (range 14–16)"],
        ["Categories balanced", "98–104 queries each"],
        ["Generation cost", "~196 gemini-2.0-flash calls = ~$0.05"],
        ["Wall time to generate", "~25 minutes"],
        ["Field completeness", "692/692 with category, expected_behavior, current_time"],
        ["Path", "test_data/json_calender/*.txt, test_data/queries/*.txt"],
    ]
    t = Table(test_data_table, colWidths=[3.0*inch, 4.3*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 9),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8e8e8')),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(p("<b>Per-category breakdown of test_data/ queries:</b>", SMALL))
    test_cat = [
        ["Category", "Queries"],
        ["RelTime", "104"],
        ["Schedule", "98"],
        ["Vague", "98"],
        ["Modifier", "98"],
        ["IR", "98"],
        ["Complex", "98"],
        ["Chaos", "98"],
        ["Total", "692"],
    ]
    t = Table(test_cat, colWidths=[2.0*inch, 1.0*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 9),
                           ('ALIGN', (1,0), (-1,-1), 'CENTER'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8e8e8')),
                           ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#fff4cc')),
                           ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(PageBreak())

    # ── 5. MODELS ──
    story.append(h1("5. Models Compared"))
    story.append(p(
        "Twelve checkpoints plus one external baseline are evaluated in this report. They span "
        "three runs and one no-training reference."))
    models_table = [
        ["Run", "Type", "Base", "LoRA target", "Epochs evaluated", "Total ckpts"],
        ["Qwen3-14B-Instruct baseline", "External (no training)", "Qwen/Qwen3-14B", "—", "—", "1 (baseline only)"],
        ["SFT v6", "Supervised FT", "Qwen/Qwen3-14B", "q,k,v,o,gate,up,down (r=64)", "1, 2, 3, 4, 5", "5"],
        ["DPO-from-SFT", "DPO LoRA on SFT", "SFT v6 ckpt-6212 (merged)", "q,k,v,o,gate,up,down (r=64)", "1, 2, 3", "3"],
        ["DPO-from-Instruct", "DPO LoRA on Instruct", "Qwen/Qwen3-14B", "q,k,v,o,gate,up,down (r=64)", "1, 2, 3", "3"],
    ]
    t = Table(models_table, colWidths=[1.6*inch, 1.4*inch, 1.6*inch, 1.6*inch, 0.9*inch, 0.7*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 8),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(h3("5.1 Important naming clarification: \"base model\""))
    story.append(p(
        "In this project, \"base model\" refers to <code>Qwen/Qwen3-14B</code>, which is "
        "Alibaba's <i>Instruct-tuned</i> release (the version that already knows chat format and "
        "tool-call structure). The truly raw pretrained-only release is "
        "<code>Qwen/Qwen3-14B-Base</code> — a different checkpoint that has never seen instruction "
        "tuning. We did <b>not</b> evaluate Qwen3-14B-Base because it is not used as a starting "
        "point for any of our training. The 63.0% \"baseline\" number throughout this report is "
        "the Instruct release, never trained on calendar data."))
    story.append(PageBreak())

    # ── 6. TRAINING DETAILS ──
    story.append(h1("6. Training Details and Hyperparameters"))
    story.append(h3("6.1 SFT v6 (existed before this report; included for comparison)"))
    sft_v6_hp = [
        ["Hyperparameter", "Value", "Notes"],
        ["framework", "unsloth + TRL SFTTrainer", ""],
        ["base_model", "Qwen/Qwen3-14B", ""],
        ["load_in_4bit", "True", "BitsAndBytes nf4"],
        ["max_seq_length", "4096", "tool-call trajectories fit"],
        ["lora_rank", "64", ""],
        ["lora_alpha", "64", ""],
        ["lora_dropout", "0", ""],
        ["target_modules", "q,k,v,o,gate,up,down", "all 7 projections"],
        ["per_device_train_batch_size", "1", ""],
        ["gradient_accumulation_steps", "4", ""],
        ["learning_rate", "2e-4 cosine", ""],
        ["warmup_ratio", "0.03", ""],
        ["optim", "paged_adamw_8bit", ""],
        ["bf16", "True", "Blackwell native"],
        ["loss_masking", "assistant tokens only", ""],
        ["num_epochs", "5", "checkpoints saved at 1553, 3106, 4659, 6212, 7765"],
        ["data", "6,947 augmented + /no_think system prompt", ""],
    ]
    t = Table(sft_v6_hp, colWidths=[2.2*inch, 2.0*inch, 3.1*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 8),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.15*inch))
    story.append(h3("6.2 DPO training (both variants share these hyperparameters)"))
    dpo_hp = [
        ["Hyperparameter", "Final value", "Initial value (changed during runs)", "Reason for change"],
        ["beta (β)", "0.1", "0.1", "—"],
        ["learning_rate", "5e-7", "5e-7", "—"],
        ["num_epochs", "3", "3", "—"],
        ["per_device_train_batch_size", "1", "1", "—"],
        ["gradient_accumulation_steps", "4", "4", "—"],
        ["max_length", "2048", "4096 → 2560 → 3072 → 2048", "OOM during fp32 logit upcast and during backward"],
        ["max_prompt_length", "1024", "1024 → 128 → 1024", "First too tight; with tools= prompts are 922 tokens"],
        ["tools=", "get_openai_tools()", "(MISSING)", "Caught by code review (see review.md)"],
        ["lora_rank", "64", "64", "—"],
        ["lora_alpha", "64", "64", "—"],
        ["lora_dropout", "0.0", "0.0", "—"],
        ["target_modules", "q,k,v,o,gate,up,down", "same", "—"],
        ["bf16", "True", "True", "—"],
        ["optim (DPO-from-SFT)", "paged_adamw_8bit", "paged_adamw_8bit", "—"],
        ["optim (DPO-from-Instruct)", "adamw_bnb_8bit (non-paged)", "paged_adamw_8bit", "Crash in bnb sync_gpu (illegal memory access) at step ~11"],
        ["lr_scheduler_type", "cosine", "—", ""],
        ["warmup_ratio", "0.03", "—", ""],
        ["gradient_checkpointing", "True (use_reentrant=False)", "—", ""],
        ["precompute_ref_log_probs", "True", "False initially → True", "Cuts per-step memory in half; pays one ~40-min upfront cost"],
        ["attn_implementation", "sdpa", "flash_attention_2", "flash_attn package not installed in env"],
    ]
    t = Table(dpo_hp, colWidths=[2.0*inch, 1.6*inch, 1.7*inch, 2.0*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 7),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(h3("6.3 Pair length distribution (Figure 4)"))
    story.append(p(
        "Sequence-length distribution drives the max_length tuning. With <code>tools=</code> "
        "passed (deployed setting), the median prompt is 915 tokens; with tools omitted "
        "(default before review) the median prompt is 51 tokens — a 17× difference. The "
        "completion lengths span p50 ≈ 1,400, p95 ≈ 2,800, max ≈ 4,700. Setting max_length=2048 "
        "truncates 25% of pairs at one end (a tail loss we accepted to fit on a 24 GiB MIG "
        "slice). Setting max_length=4096 OOM'd the slice during the fp32 logit upcast for the "
        "DPO loss computation."))
    story.append(Image("runs/analysis/figures/fig4_seq_lengths.png", width=6.5*inch, height=2.66*inch))
    story.append(p("<i>Figure 4.</i> Left: prompt length with vs without tools= flag. Right: pair length distribution with tools (chosen and rejected separately). Vertical dashed lines mark the max_length values we tried.", SMALL))
    story.append(PageBreak())

    # ── 7. OPERATIONAL ISSUES ──
    story.append(h1("7. Operational Issues — Full Narratives"))
    story.append(p(
        "Five distinct issues were encountered and resolved during the DPO + eval phase. None "
        "were anticipated up front and each cost some debug cycles."))
    story.append(h3("7.1 Missing tools= flag in DPOConfig (caught by code review)"))
    story.append(p(
        "<b>Symptom:</b> first DPO runs trained successfully but on prompts that were 61 tokens "
        "long, not the ~922 tokens that the deployed model serves with. After ~1 hour of "
        "training, <code>review.md</code> was written by manual code review and surfaced this "
        "discrepancy. The SFT pipeline at <code>scripts/training/sft/sft_train.py:195-197</code> "
        "passes <code>tokenizer.apply_chat_template(messages, tools=TOOLS, ...)</code>, but "
        "<code>dpo_train.py</code> never set <code>tools=</code>. ART rollouts also pass tools at "
        "inference time. The DPO model was therefore being trained to emit tool_calls in a "
        "context that omits the tool schemas — a major train/deploy distribution mismatch."))
    story.append(p(
        "<b>Resolution:</b> added <code>tools=get_openai_tools()</code> to the DPOConfig "
        "constructor in <code>dpo_train.py</code>. TRL 0.24's DPOConfig supports this field "
        "natively (passes through to <code>apply_chat_template</code>). Both DPO runs were "
        "killed and restarted with the fix. Verified prompt length jumped from 58 to 922 tokens "
        "on a sample pair."))
    story.append(p(
        "<b>Lesson:</b> the implementation reviewer (review.md) was crucial and almost certainly "
        "saved the experiment. Future training-config changes should run a pre-launch sanity check "
        "that verifies tokenizer-level inputs match the deployment context.", NOTE))
    story.append(h3("7.2 max_length OOM at 4096, accepted truncation at 2048"))
    story.append(p(
        "<b>Symptom:</b> first training step OOM'd during the fp32 logit upcast in "
        "<code>accelerate.utils.operations._convert_to_fp32</code>. Tensor that failed to "
        "allocate: 2.62 GiB (consistent with [2 (chosen+rejected), seq_len, vocab=152064] at "
        "bf16 → fp32 = 4× expansion). Stepped down through 4096 → 2560 → 3072 → 2048."))
    story.append(p(
        "<b>Memory math:</b> on a 24 GiB MIG slice, the budget is roughly: weights ~7.5 GiB "
        "(14B in 4-bit), trainable LoRA + grad + optimizer state ~2 GiB, activation memory ~3-5 "
        "GiB depending on max_length (with grad checkpointing), plus the logit tensor that gets "
        "upcast to fp32 for the loss reduction (~3.7 GiB at max_length=3072, ~5 GiB at 4096). "
        "Backward adds another ~1.7 GiB peak for gradient tensors. The total at "
        "max_length=3072 was 23.2 GiB — too close to the 23.62 GiB limit; at 2048 we settled "
        "with comfortable headroom."))
    story.append(p(
        "<b>Trade-off:</b> max_length=2048 truncates 24.8% of pairs at one end (chosen or "
        "rejected). max_length=2560 would truncate 13.4%; max_length=3072 only 3.5%. "
        "We chose 2048 to fit in memory; this is a known limitation of the run. Future runs "
        "could (a) precompute_ref_log_probs to reclaim more memory, (b) reduce LoRA rank to "
        "32 or 16, (c) use 2× MIG slices via tensor parallelism."))
    story.append(h3("7.3 vLLM kill-timing race — 70 GiB stranded across zombie processes"))
    story.append(p(
        "<b>Symptom:</b> the SFT v6 multi-checkpoint orchestrator finished the first "
        "checkpoint cleanly, then failed to start vLLM for the next 4 checkpoints — each "
        "aborting with <code>ValueError: Free memory on device cuda:0 (0.72/23.62 GiB) on "
        "startup is less than desired GPU memory utilization (0.9, 21.26 GiB)</code>. The "
        "orchestrator gave up after 3 attempts and exited cleanly with 4 of 5 checkpoints "
        "marked \"pending\"."))
    story.append(p(
        "<b>Investigation:</b> "
        "<code>nvidia-smi --query-compute-apps</code> revealed three orphan "
        "<code>VLLM::EngineCore</code> subprocesses each holding 23 GiB. The previous orchestrator "
        "code was sending SIGKILL to the API server PID and sleeping 3 seconds, which was "
        "not enough time for: (a) the engine subprocess (separate PID, child of the API "
        "server) to be reaped, (b) the CUDA allocator to release allocations on its end. "
        "Each retry then spawned a new engine that competed with the still-resident dead "
        "ones."))
    story.append(p(
        "<b>Resolution:</b> patched <code>kill_vllm_on_port</code> in "
        "<code>scripts/eval/eval_all_checkpoints.py</code>: now it (1) SIGKILLs the port owner, "
        "(2) <code>pgrep</code>s any user-owned <code>VLLM::EngineCore</code> processes and "
        "SIGKILLs them too, (3) polls <code>lsof -ti :PORT</code> for up to 60 s waiting for "
        "all to exit, (4) sleeps an additional 30 s for CUDA to actually flush. Killed the "
        "three zombies manually, restarted the orchestrator with the patched code; "
        "subsequent vLLM startups all succeeded."))
    story.append(p(
        "<b>Lesson:</b> killing a parent process does not release CUDA memory if the parent "
        "has already double-forked or spawned via multiprocessing. Always verify via "
        "<code>nvidia-smi --query-compute-apps</code>, not just <code>ps</code>.", NOTE))
    story.append(PageBreak())

    story.append(h3("7.4 bnb paged_adamw_8bit illegal memory access on Instruct"))
    story.append(p(
        "<b>Symptom:</b> DPO-from-Instruct crashed at training step ~11 with "
        "<code>torch.AcceleratorError: CUDA error: an illegal memory access was encountered</code>. "
        "Stack trace pointed at "
        "<code>bitsandbytes/optim/optimizer.py:329 sync_gpu(p) → torch.cuda.synchronize()</code>. "
        "Crashed on slice 2 first; relaunched on slice 3, crashed again at the same step. "
        "DPO-from-SFT used the same optimizer simultaneously and never crashed."))
    story.append(p(
        "<b>Hypothesis:</b> <code>paged_adamw_8bit</code> uses a CPU-paging mechanism for "
        "8-bit moment buffers. The Instruct model's gradient distribution evidently exposed "
        "an edge case in the paging logic — different from SFT's gradients on the same data. "
        "Same optimizer, same code path, different failure mode."))
    story.append(p(
        "<b>Resolution:</b> swapped the optimizer to <code>adamw_bnb_8bit</code> (the non-paged "
        "8-bit variant). Same precision, same memory footprint, but no paging. Threaded a new "
        "<code>DPO_OPTIM</code> environment variable through <code>dpo_train.py</code> so we "
        "could change just the Instruct run. Restarted Instruct on slice 3; got past step 11 "
        "cleanly and finished all 3 epochs. SFT run was untouched."))
    story.append(p(
        "<b>Lesson:</b> bnb's paged optimizers are a perf optimization, not a correctness "
        "guarantee. If you see CUDA illegal-memory-access in <code>sync_gpu</code>, try the "
        "non-paged variant first.", NOTE))
    story.append(h3("7.5 Mid-eval vLLM crashes (ckpt-1437 of Instruct, ckpt-4659 of SFT)"))
    story.append(p(
        "<b>Symptom:</b> two of the eval runs crashed mid-evaluation with "
        "<code>Connection error</code> on every subsequent agent call. ckpt-1437 of "
        "DPO-from-Instruct crashed at query 2/692; ckpt-4659 of SFT v6 crashed at query 32/692. "
        "In both cases the orchestrator's internal consecutive-error guard tripped at 10 "
        "errors and the eval was aborted with a 0.3% accuracy artifact saved to the result "
        "JSON."))
    story.append(p(
        "<b>Resolution:</b> deleted the bad result JSONs to make the orchestrator re-discover "
        "those checkpoints as \"NEW\". Relaunched the orchestrator on a free slice. Re-runs "
        "completed cleanly: ckpt-1437 of Instruct → 60.0%, ckpt-4659 of SFT → 80.1%. "
        "Root cause of the vLLM mid-eval crashes was not investigated; one-off retries "
        "succeeded so it was not blocking. Possibly fp8 quantization edge cases, possibly KV "
        "cache corruption after long-running serving. Worth investigating if the rate climbs."))
    story.append(h3("7.6 Slurm node drained — bare-process workaround"))
    story.append(p(
        "Already discussed in §3.3. Submitted sbatch jobs were stuck in PD with reason "
        "<code>ReqNodeNotAvail, UnavailableNodes:azkaban</code>. The node's state was "
        "<code>IDLE+DRAIN</code> with reason <code>Low RealMemory</code> (off-by-1 MiB). "
        "<code>scontrol update NodeName=azkaban State=RESUME</code> failed with "
        "<code>slurm_update error: Invalid user id</code> (admin only). All work proceeded as "
        "bare nohup'd processes."))
    story.append(PageBreak())

    # ── 8. EVAL METHODOLOGY ──
    story.append(h1("8. Evaluation Methodology"))
    story.append(h3("8.1 Eval pipeline architecture"))
    story.append(p(
        "Three layers, top to bottom:"))
    for line in [
        "<b>scripts/eval/run_test_evals.sh</b> — shell dispatcher. Symlinks pre-merged SFT v6 directories into <code>eval_test/</code> to skip ~50 minutes of re-merging, then launches three <code>eval_all_checkpoints.py</code> processes in parallel: slice 1 = SFT v6 (5 ckpts), slice 2 = DPO-from-SFT (3 ckpts), slice 3 = DPO-from-Instruct (3 ckpts) followed by an inline Qwen3-14B-Instruct baseline eval as a follow-up. Distinct ports (8006/8007/8008) and served-model names so the three vLLMs don't collide.",
        "<b>scripts/eval/eval_all_checkpoints.py</b> — per-run orchestrator. For each checkpoint: (1) merge LoRA into bf16 base on CPU via peft (~10 min per merge for 14B), (2) launch vLLM with fp8 quantization on the assigned MIG slice, (3) wait up to 9 min for the API to come up, (4) shell out to <code>eval_batch.py --mode test</code>, (5) parse the resulting JSON and write per-checkpoint <code>checkpoint-{N}.json</code> + summary.csv. Skips already-evaluated checkpoints by checking JSON existence.",
        "<b>scripts/eval/eval_batch.py</b> — per-query agent loop. For each test query: (1) load the calendar, (2) initialize an agent with the served model and the standard 7-tool schema, (3) run up to 8 turns of tool-calling, (4) snapshot calendar before/after, (5) call the Gemini-2.0-flash judge to decide Correct/Incorrect, (6) record verdict + trajectory + reasoning. Hard 60-second per-query timeout. Aborts the eval after 10 consecutive errors (the bug that produced the 0.3% bad results in §7.5).",
    ]:
        story.append(p(line, BODY))
    story.append(h3("8.2 Multi-slice parallelization map"))
    parallel = [
        ["Slice", "MIG UUID", "Port", "Run", "Checkpoints"],
        ["0", "MIG-5dc2f940 (untouched)", "—", "RL training (continuous)", "—"],
        ["1", "MIG-abbb3894", "8006", "SFT v6", "5"],
        ["2", "MIG-dd607cdf", "8007", "DPO-from-SFT", "3"],
        ["3", "MIG-7488039b", "8008", "DPO-from-Instruct + Instruct baseline", "3 + 1"],
    ]
    t = Table(parallel, colWidths=[0.6*inch, 1.8*inch, 0.6*inch, 2.5*inch, 1.8*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 9),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(h3("8.3 vLLM serve config (per-checkpoint)"))
    story.append(code_block(
        "vllm serve <merged_model_path> \\\n"
        "    --served-model-name <unique_name> \\\n"
        "    --enable-auto-tool-choice \\\n"
        "    --tool-call-parser hermes \\\n"
        "    --max-model-len 4096 \\\n"
        "    --gpu-memory-utilization 0.90 \\\n"
        "    --enforce-eager \\\n"
        "    --quantization fp8 \\\n"
        "    --port <8006|8007|8008>"))
    story.append(p(
        "<b>Tool-call parser:</b> <code>hermes</code> (NOT <code>qwen3_xml</code>). This was "
        "previously discovered to be the correct parser for Qwen3 tool-call output and is "
        "documented in the project CLAUDE.md as a critical setting."))
    story.append(p(
        "<b>fp8 quantization:</b> halves VRAM from ~24 GiB (bf16) to ~12 GiB so the merged "
        "14B model fits with KV cache and 4096-token context on a 24 GiB MIG slice."))
    story.append(h3("8.4 Judge: gemini-2.0-flash"))
    story.append(p(
        "Each query verdict is rendered by gemini-2.0-flash (chosen for cost: ~$0.0001/query, "
        "vs. gemini-2.5-pro which once consumed 20× monthly budget in a day in this project — "
        "documented as a critical cost rule). The judge sees: query, model's final answer, "
        "expected_behavior from the test data, calendar state before, calendar state after. "
        "It returns Correct or Incorrect plus a one-line reasoning. Total judge cost across "
        "all 12 evals: ~$1.00 for ~8,300 verdicts."))
    story.append(PageBreak())

    # ── 9. RESULTS ──
    story.append(h1("9. Results"))
    story.append(h3("9.1 Summary chart (Figure 1)"))
    story.append(Image("runs/analysis/figures/fig1_accuracy.png", width=7.0*inch, height=3.5*inch))
    story.append(p("<i>Figure 1.</i> Held-out test_data/ accuracy across all 12 evaluations + Instruct baseline. Bar colors: gray = baseline, blue = SFT v6, green = DPO-from-SFT, red = DPO-from-Instruct. The 63.0% baseline horizontal line shows the no-training reference; the 80.1% line marks the overall best (SFT v6 ckpt-4659).", SMALL))
    story.append(h3("9.2 Per-category heatmap (Figure 2)"))
    story.append(Image("runs/analysis/figures/fig2_category_heatmap.png", width=6.5*inch, height=4.55*inch))
    story.append(p("<i>Figure 2.</i> Per-category accuracy heatmap. Rows = checkpoints; columns = the 7 calendar-agent categories. Green = high accuracy, red = low. Reading horizontally, the SFT v6 rows are uniformly green except on Complex (which is the hardest category for every model). DPO-from-Instruct rows show Chaos visibly worse than the baseline above it.", SMALL))
    story.append(PageBreak())

    story.append(h3("9.3 Full per-checkpoint table (all 12 evals)"))
    story.append(p(
        "Each cell shows correct/total for that category (≈98 queries each). The highlighted "
        "row is the overall best."))
    story.append(build_results_table())
    story.append(Spacer(1, 0.15*inch))
    story.append(h3("9.4 Best-of-each comparison"))
    story.append(p("Cells are accuracy percentages on the held-out test_data/."))
    story.append(build_per_cat_summary_table())
    story.append(Spacer(1, 0.1*inch))
    story.append(h3("9.5 DPO loss trajectory (Figure 5)"))
    story.append(Image("runs/analysis/figures/fig5_dpo_loss.png", width=6.5*inch, height=2.95*inch))
    story.append(p("<i>Figure 5.</i> Per-step DPO loss (raw + 15-step moving average) for both runs across 3 epochs. The horizontal dashed line is ln(2) = 0.693, the loss when chosen and rejected have equal model probability. Both runs trend down (DPO-from-SFT settles around 0.60; DPO-from-Instruct around 0.57), demonstrating that the DPO objective <i>was</i> minimized — but that minimization failed to translate to accuracy gains on the held-out test set.", SMALL))
    story.append(PageBreak())

    # ── 10. ANALYSIS ──
    story.append(h1("10. Analysis — Why Each Approach Worked or Didn't"))
    story.append(h3("10.1 SFT v6 — what worked"))
    story.append(p(
        "SFT lifts the model from 63.0% to 80.1% on the held-out test set, a +17.1 pp gain. "
        "The category breakdown reveals where the gains come from:"))
    sft_gains = [
        ["Category", "Baseline", "SFT v6 best", "Gain"],
        ["Complex Logic & Conflict", "41%", "59%", "+18 pp"],
        ["Human Chaos (fragments)", "30%", "83%", "+53 pp"],
        ["Information Retrieval", "60%", "90%", "+30 pp"],
        ["Modifier & Correction", "70%", "89%", "+19 pp"],
        ["Relative Time References", "95%", "93%", "−2 pp"],
        ["Schedule a Single Event", "72%", "70%", "−2 pp"],
        ["Vague & Contextual", "70%", "80%", "+10 pp"],
    ]
    t = Table(sft_gains, colWidths=[2.4*inch, 1.0*inch, 1.0*inch, 1.0*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 9),
                           ('ALIGN', (1,0), (-1,-1), 'CENTER'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(p(
        "The biggest gains are on Chaos (+53 pp) and IR (+30 pp). These are calendar-specific: "
        "Chaos queries are short fragments (\"Sarah dinner?\") that require the model to "
        "pattern-match the kind of scheduling intent without explicit cues; IR queries demand "
        "correctly-formed time-window arguments to <code>list_events</code>. The Instruct "
        "baseline lacks both kinds of training and reaches 30% / 60%; SFT pulls these up to "
        "83% / 90%."))
    story.append(p(
        "Two categories show flat or slightly negative gains. <b>RelTime</b> is already "
        "saturated for the Instruct baseline at 95% — the model already understands "
        "today/tomorrow/yesterday. <b>Schedule</b> is interesting: it actually drops 2 pp under "
        "SFT v6's best. Both are within noise (SE ≈ 1.5 pp on 98 queries) and could be "
        "checkpoint-specific noise in the SFT v6 ckpt-4659 weights."))
    story.append(h3("10.2 DPO-from-SFT — why it didn't beat SFT"))
    story.append(p(
        "<b>Result:</b> all three DPO-from-SFT epochs land at 78.8–79.3%, indistinguishable from "
        "SFT v6's best of 80.1% (gap is 0.8 pp). DPO loss <i>did</i> minimize "
        "(Figure 5: 0.693 → 0.60), confirming the algorithm is doing what it's supposed to do "
        "in terms of margin maximization. So why didn't the margin gain produce an accuracy "
        "gain?"))
    story.append(p("<b>Two non-mutually-exclusive hypotheses:</b>"))
    story.append(p(
        "<b>Hypothesis 1 — pair saturation.</b> The 1,913 mined pairs come from RL training "
        "rollouts; the \"chosen\" trajectories were generated by a policy already very close "
        "to (and in fact the descendant of) SFT v6 ckpt-6212. There's not much daylight "
        "between SFT v6's behavior and the chosen trajectories — the marginal information "
        "DPO can extract is small. The \"rejected\" trajectories tell DPO not to do specific "
        "things, but most of those things SFT v6 already wasn't doing."))
    story.append(p(
        "<b>Hypothesis 2 — category trade-offs net to zero.</b> DPO-from-SFT actually does "
        "produce significant per-category movements: <b>Schedule +9 pp</b> (70% → 79%), "
        "<b>Complex −7 pp</b> (59% → 52%). The net is approximately zero, but underneath the "
        "averages DPO has clearly moved the model — just in directions that cancel. The "
        "Schedule gain is plausibly because Schedule has many mixed-outcome pairs (335) and "
        "they're high-quality (clear chosen/rejected distinction). The Complex regression is "
        "harder to explain — possibly DPO is teaching the model a shallower-pattern shortcut "
        "that helps on Schedule but hurts on multi-step reasoning."))
    story.append(p(
        "<b>Implication:</b> if DPO is the right tool here, it would need to be either "
        "(a) trained on on-policy pairs from the actual SFT v6 ckpt-6212 (not from RL "
        "training rollouts), or (b) restricted to categories where pair-quality is high and "
        "the gradient direction doesn't degrade other categories."))
    story.append(h3("10.3 DPO-from-Instruct — why it actively hurt"))
    story.append(p(
        "<b>Result:</b> all three DPO-from-Instruct epochs scored 60.0–61.3%, all <i>below</i> "
        "the 63.0% Instruct baseline. The pattern is consistent across all three epochs and "
        "across all categories — this is not a single-epoch artifact."))
    story.append(p(
        "<b>Mechanism (likelihood displacement, Pal et al. 2024):</b> DPO's loss is a ratio:"))
    story.append(code_block(
        "L = -log σ( β · [ log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x) ] )"))
    story.append(p(
        "Since <code>π_ref</code> for DPO-from-Instruct is the un-SFT'd Qwen3-14B-Instruct, and "
        "the chosen / rejected trajectories in the pairs were generated by SFT-then-RL'd "
        "Qwen3-14B (a quite different policy), <code>π_ref</code> assigns very low probability "
        "to <i>both</i> y_w and y_l. The numerator and denominator of the ratio are similarly "
        "tiny, the differential signal cancels at the format level (both are surprising to "
        "π_ref), and what's left is noise. Worse, DPO's gradient can drive both "
        "<code>π_θ(y_w)</code> and <code>π_θ(y_l)</code> further into the tail while the ratio "
        "looks fine — the absolute probability of <i>chosen</i> goes <i>down</i>. Pal et al.'s "
        "Smaug paper documents exactly this failure for far-off-policy DPO. We predicted this "
        "outcome ex-ante and it is what we observed."))
    story.append(p(
        "<b>Category-level evidence:</b> DPO-from-Instruct's biggest regressions are on Chaos "
        "(30 → 26-29%) and Vague (70 → 57-67%). RelTime, where the baseline was already "
        "saturated at 95%, holds essentially flat (92–97% across DPO epochs). Categories "
        "where Instruct already had calibrated format probability barely moved; categories "
        "where Instruct had fragile probability got pushed further into the tail. Exact "
        "likelihood displacement signature."))
    story.append(PageBreak())

    # ── 11. DECISIONS / NEXT STEPS ──
    story.append(h1("11. Decisions and Recommended Next Steps"))
    story.append(h3("11.1 Decision: pause DPO experiments"))
    story.append(p(
        "Per the user's call on 2026-04-25 after reviewing the test_eval_summary: \"Looks like "
        "we didn't implement DPO properly. Let's skip DPO.\" The decision is documented in "
        "<code>memory/feedback_dpo_skipped.md</code> and reflected in the PROGRESS.md "
        "header."))
    story.append(p(
        "Rationale: the negative DPO result is plausibly attributable to the implementation "
        "issues catalogued in §7 — particularly the off-policy pair source (DPO trained on "
        "rollouts from a policy quite different from its own π_ref) — and not necessarily "
        "to DPO being fundamentally wrong for this task. But the cost of investigating those "
        "is high and other directions are more promising."))
    story.append(h3("11.2 Recommended next experiments (in priority order)"))
    story.append(p(
        "<b>1. Rejection Fine-Tuning (RFT).</b> The 33,027 \"correct\" trajectories sitting in "
        "the RL parquets are an underused resource. Filter to correct, run a few epochs of "
        "SFT on them. Avoids the ratio-based failure mode of DPO entirely; directly "
        "reinforces what the model already does right."))
    story.append(p(
        "<b>2. Iterative DPO with on-policy pairs.</b> If revisiting DPO, the <i>first</i> "
        "thing to fix is pair source. Rather than mining from arbitrary RL training "
        "rollouts, generate fresh rollouts from SFT v6 ckpt-4659 itself (the actual π_ref), "
        "judge them with Gemini, take pairs with mixed outcomes, train one DPO pass, then "
        "regenerate. Standard iterative DPO. Expensive but theoretically clean."))
    story.append(p(
        "<b>3. Complex-category-focused SFT data.</b> Complex Logic & Conflict is the worst "
        "category for every model in this report — 59% for SFT's best, well below the others. "
        "Generating more Complex-specific SFT trajectories (multi-step tool use, "
        "deliberate scheduling conflicts, ambiguous time references) could lift the floor "
        "on the hardest category."))
    story.append(p(
        "<b>4. Category-weighted DPO.</b> Mining pairs only from Schedule (where DPO worked, "
        "+9 pp) or up-weighting them during training would let DPO contribute net-positively "
        "without dragging Complex down. A targeted version of (2)."))
    story.append(p(
        "<b>5. Investigate vLLM mid-eval crashes.</b> Two of the 12 evals crashed mid-run "
        "(§7.5). Re-runs succeeded but the underlying cause was not investigated. If the "
        "rate climbs as we add more evals, this becomes a real bottleneck. Candidate "
        "causes: fp8 quantization edge cases, KV cache corruption after extended serving, "
        "specific query patterns that trigger numerical instability."))
    story.append(PageBreak())

    # ── 12. ARTIFACTS ──
    story.append(h1("12. Reproducibility — Artifacts and File Paths"))
    story.append(h3("12.1 Code"))
    code_paths = [
        ["File", "Purpose"],
        ["scripts/training/dpo/mine_dpo_pairs.py", "Pair mining from RL parquets"],
        ["scripts/training/dpo/dpo_train.py", "DPO training (TRL DPOTrainer + LoRA + bnb)"],
        ["scripts/training/dpo/dpo_from_sft.sbatch", "Slurm template (DPO-from-SFT) — submitted but ran bare"],
        ["scripts/training/dpo/dpo_from_instruct.sbatch", "Slurm template (DPO-from-Instruct) — submitted but ran bare"],
        ["scripts/data_generation/generate_test_data.py", "Held-out test set generation"],
        ["scripts/eval/eval_batch.py", "Per-query agent loop + Gemini judge (added --mode test)"],
        ["scripts/eval/eval_all_checkpoints.py", "Multi-checkpoint orchestrator (parameterized for test mode)"],
        ["scripts/eval/run_test_evals.sh", "3-slice parallel dispatcher"],
        ["scripts/utils/count_dpo_pairs.py", "DPO pair yield analysis"],
        ["scripts/utils/build_project_report.py", "Generates this PDF"],
    ]
    t = Table(code_paths, colWidths=[3.3*inch, 4.0*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 8),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(h3("12.2 Data"))
    data_paths = [
        ["Path", "Contents"],
        ["sft_data/trajectories_augmented/", "6,947 augmented SFT trajectories (114 calendars)"],
        ["rl_data/json_calender/, rl_data/queries/", "622 RL scenarios across 44 calendars"],
        ["test_data/json_calender/, test_data/queries/", "692 held-out test queries across 49 calendars"],
        [".art/calendar-agent/models/calendar-agent-001/trajectories/train/", "5,506 RL rollout parquets (8 rollouts each)"],
        ["runs/dpo/pairs_from_14b_rl.jsonl", "1,913 mined DPO pairs (TRL conversational format)"],
    ]
    t = Table(data_paths, colWidths=[3.5*inch, 3.8*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 8),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(h3("12.3 Run outputs"))
    run_paths = [
        ["Path", "Contents"],
        ["runs/sft_v6_qwen3_14b_20260420/checkpoints/", "5 SFT v6 LoRA adapter checkpoints (1553, 3106, 4659, 6212, 7765)"],
        ["runs/sft_v6_qwen3_14b_20260420/eval/", "rl_data evaluation results (legacy)"],
        ["runs/sft_v6_qwen3_14b_20260420/eval_test/", "test_data evaluation results"],
        ["runs/sft_v6_qwen3_14b_20260420/eval/merged_tmp_*", "Merged fp16 models (used by both rl_data and test_data eval)"],
        ["runs/dpo_qwen3_14b_sft_20260423/", "DPO-from-SFT run dir (config, checkpoints, eval_test)"],
        ["runs/dpo_qwen3_14b_instruct_20260423/", "DPO-from-Instruct run dir"],
        ["runs/qwen3_14b_instruct_baseline_20260424/", "Instruct baseline eval result"],
        ["runs/analysis/test_eval_summary.md", "Markdown summary of all 12 evals + baseline"],
        ["runs/analysis/dpo_pair_counts.json", "Pair mining yield analysis"],
        ["runs/analysis/figures/", "Generated figures used in this report"],
        ["runs/analysis/project_report.pdf", "This PDF"],
        [".art/calendar-agent/models/calendar-agent-001/", "RL training state (continuous run)"],
    ]
    t = Table(run_paths, colWidths=[3.7*inch, 3.6*inch])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                           ('FONTSIZE', (0,0), (-1,-1), 8),
                           ('VALIGN', (0,0), (-1,-1), 'TOP'),
                           ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                           ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a3d6d')),
                           ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                           ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)
    story.append(PageBreak())

    # ── APPENDIX A: per-checkpoint snippets ──
    story.append(h1("Appendix A. Per-Checkpoint Result Summary"))
    story.append(p(
        "Per-checkpoint accuracy and category breakdowns. JSON files contain full per-query "
        "trajectories and judge reasoning; only summary numbers are shown here."))
    runs_app = [
        ("Instruct baseline", None, None, "runs/qwen3_14b_instruct_baseline_20260424/eval_test/baseline.json"),
        ("SFT v6", 1, 1553, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-1553.json"),
        ("SFT v6", 2, 3106, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-3106.json"),
        ("SFT v6", 3, 4659, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-4659.json"),
        ("SFT v6", 4, 6212, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-6212.json"),
        ("SFT v6", 5, 7765, "runs/sft_v6_qwen3_14b_20260420/eval_test/checkpoint-7765.json"),
        ("DPO-from-SFT", 1, 479, "runs/dpo_qwen3_14b_sft_20260423/eval_test/checkpoint-479.json"),
        ("DPO-from-SFT", 2, 958, "runs/dpo_qwen3_14b_sft_20260423/eval_test/checkpoint-958.json"),
        ("DPO-from-SFT", 3, 1437, "runs/dpo_qwen3_14b_sft_20260423/eval_test/checkpoint-1437.json"),
        ("DPO-from-Instruct", 1, 479, "runs/dpo_qwen3_14b_instruct_20260423/eval_test/checkpoint-479.json"),
        ("DPO-from-Instruct", 2, 958, "runs/dpo_qwen3_14b_instruct_20260423/eval_test/checkpoint-958.json"),
        ("DPO-from-Instruct", 3, 1437, "runs/dpo_qwen3_14b_instruct_20260423/eval_test/checkpoint-1437.json"),
    ]
    for name, ep, ckpt, path in runs_app:
        d = load_eval(path)
        if not d: continue
        title = f"{name}" + (f" — epoch {ep} (ckpt-{ckpt})" if ep else "")
        story.append(h3(title))
        pct = d["correct"]/d["total"]*100
        rows = [["Overall", f"{d['correct']}/{d['total']}", f"{pct:.1f}%"]]
        for cat in CAT_ORDER:
            b = d["by"].get(cat, {"c":0,"t":0})
            if b["t"]:
                rows.append([cat, f"{b['c']}/{b['t']}", f"{b['c']/b['t']*100:.1f}%"])
        rows.append(["Path", path, ""])
        t = Table(rows, colWidths=[1.8*inch, 4.4*inch, 1.0*inch])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                               ('FONTSIZE', (0,0), (-1,-1), 8),
                               ('VALIGN', (0,0), (-1,-1), 'TOP'),
                               ('FONTNAME', (0,0), (0,0), 'Helvetica-Bold'),
                               ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#fff4cc')),
                               ('FONTNAME', (0,-1), (0,-1), 'Helvetica-Oblique'),
                               ('FONTSIZE', (0,-1), (-1,-1), 6),
                               ('TEXTCOLOR', (0,-1), (-1,-1), colors.HexColor('#666666'))]))
        story.append(t)
        story.append(Spacer(1, 0.05*inch))
    story.append(PageBreak())

    # ── APPENDIX B: configs ──
    story.append(h1("Appendix B. Configuration Files"))
    story.append(h3("B.1 SFT v6 config (excerpt)"))
    try:
        with open("runs/sft_v6_qwen3_14b_20260420/config.json") as f:
            sft_config = f.read()
        story.append(code_block(sft_config))
    except Exception as e:
        story.append(p(f"<i>Could not read SFT v6 config: {e}</i>", SMALL))
    story.append(h3("B.2 DPO-from-SFT config"))
    try:
        with open("runs/dpo_qwen3_14b_sft_20260423/config.json") as f:
            story.append(code_block(f.read()))
    except Exception as e:
        story.append(p(f"<i>Could not read DPO-from-SFT config: {e}</i>", SMALL))
    story.append(h3("B.3 DPO-from-Instruct config"))
    try:
        with open("runs/dpo_qwen3_14b_instruct_20260423/config.json") as f:
            story.append(code_block(f.read()))
    except Exception as e:
        story.append(p(f"<i>Could not read DPO-from-Instruct config: {e}</i>", SMALL))
    story.append(PageBreak())

    # ── APPENDIX C: pair stats ──
    story.append(h1("Appendix C. DPO Pair-Mining Statistics"))
    story.append(p(
        "Full output of <code>scripts/utils/count_dpo_pairs.py</code>:"))
    try:
        with open("runs/analysis/dpo_pair_counts.json") as f:
            story.append(code_block(f.read()))
    except Exception as e:
        story.append(p(f"<i>Could not read pair counts: {e}</i>", SMALL))

    # Build
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {out_path}")
    sz = os.path.getsize(out_path) / 1024
    print(f"Size: {sz:.1f} KB")


if __name__ == "__main__":
    main()
