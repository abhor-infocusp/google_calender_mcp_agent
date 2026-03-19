#!/usr/bin/env python3
"""Evaluate all SFT checkpoints: merge LoRA, serve via vLLM, run eval_batch.

Usage:
    PYTHONPATH=src python scripts/eval/eval_all_checkpoints.py
"""

import csv
import json
import os
import subprocess
import sys
import time

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from calendar_agent.paths import PROJECT_ROOT, SFT_OUTPUT_DIR

PYTHON = sys.executable
PROJECT = str(PROJECT_ROOT)
SFT_OUTPUT = str(SFT_OUTPUT_DIR)
MERGED_DIR = os.path.join(SFT_OUTPUT, "merged_tmp")
RESULTS_CSV = os.path.join(SFT_OUTPUT, "checkpoint_eval_results.csv")
PORT = 8005

CHECKPOINTS = [234, 468, 699, 933, 1167, 1401, 1635]

# Loss data: checkpoint -> (epoch, train_loss, eval_loss)
LOSS_DATA = {
    234:  (1, 0.1835, 0.1009),
    468:  (2, 0.0685, 0.0818),
    699:  (3, 0.0615, 0.0999),
    933:  (4, 0.0400, 0.0865),
    1167: (5, 0.0413, 0.0970),
    1401: (6, 0.0200, 0.0970),
    1635: (7, 0.0264, 0.1042),
}


def kill_vllm():
    """Kill any vLLM process on PORT and free GPU memory."""
    subprocess.run(f"lsof -ti :{PORT} | xargs -r kill -9", shell=True, capture_output=True)
    time.sleep(2)
    # Also kill any leftover GPU processes owned by us
    result = subprocess.run(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
        shell=True, capture_output=True, text=True,
    )
    for line in result.stdout.strip().split("\n"):
        pid = line.strip()
        if not pid:
            continue
        owner = subprocess.run(f"ps -p {pid} -o user=", shell=True, capture_output=True, text=True)
        if "abhor" in owner.stdout:
            subprocess.run(f"kill -9 {pid}", shell=True, capture_output=True)
    time.sleep(3)


def merge_checkpoint(ckpt_num):
    """Merge LoRA checkpoint into fp16 model."""
    ckpt_path = os.path.join(SFT_OUTPUT, f"checkpoint-{ckpt_num}")
    print(f"\n{'='*60}")
    print(f"Merging checkpoint-{ckpt_num}...")
    print(f"{'='*60}")

    # Write a small merge script to avoid import issues
    merge_script = f"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{ckpt_path}",
    max_seq_length=4096,
    load_in_4bit=True,
)
output = "{MERGED_DIR}"
model.save_pretrained_merged(output, tokenizer, save_method="merged_16bit")
print("Merge complete")
"""
    script_path = os.path.join(SFT_OUTPUT, "_merge_tmp.py")
    with open(script_path, "w") as f:
        f.write(merge_script)

    result = subprocess.run(
        [PYTHON, script_path],
        env={**os.environ, "PYTHONPATH": os.path.join(PROJECT, "src"),
             "PYTHONUNBUFFERED": "1"},
        timeout=300,
    )
    os.remove(script_path)

    if result.returncode != 0:
        print(f"MERGE FAILED (exit code {result.returncode})")
        return False
    print("Merge OK")
    return True


def start_vllm():
    """Start vLLM server and wait for it to be ready."""
    print(f"Starting vLLM on port {PORT}...")
    proc = subprocess.Popen(
        [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MERGED_DIR,
         "--served-model-name", "sft-v2",
         "--enable-auto-tool-choice",
         "--tool-call-parser", "hermes",
         "--max-model-len", "3076",
         "--gpu-memory-utilization", "0.80",
         "--port", str(PORT)],
        env={**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # Wait for server to be ready
    for i in range(120):
        time.sleep(3)
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2)
            if req.status == 200:
                print("vLLM ready!")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            print(f"vLLM exited early:\n{out[-500:]}")
            return None

    print("vLLM failed to start in 360s")
    proc.kill()
    return None


def run_eval(mode, num_calendars=10, max_queries=40):
    """Run eval_batch.py (Gemini judge) and return results."""
    save_path = os.path.join(SFT_OUTPUT, f"_eval_tmp_{mode}.json")
    cmd = [PYTHON, "-u", os.path.join(PROJECT, "scripts/eval/eval_batch.py"),
         "--mode", mode,
         "--model", "sft-v2",
         "--base-url", f"http://localhost:{PORT}/v1",
         "--num-calendars", str(num_calendars),
         "--save", save_path]
    if max_queries > 0:
        cmd.extend(["--max-queries", str(max_queries)])
    result = subprocess.run(
        cmd,
        env={**os.environ, "PYTHONPATH": os.path.join(PROJECT, "src"),
             "PYTHONUNBUFFERED": "1"},
        timeout=14400,
    )
    if os.path.exists(save_path):
        with open(save_path) as f:
            data = json.load(f)
        os.remove(save_path)
        return data
    return None


def main():
    # Check for already-completed results to allow resume
    completed = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ckpt = int(row["checkpoint"])
                if row.get("sft_correct") and row["sft_correct"] not in ("", "ERR"):
                    completed[ckpt] = row
        print(f"Found {len(completed)} already-evaluated checkpoints: {sorted(completed.keys())}")

    all_rows = []

    for ckpt in CHECKPOINTS:
        epoch, train_loss, eval_loss = LOSS_DATA[ckpt]

        # Check if already done
        if ckpt in completed:
            row = completed[ckpt]
            all_rows.append([ckpt, epoch, train_loss, eval_loss,
                             row["sft_correct"], row["sft_total"], row["sft_pct"],
                             row["rl_correct"], row["rl_total"], row["rl_pct"]])
            print(f"\nCheckpoint {ckpt} (epoch {epoch}): already evaluated, skipping")
            continue

        # Merge
        kill_vllm()
        if not merge_checkpoint(ckpt):
            all_rows.append([ckpt, epoch, train_loss, eval_loss, "ERR", "", "", "ERR", "", ""])
            continue

        # Start vLLM
        vllm_proc = start_vllm()
        if vllm_proc is None:
            all_rows.append([ckpt, epoch, train_loss, eval_loss, "ERR", "", "", "ERR", "", ""])
            continue

        # Run SFT eval (all 161 queries, Gemini judge)
        print(f"\n--- Evaluating checkpoint-{ckpt} on SFT training data ---")
        sft_data = run_eval("sft", max_queries=0)
        sft_correct = sft_data["sft"]["correct"] if sft_data and "sft" in sft_data else 0
        sft_total = sft_data["sft"]["total"] if sft_data and "sft" in sft_data else 0
        sft_pct = f"{sft_correct/sft_total*100:.1f}" if sft_total > 0 else "0"

        # Run RL eval (20 calendars, Gemini judge)
        print(f"\n--- Evaluating checkpoint-{ckpt} on RL data ---")
        rl_data = run_eval("rl", num_calendars=20, max_queries=0)
        rl_correct = rl_data["rl"]["correct"] if rl_data and "rl" in rl_data else 0
        rl_total = rl_data["rl"]["total"] if rl_data and "rl" in rl_data else 0
        rl_pct = f"{rl_correct/rl_total*100:.1f}" if rl_total > 0 else "0"

        all_rows.append([ckpt, epoch, train_loss, eval_loss,
                         sft_correct, sft_total, sft_pct,
                         rl_correct, rl_total, rl_pct])

        # Kill vLLM
        vllm_proc.kill()
        vllm_proc.wait()
        kill_vllm()

        # Write results incrementally
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["checkpoint", "epoch", "train_loss", "eval_loss",
                              "sft_correct", "sft_total", "sft_pct",
                              "rl_correct", "rl_total", "rl_pct"])
            writer.writerows(all_rows)

        print(f"\n>>> Checkpoint {ckpt} (epoch {epoch}): SFT={sft_correct}/{sft_total} ({sft_pct}%), RL={rl_correct}/{rl_total} ({rl_pct}%)")

    # Build lookup from evaluated results
    eval_lookup = {}
    for row in all_rows:
        eval_lookup[row[0]] = row

    # Final summary — all checkpoints, with eval data where available
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"{'Ckpt':>6} {'Epoch':>5} {'TrainLoss':>10} {'EvalLoss':>10} {'SFT':>15} {'RL':>15}")
    print("-" * 70)
    for ckpt in CHECKPOINTS:
        epoch, tl, el = LOSS_DATA[ckpt]
        if ckpt in eval_lookup:
            row = eval_lookup[ckpt]
            sft_c, sft_t, sft_p = row[4], row[5], row[6]
            rl_c, rl_t, rl_p = row[7], row[8], row[9]
            sft_str = f"{sft_c}/{sft_t} ({sft_p}%)" if sft_t else "ERR"
            rl_str = f"{rl_c}/{rl_t} ({rl_p}%)" if rl_t else "ERR"
        else:
            sft_str = "-"
            rl_str = "-"
        print(f"{ckpt:>6} {epoch:>5} {tl:>10.4f} {el:>10.4f} {sft_str:>15} {rl_str:>15}")

    print(f"\nResults saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
