"""
models/evaluate.py
==================
Sanity-check evaluation benchmark for the adapted remote sensing vision-language model.

Runs inference across a representative sample of questions from:
  - VRSBench (scene reasoning & visual description)
  - RSVQA (presence, count, area, and comparison queries)
  - BigEarthNet (land cover identification)

Prints side-by-side question / expected-answer / model-answer comparisons for quick inspection.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_vrsbench, load_rsvqa, load_bigearthnet
from models.inference import answer_question


def run_evaluation():
    print("=" * 80)
    print("SatQuery AI - Vision-Language Model Adaptation Evaluation")
    print("=" * 80)

    test_cases: List[Dict[str, Any]] = []

    # 1. VRSBench Samples
    try:
        vrs_samples = load_vrsbench()
        for s in vrs_samples[:3]:
            # VRSBench has multiple QA pairs per image
            for qa in s.get("qa_pairs", [])[:1]:
                test_cases.append({
                    "benchmark": "VRSBench",
                    "image_path": s["image_path"],
                    "question": qa.get("question", s.get("question")),
                    "expected": qa.get("answer", s.get("answer")),
                })
    except Exception as e:
        print(f"[!] Warning loading VRSBench samples: {e}")

    # 2. RSVQA Samples (Presence, Count, Comparison)
    try:
        rsvqa_samples = load_rsvqa()
        for s in rsvqa_samples[:4]:
            test_cases.append({
                "benchmark": f"RSVQA ({s.get('question_type', 'general')})",
                "image_path": s["image_path"],
                "question": s.get("question"),
                "expected": s.get("answer"),
            })
    except Exception as e:
        print(f"[!] Warning loading RSVQA samples: {e}")

    # 3. BigEarthNet Samples (Land cover query)
    try:
        ben_samples = load_bigearthnet(split="train")
        for s in ben_samples[:2]:
            test_cases.append({
                "benchmark": "BigEarthNet",
                "image_path": s["image_path"],
                "question": s.get("question"),
                "expected": s.get("answer"),
            })
    except Exception as e:
        print(f"[!] Warning loading BigEarthNet samples: {e}")

    if not test_cases:
        print("[-] Error: No test samples could be assembled from data loaders!")
        return

    print(f"\n[*] Evaluating {len(test_cases)} sample queries...\n")

    results = []
    latencies = []

    for i, item in enumerate(test_cases, 1):
        img_name = Path(item["image_path"]).name
        q = item["question"]
        expected = item["expected"]

        start_time = time.perf_counter()
        pred = answer_question(item["image_path"], q)
        latency = time.perf_counter() - start_time
        latencies.append(latency)

        results.append({
            "idx": i,
            "benchmark": item["benchmark"],
            "image": img_name,
            "question": q,
            "expected": str(expected),
            "model_answer": str(pred),
            "latency": latency,
        })

    # Print Table
    col_w = {
        "idx": 3,
        "bm": 15,
        "img": 18,
        "q": 35,
        "exp": 25,
        "ans": 25,
        "lat": 7,
    }

    sep_line = (
        f"+{'-'*(col_w['idx']+2)}+{'-'*(col_w['bm']+2)}+{'-'*(col_w['img']+2)}"
        f"+{'-'*(col_w['q']+2)}+{'-'*(col_w['exp']+2)}+{'-'*(col_w['ans']+2)}+{'-'*(col_w['lat']+2)}+"
    )

    print(sep_line)
    print(
        f"| {'#':<{col_w['idx']}} | {'Benchmark':<{col_w['bm']}} | {'Image':<{col_w['img']}} "
        f"| {'Question':<{col_w['q']}} | {'Expected Answer':<{col_w['exp']}} | {'Model Answer':<{col_w['ans']}} | {'Time(s)':<{col_w['lat']}} |"
    )
    print(sep_line)

    for r in results:
        # Truncate strings to fit table nicely
        q_trunc = (r["question"][:col_w["q"]-3] + "...") if len(r["question"]) > col_w["q"] else r["question"]
        exp_trunc = (r["expected"][:col_w["exp"]-3] + "...") if len(r["expected"]) > col_w["exp"] else r["expected"]
        ans_trunc = (r["model_answer"][:col_w["ans"]-3] + "...") if len(r["model_answer"]) > col_w["ans"] else r["model_answer"]
        img_trunc = (r["image"][:col_w["img"]-3] + "...") if len(r["image"]) > col_w["img"] else r["image"]
        bm_trunc = (r["benchmark"][:col_w["bm"]-3] + "...") if len(r["benchmark"]) > col_w["bm"] else r["benchmark"]

        print(
            f"| {r['idx']:<{col_w['idx']}} | {bm_trunc:<{col_w['bm']}} | {img_trunc:<{col_w['img']}} "
            f"| {q_trunc:<{col_w['q']}} | {exp_trunc:<{col_w['exp']}} | {ans_trunc:<{col_w['ans']}} | {r['latency']:<{col_w['lat']}.2f} |"
        )

    print(sep_line)

    # Compute summary metrics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    non_empty = sum(1 for r in results if r["model_answer"].strip() and not r["model_answer"].startswith("Error"))
    print("\n--- Summary Performance ---")
    print(f"Total Samples Evaluated : {len(results)}")
    print(f"Valid Responses Generated: {non_empty}/{len(results)} ({non_empty/len(results)*100:.1f}%)")
    print(f"Average Inference Latency: {avg_latency:.2f} seconds/query")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_evaluation()
