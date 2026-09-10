"""
tests/test_vqa_captioning.py
============================
Automated verification script for SatQuery AI Phase 3 tools:
- Visual Question Answering (VQA): run_vqa
- Scene Captioning: run_captioning
- Text-Guided Region Grounding: run_grounding

Tests tools against sample images from data/sample/optical/ and validates
schema structure, confidence boundaries, and execution reliability.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vqa_tool import run_vqa
from backend.captioning_tool import run_captioning
from backend.grounding_tool import run_grounding


def main():
    print("=" * 80)
    print(" SatQuery AI - Baseline Tools Verification (VQA, Captioning, Grounding)")
    print("=" * 80)

    test_samples = [
        {
            "image_path": "data/sample/optical/sample_optical_001.png",
            "terrain": "Forest",
            "vqa_question": "What terrain type is depicted in this optical satellite observation?",
            "grounding_query": "forest and green vegetation",
        },
        {
            "image_path": "data/sample/optical/sample_optical_002.png",
            "terrain": "Water",
            "vqa_question": "Is there a water body visible in this image?",
            "grounding_query": "the water body",
        },
        {
            "image_path": "data/sample/optical/sample_optical_003.png",
            "terrain": "Urban",
            "vqa_question": "What kind of environment or structures are shown here?",
            "grounding_query": "urban commercial buildings",
        },
        {
            "image_path": "data/sample/optical/sample_optical_004.png",
            "terrain": "Agriculture",
            "vqa_question": "What land use is visible across these parcels?",
            "grounding_query": "agricultural fields",
        },
    ]

    all_passed = True
    vqa_results = []
    caption_results = []
    grounding_results = []

    for idx, sample in enumerate(test_samples, 1):
        img_path = sample["image_path"]
        terrain = sample["terrain"]
        vqa_q = sample["vqa_question"]
        grd_q = sample["grounding_query"]

        print(f"\n[{idx}/{len(test_samples)}] Testing Sample: {img_path} ({terrain})")
        print("-" * 80)

        # 1. Test VQA
        print(f"  [1] VQA Query: '{vqa_q}'")
        vqa_res = run_vqa(img_path, vqa_q)
        print(f"      Answer:     {vqa_res.get('answer')}")
        print(f"      Confidence: {vqa_res.get('confidence')}")

        # Assertions for VQA
        assert vqa_res.get("task") == "vqa", f"Expected task 'vqa', got {vqa_res.get('task')}"
        assert isinstance(vqa_res.get("answer"), str) and len(vqa_res.get("answer")) > 0
        assert 0.0 <= vqa_res.get("confidence") <= 1.0, f"Confidence {vqa_res.get('confidence')} out of bounds"
        assert vqa_res.get("image_path") == img_path
        vqa_results.append(vqa_res)

        # 2. Test Captioning
        print(f"  [2] Scene Captioning...")
        cap_res = run_captioning(img_path)
        print(f"      Caption:    {cap_res.get('caption')}")
        print(f"      Confidence: {cap_res.get('confidence')}")

        # Assertions for Captioning
        assert cap_res.get("task") == "captioning", f"Expected task 'captioning', got {cap_res.get('task')}"
        assert isinstance(cap_res.get("caption"), str) and len(cap_res.get("caption")) > 0
        assert 0.0 <= cap_res.get("confidence") <= 1.0, f"Confidence {cap_res.get('confidence')} out of bounds"
        assert cap_res.get("image_path") == img_path
        caption_results.append(cap_res)

        # 3. Test Grounding
        print(f"  [3] Region Grounding Query: '{grd_q}'")
        grd_res = run_grounding(img_path, grd_q)
        print(f"      BBox:       {grd_res.get('bbox')} [x1, y1, x2, y2]")
        print(f"      Confidence: {grd_res.get('confidence')}")

        # Assertions for Grounding
        assert grd_res.get("task") == "grounding", f"Expected task 'grounding', got {grd_res.get('task')}"
        bbox = grd_res.get("bbox")
        assert isinstance(bbox, list) and len(bbox) == 4, f"Invalid bbox {bbox}"
        assert all(isinstance(c, int) for c in bbox), "BBox coordinates must be integers"
        assert 0.0 <= grd_res.get("confidence") <= 1.0, f"Confidence {grd_res.get('confidence')} out of bounds"
        assert grd_res.get("image_path") == img_path
        grounding_results.append(grd_res)

    print("\n" + "=" * 80)
    print(" SUMMARY OF EXECUTION RESULTS")
    print("=" * 80)
    print(f"{'Image':<38} | {'Task':<10} | {'Output':<30} | {'Conf':<6}")
    print("-" * 90)

    for v, c, g in zip(vqa_results, caption_results, grounding_results):
        img_name = Path(v['image_path']).name
        ans_trunc = (v['answer'][:27] + '...') if len(v['answer']) > 30 else v['answer']
        cap_trunc = (c['caption'][:27] + '...') if len(c['caption']) > 30 else c['caption']
        bbox_str = str(g['bbox'])

        print(f"{img_name:<38} | {'VQA':<10} | {ans_trunc:<30} | {v['confidence']:<6.2f}")
        print(f"{img_name:<38} | {'Caption':<10} | {cap_trunc:<30} | {c['confidence']:<6.2f}")
        print(f"{img_name:<38} | {'Grounding':<10} | {bbox_str:<30} | {g['confidence']:<6.2f}")
        print("-" * 90)

    print("\n[+] All single-image baseline tools executed successfully with valid schemas and constraints!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
