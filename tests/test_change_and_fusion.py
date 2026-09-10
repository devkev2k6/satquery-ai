"""
tests/test_change_and_fusion.py
===============================
Automated verification test suite for SatQuery AI Phase 4 tools:
- Bi-Temporal Change Detection Tool: run_change_detection
- Cross-Modal Optical+SAR Fusion Tool: run_optical_sar_fusion

Validates:
1. Exact dictionary schema contracts and return keys.
2. General change description (question=None) and change VQA queries.
3. Identical image pair verification (asserting change_detected is False).
4. Real CDVQA multitemporal benchmark pairs and synthetic pairs.
5. Cross-modal Optical + SAR fusion with and without targeted user queries.
6. Robustness on missing files and invalid paths.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.change_detection_tool import run_change_detection
from backend.fusion_tool import run_optical_sar_fusion


def main():
    print("=" * 85)
    print(" SatQuery AI - Phase 4 Multi-Image Tools Verification (Change Detection & Fusion)")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # PART 1: CHANGE DETECTION TESTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [SECTION 1] TESTING BI-TEMPORAL CHANGE DETECTION TOOL (run_change_detection)")
    print("=" * 85)

    change_test_cases = [
        {
            "id": "CD-01 (Synthetic Pair - General)",
            "before": "data/sample/pairs/sample_pair_001_t1.png",
            "after": "data/sample/pairs/sample_pair_001_t2.png",
            "question": None,
            "expect_change": True,
        },
        {
            "id": "CD-02 (Synthetic Pair - Change VQA)",
            "before": "data/sample/pairs/sample_pair_001_t1.png",
            "after": "data/sample/pairs/sample_pair_001_t2.png",
            "question": "What environmental or infrastructure change occurred between Time 1 and Time 2?",
            "expect_change": True,
        },
        {
            "id": "CD-03 (Synthetic Pair - Deforestation)",
            "before": "data/sample/pairs/sample_pair_002_t1.png",
            "after": "data/sample/pairs/sample_pair_002_t2.png",
            "question": "Was there deforestation or land clearance between Time 1 and Time 2?",
            "expect_change": True,
        },
        {
            "id": "CD-04 (Identical Pair - Negative Control)",
            "before": "data/sample/pairs/sample_pair_001_t1.png",
            "after": "data/sample/pairs/sample_pair_001_t1.png",
            "question": None,
            "expect_change": False,
        },
        {
            "id": "CD-05 (CDVQA Benchmark Pair)",
            "before": "data/cdvqa/image_pairs/cdvqa_001_t1.png",
            "after": "data/cdvqa/image_pairs/cdvqa_001_t2.png",
            "question": "Did new building structures appear between Time 1 and Time 2?",
            "expect_change": True,
        },
    ]

    change_results = []
    for idx, tc in enumerate(change_test_cases, 1):
        print(f"\n[{idx}/{len(change_test_cases)}] Test Case: {tc['id']}")
        print(f"    Before:   {tc['before']}")
        print(f"    After:    {tc['after']}")
        print(f"    Question: {tc['question']}")

        res = run_change_detection(tc["before"], tc["after"], question=tc["question"])

        print(f"    -> Task:            {res.get('task')}")
        print(f"    -> Change Detected: {res.get('change_detected')}")
        print(f"    -> Confidence:      {res.get('confidence')}")
        print(f"    -> Description:     {res.get('description')}")

        # Contract assertions
        assert "task" in res and res["task"] == "change_detection", f"Invalid task: {res.get('task')}"
        assert "description" in res and isinstance(res["description"], str) and len(res["description"]) > 0
        assert "change_detected" in res and isinstance(res["change_detected"], bool)
        assert "confidence" in res and isinstance(res["confidence"], float) and 0.0 <= res["confidence"] <= 1.0
        assert "before_path" in res and res["before_path"] == tc["before"]
        assert "after_path" in res and res["after_path"] == tc["after"]

        if tc["expect_change"] is False:
            assert res["change_detected"] is False, "Expected change_detected=False for identical images"

        change_results.append((tc, res))

    # Error handling test for change detection
    print("\n[*] Testing Change Detection Error Handling (Missing File)...")
    err_res = run_change_detection("non_existent_before.png", "data/sample/pairs/sample_pair_001_t2.png")
    assert err_res["confidence"] == 0.0
    assert err_res["change_detected"] is False
    assert err_res["description"].startswith("Error:")
    print("    [+] Correctly handled missing input image with confidence=0.0 and error message.")

    # -------------------------------------------------------------------------
    # PART 2: OPTICAL + SAR FUSION TESTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [SECTION 2] TESTING OPTICAL + SAR FUSION TOOL (run_optical_sar_fusion)")
    print("=" * 85)

    fusion_test_cases = [
        {
            "id": "FUS-01 (Sample Optical + SAR - General)",
            "optical": "data/sample/optical/sample_optical_001.png",
            "sar": "data/sample/sar/sample_sar_001.png",
            "question": None,
        },
        {
            "id": "FUS-02 (Sample Optical + SAR - Built-up & Water Query)",
            "optical": "data/sample/optical/sample_optical_003.png",
            "sar": "data/sample/sar/sample_sar_003.png",
            "question": "Identify built-up and water-covered regions using both images together.",
        },
        {
            "id": "FUS-03 (Sample Optical + SAR - Surface Roughness Query)",
            "optical": "data/sample/optical/sample_optical_002.png",
            "sar": "data/sample/sar/sample_sar_002.png",
            "question": "Is there a calm water body visible with low radar backscatter?",
        },
        {
            "id": "FUS-04 (BigEarthNet Optical + SAR Pair)",
            "optical": "data/bigearthnet/images/S2A_MSIL2A_20170717T113321_01.png",
            "sar": "data/bigearthnet/images/S2A_MSIL2A_20170717T113321_10.png",
            "question": "What land cover classes and radar backscatter signatures are present?",
        },
    ]

    fusion_results = []
    for idx, tc in enumerate(fusion_test_cases, 1):
        print(f"\n[{idx}/{len(fusion_test_cases)}] Test Case: {tc['id']}")
        print(f"    Optical:  {tc['optical']}")
        print(f"    SAR:      {tc['sar']}")
        print(f"    Question: {tc['question']}")

        res = run_optical_sar_fusion(tc["optical"], tc["sar"], question=tc["question"])

        print(f"    -> Task:       {res.get('task')}")
        print(f"    -> Confidence: {res.get('confidence')}")
        print(f"    -> Answer:     {res.get('answer')}")

        # Contract assertions
        assert "task" in res and res["task"] == "fusion", f"Invalid task: {res.get('task')}"
        assert "answer" in res and isinstance(res["answer"], str) and len(res["answer"]) > 0
        assert "confidence" in res and isinstance(res["confidence"], float) and 0.0 <= res["confidence"] <= 1.0
        assert "optical_path" in res and res["optical_path"] == tc["optical"]
        assert "sar_path" in res and res["sar_path"] == tc["sar"]

        fusion_results.append((tc, res))

    # Error handling test for optical-SAR fusion
    print("\n[*] Testing Optical-SAR Fusion Error Handling (Missing File)...")
    err_fus = run_optical_sar_fusion("non_existent_optical.png", "data/sample/sar/sample_sar_001.png")
    assert err_fus["confidence"] == 0.0
    assert err_fus["answer"].startswith("Error:")
    print("    [+] Correctly handled missing optical file with confidence=0.0 and error message.")

    # -------------------------------------------------------------------------
    # PART 3: SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print(" SUMMARY OF PHASE 4 MULTI-IMAGE EXECUTION RESULTS")
    print("=" * 95)
    print(f"{'Test Case ID':<35} | {'Task':<16} | {'Change?':<8} | {'Conf':<6} | {'Status':<6}")
    print("-" * 95)

    for tc, r in change_results:
        ch_str = str(r['change_detected'])
        print(f"{tc['id']:<35} | {r['task']:<16} | {ch_str:<8} | {r['confidence']:<6.2f} | PASS")

    for tc, r in fusion_results:
        print(f"{tc['id']:<35} | {r['task']:<16} | {'N/A':<8} | {r['confidence']:<6.2f} | PASS")

    print("-" * 95)
    print("\n[+] All Phase 4 multi-image test cases PASSED strict contract and boundary checks!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
