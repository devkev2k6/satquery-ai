"""
tests/test_agent_controller.py
==============================
Comprehensive Verification Suite for SatQuery AI Agent Controller.

Tests:
1. ALL FIVE representative example queries from the problem statement:
   - Query 1 (Captioning): "Describe the land-cover and major objects visible in this image."
   - Query 2 (Grounding): "Highlight the water body referred to in the query."
   - Query 3 (Change Detection): "What changed between these two dates, and where did the change occur?"
   - Query 4 (Cross-Modal Fusion): "Use the optical and SAR images together to identify built-up and water-covered regions."
   - Query 5 (Change Detection VQA): "Has the built-up area increased, decreased, or remained unchanged?"
2. Standard Single-Image Visual Question Answering (VQA).
3. Input validation failure modes (image count mismatch, invalid modality, missing files).

Outputs:
- Full formatted JSON payloads including the auditable `execution_summary`.
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import process_query


def print_header(title: str):
    print("\n" + "=" * 90)
    print(f" {title}")
    print("=" * 90)


def print_case_result(case_id: str, query: str, res: dict):
    print(f"\n--- [{case_id}] ---")
    print(f"Query: \"{query}\"")
    print(f"Success: {res.get('success')}")
    print("Execution Summary:")
    print(json.dumps(res.get("execution_summary"), indent=2))
    print("Result Payload:")
    print(json.dumps(res.get("result"), indent=2))


def main():
    print_header("SATQUERY AI - AGENT CONTROLLER VERIFICATION SUITE")
    print("Validating End-to-End Task Routing, Tool Execution, and Execution Summaries\n")

    all_passed = True

    # =========================================================================
    # PART 1: THE FIVE CORE REPRESENTATIVE QUERIES
    # =========================================================================
    print_header("PART 1: TESTING THE 5 REPRESENTATIVE PROBLEM STATEMENT QUERIES")

    test_cases = [
        {
            "id": "QUERY 1 (CAPTIONING)",
            "query": "Describe the land-cover and major objects visible in this image.",
            "image_paths": ["data/sample/optical/sample_optical_001.png"],
            "image_types": ["optical"],
            "expected_task": "captioning",
            "expected_tool": "run_captioning",
        },
        {
            "id": "QUERY 2 (GROUNDING)",
            "query": "Highlight the water body referred to in the query.",
            "image_paths": ["data/sample/optical/sample_optical_002.png"],
            "image_types": ["optical"],
            "expected_task": "grounding",
            "expected_tool": "run_grounding",
        },
        {
            "id": "QUERY 3 (CHANGE DETECTION)",
            "query": "What changed between these two dates, and where did the change occur?",
            "image_paths": [
                "data/sample/pairs/sample_pair_001_t1.png",
                "data/sample/pairs/sample_pair_001_t2.png"
            ],
            "image_types": ["optical", "optical"],
            "expected_task": "change_detection",
            "expected_tool": "run_change_detection",
        },
        {
            "id": "QUERY 4 (CROSS-MODAL FUSION)",
            "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "image_paths": [
                "data/sample/optical/sample_optical_003.png",
                "data/sample/sar/sample_sar_003.png"
            ],
            "image_types": ["optical", "sar"],
            "expected_task": "fusion",
            "expected_tool": "run_optical_sar_fusion",
        },
        {
            "id": "QUERY 5 (CHANGE DETECTION VQA)",
            "query": "Has the built-up area increased, decreased, or remained unchanged?",
            "image_paths": [
                "data/sample/pairs/sample_pair_001_t1.png",
                "data/sample/pairs/sample_pair_001_t2.png"
            ],
            "image_types": ["optical", "optical"],
            "expected_task": "change_detection",
            "expected_tool": "run_change_detection",
        },
    ]

    for tc in test_cases:
        res = process_query(
            query=tc["query"],
            image_paths=tc["image_paths"],
            image_types=tc["image_types"]
        )

        print_case_result(tc["id"], tc["query"], res)

        # Verification asserts
        summary = res.get("execution_summary", {})
        task_match = summary.get("task") == tc["expected_task"]
        tool_match = summary.get("tool_used") == tc["expected_tool"]
        success = res.get("success") is True
        has_timestamp = bool(summary.get("timestamp"))

        case_ok = task_match and tool_match and success and has_timestamp
        if not case_ok:
            all_passed = False
            print(f"[-] FAILED {tc['id']}: task_match={task_match}, tool_match={tool_match}, success={success}")
        else:
            print(f"[+] PASSED {tc['id']} -> Routed to '{summary.get('task')}' via '{summary.get('tool_used')}'")

    # =========================================================================
    # PART 2: SINGLE-IMAGE VQA
    # =========================================================================
    print_header("PART 2: TESTING STANDARD SINGLE-IMAGE VQA")

    vqa_case = {
        "id": "QUERY 6 (SINGLE-IMAGE VQA)",
        "query": "What terrain type or surface features are present in this satellite image?",
        "image_paths": ["data/sample/optical/sample_optical_001.png"],
        "image_types": ["optical"],
        "expected_task": "vqa",
        "expected_tool": "run_vqa",
    }

    res_vqa = process_query(
        query=vqa_case["query"],
        image_paths=vqa_case["image_paths"],
        image_types=vqa_case["image_types"]
    )
    print_case_result(vqa_case["id"], vqa_case["query"], res_vqa)

    vqa_summary = res_vqa.get("execution_summary", {})
    if (
        vqa_summary.get("task") == vqa_case["expected_task"]
        and vqa_summary.get("tool_used") == vqa_case["expected_tool"]
        and res_vqa.get("success") is True
    ):
        print(f"[+] PASSED {vqa_case['id']} -> Routed to 'vqa' via 'run_vqa'")
    else:
        all_passed = False
        print(f"[-] FAILED {vqa_case['id']}")

    # =========================================================================
    # PART 3: INPUT VALIDATION AND ERROR HANDLING
    # =========================================================================
    print_header("PART 3: TESTING INPUT VALIDATION & ERROR HANDLING")

    validation_cases = [
        {
            "id": "VAL-01 (Change Detection with only 1 image)",
            "query": "What changed between these two dates?",
            "image_paths": ["data/sample/optical/sample_optical_001.png"],
            "image_types": ["optical"],
            "expect_error": True,
        },
        {
            "id": "VAL-02 (Fusion with mismatched modalities)",
            "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "image_paths": [
                "data/sample/optical/sample_optical_001.png",
                "data/sample/optical/sample_optical_002.png"
            ],
            "image_types": ["optical", "optical"],
            "expect_error": True,
        },
        {
            "id": "VAL-03 (Missing image file)",
            "query": "Describe this image.",
            "image_paths": ["data/sample/optical/non_existent_file_999.png"],
            "image_types": ["optical"],
            "expect_error": True,
        },
    ]

    for vc in validation_cases:
        res = process_query(
            query=vc["query"],
            image_paths=vc["image_paths"],
            image_types=vc["image_types"]
        )

        print_case_result(vc["id"], vc["query"], res)

        is_error = (res.get("success") is False) and ("error" in res.get("result", {}))
        if is_error:
            print(f"[+] PASSED {vc['id']} -> Correctly caught validation error: {res['result']['error']}")
        else:
            all_passed = False
            print(f"[-] FAILED {vc['id']} -> Validation error was not caught.")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("FINAL VERIFICATION VERDICT")
    if all_passed:
        print("[+] ALL AGENT CONTROLLER TESTS PASSED (100% SUCCESS)")
        print("[+] All 5 problem statement queries correctly routed and executed.")
        print("[+] Input validation and error summary contracts fully satisfied.")
    else:
        print("[-] SOME TESTS FAILED. Please review the trace output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
