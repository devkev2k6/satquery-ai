"""
tests/test_full_demo.py
=======================
Full End-to-End Demo Verification Suite for SatQuery AI.

Simulates the exact execution flow of the frontend Streamlit dashboard:
1. Dispatches the 5 official problem statement queries through process_query().
2. Verifies successful agent orchestration, task classification, and tool execution.
3. Automatically compiles and validates a downloadable PDF Intelligence Report for each query.
4. Confirms zero failures across the entire system before the live competition demo.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import process_query
from frontend.report_generator import generate_pdf_report


def print_banner(text: str):
    print("\n" + "=" * 95)
    print(f" {text}")
    print("=" * 95)


def main():
    print_banner("SATQUERY AI - FINAL FULL DEMO END-TO-END VERIFICATION")
    print("Simulating User Interactions: Ingestion -> Orchestration -> Execution -> PDF Reporting\n")

    demo_cases = [
        {
            "id": "DEMO-01 (SCENE CAPTIONING)",
            "query": "Describe the land-cover and major objects visible in this image.",
            "image_paths": ["data/sample/optical/sample_optical_001.png"],
            "image_types": ["optical"],
            "expected_task": "captioning",
            "expected_tool": "run_captioning",
        },
        {
            "id": "DEMO-02 (TEXT-GUIDED GROUNDING)",
            "query": "Highlight the water body referred to in the query.",
            "image_paths": ["data/sample/optical/sample_optical_002.png"],
            "image_types": ["optical"],
            "expected_task": "grounding",
            "expected_tool": "run_grounding",
        },
        {
            "id": "DEMO-03 (BI-TEMPORAL CHANGE DETECTION)",
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
            "id": "DEMO-04 (OPTICAL + SAR CROSS-MODAL FUSION)",
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
            "id": "DEMO-05 (CHANGE TREND / CDVQA)",
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

    all_passed = True
    summary_records = []

    for idx, case in enumerate(demo_cases, 1):
        print(f"\n--- [{idx}/5] Executing {case['id']} ---")
        print(f"Query:       \"{case['query']}\"")
        print(f"Images:      {case['image_paths']}")
        print(f"Modalities:  {case['image_types']}")

        # 1. Dispatch through Agent Controller (exact same entry point as Streamlit app)
        result = process_query(
            query=case["query"],
            image_paths=case["image_paths"],
            image_types=case["image_types"]
        )

        tool_result = result.get("result", {})
        exec_summary = result.get("execution_summary", {})
        success = result.get("success", False)

        task_selected = exec_summary.get("task")
        tool_called = exec_summary.get("tool_used")

        # 2. Generate PDF Report (exact same mechanism as UI "Download Report" button)
        report_dest = PROJECT_ROOT / "outputs" / "reports" / f"demo_case_{idx}_{task_selected}.pdf"
        pdf_path = generate_pdf_report(case["query"], result, output_path=str(report_dest))
        pdf_file = Path(pdf_path)
        pdf_valid = pdf_file.is_file() and pdf_file.stat().st_size > 1000

        # Assertions
        task_match = (task_selected == case["expected_task"])
        tool_match = (tool_called == case["expected_tool"])
        case_passed = success and task_match and tool_match and pdf_valid

        if not case_passed:
            all_passed = False
            print(f"[-] FAILED {case['id']}: success={success}, task_match={task_match}, tool_match={tool_match}, pdf_valid={pdf_valid}")
        else:
            print(f"[+] PASSED: Routed to '{task_selected}' via '{tool_called}'")
            print(f"    Confidence:  {tool_result.get('confidence')}")
            print(f"    PDF Report:  {pdf_path} ({pdf_file.stat().st_size} bytes)")

        # Extract brief narrative for summary table
        narrative = tool_result.get("caption") or tool_result.get("description") or tool_result.get("answer") or ""
        summary_records.append({
            "id": case["id"].split()[0],
            "task": task_selected,
            "tool": tool_called,
            "conf": tool_result.get("confidence", 0.0),
            "pdf_size": pdf_file.stat().st_size if pdf_valid else 0,
            "status": "PASS" if case_passed else "FAIL",
            "narrative": narrative[:55] + "..." if len(narrative) > 55 else narrative
        })

    # Summary Table
    print_banner("FINAL FULL DEMO SUMMARY REPORT")
    print(f"{'Case ID':<10} | {'Task':<17} | {'Tool Used':<24} | {'Conf':<5} | {'PDF Size':<9} | {'Status':<6} | {'Findings Preview'}")
    print("-" * 115)
    for r in summary_records:
        print(f"{r['id']:<10} | {r['task']:<17} | {r['tool']:<24} | {r['conf']:<5.2f} | {r['pdf_size']:<6} B | {r['status']:<6} | {r['narrative']}")

    print_banner("FINAL DEMO VERDICT")
    if all_passed:
        print("[+] 100% FULL DEMO SUCCESS: All 5 representative benchmark queries executed end-to-end!")
        print("[+] All 5 downloadable PDF intelligence reports generated and validated successfully.")
        print("[+] SatQuery AI is verified and ready for live presentation.\n")
    else:
        print("[-] Verification failed on one or more demo cases. Review trace above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
