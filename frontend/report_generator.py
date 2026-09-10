"""
frontend/report_generator.py
============================
Automated PDF Intelligence Report Generator for SatQuery AI.

Provides:
    generate_pdf_report(
        query: str,
        result: dict,
        output_path: Optional[str] = None
    ) -> str

Generates a publication-grade Earth Observation intelligence PDF report summarizing:
- Natural-language query asked
- Agent task classification
- Backend tool and model invoked
- Extracted semantic answer / change narrative / grounding bounding box
- Sensor modalities analyzed
- Confidence metrics
- Auditable execution parameters & ISO-8601 timestamp
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fpdf import FPDF
from fpdf.enums import XPos, YPos


class SatQueryPDF(FPDF):
    """Custom FPDF layout for SatQuery AI Intelligence Reports."""

    def header(self):
        # Header banner
        self.set_fill_color(24, 43, 73)  # Deep satellite blue
        self.rect(0, 0, 210, 24, "F")

        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(12, 5)
        self.cell(0, 7, "SATQUERY AI - EARTH OBSERVATION INTELLIGENCE REPORT", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("Helvetica", "I", 8)
        self.set_text_color(180, 205, 237)
        self.set_x(12)
        self.cell(0, 5, "Multimodal Satellite Question Answering, Grounding & Cross-Modal Fusion", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(28)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"SatQuery AI Autonomous Agent Pipeline  |  Page {self.page_no()}/{{nb}}", align="C")


def generate_pdf_report(
    query: str,
    result: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    Generates a structured PDF intelligence report from query execution results.

    Args:
        query (str): User natural language question or instruction.
        result (dict): Complete payload returned by process_query().
        output_path (str, optional): Custom file path to save PDF.

    Returns:
        str: Absolute file path to the generated PDF.
    """
    # 1. Resolve output destination
    if output_path is None:
        reports_dir = PROJECT_ROOT / "outputs" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = reports_dir / f"satquery_report_{timestamp_slug}.pdf"
    else:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

    # 2. Extract result metadata
    tool_result = result.get("result", {})
    exec_summary = result.get("execution_summary", {})
    success = result.get("success", False)

    task = str(exec_summary.get("task", tool_result.get("task", "Unknown"))).upper()
    tool_used = str(exec_summary.get("tool_used", "N/A"))
    timestamp = str(exec_summary.get("timestamp", datetime.now(timezone.utc).isoformat()))
    params = exec_summary.get("parameters", {})
    confidence = tool_result.get("confidence", 0.0)

    # 3. Initialize PDF
    pdf = SatQueryPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # SECTION 1: EXECUTIVE SUMMARY TABLE
    pdf.set_text_color(33, 37, 41)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "1. Executive Query & Routing Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    label_w = 48
    val_w = pdf.epw - label_w

    metadata_rows = [
        ("User Query:", str(query)),
        ("Orchestrated Task:", f"{task} (Autonomous Agent Routing)"),
        ("Tool / Model Dispatched:", str(tool_used)),
        ("Execution Status:", "SUCCESSFUL" if success else "FAILED / VALIDATION ERROR"),
        ("Confidence Score:", f"{float(confidence):.2f}" if confidence else "N/A"),
        ("Timestamp (UTC):", str(timestamp)),
    ]

    for label, val in metadata_rows:
        pdf.set_fill_color(245, 247, 250)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(label_w, 6.5, label, border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.multi_cell(val_w, 6.5, val, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(4)

    # SECTION 2: INTELLIGENCE ANALYSIS & CORE FINDINGS
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Intelligence Analysis & Extracted Findings", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    primary_text = ""
    if "caption" in tool_result:
        primary_text = tool_result["caption"]
    elif "description" in tool_result:
        primary_text = tool_result["description"]
    elif "answer" in tool_result:
        primary_text = tool_result["answer"]
    elif "error" in tool_result:
        primary_text = f"Error: {tool_result['error']}"
    else:
        primary_text = "No narrative output generated."

    pdf.set_fill_color(238, 242, 248)
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.multi_cell(pdf.epw, 6, primary_text, border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(2)
    if "change_detected" in tool_result:
        pdf.set_font("Helvetica", "B", 8.5)
        cd_val = "YES (Significant Surface Alteration Detected)" if tool_result["change_detected"] else "NO (No Significant Change)"
        pdf.cell(50, 5.5, "Bi-Temporal Change Status:", border=0)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(0, 5.5, cd_val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if "bbox" in tool_result:
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(50, 5.5, "Region Grounding [x1, y1, x2, y2]:", border=0)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(0, 5.5, str(tool_result["bbox"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(4)

    # SECTION 3: SENSOR & IMAGE ASSETS ANALYZED
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "3. Remote Sensing Imagery Assets", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    image_paths = []
    if isinstance(params, dict):
        raw_paths = params.get("image_paths", [])
        if isinstance(raw_paths, (list, tuple)):
            image_paths.extend(raw_paths)
        for k in ["image_path", "optical_path", "sar_path", "before_path", "after_path", "image_path_before", "image_path_after"]:
            if k in params and params[k] not in image_paths:
                image_paths.append(params[k])

    if not image_paths and isinstance(tool_result, dict):
        for k in ["image_path", "optical_path", "sar_path", "before_path", "after_path"]:
            if k in tool_result and tool_result[k] not in image_paths:
                image_paths.append(tool_result[k])

    pdf.set_font("Helvetica", "", 8.5)
    if image_paths:
        for idx, p in enumerate(image_paths, 1):
            pdf.cell(10, 5.5, f"[{idx}]", border=0)
            pdf.multi_cell(pdf.epw - 10, 5.5, f"Path: {p}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.cell(0, 5.5, "No explicit file paths recorded in execution context.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(4)

    # SECTION 4: TECHNICAL AUDIT TRAIL
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "4. Technical Audit & Execution Trace", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(70, 80, 95)
    pdf.set_fill_color(250, 250, 250)

    audit_info = (
        f"Pipeline Version: SatQuery AI v1.0 (6-Person Multimodal System)\n"
        f"Classification Method: Explainable Rule-Based Deterministic Router\n"
        f"Vision-Language Model: Salesforce/blip-vqa-base with Low-Rank Adaptation (LoRA)\n"
        f"Parameters Dispatched: {str(params)}\n"
        f"Raw Backend Payload: {str(tool_result)}"
    )
    pdf.multi_cell(pdf.epw, 4.5, audit_info, border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 4. Save PDF to disk
    pdf.output(str(out_file))
    return str(out_file)


if __name__ == "__main__":
    dummy_query = "What changed between these two dates, and where did the change occur?"
    dummy_result = {
        "result": {
            "task": "change_detection",
            "description": "Visual alteration was detected. Surface variance indicates infrastructure expansion.",
            "change_detected": True,
            "confidence": 0.91,
            "before_path": "data/sample/pairs/sample_pair_001_t1.png",
            "after_path": "data/sample/pairs/sample_pair_001_t2.png"
        },
        "execution_summary": {
            "task": "change_detection",
            "tool_used": "run_change_detection",
            "parameters": {
                "image_paths": ["data/sample/pairs/sample_pair_001_t1.png", "data/sample/pairs/sample_pair_001_t2.png"],
                "query": dummy_query
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "success": True
    }
    saved_path = generate_pdf_report(dummy_query, dummy_result)
    print(f"[+] Generated self-test report successfully: {saved_path}")
