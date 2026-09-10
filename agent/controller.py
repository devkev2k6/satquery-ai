"""
agent/controller.py
===================
Agentic Model & Tool Controller for SatQuery AI.

Provides:
    process_query(
        query: str,
        image_paths: list,
        image_types: list = None
    ) -> dict

Fulfills the core problem statement requirement for "Agentic Model and Tool Orchestration":
    - Interprets natural-language user queries.
    - Inspects image quantities and sensor modalities (Optical vs SAR).
    - Classifies the user intent into one of five operational pipelines:
        1. "vqa"              -> backend.vqa_tool.run_vqa
        2. "captioning"        -> backend.captioning_tool.run_captioning
        3. "grounding"         -> backend.grounding_tool.run_grounding
        4. "change_detection"  -> backend.change_detection_tool.run_change_detection
        5. "fusion"            -> backend.fusion_tool.run_optical_sar_fusion
    - Validates inputs before tool execution.
    - Executes the corresponding tool(s).
    - Produces a transparent, auditable execution summary detailing selected task,
      tool used, parameters, and UTC timestamp.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.task_classifier import classify_task
from agent.input_validator import validate_input, SAR_MODALITIES
from backend.vqa_tool import run_vqa
from backend.captioning_tool import run_captioning
from backend.grounding_tool import run_grounding
from backend.change_detection_tool import run_change_detection
from backend.fusion_tool import run_optical_sar_fusion


def _resolve_fusion_modalities(
    image_paths: List[str],
    image_types: Optional[List[str]] = None
) -> Tuple[str, str]:
    """
    Identifies which image is optical and which is SAR.

    Returns:
        (optical_path, sar_path)
    """
    if image_types and len(image_types) == 2:
        type0 = str(image_types[0]).strip().lower()
        if any(m in type0 for m in SAR_MODALITIES):
            return image_paths[1], image_paths[0]
        else:
            return image_paths[0], image_paths[1]

    # Fallback to filename heuristics
    p0_lower = str(image_paths[0]).lower()
    p1_lower = str(image_paths[1]).lower()

    if ("sar" in p0_lower or "radar" in p0_lower) and not ("sar" in p1_lower or "radar" in p1_lower):
        return image_paths[1], image_paths[0]

    return image_paths[0], image_paths[1]


def _evaluate_execution_success(tool_result: Dict[str, Any]) -> bool:
    """
    Determines if the backend tool executed successfully based on confidence and errors.
    """
    if not isinstance(tool_result, dict):
        return False

    # If confidence is 0.0, an execution or file error occurred
    if tool_result.get("confidence", 1.0) == 0.0:
        return False

    # Check for explicit error strings in common result keys
    for key in ["answer", "caption", "description"]:
        val = str(tool_result.get(key, "")).strip().lower()
        if val.startswith("error:") or val.startswith("inference error:"):
            return False

    return True


def process_query(
    query: Optional[str],
    image_paths: List[str],
    image_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main orchestrator entry point for natural language satellite image analysis.

    Args:
        query (str, optional): Natural language query, question, or instruction.
        image_paths (list): List of file paths to remote sensing images.
        image_types (list, optional): List of image modalities (e.g. ['optical', 'sar']).

    Returns:
        dict: Standardized orchestration payload containing:
            - result (dict): Output payload from the executed backend tool.
            - execution_summary (dict): Auditable trace containing:
                - task (str): Selected task ("vqa", "captioning", "grounding", "change_detection", "fusion").
                - tool_used (str): Name of tool function called.
                - parameters (dict): Key input parameters dispatched to the tool.
                - timestamp (str): ISO-8601 UTC timestamp of execution.
            - success (bool): True if input was valid and tool executed without error.
    """
    timestamp_str = datetime.now(timezone.utc).isoformat()
    num_images = len(image_paths) if isinstance(image_paths, (list, tuple)) else 0

    # 1. Classify Task
    task = classify_task(query, num_images, image_types)

    # 2. Validate Inputs
    validation = validate_input(query, image_paths, image_types, task=task)
    if not validation["valid"]:
        return {
            "result": {
                "error": validation["error"],
                "task": task
            },
            "execution_summary": {
                "task": task,
                "tool_used": None,
                "parameters": {
                    "query": query,
                    "image_paths": image_paths,
                    "image_types": image_types
                },
                "timestamp": timestamp_str
            },
            "success": False
        }

    # 3. Dispatch to Appropriate Backend Tool
    tool_used = None
    parameters = {}
    tool_result = {}

    if task == "captioning":
        tool_used = "run_captioning"
        prompt_arg = query.strip() if query and query.strip() else None
        parameters = {
            "image_path": image_paths[0],
            "prompt": prompt_arg
        }
        tool_result = run_captioning(
            image_path=image_paths[0],
            prompt=prompt_arg
        )

    elif task == "grounding":
        tool_used = "run_grounding"
        parameters = {
            "image_path": image_paths[0],
            "query": query
        }
        tool_result = run_grounding(
            image_path=image_paths[0],
            query=query
        )

    elif task == "vqa":
        tool_used = "run_vqa"
        parameters = {
            "image_path": image_paths[0],
            "question": query
        }
        tool_result = run_vqa(
            image_path=image_paths[0],
            question=query
        )

    elif task == "change_detection":
        tool_used = "run_change_detection"
        parameters = {
            "image_path_before": image_paths[0],
            "image_path_after": image_paths[1],
            "question": query
        }
        tool_result = run_change_detection(
            image_path_before=image_paths[0],
            image_path_after=image_paths[1],
            question=query
        )

    elif task == "fusion":
        tool_used = "run_optical_sar_fusion"
        opt_path, sar_path = _resolve_fusion_modalities(image_paths, image_types)
        parameters = {
            "optical_path": opt_path,
            "sar_path": sar_path,
            "question": query
        }
        tool_result = run_optical_sar_fusion(
            optical_path=opt_path,
            sar_path=sar_path,
            question=query
        )

    else:
        # Unexpected fallback
        return {
            "result": {"error": f"Unknown task '{task}'"},
            "execution_summary": {
                "task": task,
                "tool_used": None,
                "parameters": {"query": query, "image_paths": image_paths},
                "timestamp": timestamp_str
            },
            "success": False
        }

    # 4. Success Evaluation
    success = _evaluate_execution_success(tool_result)

    # 5. Return Full Auditable Payload
    return {
        "result": tool_result,
        "execution_summary": {
            "task": task,
            "tool_used": tool_used,
            "parameters": parameters,
            "timestamp": timestamp_str
        },
        "success": success
    }
