"""
backend/change_detection_tool.py
================================
Bi-temporal change detection and multitemporal Visual Question Answering (VQA)
tool for SatQuery AI.

Provides:
    run_change_detection(
        image_path_before: str,
        image_path_after: str,
        question: Optional[str] = None
    ) -> dict

Design & Architectural Approach:
---------------------------------
Because M2's adapted Vision-Language Model (Salesforce/blip-vqa-base with LoRA)
is a single-image encoder-decoder model, direct multi-image input is not natively
supported in a single forward pass.

This tool implements a multi-stage comparative analysis pipeline:
1. Radiometric & Structural Change Metric: Evaluates mean absolute pixel difference
   and channel variance between T1 (before) and T2 (after) using NumPy / PIL.
2. Temporal Semantic Inspection: Queries M2's adapted VLM (answer_question) on both
   T1 and T2 images with targeted land cover and feature prompts.
3. Cross-Temporal Synthesis:
   - If a specific question is provided (Change VQA): Evaluates the question across
     both temporal states and synthesizes a direct comparative answer.
   - If no question is provided (General Change Detection): Generates a natural language
     description detailing the transition from Time 1 to Time 2 and flags whether
     meaningful physical change occurred.

Returns:
    {
        "task": "change_detection",
        "description": "...",
        "change_detected": true/false,
        "confidence": 0.0-1.0,
        "before_path": "...",
        "after_path": "..."
    }
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.inference import answer_question


def compute_pixel_change_metric(path_before: Path, path_after: Path) -> Tuple[float, bool]:
    """
    Computes normalized mean absolute pixel difference between two co-registered images.

    Returns:
        (diff_score, is_visually_altered):
            - diff_score (float): Normalized pixel difference in [0.0, 1.0].
            - is_visually_altered (bool): True if difference exceeds change threshold.
    """
    try:
        im1 = Image.open(path_before).convert("RGB")
        im2 = Image.open(path_after).convert("RGB")

        # Resize if dimensions slightly differ
        if im1.size != im2.size:
            im2 = im2.resize(im1.size, Image.Resampling.BILINEAR)

        arr1 = np.array(im1, dtype=np.float32) / 255.0
        arr2 = np.array(im2, dtype=np.float32) / 255.0

        abs_diff = np.abs(arr1 - arr2)
        mean_diff = float(np.mean(abs_diff))

        # Threshold: ~0.04 represents noticeable land cover or structural alterations
        # while absorbing minor sensor illumination or compression noise
        is_altered = mean_diff > 0.04
        return round(mean_diff, 4), is_altered
    except Exception:
        return 0.0, False


def calculate_change_confidence(
    description: str,
    raw_before: str,
    raw_after: str,
    change_detected: bool,
    pixel_diff: float
) -> float:
    """
    Computes a heuristic confidence score in [0.0, 1.0] for change detection results.

    Criteria:
    - 0.00: Missing file, read error, or model failure.
    - 0.35: Explicit uncertainty phrases ("uncertain", "unclear", "cannot tell").
    - 0.90+: High agreement between visual difference metric and semantic descriptions.
    - 0.80 - 0.88: Standard comparative descriptions.
    """
    if not description or not isinstance(description, str):
        return 0.0

    lower_desc = description.lower()
    if lower_desc.startswith("error:") or lower_desc.startswith("inference error:"):
        return 0.0

    uncertainty_tokens = ["uncertain", "unclear", "unknown", "maybe", "not sure", "difficult to determine"]
    if any(t in lower_desc for t in uncertainty_tokens):
        return 0.35

    confidence = 0.85

    # Check for consistency between pixel difference and detected change
    if change_detected and pixel_diff > 0.05:
        confidence += 0.06
    elif not change_detected and pixel_diff < 0.02:
        confidence += 0.07

    # Check for informative answers from VLM
    informative_tokens = {"water", "forest", "urban", "building", "buildings", "trees", "agriculture", "road", "desert", "soil", "cleared", "constructed"}
    words = set(lower_desc.replace(".", "").replace(",", "").split())
    if len(words.intersection(informative_tokens)) >= 2:
        confidence += 0.04

    return round(min(0.96, max(0.20, confidence)), 2)


def run_change_detection(
    image_path_before: str,
    image_path_after: str,
    question: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes bi-temporal change detection or change-based VQA on an image pair.

    Args:
        image_path_before (str): Filesystem path to the pre-change (Time 1) image.
        image_path_after (str): Filesystem path to the post-change (Time 2) image.
        question (str, optional): Specific question inquiring what changed between
            the two passes. If None, produces a comprehensive change description.

    Returns:
        dict: Standardized payload conforming to pipeline specifications:
            - task (str): "change_detection"
            - description (str): Natural language summary of alterations or QA answer.
            - change_detected (bool): Boolean flag indicating presence of change.
            - confidence (float): Heuristic confidence metric in range [0.0, 1.0].
            - before_path (str): Preserved before image path.
            - after_path (str): Preserved after image path.
    """
    # 1. Resolve paths
    p_before = Path(image_path_before)
    if not p_before.is_absolute():
        p_before = (PROJECT_ROOT / p_before).resolve()

    p_after = Path(image_path_after)
    if not p_after.is_absolute():
        p_after = (PROJECT_ROOT / p_after).resolve()

    # Validation
    if not p_before.is_file():
        return {
            "task": "change_detection",
            "description": f"Error: Pre-change image file not found at '{image_path_before}'",
            "change_detected": False,
            "confidence": 0.0,
            "before_path": str(image_path_before),
            "after_path": str(image_path_after),
        }

    if not p_after.is_file():
        return {
            "task": "change_detection",
            "description": f"Error: Post-change image file not found at '{image_path_after}'",
            "change_detected": False,
            "confidence": 0.0,
            "before_path": str(image_path_before),
            "after_path": str(image_path_after),
        }

    # 2. Visual radiometric difference metric
    pixel_diff, is_visually_altered = compute_pixel_change_metric(p_before, p_after)

    # Check for identical files
    if p_before == p_after or pixel_diff == 0.0:
        desc = "No change detected between Time 1 and Time 2. Both observations are identical."
        if question:
            desc = f"No change detected. In response to '{question}': Both images depict the exact same scene with zero observable alterations."
        return {
            "task": "change_detection",
            "description": desc,
            "change_detected": False,
            "confidence": 0.98,
            "before_path": str(image_path_before),
            "after_path": str(image_path_after),
        }

    # 3. Model inference on both images
    # Query land cover / dominant features for T1 and T2
    lc_prompt = "What land cover or surface features are visible in this satellite image?"
    t1_landcover = answer_question(str(p_before), lc_prompt)
    t2_landcover = answer_question(str(p_after), lc_prompt)

    # Check for errors in model output
    if t1_landcover.startswith("Error:") or t1_landcover.startswith("Inference error:"):
        return {
            "task": "change_detection",
            "description": f"Error querying pre-change image: {t1_landcover}",
            "change_detected": False,
            "confidence": 0.0,
            "before_path": str(image_path_before),
            "after_path": str(image_path_after),
        }

    if t2_landcover.startswith("Error:") or t2_landcover.startswith("Inference error:"):
        return {
            "task": "change_detection",
            "description": f"Error querying post-change image: {t2_landcover}",
            "change_detected": False,
            "confidence": 0.0,
            "before_path": str(image_path_before),
            "after_path": str(image_path_after),
        }

    # Clean outputs
    t1_clean = t1_landcover.strip().rstrip(".")
    t2_clean = t2_landcover.strip().rstrip(".")

    # Evaluate semantic change
    semantic_changed = (t1_clean.lower() != t2_clean.lower())
    change_detected = bool(semantic_changed or is_visually_altered)

    # 4. Formulate output depending on whether a question was asked
    if question and question.strip():
        q_clean = question.strip()
        # Query both images with the user's question to evaluate temporal transition
        q_t1 = answer_question(str(p_before), q_clean)
        q_t2 = answer_question(str(p_after), q_clean)

        if not change_detected:
            description = (
                f"No significant change detected between passes. In Time 1, the scene showed {t1_clean}, "
                f"and in Time 2 it remains {t2_clean}. Regarding the query: {q_t2}."
            )
        else:
            # Build synthesized comparative answer
            if q_t1.lower() == q_t2.lower() and not semantic_changed:
                description = (
                    f"Visual alteration was detected (pixel variance: {pixel_diff:.3f}). "
                    f"Both passes primarily show {t1_clean}, with query assessment: {q_t2}."
                )
            else:
                description = (
                    f"Change detected: Prior observation (Time 1) depicted {t1_clean} ({q_t1}), "
                    f"whereas subsequent observation (Time 2) depicts {t2_clean} ({q_t2})."
                )
    else:
        # General change description
        if not change_detected:
            description = (
                f"No significant land cover change detected between Time 1 and Time 2. "
                f"Both observations consistently depict {t1_clean}."
            )
        else:
            if semantic_changed:
                description = (
                    f"Significant land cover change detected: Time 1 depicted {t1_clean}, "
                    f"which transitioned to {t2_clean} at Time 2."
                )
            else:
                description = (
                    f"Surface alteration detected: While primary land cover category remains {t1_clean}, "
                    f"structural and textural variance (metric: {pixel_diff:.3f}) indicates localized activity between passes."
                )

    confidence = calculate_change_confidence(
        description=description,
        raw_before=t1_landcover,
        raw_after=t2_landcover,
        change_detected=change_detected,
        pixel_diff=pixel_diff
    )

    return {
        "task": "change_detection",
        "description": description,
        "change_detected": change_detected,
        "confidence": confidence,
        "before_path": str(image_path_before),
        "after_path": str(image_path_after),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SatQuery AI - Bi-Temporal Change Detection Tool")
    parser.add_argument("--before", type=str, required=True, help="Path to pre-change (T1) image")
    parser.add_argument("--after", type=str, required=True, help="Path to post-change (T2) image")
    parser.add_argument("--question", type=str, default=None, help="Optional change question")
    args = parser.parse_args()

    result = run_change_detection(args.before, args.after, question=args.question)
    print(result)
