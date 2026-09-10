"""
backend/vqa_tool.py
===================
Visual Question Answering (VQA) baseline tool for SatQuery AI.

Provides:
    run_vqa(image_path: str, question: str) -> dict

Builds on top of M2's adapted Vision-Language Model in models/inference.py.
Computes a heuristic confidence score based on answer certainty phrasing,
brevity, and error detection.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.inference import answer_question


def calculate_vqa_confidence(answer: str) -> float:
    """
    Computes a heuristic confidence score for VQA answers in [0.0, 1.0].

    Heuristic Criteria:
    - 0.00: Model or runtime errors (e.g., missing file, exception).
    - 0.10: Uninformative default fallback ("No discernible feature detected").
    - 0.35: Uncertainty phrases ("uncertain", "unclear", "unknown", "maybe", "not sure").
    - 0.90: Concise categorical answers (1-4 words like "water", "forest", "yes", "no"),
            characteristic of high-certainty VLM predictions.
    - 0.82: Standard descriptive multi-word answers.
    - 0.70: Excessively verbose answers (>25 words) which tend to hallucinate details.
    """
    if not answer or not isinstance(answer, str):
        return 0.0

    cleaned = answer.strip()
    lower_ans = cleaned.lower()

    # Error conditions
    if lower_ans.startswith("error:") or lower_ans.startswith("inference error:"):
        return 0.0

    if lower_ans == "no discernible feature detected.":
        return 0.10

    # Uncertainty markers
    uncertainty_tokens = ["uncertain", "unclear", "unknown", "maybe", "not sure", "cannot tell", "difficult to determine"]
    if any(token in lower_ans for token in uncertainty_tokens):
        return 0.35

    # Length-based heuristic
    words = cleaned.split()
    word_count = len(words)

    if 1 <= word_count <= 4:
        confidence = 0.90
    elif 5 <= word_count <= 15:
        confidence = 0.85
    elif 16 <= word_count <= 25:
        confidence = 0.80
    else:
        confidence = 0.70

    # Bonus for standard remote sensing categorical presence terms
    rs_high_confidence_terms = {"water", "forest", "trees", "urban", "vegetation", "agriculture", "airport", "plane", "yes", "no", "river", "bare soil", "road"}
    if lower_ans in rs_high_confidence_terms or any(term in words for term in rs_high_confidence_terms):
        confidence = min(0.95, confidence + 0.04)

    return round(confidence, 2)


def run_vqa(image_path: str, question: str) -> Dict[str, Any]:
    """
    Executes Visual Question Answering on a single remote sensing image.

    Args:
        image_path (str): Path to the satellite image file (PNG, JPG, TIF).
        question (str): Natural language question about the image.

    Returns:
        dict: Standardized result containing:
            - task (str): "vqa"
            - answer (str): Generated natural language answer.
            - confidence (float): Heuristic confidence metric in range [0.0, 1.0].
            - image_path (str): Preserved image path string.
    """
    resolved_path = Path(image_path)
    if not resolved_path.is_absolute():
        resolved_path = (PROJECT_ROOT / resolved_path).resolve()

    if not resolved_path.is_file():
        err_msg = f"Error: Image file not found at '{image_path}'"
        return {
            "task": "vqa",
            "answer": err_msg,
            "confidence": 0.0,
            "image_path": str(image_path),
        }

    raw_answer = answer_question(str(resolved_path), question)
    confidence = calculate_vqa_confidence(raw_answer)

    return {
        "task": "vqa",
        "answer": raw_answer,
        "confidence": confidence,
        "image_path": str(image_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SatQuery AI - Single-Image VQA Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to satellite image")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    args = parser.parse_args()

    res = run_vqa(args.image, args.question)
    print(res)
