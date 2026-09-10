"""
backend/captioning_tool.py
==========================
Captioning and scene description baseline tool for SatQuery AI.

Provides:
    run_captioning(image_path: str, prompt: Optional[str] = None) -> dict

Generates a natural-language description of land cover and major visible objects
in single remote sensing images using M2's adapted Vision-Language Model.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.inference import answer_question

DEFAULT_CAPTION_PROMPT = "Describe the land cover and major objects visible in this image."


def calculate_caption_confidence(caption_text: str) -> float:
    """
    Computes a heuristic confidence score for caption generation in [0.0, 1.0].

    Heuristic Criteria:
    - 0.00: File errors or model failure.
    - 0.10: Uninformative or fallback responses.
    - 0.35: Explicit uncertainty phrases.
    - 0.85: Baseline confidence for successful VLM scene descriptions.
    - 0.90+: Detailed descriptions containing recognized remote sensing features.
    """
    if not caption_text or not isinstance(caption_text, str):
        return 0.0

    cleaned = caption_text.strip()
    lower_text = cleaned.lower()

    if lower_text.startswith("error:") or lower_text.startswith("inference error:"):
        return 0.0

    if "no discernible feature detected" in lower_text:
        return 0.10

    uncertainty_tokens = ["uncertain", "unclear", "unknown", "maybe", "not sure", "cannot tell"]
    if any(token in lower_text for token in uncertainty_tokens):
        return 0.35

    confidence = 0.85
    words = cleaned.split()

    # Reward descriptive specificity
    rs_features = {
        "water", "forest", "trees", "urban", "vegetation", "agriculture",
        "crop", "grass", "river", "coastal", "commercial", "residential",
        "building", "buildings", "road", "roads", "airport", "runway", "soil"
    }
    matches = sum(1 for w in words if w.lower().strip(".,;:()") in rs_features)
    if matches >= 2:
        confidence += 0.07
    elif matches == 1:
        confidence += 0.03

    if len(words) >= 4:
        confidence += 0.03

    return round(min(0.96, confidence), 2)


def format_scene_caption(raw_description: str) -> str:
    """
    Formats the raw VLM description into a natural language sentence.

    Examples:
        'trees and grass' -> 'Satellite imagery showing trees and grass.'
        'commercial buildings and road networks' -> 'Satellite imagery showing commercial buildings and road networks.'
    """
    cleaned = raw_description.strip()
    if not cleaned:
        return "No discernible features detected."

    # If it's an error message or already a full sentence, leave intact
    lower = cleaned.lower()
    if lower.startswith("error:") or lower.startswith("satellite ") or lower.startswith("a high-resolution") or lower.startswith("an aerial") or lower.startswith("the image"):
        return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."

    # Format into fluent sentence
    # Remove any trailing period before formatting
    cleaned = cleaned.rstrip(".")
    return f"Satellite imagery showing {cleaned}."


def run_captioning(image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a natural-language description of land cover and major objects
    visible in a single satellite image.

    Args:
        image_path (str): Path to satellite image file (.png, .jpg, .tif).
        prompt (str, optional): Custom captioning prompt. Defaults to
            'Describe the land cover and major objects visible in this image.'

    Returns:
        dict: Standardized result containing:
            - task (str): "captioning"
            - caption (str): Natural language description.
            - confidence (float): Heuristic confidence metric in range [0.0, 1.0].
            - image_path (str): Preserved image path string.
    """
    resolved_path = Path(image_path)
    if not resolved_path.is_absolute():
        resolved_path = (PROJECT_ROOT / resolved_path).resolve()

    if not resolved_path.is_file():
        err_msg = f"Error: Image file not found at '{image_path}'"
        return {
            "task": "captioning",
            "caption": err_msg,
            "confidence": 0.0,
            "image_path": str(image_path),
        }

    active_prompt = prompt if prompt else DEFAULT_CAPTION_PROMPT
    raw_desc = answer_question(str(resolved_path), active_prompt)

    if raw_desc.startswith("Error:") or raw_desc.startswith("Inference error:"):
        return {
            "task": "captioning",
            "caption": raw_desc,
            "confidence": 0.0,
            "image_path": str(image_path),
        }

    formatted_caption = format_scene_caption(raw_desc)
    confidence = calculate_caption_confidence(formatted_caption)

    return {
        "task": "captioning",
        "caption": formatted_caption,
        "confidence": confidence,
        "image_path": str(image_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SatQuery AI - Single-Image Captioning Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to satellite image")
    parser.add_argument("--prompt", type=str, default=None, help="Optional captioning prompt")
    args = parser.parse_args()

    res = run_captioning(args.image, prompt=args.prompt)
    print(res)
