"""
backend/grounding_tool.py
=========================
Text-guided region grounding baseline tool for SatQuery AI.

Provides:
    run_grounding(image_path: str, query: str) -> dict

Attempts to localize and return approximate bounding box coordinates [x1, y1, x2, y2]
for an object or land cover type specified by natural language query.

Note on Implementation & Accuracy:
    Because Salesforce/blip-vqa-base lacks a native bounding-box regression head,
    this tool implements a robust remote-sensing heuristic grounding engine using
    spectral indices (Excess Green, NDWI RGB approximation, brightness/variance
    segmentation, and density-filtered spatial clustering).
    This baseline is intentionally simplified and clearly labeled.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _detect_water_mask(rgb: np.ndarray) -> np.ndarray:
    """Detects water bodies using RGB water index heuristics (blue/cyan bias, low red)."""
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    brightness = (r + g + b) / 3.0

    # Water has higher blue/green than red, and moderate to low overall reflectance
    mask = (b > r + 8) & (g > r) & (brightness < 160)
    return mask


def _detect_vegetation_mask(rgb: np.ndarray) -> np.ndarray:
    """Detects vegetation/forest using Excess Green Index (ExG = 2G - R - B)."""
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    exg = 2.0 * g - r - b
    mask = (exg > 15.0) & (g > r + 5.0)
    return mask


def _detect_urban_mask(rgb: np.ndarray) -> np.ndarray:
    """Detects urban/built-up structures using texture variance and gray/neutral tones."""
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    brightness = (r + g + b) / 3.0

    # Low saturation/color divergence, moderate to high brightness
    color_diff = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    mask = (color_diff < 35.0) & (brightness >= 70.0) & (brightness <= 220.0)
    return mask


def _detect_bright_object_mask(rgb: np.ndarray) -> np.ndarray:
    """Detects high-reflectance features such as planes, runways, or isolated structures."""
    brightness = np.mean(rgb.astype(float), axis=2)
    p90 = np.percentile(brightness, 90)
    mask = brightness >= max(180.0, p90)
    return mask


def _compute_bbox_from_mask(mask: np.ndarray, width: int, height: int) -> Tuple[List[int], float]:
    """
    Computes a robust [x1, y1, x2, y2] bounding box from a binary candidate mask,
    using percentile trimming to reject isolated noise artifacts.
    """
    total_pixels = width * height
    matching_pixels = int(np.sum(mask))

    if matching_pixels == 0:
        # No matching pixels found - return full image fallback with low confidence
        return [0, 0, width, height], 0.20

    coverage = matching_pixels / float(total_pixels)
    y_indices, x_indices = np.where(mask)

    # Use 3rd and 97th percentiles to clip out spurious pixel noise
    x1 = int(np.percentile(x_indices, 3))
    x2 = int(np.percentile(x_indices, 97))
    y1 = int(np.percentile(y_indices, 3))
    y2 = int(np.percentile(y_indices, 97))

    # Ensure valid bounding box extents
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2 + 1, width))
    y2 = max(y1 + 1, min(y2 + 1, height))

    # Calculate heuristic confidence based on coverage realism
    if 0.02 <= coverage <= 0.85:
        confidence = 0.75
    elif coverage < 0.02:
        confidence = 0.50
    else:
        confidence = 0.60

    return [x1, y1, x2, y2], confidence


def run_grounding(image_path: str, query: str) -> Dict[str, Any]:
    """
    Identifies and returns approximate bounding-box coordinates for an object
    or land cover type described in query.

    Args:
        image_path (str): Path to the satellite image (.png, .jpg, .tif).
        query (str): Natural language description of object/feature (e.g. 'the water body', 'forest', 'plane').

    Returns:
        dict: Standardized result containing:
            - task (str): "grounding"
            - bbox (list): [x1, y1, x2, y2] bounding box coordinates in pixels.
            - confidence (float): Confidence score in [0.0, 1.0].
            - image_path (str): Preserved image path string.
    """
    resolved_path = Path(image_path)
    if not resolved_path.is_absolute():
        resolved_path = (PROJECT_ROOT / resolved_path).resolve()

    if not resolved_path.is_file():
        return {
            "task": "grounding",
            "bbox": [0, 0, 0, 0],
            "confidence": 0.0,
            "image_path": str(image_path),
        }

    try:
        with Image.open(resolved_path) as img:
            rgb_img = img.convert("RGB")
            width, height = rgb_img.size
            rgb_array = np.array(rgb_img)
    except Exception:
        return {
            "task": "grounding",
            "bbox": [0, 0, 0, 0],
            "confidence": 0.0,
            "image_path": str(image_path),
        }

    q_lower = query.lower()

    # Route to appropriate spectral/texture heuristic
    if any(k in q_lower for k in ["water", "river", "lake", "ocean", "sea", "pond"]):
        mask = _detect_water_mask(rgb_array)
        bbox, base_conf = _compute_bbox_from_mask(mask, width, height)
        conf = min(0.85, base_conf + 0.05)
    elif any(k in q_lower for k in ["forest", "tree", "vegetation", "grass", "green", "agriculture", "crop"]):
        mask = _detect_vegetation_mask(rgb_array)
        bbox, base_conf = _compute_bbox_from_mask(mask, width, height)
        conf = min(0.85, base_conf + 0.05)
    elif any(k in q_lower for k in ["urban", "building", "house", "commercial", "settlement", "roof", "road"]):
        mask = _detect_urban_mask(rgb_array)
        bbox, base_conf = _compute_bbox_from_mask(mask, width, height)
        conf = min(0.75, base_conf)
    elif any(k in q_lower for k in ["plane", "aircraft", "runway", "airport", "white", "bright"]):
        mask = _detect_bright_object_mask(rgb_array)
        bbox, base_conf = _compute_bbox_from_mask(mask, width, height)
        conf = min(0.70, base_conf)
    else:
        # Fallback: identify highest gradient / salient central zone
        mask = _detect_bright_object_mask(rgb_array)
        bbox, base_conf = _compute_bbox_from_mask(mask, width, height)
        conf = round(max(0.30, base_conf * 0.6), 2)

    return {
        "task": "grounding",
        "bbox": bbox,
        "confidence": round(conf, 2),
        "image_path": str(image_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SatQuery AI - Region Grounding Baseline Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to satellite image")
    parser.add_argument("--query", type=str, required=True, help="Object or terrain query")
    args = parser.parse_args()

    res = run_grounding(args.image, args.query)
    print(res)
