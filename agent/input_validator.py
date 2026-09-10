"""
agent/input_validator.py
========================
Input Validation & Schema Verification for SatQuery AI.

Provides:
    validate_input(
        query: str,
        image_paths: list,
        image_types: list = None,
        task: str = None
    ) -> dict

Validates:
    1. Correct number of images for the target task (e.g. exactly 2 images for
       change detection and fusion; 1 image for VQA, captioning, grounding).
    2. File format validity (.png, .jpg, .jpeg, .tif, .tiff) and physical existence on disk.
    3. Sensor modality compatibility (e.g. fusion requires one optical and one SAR image).
    4. Query string adequacy for task context.

Returns:
    {"valid": bool, "error": Optional[str]}
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

from agent.task_classifier import classify_task

# Supported remote sensing image extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Modality sets
OPTICAL_MODALITIES = {"optical", "rgb", "multispectral", "visual"}
SAR_MODALITIES = {"sar", "radar", "sentinel-1", "backscatter"}


def validate_input(
    query: Optional[str],
    image_paths: List[str],
    image_types: Optional[List[str]] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validates input parameters before executing backend model tools.

    Args:
        query (str, optional): User's query string.
        image_paths (list): List of paths to input images.
        image_types (list, optional): List of image modalities (e.g. ['optical', 'sar']).
        task (str, optional): Pre-classified task. If None, dynamically determined.

    Returns:
        dict: {"valid": bool, "error": Optional[str]}
    """
    # 1. Basic path structure validation
    if not isinstance(image_paths, (list, tuple)):
        return {
            "valid": False,
            "error": f"Invalid image_paths: Expected list of file paths, received {type(image_paths).__name__}."
        }

    num_images = len(image_paths)
    if num_images == 0:
        return {
            "valid": False,
            "error": "No input images provided. At least one image path is required."
        }

    # Determine task if not explicitly passed
    if task is None:
        task = classify_task(query, num_images, image_types)

    # 2. Image count checks per task
    if task == "change_detection":
        if num_images != 2:
            return {
                "valid": False,
                "error": f"Task 'change_detection' requires exactly 2 images (before and after), but received {num_images}."
            }

    elif task == "fusion":
        if num_images != 2:
            return {
                "valid": False,
                "error": f"Task 'fusion' requires exactly 2 images (one optical and one SAR), but received {num_images}."
            }

    elif task in ["vqa", "grounding", "captioning"]:
        if num_images != 1:
            return {
                "valid": False,
                "error": f"Task '{task}' requires exactly 1 image, but received {num_images}."
            }

    # 3. File existence and format validation
    project_root = Path(__file__).resolve().parent.parent

    for idx, path_str in enumerate(image_paths):
        if not path_str or not isinstance(path_str, str):
            return {
                "valid": False,
                "error": f"Image path at index {idx} is invalid or empty: {path_str!r}."
            }

        p = Path(path_str)
        ext = p.suffix.lower()

        # Format validation
        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "valid": False,
                "error": f"Unsupported file format '{ext}' for image '{path_str}'. Supported extensions: {sorted(list(SUPPORTED_EXTENSIONS))}."
            }

        # Existence validation
        resolved = p if p.is_absolute() else (project_root / p).resolve()
        if not resolved.is_file():
            return {
                "valid": False,
                "error": f"Image file not found at '{path_str}' (resolved to '{resolved}')."
            }

    # 4. Modality compatibility checks
    if task == "fusion":
        if image_types is not None and len(image_types) == 2:
            types_lower = [str(t).strip().lower() for t in image_types]
            has_optical = any(t in OPTICAL_MODALITIES for t in types_lower)
            has_sar = any(t in SAR_MODALITIES for t in types_lower)

            if not (has_optical and has_sar):
                return {
                    "valid": False,
                    "error": (
                        f"Task 'fusion' requires complementary optical and SAR sensor modalities. "
                        f"Provided image_types: {image_types}."
                    )
                }

    # 5. Query adequacy checks
    if task in ["vqa", "grounding"]:
        if not query or not query.strip():
            return {
                "valid": False,
                "error": f"Task '{task}' requires a non-empty natural language query."
            }

    return {"valid": True, "error": None}
