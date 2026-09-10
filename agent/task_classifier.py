"""
agent/task_classifier.py
========================
Intelligent Rule-Based Task Classifier for SatQuery AI.

Provides:
    classify_task(query: str, num_images: int, image_types: list = None) -> str

Maps a natural-language query and image metadata to one of the five core tasks:
    - "vqa"              : Single-image Visual Question Answering
    - "captioning"        : Single-image scene description / land-cover summary
    - "grounding"         : Single-image text-guided bounding box localization
    - "change_detection"  : Bi-temporal multitemporal change analysis
    - "fusion"            : Cross-modal optical and SAR complementary analysis

Design Rationale:
-----------------
Uses an explainable, deterministic rule-based heuristic with clear keyword
dictionaries and sensor modality verification. This provides transparent,
auditable task routing that is easily defendable and predictable under evaluation.
"""

from typing import List, Optional


def classify_task(
    query: Optional[str],
    num_images: int,
    image_types: Optional[List[str]] = None
) -> str:
    """
    Classifies an incoming query and image context into an execution task.

    Args:
        query (str, optional): User's natural language question or instruction.
        num_images (int): Count of images provided in the request.
        image_types (list, optional): List of modalities (e.g. ['optical', 'sar']).

    Returns:
        str: One of ["vqa", "captioning", "grounding", "change_detection", "fusion"].
    """
    # Normalize inputs
    q = (query or "").strip().lower()
    types_clean = [str(t).strip().lower() for t in (image_types or []) if t]

    has_optical = any(t in ["optical", "rgb", "multispectral"] for t in types_clean)
    has_sar = any(t in ["sar", "radar", "sentinel-1"] for t in types_clean)

    # -------------------------------------------------------------------------
    # 1. Cross-Modal Optical + SAR Fusion ("fusion")
    # -------------------------------------------------------------------------
    # Explicit sensor condition: 2 images where one is optical and one is SAR
    if num_images == 2 and (has_optical and has_sar):
        return "fusion"

    # Query-level trigger for fusion / joint optical and SAR analysis
    fusion_keywords = [
        "optical and sar",
        "sar and optical",
        "optical and radar",
        "radar and optical",
        "sensor fusion",
        "cross-modal",
        "multimodal fusion",
        "fuse optical",
        "fuse sar",
        "both optical and sar",
        "optical and synthetic aperture radar",
    ]
    if any(kw in q for kw in fusion_keywords):
        return "fusion"

    # -------------------------------------------------------------------------
    # 2. Bi-Temporal Change Detection ("change_detection")
    # -------------------------------------------------------------------------
    change_keywords = [
        "changed",
        "change",
        "between these two",
        "between two",
        "between the two",
        "between dates",
        "between time",
        "before and after",
        "before & after",
        "increased, decreased, or remained unchanged",
        "increased",
        "decreased",
        "remained unchanged",
        "has the built-up area increased",
        "difference between",
        "transition",
        "deforestation",
        "land clearance",
        "altered",
        "temporal",
        "time 1",
        "time 2",
        "prior observation",
        "subsequent observation",
    ]

    has_change_query = any(kw in q for kw in change_keywords)

    # If 2 images are provided and not classified as fusion, default to change detection
    if num_images == 2:
        return "change_detection"

    # If only 1 image or count unspecified, but query strongly indicates change detection
    if has_change_query:
        return "change_detection"

    # -------------------------------------------------------------------------
    # 3. Visual Grounding / Region Localization ("grounding")
    # -------------------------------------------------------------------------
    grounding_keywords = [
        "highlight",
        "highlighting",
        "where is",
        "where are",
        "locate",
        "locating",
        "localize",
        "localization",
        "bounding box",
        "bounding-box",
        "bbox",
        "coordinates",
        "pinpoint",
        "box around",
        "find the location",
    ]
    if any(kw in q for kw in grounding_keywords):
        return "grounding"

    # -------------------------------------------------------------------------
    # 4. Scene Captioning ("captioning")
    # -------------------------------------------------------------------------
    # No query provided -> default to captioning
    if not q:
        return "captioning"

    caption_keywords = [
        "describe",
        "description",
        "caption",
        "overview",
        "summarize",
        "summary",
        "what does this image show",
        "tell me about this image",
        "tell me about this scene",
        "land-cover and major objects",
        "land cover and major objects",
    ]

    is_caption_query = any(kw in q for kw in caption_keywords)

    if is_caption_query and not (q.endswith("?") and any(q.startswith(w) for w in ["what is", "is there", "are there", "how many", "which"])):
        return "captioning"

    # -------------------------------------------------------------------------
    # 5. Visual Question Answering ("vqa")
    # -------------------------------------------------------------------------
    # Single image with an interrogative question or any remaining query
    return "vqa"
