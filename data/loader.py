"""
SatQuery AI - Unified Remote Sensing Data Loader
=================================================
This module provides simple, standard interfaces to load remote sensing datasets:
  - BigEarthNet (multi-spectral & SAR land cover image-text adaptation)
  - VRSBench (captioning, visual grounding, and VQA)
  - RSVQA (single-image visual question answering)
  - CDVQA (multitemporal change detection visual question answering)
  - Sample / Synthetic fallback data (optical, SAR, and change pairs)

All loader functions return a list of standard dictionaries containing at least:
  - "image_path": str (path to primary image file)
  - "question": str (query or task prompt)
  - "answer": str (ground truth answer, labels, or caption)
plus dataset-specific auxiliary metadata.

Usage:
    from data.loader import (
        load_bigearthnet,
        load_vrsbench,
        load_rsvqa,
        load_cdvqa,
        load_sample_data,
    )

    bigearth_data = load_bigearthnet()
    sample_data = load_sample_data()
"""

import os
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Default base directories relative to this file
DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent


def _resolve_path(base_dir: Optional[str or Path], default_dir: Path) -> Path:
    if base_dir is None:
        return default_dir
    return Path(base_dir).resolve()


def load_bigearthnet(
    data_dir: Optional[str or Path] = None,
    split: Optional[str] = None,
    modality: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Loads BigEarthNet sample dataset.
    
    Expected directory structure:
      data/bigearthnet/
        ├── images/
        │   ├── <patch_id>.png (or .tif)
        └── labels.csv

    Returns:
      List[Dict[str, Any]]: Each item has:
        - image_path (str): Path to image file
        - patch_id (str): Identifier of the image patch
        - labels (List[str]): List of multi-label land cover classes
        - modality (str): 'optical' or 'sar'
        - split (str): 'train', 'val', or 'test'
        - question (str): Formatted query prompt for image-text adaptation / VQA
        - answer (str): Comma-separated land cover classes
        - caption (str): Natural language sentence describing land cover
    """
    root = _resolve_path(data_dir, DATA_DIR / "bigearthnet")
    images_dir = root / "images"
    labels_csv = root / "labels.csv"

    samples = []
    if not labels_csv.is_file():
        # If labels.csv doesn't exist yet, inspect images_dir
        if images_dir.is_dir():
            for img_file in sorted(images_dir.glob("*.*")):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                    patch_name = img_file.stem
                    samples.append({
                        "image_path": str(img_file),
                        "patch_id": patch_name,
                        "labels": ["Unknown"],
                        "modality": "optical",
                        "split": "train",
                        "question": "What land cover classes are present in this satellite image?",
                        "answer": "Unknown",
                        "caption": f"Satellite patch {patch_name}.",
                    })
        return samples[:limit] if limit else samples

    with open(labels_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = row.get("split", "train").strip()
            row_modality = row.get("modality", "optical").strip()

            if split and row_split.lower() != split.lower():
                continue
            if modality and row_modality.lower() != modality.lower():
                continue

            img_name = row.get("image_file", "").strip()
            img_path = images_dir / img_name
            raw_labels = row.get("labels", "").strip()
            labels_list = [l.strip() for l in raw_labels.split("|") if l.strip()] if raw_labels else []
            answer_str = ", ".join(labels_list) if labels_list else "None"
            patch_id = Path(img_name).stem

            caption_str = row.get("caption") or (
                f"A {row_modality} satellite patch displaying {answer_str}."
                if labels_list else f"A satellite patch {patch_id}."
            )

            samples.append({
                "image_path": str(img_path),
                "patch_id": patch_id,
                "labels": labels_list,
                "modality": row_modality,
                "split": row_split,
                "question": "What land cover classes are present in this satellite image?",
                "answer": answer_str,
                "caption": caption_str,
            })

            if limit and len(samples) >= limit:
                break

    return samples


def load_vrsbench(
    data_dir: Optional[str or Path] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Loads VRSBench (Visual Reasoning Benchmark) sample dataset.

    Expected directory structure:
      data/vrsbench/
        ├── images/
        │   ├── <img_id>.png
        └── annotations.json

    Returns:
      List[Dict[str, Any]]: Each item has:
        - image_path (str): Path to image file
        - image_id (str): Unique image identifier
        - caption (str): Natural language caption describing the scene
        - question (str): Primary visual reasoning question
        - answer (str): Ground truth answer
        - qa_pairs (List[Dict]): All question-answer pairs for this image
        - grounding (List[Dict]): Object bounding boxes and categories
    """
    root = _resolve_path(data_dir, DATA_DIR / "vrsbench")
    images_dir = root / "images"
    annot_file = root / "annotations.json"

    samples = []
    if not annot_file.is_file():
        return samples

    with open(annot_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        img_name = item.get("image_file") or f"{item.get('image_id', '')}.png"
        img_path = images_dir / img_name
        qa_pairs = item.get("qa_pairs", [])
        first_q = qa_pairs[0].get("question", "") if qa_pairs else "Describe this remote sensing image."
        first_a = qa_pairs[0].get("answer", "") if qa_pairs else item.get("caption", "")

        samples.append({
            "image_path": str(img_path),
            "image_id": item.get("image_id", Path(img_name).stem),
            "caption": item.get("caption", ""),
            "question": first_q,
            "answer": first_a,
            "qa_pairs": qa_pairs,
            "grounding": item.get("grounding", []),
        })

        if limit and len(samples) >= limit:
            break

    return samples


def load_rsvqa(
    data_dir: Optional[str or Path] = None,
    question_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Loads RSVQA (Remote Sensing Visual Question Answering) sample dataset.

    Expected directory structure:
      data/rsvqa/
        ├── images/
        │   ├── <id>.png
        └── qa_pairs.json

    Returns:
      List[Dict[str, Any]]: Each item has:
        - image_path (str): Path to image file
        - id (str or int): QA record ID
        - question (str): Visual question about presence, count, area, or comparison
        - answer (str): Target answer
        - question_type (str): Type of question (presence, count, area, comparison)
    """
    root = _resolve_path(data_dir, DATA_DIR / "rsvqa")
    images_dir = root / "images"
    qa_file = root / "qa_pairs.json"

    samples = []
    if not qa_file.is_file():
        return samples

    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        q_type = item.get("type", "general")
        if question_type and q_type.lower() != question_type.lower():
            continue

        img_name = item.get("image_file", "")
        img_path = images_dir / img_name

        samples.append({
            "image_path": str(img_path),
            "id": item.get("id"),
            "question": item.get("question", ""),
            "answer": str(item.get("answer", "")),
            "question_type": q_type,
        })

        if limit and len(samples) >= limit:
            break

    return samples


def load_cdvqa(
    data_dir: Optional[str or Path] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Loads CDVQA (Change Detection Visual Question Answering) multitemporal sample dataset.

    Expected directory structure:
      data/cdvqa/
        ├── image_pairs/
        │   ├── <pair_id>_t1.png
        │   ├── <pair_id>_t2.png
        └── qa_pairs.json

    Returns:
      List[Dict[str, Any]]: Each item has:
        - pair_id (str): Identifier for the multitemporal pair
        - image_path (str): Points to post-change (T2) image (for single-image loader compatibility)
        - image_t1_path (str): Pre-change image (Time 1)
        - image_t2_path (str): Post-change image (Time 2)
        - question (str): Change detection query between T1 and T2
        - answer (str): Target answer describing the temporal change
        - change_type (str): Category of change (e.g., urban_construction, vegetation, water)
    """
    root = _resolve_path(data_dir, DATA_DIR / "cdvqa")
    pairs_dir = root / "image_pairs"
    qa_file = root / "qa_pairs.json"

    samples = []
    if not qa_file.is_file():
        return samples

    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        t1_name = item.get("image_t1", "")
        t2_name = item.get("image_t2", "")

        t1_path = pairs_dir / t1_name
        t2_path = pairs_dir / t2_name

        samples.append({
            "pair_id": item.get("pair_id", ""),
            "image_path": str(t2_path),
            "image_t1_path": str(t1_path),
            "image_t2_path": str(t2_path),
            "question": item.get("question", ""),
            "answer": str(item.get("answer", "")),
            "change_type": item.get("change_type", "general"),
        })

        if limit and len(samples) >= limit:
            break

    return samples


def load_sample_data(
    data_dir: Optional[str or Path] = None,
    category: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Loads synthetic fallback sample dataset (optical, SAR, multitemporal pairs).

    Expected directory structure:
      data/sample/
        ├── optical/
        ├── sar/
        └── pairs/

    Returns:
      List[Dict[str, Any]]: Standardized sample dictionaries with image paths,
      prompt questions, expected answers, and modality metadata.
    """
    root = _resolve_path(data_dir, DATA_DIR / "sample")
    samples = []

    valid_categories = ["optical", "sar", "pairs"]
    cats_to_load = [category] if (category and category in valid_categories) else valid_categories

    for cat in cats_to_load:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue

        if cat == "pairs":
            # Pairs are named *_t1.png and *_t2.png
            t1_files = sorted(cat_dir.glob("*_t1.png"))
            for t1_file in t1_files:
                base_stem = t1_file.stem[:-3]  # Strip '_t1'
                t2_file = cat_dir / f"{base_stem}_t2.png"
                meta_file = cat_dir / f"{base_stem}_meta.json"

                q = "What difference is visible between Time 1 and Time 2?"
                a = "Surface alterations detected."
                change_type = "synthetic_change"

                if meta_file.is_file():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            q = meta.get("question", q)
                            a = meta.get("answer", a)
                            change_type = meta.get("change_type", change_type)
                    except Exception:
                        pass

                samples.append({
                    "image_path": str(t2_file if t2_file.is_file() else t1_file),
                    "pair_id": base_stem,
                    "image_t1_path": str(t1_file),
                    "image_t2_path": str(t2_file) if t2_file.is_file() else None,
                    "category": "pairs",
                    "modality": "multitemporal",
                    "question": q,
                    "answer": a,
                    "change_type": change_type,
                })
                if limit and len(samples) >= limit:
                    return samples
        else:
            # Optical or SAR
            img_files = sorted([f for f in cat_dir.glob("*.*") if f.suffix.lower() in [".png", ".jpg", ".tif"]])
            for img_file in img_files:
                meta_file = img_file.with_suffix(".json")
                q = (
                    "What features are visible in this optical satellite image?"
                    if cat == "optical"
                    else "Describe the radar backscatter characteristics in this SAR image."
                )
                a = f"Synthetic {cat} satellite sample displaying simulated ground texture."
                labels = [cat]

                if meta_file.is_file():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            q = meta.get("question", q)
                            a = meta.get("answer", a)
                            labels = meta.get("labels", labels)
                    except Exception:
                        pass

                samples.append({
                    "image_path": str(img_file),
                    "image_id": img_file.stem,
                    "category": cat,
                    "modality": cat,
                    "question": q,
                    "answer": a,
                    "labels": labels,
                })
                if limit and len(samples) >= limit:
                    return samples

    return samples


if __name__ == "__main__":
    print("=" * 60)
    print("SatQuery AI - Dataset Loader Sanity Test")
    print("=" * 60)

    loaders = [
        ("BigEarthNet", load_bigearthnet),
        ("VRSBench", load_vrsbench),
        ("RSVQA", load_rsvqa),
        ("CDVQA", load_cdvqa),
        ("Synthetic Sample", load_sample_data),
    ]

    for name, fn in loaders:
        try:
            data = fn(limit=3)
            print(f"\n[+] {name}: Loaded {len(data)} sample(s)")
            if data:
                print(f"    First item keys: {list(data[0].keys())}")
                print(f"    Image: {os.path.basename(data[0].get('image_path', ''))}")
                print(f"    Question: {data[0].get('question')}")
                print(f"    Answer: {data[0].get('answer')}")
            else:
                print("    (No samples found)")
        except Exception as e:
            print(f"[-] {name} failed with error: {e}")

    print("\n" + "=" * 60)
    print("All loader functions executed successfully.")
    print("=" * 60)
