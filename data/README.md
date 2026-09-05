# SatQuery AI - Data Layer

This directory contains the remote sensing datasets, synthetic fallback benchmarks, and loading utilities for the **SatQuery AI** multimodal satellite question answering and reasoning system.

---

## 1. Directory Structure

```text
data/
├── DOWNLOAD_INSTRUCTIONS.md   # Complete guide to obtaining full academic datasets
├── README.md                  # This file
├── loader.py                  # Unified Python loader utility for all datasets
│
├── bigearthnet/               # Primary Image-Text Adaptation dataset (Sentinel-2 / Sentinel-1)
│   ├── images/                # Sample image patches (.png / .tif)
│   └── labels.csv             # Patch IDs, CLC land cover labels, modalities, splits
│
├── vrsbench/                  # Visual Reasoning Benchmark (Captioning, Grounding, VQA)
│   ├── images/                # High-resolution optical satellite images
│   └── annotations.json       # Captions, questions, answers, and bounding boxes
│
├── rsvqa/                     # Remote Sensing Visual Question Answering
│   ├── images/                # Sentinel-2 / aerial scene images
│   └── qa_pairs.json          # Presence, count, area, and comparison QA pairs
│
├── cdvqa/                     # Change Detection Visual Question Answering
│   ├── image_pairs/           # Multitemporal pairs (pair_*_t1.png, pair_*_t2.png)
│   └── qa_pairs.json          # Bi-temporal change detection QA pairs
│
└── sample/                    # Synthetic Fallback Data (Ready out-of-the-box)
    ├── optical/               # Synthetic multispectral optical satellite tiles
    ├── sar/                   # Synthetic Sentinel-1 radar backscatter tiles
    └── pairs/                 # Synthetic bi-temporal change detection pairs
```

---

## 2. Dataset Purpose & Description

### BigEarthNet
- **Purpose**: Primary benchmark for remote sensing image-text adaptation and multi-label classification.
- **Sensor Types**: Sentinel-2 Multi-Spectral Instrument (12 bands) and Sentinel-1 SAR (dual-pol VV/VH).
- **Classes**: 19-class or 43-class CORINE Land Cover (CLC) categories (e.g., *Coniferous forest*, *Industrial or commercial units*, *Water bodies*).

### VRSBench (Visual Reasoning in Remote Sensing)
- **Purpose**: Evaluates high-level visual reasoning, dense captioning, fine-grained object grounding (bounding boxes), and visual question answering on high-resolution Earth observation data.

### RSVQA (Remote Sensing Visual Question Answering)
- **Purpose**: Evaluates single-image VQA for satellite and aerial imagery. Includes specific question types:
  - *Presence*: "Is there an airport present?"
  - *Count*: "How many storage tanks are in the harbor?"
  - *Area Comparison*: "Is the agricultural area greater than the urban area?"

### CDVQA (Change Detection Visual Question Answering)
- **Purpose**: Evaluates multitemporal change understanding between bi-temporal image pairs ($T_1$ and $T_2$). Addresses questions such as "What structures were built between Time 1 and Time 2?" or "Did deforestation occur in the western section?".

### Sample (Synthetic Fallback)
- **Purpose**: Provides realistic, lightweight synthetic satellite tiles (optical, SAR, and change pairs) immediately without requiring external multi-gigabyte downloads. Allows pipeline development, training loop testing, and API integration from day one.

---

## 3. How to Obtain Full Versions

Full datasets are large (ranging from ~1.5 GB to ~80 GB) and some require free user registration.

Detailed, step-by-step download instructions, URLs, citation links, and extraction guides are documented in:
👉 [data/DOWNLOAD_INSTRUCTIONS.md](DOWNLOAD_INSTRUCTIONS.md)

Quick Links:
- **BigEarthNet**: [bigearth.net](https://bigearth.net/) / [Zenodo S2 Archive](https://zenodo.org/records/4893574)
- **VRSBench**: [VRSBench GitHub](https://github.com/Visual-Intelligence-Laboratory/VRSBench)
- **RSVQA**: [RSVQA Zenodo](https://zenodo.org/records/6344334)
- **CDVQA**: [CDVQA GitHub Repository](https://github.com/ggs-whu/CDVQA)

---

## 4. Unified Python Data Loader (`data/loader.py`)

All teammates (particularly M2 for model fine-tuning) should import and use the standard loaders in [data/loader.py](loader.py).

### Common Return Format
All loaders return a standard Python `List[Dict[str, Any]]`. Every dictionary contains the core keys:
- `"image_path"`: `str` — Absolute or relative path to the image file.
- `"question"`: `str` — Query prompt, VQA question, or image-text adaptation prompt.
- `"answer"`: `str` — Ground truth answer or label string.

### Available Loader Functions

```python
from data.loader import (
    load_bigearthnet,
    load_vrsbench,
    load_rsvqa,
    load_cdvqa,
    load_sample_data,
)

# 1. BigEarthNet (returns image_path, patch_id, labels, modality, split, question, answer)
train_samples = load_bigearthnet(split="train", limit=100)

# 2. VRSBench (returns image_path, image_id, caption, question, answer, qa_pairs, grounding)
vrs_samples = load_vrsbench(limit=50)

# 3. RSVQA (returns image_path, id, question, answer, question_type)
rsvqa_samples = load_rsvqa(question_type="count", limit=50)

# 4. CDVQA (returns pair_id, image_path, image_t1_path, image_t2_path, question, answer, change_type)
cdvqa_samples = load_cdvqa(limit=50)

# 5. Synthetic Fallback (returns image_path, category, modality, question, answer, etc.)
synthetic_optical = load_sample_data(category="optical")
synthetic_sar = load_sample_data(category="sar")
synthetic_pairs = load_sample_data(category="pairs")
```

---

## 5. Known Limitations of Sample Data

1. **Sample Scale**: The committed sample data contains representative subsets designed for development, sanity checks, and unit testing under 500MB total.
2. **Synthetic Textures**: The images in `data/sample/` simulate optical reflectance (NDVI-like greens, water absorption, urban grids) and SAR backscatter speckle noise. While mathematically structured to emulate satellite distributions, they should be replaced with official BigEarthNet / VRSBench patches for production model fine-tuning.
3. **GeoTIFF Bands**: Real BigEarthNet S2 contains 12 spectral bands at 10m, 20m, and 60m resolution. Sample images are stored as standard 3-band RGB/greyscale PNGs for universal compatibility without requiring GDAL C-library binaries during initial scaffolding.
