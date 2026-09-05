# Dataset Download & Acquisition Guide (SatQuery AI)

This document provides detailed instructions for acquiring the remote sensing datasets used in the **SatQuery AI** pipeline. 

To prevent disk saturation and ensure rapid development across the 6-member team, small sample subsets and synthetic fallback generators are provided in the repository. When preparing for full model fine-tuning and evaluation, use the instructions below to download the official datasets.

---

## 1. Overview of Required Datasets

| Dataset | Modality & Type | Primary Task in SatQuery AI | Official Source | Full Size |
| :--- | :--- | :--- | :--- | :--- |
| **BigEarthNet** | Multi-spectral (Sentinel-2) & Dual-Pol SAR (Sentinel-1) | Image-Text Adaptation, Multimodal Foundation Pretraining | [bigearth.net](https://bigearth.net/) / [Zenodo](https://zenodo.org/records/4893574) | ~66 GB (S2), ~13 GB (S1) |
| **VRSBench** | High-Resolution Optical RGB | Image Captioning, Visual Grounding, Multimodal Reasoning | [GitHub / Hugging Face](https://github.com/Visual-Intelligence-Laboratory/VRSBench) | ~12 GB |
| **RSVQA** | Sentinel-2 (LR) & Aerial Orthophotos (HR) | Remote Sensing Visual Question Answering | [Zenodo](https://zenodo.org/records/6344334) | ~1.5 GB (LR), ~15 GB (HR) |
| **CDVQA** | Multitemporal Bi-temporal Optical / Aerial Image Pairs | Change Detection VQA, Temporal Reasoning | [IEEE / GitHub](https://github.com/ggs-whu/CDVQA) | ~3 GB |

---

## 2. Dataset Specific Download Instructions

### A. BigEarthNet (Mandatory per Problem Statement)

BigEarthNet consists of 590,326 Sentinel-2 image patches across 10 European countries annotated with 19 (or 43) CORINE Land Cover (CLC) classes. A paired Sentinel-1 SAR dataset (BigEarthNet-S1) is also available.

#### Option 1: Official Portal (Requires Free Registration)
1. Go to [https://bigearth.net/](https://bigearth.net/).
2. Create an account and verify your email.
3. Under the **Downloads** section:
   - `BigEarthNet-S2 (v1.0)`: 12-band Sentinel-2 GeoTIFF patches (`BigEarthNet-v1.0.tar.gz`, ~66 GB).
   - Alternatively, download sample packages (e.g., regional subsets or 10,000-patch mini archive).
4. Extract the sample patches into `satquery-ai/data/bigearthnet/images/`.
5. Ensure `labels.csv` maps each image filename to its associated land cover labels.

#### Option 2: Hugging Face Datasets (Direct programmatic access)
```python
from datasets import load_dataset
# Streaming small sample of BigEarthNet
dataset = load_dataset("BIFROST-AI/bigearthnet-s2", split="train", streaming=True)
# Save first 200 samples locally
```

#### Option 3: Zenodo Mirror
- S2 Archive: [https://zenodo.org/records/4893574](https://zenodo.org/records/4893574)
- S1 Archive: [https://zenodo.org/records/5569420](https://zenodo.org/records/5569420)

Expected target folder structure:
```text
data/bigearthnet/
├── images/
│   ├── S2A_MSIL2A_20170717T113321_0_54.png (or .tif)
│   ├── S2A_MSIL2A_20170717T113321_12_31.png
│   └── ...
└── labels.csv
```
`labels.csv` format:
```csv
image_file,labels,modality,split
S2A_MSIL2A_20170717T113321_0_54.png,"Coniferous forest|Pastures",optical,train
```

---

### B. VRSBench (Visual Reasoning in Remote Sensing)

VRSBench covers detailed text captions, question-answer pairs, and object-level bounding boxes for visual grounding on high-resolution remote sensing imagery.

1. Source repository: [https://github.com/Visual-Intelligence-Laboratory/VRSBench](https://github.com/Visual-Intelligence-Laboratory/VRSBench)
2. Follow their Google Drive or Baidu Pan download links for `images.tar.gz` and `annotations.json`.
3. For small-scale evaluation, download the validation subset (~1,000 images, ~1.2 GB).
4. Place images in `data/vrsbench/images/` and the annotation JSON in `data/vrsbench/annotations.json`.

Expected structure:
```text
data/vrsbench/
├── images/
│   ├── vrs_0001.png
│   ├── vrs_0002.png
│   └── ...
└── annotations.json
```
`annotations.json` format:
```json
[
  {
    "image_id": "vrs_0001",
    "image_file": "vrs_0001.png",
    "caption": "A high resolution satellite view of an industrial port with storage tanks.",
    "qa_pairs": [
      {"question": "How many circular storage tanks are visible?", "answer": "4"}
    ],
    "grounding": [
      {"object": "storage tank", "bbox": [120, 85, 180, 145]}
    ]
  }
]
```

---

### C. RSVQA (Remote Sensing Visual Question Answering)

RSVQA includes low-resolution (Sentinel-2, 10m-20m) and high-resolution (aerial orthophoto, 15cm) datasets with questions covering presence, count, area comparison, and rural/urban characteristics.

1. Source repository: [https://github.com/syvlo/rsvqa](https://github.com/syvlo/rsvqa)
2. Zenodo download links:
   - Low-Resolution (LR): [https://zenodo.org/records/6344334](https://zenodo.org/records/6344334) (~1.5 GB total, includes LR images and questions/answers JSON files).
   - High-Resolution (HR): [https://zenodo.org/records/6344367](https://zenodo.org/records/6344367).
3. Unpack sample images to `data/rsvqa/images/` and consolidate QA pairs into `data/rsvqa/qa_pairs.json`.

Expected structure:
```text
data/rsvqa/
├── images/
│   ├── rsvqa_0001.png
│   └── ...
└── qa_pairs.json
```
`qa_pairs.json` format:
```json
[
  {
    "id": 1001,
    "image_file": "rsvqa_0001.png",
    "question": "Is there a river crossing the residential area?",
    "answer": "Yes",
    "type": "presence"
  }
]
```

---

### D. CDVQA (Change Detection Visual Question Answering)

CDVQA focuses on multitemporal bi-temporal image pairs ($T_1$ and $T_2$) captured over the same geographical coordinates at different dates, with questions querying human-induced and environmental changes.

1. Source: [https://github.com/ggs-whu/CDVQA](https://github.com/ggs-whu/CDVQA)
2. Download the bi-temporal image pair archive and question-answer sets.
3. Save paired images under `data/cdvqa/image_pairs/` (e.g. `pair_0001_t1.png` and `pair_0001_t2.png`).
4. Save the QA annotations into `data/cdvqa/qa_pairs.json`.

Expected structure:
```text
data/cdvqa/
├── image_pairs/
│   ├── pair_0001_t1.png
│   ├── pair_0001_t2.png
│   └── ...
└── qa_pairs.json
```
`qa_pairs.json` format:
```json
[
  {
    "pair_id": "pair_0001",
    "image_t1": "pair_0001_t1.png",
    "image_t2": "pair_0001_t2.png",
    "question": "Did new residential buildings appear in the northern sector between T1 and T2?",
    "answer": "Yes, five new structures were constructed.",
    "change_type": "urban_construction"
  }
]
```

---

## 3. Synthetic Fallback Data (`data/sample/`)

To unblock development immediately without waiting for multi-gigabyte downloads or portal account verifications, `satquery-ai` includes realistic synthetic samples:
- `data/sample/optical/`: Simulated multispectral RGB tiles (forests, rivers, agricultural plots, urban grids, cloud cover).
- `data/sample/sar/`: Simulated Sentinel-1 Synthetic Aperture Radar backscatter tiles (speckle noise, corner reflection hotspots, dark calm water bodies).
- `data/sample/pairs/`: Multitemporal bi-temporal pairs ($T_1$ and $T_2$) demonstrating structural, agricultural, and deforestation change dynamics.

Downstream teammates (Model Training, Backend API, Frontend UI, Agent Orchestration) can immediately call `load_sample_data()` or any of the loader functions in `data/loader.py` to test the full pipeline end-to-end.
