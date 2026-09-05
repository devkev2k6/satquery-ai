# Handoff Log

This file is updated by each team member after their phase. Read it before starting your work.

---

## Phase 1: Environment & Dataset Ingestion Setup (Completed)

### 1. What Was Built & Where It Lives
- **Environment Setup Scripts & Requirements**:
  - `requirements.txt`: Core geospatial and ML dependencies (`numpy`, `pandas`, `pillow`, `rasterio`, `requests`, `tqdm`, `scikit-learn`, `matplotlib`).
  - `scripts/setup_env.sh`: Automated environment creation and dependency installer for Linux and macOS.
  - `scripts/setup_env.bat`: Automated environment creation and dependency installer for Windows.
- **Dataset Storage & Subsets** (under `data/`, structured to stay strictly < 500MB):
  - `data/bigearthnet/`: 12 representative multi-spectral (optical) & SAR image patches in `images/`, with labels and train/val/test splits mapped in `labels.csv` according to CORINE Land Cover (CLC) classes.
  - `data/vrsbench/`: 5 high-resolution optical images in `images/` with corresponding scene captions, visual grounding bounding boxes, and QA pairs in `annotations.json`.
  - `data/rsvqa/`: 8 Sentinel-2/aerial images in `images/` with presence, count, comparison, and area visual QA pairs in `qa_pairs.json`.
  - `data/cdvqa/`: 5 multitemporal bi-temporal image pairs ($T_1$ and $T_2$) in `image_pairs/` with change detection question-answer annotations in `qa_pairs.json`.
  - `data/sample/`: Synthetic optical (`optical/`), SAR backscatter (`sar/`), and multitemporal change pairs (`pairs/`) with sidecar metadata JSONs for end-to-end pipeline testing without external dependencies.
- **Dataset Loader Utility**:
  - `data/loader.py`: Unified Python loader module exposing `load_bigearthnet()`, `load_vrsbench()`, `load_rsvqa()`, `load_cdvqa()`, and `load_sample_data()`. All functions return a standardized `List[Dict[str, Any]]` containing `"image_path"`, `"question"`, and `"answer"` keys.
- **Documentation & Download Guide**:
  - `data/DOWNLOAD_INSTRUCTIONS.md`: Complete guide to downloading full academic datasets from official sources (bigearth.net, Zenodo, GitHub mirrors).
  - `data/README.md`: Overview of dataset structures, loader API contracts, and sample data limitations.
- **Data Generation Utility**:
  - `scripts/generate_sample_data.py`: Lightweight data generation script.

---

### 2. How to Run the Setup Script

#### On Linux / macOS:
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
source .venv/bin/activate
```

#### On Windows (Command Prompt or PowerShell):
```cmd
scripts\setup_env.bat
.venv\Scripts\activate
```

---

### 3. Quickstart for Teammate M2 (Model Fine-Tuning)

You can immediately import and use the loaders from `data/loader.py` to retrieve standardized training batches without worrying about dataset-specific file formats:

```python
from data.loader import (
    load_bigearthnet,
    load_vrsbench,
    load_rsvqa,
    load_cdvqa,
    load_sample_data,
)

# 1. BigEarthNet for image-text adaptation & classification
# Returns: dict with 'image_path', 'labels', 'modality', 'split', 'question', 'answer', 'caption'
train_data = load_bigearthnet(split="train")
print(f"Loaded {len(train_data)} BigEarthNet samples")
sample = train_data[0]
print(sample["image_path"], sample["labels"], sample["caption"])

# 2. VRSBench for captioning, visual grounding, and VQA
# Returns: dict with 'image_path', 'caption', 'question', 'answer', 'qa_pairs', 'grounding'
vrs_data = load_vrsbench()
print(f"Loaded {len(vrs_data)} VRSBench samples")

# 3. RSVQA for single-image visual question answering
# Returns: dict with 'image_path', 'question', 'answer', 'question_type'
rsvqa_data = load_rsvqa(question_type="presence")
print(f"Loaded {len(rsvqa_data)} RSVQA samples")

# 4. CDVQA for multitemporal change detection VQA
# Returns: dict with 'pair_id', 'image_path', 'image_t1_path', 'image_t2_path', 'question', 'answer', 'change_type'
cdvqa_data = load_cdvqa()
print(f"Loaded {len(cdvqa_data)} CDVQA pairs")

# 5. Synthetic fallback (available out of the box for quick offline debugging)
synthetic_optical = load_sample_data(category="optical")
synthetic_sar = load_sample_data(category="sar")
synthetic_pairs = load_sample_data(category="pairs")
```

---

### 4. Blockers, Limitations & Notes for Next Steps
- **Full BigEarthNet Access**: The full BigEarthNet dataset (~66 GB for Sentinel-2, ~13 GB for Sentinel-1) requires an account registration on [bigearth.net](https://bigearth.net/) or bulk download from Zenodo. Instructions and mirrors are in `data/DOWNLOAD_INSTRUCTIONS.md`. The repository includes sample patches and the synthetic fallback generator so model development is NOT blocked.
- **Full VRSBench, RSVQA, and CDVQA**: Full academic versions are multi-gigabyte archives hosted on Zenodo and GitHub/Google Drive. Download and directory placement instructions are provided in `data/DOWNLOAD_INSTRUCTIONS.md`.
- **Reserved Directories**: The `models/`, `backend/`, `frontend/`, `agent/`, and `tests/` folders have been preserved with their `.gitkeep` markers untouched for subsequent teammates.

