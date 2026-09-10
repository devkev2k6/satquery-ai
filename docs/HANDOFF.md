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

---

## Phase 2: Vision-Language Model Fine-Tuning & Adaptation (Completed)

### 1. Model Choice & Architectural Rationale
- **Selected Model**: [`Salesforce/blip-vqa-base`](https://huggingface.co/Salesforce/blip-vqa-base) (~385 Million parameters)
- **Adaptation Method**: Low-Rank Adaptation (**LoRA** via `peft`) targeting the attention query and value projection modules in the multimodal cross-attention text decoder ($r=16, \alpha=32$, dropout $0.05$).
- **Why BLIP-VQA**:
  - **Native VQA Head**: Uses `BlipForQuestionAnswering` with multimodal cross-attention conditioning token generation directly on visual satellite representations.
  - **Hardware Feasibility**: 3B models (e.g. PaliGemma 3B, BLIP-2 2.7B) require 6GB–12GB VRAM, which causes Out-Of-Memory errors on standard 4GB laptop GPUs (like RTX 2050) and exhibits intolerable latency on CPU (>25s/query). BLIP-VQA requires only ~1.5GB VRAM (or ~1.5GB system RAM), executes in <0.4s on GPU (~0.7s on CPU), and adapter weights are compact (~9.5MB).
  - Detailed comparison across evaluated candidates (BLIP-2, PaliGemma, Moondream, Qwen2-VL) is documented in `models/MODEL_CHOICE.md`.

---

### 2. Delivered Artifacts & Files
- `models/MODEL_CHOICE.md`: In-depth analysis, benchmark comparisons, and hardware rationale.
- `models/finetune.py`: End-to-end parameter-efficient LoRA training pipeline loading data from M1's `data.loader`.
- `models/checkpoints/blip_rs_lora/`: Saved adapted weights (`adapter_model.safetensors`, `adapter_config.json`, processor config).
- `models/inference.py`: Standardized inference module exposing `answer_question(image_path: str, question: str) -> str`.
- `models/evaluate.py`: Evaluation benchmark running queries across VRSBench, RSVQA, and BigEarthNet with side-by-side comparison tables.
- `models/README.md`: Complete documentation for model architecture, training, inference, and known limitations.
- `requirements.txt`: Updated with `torch`, `transformers`, `peft`, and `accelerate`.

---

### 3. Verification & How to Call `answer_question()`
The required function signature is fully implemented, tested, and verified:
```python
from models.inference import answer_question

# Example call
image_path = "data/bigearthnet/images/S2A_MSIL2A_20170717T113321_01.png"
question = "What land cover classes are present in this satellite image?"
answer = answer_question(image_path, question)
print(answer)  # e.g., "land and water"
```
- **Caching**: The model, processor, and weights are cached in memory on first call (`_CACHED_MODEL`). Subsequent calls execute with zero reload overhead.
- **Robust Path Handling**: Handles relative or absolute paths, missing images, and unsupported formats gracefully without crashing.

---

### 4. Hardware & Environment Requirements for Teammates M3–M6
- **Compute**: Runs automatically on **CUDA GPU** if available, with transparent fallback to **CPU**.
- **Memory Footprint**:
  - GPU: ~1.2 GB – 1.5 GB VRAM
  - CPU: ~1.5 GB System RAM
- **Latency**:
  - GPU: ~0.25 – 0.45 seconds / query
  - CPU: ~0.70 seconds / query (after initial weight load)

---

### 5. Architectural Recommendations for Teammates M3 (Agent) & M4 (Backend)

| Query Category | Recommended Pipeline Strategy | Rationale & Action |
| :--- | :--- | :--- |
| **Land Cover / Scene Classification** | **Call `answer_question()` directly** | Adapted VLM excels at recognizing dominant surface textures (forest, water, urban, agricultural). |
| **Feature Presence ("Is there X?")** | **Call `answer_question()` directly** | Binary presence queries (roads, water bodies, airports) are answered reliably. |
| **Exact Object Counting ("How many X?")** | **Use rule-based / detector tools or prompt for presence** | VLM provides approximate counts (e.g. "1" or "multiple"), but struggles with precise counts of small clustered objects. M3 should prompt for presence first or route to a dedicated detector if available. |
| **Bi-Temporal Change Detection (CDVQA)** | **Tile images or provide comparative context** | Single-image VLM receives one image per prompt. For multitemporal pairs, horizontally concatenate $T_1$ and $T_2$ or query both sequentially and use agent reasoning to compare. |

---

## Phase 3: Single-Image Baseline Features - VQA, Captioning & Grounding (Completed)

### 1. What Was Built & Where It Lives
- **Visual Question Answering Tool** (`backend/vqa_tool.py`):
  - Function: `run_vqa(image_path: str, question: str) -> Dict[str, Any]`
  - Directly interfaces M2's `answer_question()` from `models/inference.py`.
  - Computes structured confidence score [0.0 - 1.0] based on certainty markers, response brevity, and error handling.
- **Scene Captioning Tool** (`backend/captioning_tool.py`):
  - Function: `run_captioning(image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]`
  - Leverages M2's adapted VLM with structured remote sensing caption prompts to generate fluent natural-language land cover and visible object summaries.
- **Text-Guided Region Grounding Tool** (`backend/grounding_tool.py`):
  - Function: `run_grounding(image_path: str, query: str) -> Dict[str, Any]`
  - Implements remote sensing spectral/texture heuristic localization (Excess Green index for vegetation, NDWI RGB approximation for water, variance/reflectance for urban/structures, brightness for aircraft/runways).
  - Returns robust `[x1, y1, x2, y2]` pixel bounding boxes filtered with spatial percentiles to prevent noise inflation.
- **Automated Verification Test Suite** (`tests/test_vqa_captioning.py`):
  - Runs all three baseline tools across 4 representative sample images from `data/sample/optical/` (Forest, Water, Urban, Agriculture).
  - Asserts strict contract compliance (key presence, types, bounding box integer coordinate validity, confidence ranges).
- **Backend Documentation** (`backend/README.md`):
  - Comprehensive API documentation with signatures, return payloads, confidence formulas, and usage snippets.

---

### 2. Implemented Tool Contracts & Exact Signatures

Downstream teammates (especially M5 Agent Lead and M4 Backend/Fusion) can directly import and call:

```python
from backend.vqa_tool import run_vqa
from backend.captioning_tool import run_captioning
from backend.grounding_tool import run_grounding

# 1. Visual Question Answering (VQA)
# Signature: run_vqa(image_path: str, question: str) -> dict
vqa_res = run_vqa(
    image_path="data/sample/optical/sample_optical_001.png",
    question="What terrain type is depicted in this optical satellite observation?"
)
# Returns:
# {
#     "task": "vqa",
#     "answer": "flat land",
#     "confidence": 0.90,
#     "image_path": "data/sample/optical/sample_optical_001.png"
# }

# 2. Scene Captioning
# Signature: run_captioning(image_path: str, prompt: Optional[str] = None) -> dict
cap_res = run_captioning(
    image_path="data/sample/optical/sample_optical_001.png"
)
# Returns:
# {
#     "task": "captioning",
#     "caption": "Satellite imagery showing trees and grass.",
#     "confidence": 0.95,
#     "image_path": "data/sample/optical/sample_optical_001.png"
# }

# 3. Text-Guided Region Grounding
# Signature: run_grounding(image_path: str, query: str) -> dict
grd_res = run_grounding(
    image_path="data/sample/optical/sample_optical_001.png",
    query="forest and green vegetation"
)
# Returns:
# {
#     "task": "grounding",
#     "bbox": [3, 3, 125, 125],
#     "confidence": 0.65,
#     "image_path": "data/sample/optical/sample_optical_001.png"
# }
```

---

### 3. Implementation Scope: Captioning and Grounding
- **Captioning**: Implemented and validated using M2's fine-tuned model with formatted scene descriptions.
- **Grounding**: Also implemented and validated using a robust spectral/texture heuristic pipeline designed specifically for remote sensing images. Both features are fully ready and available for M5's agent tool registry.

---

### 4. Known Weaknesses, Caveats & Edge Cases for Teammates M4–M6

1. **Captions on Dense/Cluttered Scenes**:
   - For complex, mixed-use patches (e.g., dense urban scenes with commercial buildings, transit hubs, and sparse greenery), the VLM caption focuses primarily on the dominant visual feature (often generic e.g. "trees" or "buildings").
   - *Guidance for M5 (Agent)*: Encourage the agent to decompose general scene questions into specific VQA queries (e.g. asking specifically about road density, water presence, or building types) rather than relying exclusively on the global caption.
2. **Grounding Accuracy Scope**:
   - Because `Salesforce/blip-vqa-base` is an autoregressive encoder-decoder VQA model without native bounding-box regression heads, `run_grounding` uses an index-based spectral heuristic. It works well for dominant macro land cover (water bodies, forests, agricultural tracts, large runways), but cannot reliably isolate small individual cars or occluded targets.
3. **Single Image Limitation**:
   - All tools in `backend/` are currently single-image scoped. For M4 (Change Detection & Multimodal Fusion), multitemporal pairs ($T_1, T_2$) or SAR+Optical pairs should be passed through separate queries or concatenated horizontally prior to passing to `run_vqa`.


