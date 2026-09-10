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

---

## Phase 4: Multi-Image Capabilities - Change Detection & Optical+SAR Fusion (Completed)

### 1. What Was Built & Where It Lives
- **Bi-Temporal Change Detection Tool** (`backend/change_detection_tool.py`):
  - Function: `run_change_detection(image_path_before: str, image_path_after: str, question: Optional[str] = None) -> Dict[str, Any]`
  - Supports both general change description generation ($T_1 \to T_2$ narrative) and targeted change-based Visual Question Answering.
  - Combines normalized mean absolute pixel variance metrics with dual-pass VLM semantic inspection.
- **Optical + SAR Cross-Modal Fusion Tool** (`backend/fusion_tool.py`):
  - Function: `run_optical_sar_fusion(optical_path: str, sar_path: str, question: Optional[str] = None) -> Dict[str, Any]`
  - Combines radiometric radar backscatter physics (specular reflection for water, double-bounce corner reflection for urban structures, diffuse scattering for vegetation) with optical multi-spectral reflectance and VLM semantic classification.
- **Automated Verification Test Suite** (`tests/test_change_and_fusion.py`):
  - Validates both tools across 9 test scenarios (synthetic change pairs, negative control identical pairs, CDVQA benchmark pairs, optical+SAR pairs, BigEarthNet patches, and error handling for missing files).
  - Verifies 100% contract compliance and confidence bounds.
- **Updated Backend Documentation** (`backend/README.md`):
  - Updated tool summary table, API contracts, JSON schemas, confidence calculations, and caveats.

---

### 2. Implemented Tool Contracts & Exact Signatures for Teammate M5 (Agent Controller)

Teammate M5 can directly register and invoke these two tools in the LangChain/LangGraph agent controller:

```python
from backend.change_detection_tool import run_change_detection
from backend.fusion_tool import run_optical_sar_fusion

# 1. Bi-Temporal Change Detection
# Exact Signature: run_change_detection(image_path_before: str, image_path_after: str, question: Optional[str] = None) -> dict
cd_res = run_change_detection(
    image_path_before="data/sample/pairs/sample_pair_001_t1.png",
    image_path_after="data/sample/pairs/sample_pair_001_t2.png",
    question="What environmental or infrastructure change occurred between Time 1 and Time 2?"
)
# Returns:
# {
#     "task": "change_detection",
#     "description": "Change detected: Prior observation (Time 1) depicted land and water (tropical climate), whereas subsequent observation (Time 2) depicts land and water (water source).",
#     "change_detected": True,
#     "confidence": 0.91,
#     "before_path": "data/sample/pairs/sample_pair_001_t1.png",
#     "after_path": "data/sample/pairs/sample_pair_001_t2.png"
# }

# 2. Optical + SAR Cross-Modal Fusion
# Exact Signature: run_optical_sar_fusion(optical_path: str, sar_path: str, question: Optional[str] = None) -> dict
fusion_res = run_optical_sar_fusion(
    optical_path="data/sample/optical/sample_optical_003.png",
    sar_path="data/sample/sar/sample_sar_003.png",
    question="Identify built-up and water-covered regions using both images together."
)
# Returns:
# {
#     "task": "fusion",
#     "answer": "Joint Optical-SAR Analysis: Optical imagery provides spectral delineation indicating temperate (squares). Co-registered SAR confirms physical dielectric properties with elevated radar backscatter with strong double-bounce reflections (characteristic of built-up urban structures or metallic targets) (water). Combining both sensors enables robust cross-modal verification...",
#     "confidence": 0.89,
#     "optical_path": "data/sample/optical/sample_optical_003.png",
#     "sar_path": "data/sample/sar/sample_sar_003.png"
# }
```

---

### 3. Architectural Clarification: Nature of Multimodal Fusion

**EXPLICIT NOTE FOR M5 AGENT CONTROLLER**:
- The fusion implementation in `backend/fusion_tool.py` is a **text-level and radiometric feature combination approach**, NOT an end-to-end pixel-level or latent tensor cross-attention network.
- **Why**: M2's fine-tuned model (`Salesforce/blip-vqa-base` with LoRA) natively accepts a single 3-channel optical image tensor. True pixel/latent fusion would require retraining a dual-branch cross-attention encoder, which is outside the operational timeline.
- **How it works**: The pipeline extracts optical semantic classifications via M2's model, computes radiometric radar backscatter distributions (mean intensity, specular water reflection ratio, double-bounce corner reflection ratio) from the SAR raster, queries the SAR modality for texture characteristics, and synthesizes these complementary physical perspectives into a unified natural language answer.
- **Implication for M5 Agent Controller**: When presenting fusion results to users or generating explanations, the agent should state: *"Based on joint cross-modal synthesis of optical multi-spectral reflectance and co-registered SAR radar backscatter..."* rather than claiming raw neural pixel-fusion.

---

### 4. Known Weaknesses & Caveats for Teammates M5 (Agent) & M6 (Frontend)

1. **Lighting & Seasonal Phenology Flagged as Change**:
   - Sun angle changes, cast shadow variations, cloud shadows, or seasonal vegetative growth/browning between $T_1$ and $T_2$ produce radiometric pixel variance that can sometimes trigger `change_detected: true` even without structural construction.
   - *Guidance for M5 (Agent)*: Instruct the agent to check whether `description` explicitly notes a categorical transition (e.g. forest $\to$ urban) versus minor surface variance before asserting physical development.

2. **SAR Speckle Noise & Interpretability**:
   - Due to coherent radar interference (Rayleigh speckle), raw SAR pixel values exhibit high spatial variance. The tool mitigates this using robust percentile filtering and area statistics, but fine-grained object boundaries are less precise in SAR than optical.

3. **Single-Image VLM Comparative Prompting**:
   - For change-based VQA, `answer_question` is called on both images independently. If the user question is highly complex, decomposing it into sub-questions about $T_1$ and $T_2$ via the M5 Agent Controller will produce superior analytical depth.

---

### 5. Verification
Run the Phase 4 verification test suite:
```bash
python tests/test_change_and_fusion.py
```

---

## Phase 5: Agentic Model & Tool Orchestration (Completed)

### 1. What Was Built & Where It Lives
- **Rule-Based Task Classifier** (`agent/task_classifier.py`):
  - Function: `classify_task(query: Optional[str], num_images: int, image_types: Optional[List[str]] = None) -> str`
  - Deterministic keyword and sensor modality matching routing to `"vqa"`, `"captioning"`, `"grounding"`, `"change_detection"`, or `"fusion"`.
- **Input Validator** (`agent/input_validator.py`):
  - Function: `validate_input(query: Optional[str], image_paths: List[str], image_types: Optional[List[str]] = None, task: Optional[str] = None) -> Dict[str, Any]`
  - Validates image count rules per task, supported raster formats (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`), disk existence, and optical/SAR sensor compatibility before running models.
- **Agent Controller** (`agent/controller.py`):
  - Function: `process_query(query: Optional[str], image_paths: List[str], image_types: Optional[List[str]] = None) -> Dict[str, Any]`
  - Main orchestrator: classifies intent, validates inputs, dynamically executes the matching backend tool, and returns the result with a full auditable `execution_summary`.
- **Comprehensive Agent Test Suite** (`tests/test_agent_controller.py`):
  - Validates end-to-end routing and execution across all 5 representative queries from the original problem statement, single-image VQA, and negative validation edge cases.
- **Agent Architecture Documentation** (`agent/README.md`):
  - Comprehensive guide covering architecture, decision tree, input validation rules, and instructions on extending the controller with new tools.

---

### 2. Exact Function Signature & Return Schema for Teammate M6 (Frontend)

Teammate M6 (Streamlit / Web UI Lead) should import and invoke `process_query` as the single entry point for all user interactions:

```python
from agent.controller import process_query

# Example 1: Scene Captioning
res_caption = process_query(
    query="Describe the land-cover and major objects visible in this image.",
    image_paths=["data/sample/optical/sample_optical_001.png"],
    image_types=["optical"]  # Optional: can be None or omitted
)

# Example 2: Text-Guided Grounding
res_grounding = process_query(
    query="Highlight the water body referred to in the query.",
    image_paths=["data/sample/optical/sample_optical_002.png"],
    image_types=["optical"]
)

# Example 3: Bi-Temporal Change Detection
res_change = process_query(
    query="What changed between these two dates, and where did the change occur?",
    image_paths=[
        "data/sample/pairs/sample_pair_001_t1.png",
        "data/sample/pairs/sample_pair_001_t2.png"
    ],
    image_types=["optical", "optical"]
)

# Example 4: Joint Optical + SAR Fusion
res_fusion = process_query(
    query="Use the optical and SAR images together to identify built-up and water-covered regions.",
    image_paths=[
        "data/sample/optical/sample_optical_003.png",
        "data/sample/sar/sample_sar_003.png"
    ],
    image_types=["optical", "sar"]
)
```

#### Standard Return Schema
Every invocation returns a JSON-serializable dictionary with 3 top-level keys:
```json
{
  "result": {
    "task": "captioning",
    "caption": "Satellite imagery showing flat land.",
    "confidence": 0.88,
    "image_path": "data/sample/optical/sample_optical_001.png"
  },
  "execution_summary": {
    "task": "captioning",
    "tool_used": "run_captioning",
    "parameters": {
      "image_path": "data/sample/optical/sample_optical_001.png",
      "prompt": "Describe the land-cover and major objects visible in this image."
    },
    "timestamp": "2026-09-10T07:15:30.123456+00:00"
  },
  "success": true
}
```

In case of input validation errors:
```json
{
  "result": {
    "task": "change_detection",
    "error": "Task 'change_detection' requires exactly 2 images (before and after), but received 1."
  },
  "execution_summary": {
    "task": "change_detection",
    "tool_used": null,
    "parameters": { ... },
    "timestamp": "2026-09-10T07:15:30.123456+00:00"
  },
  "success": false
}
```

---

### 3. Known Misclassification Cases & Limitations for Frontend Demo (M6)

1. **Ambiguous Hybrid Queries (e.g., "Compare" + "Highlight")**:
   - If a user provides two images and asks: *"Compare these two images and highlight the difference"*, the classifier gives precedence to `change_detection` due to the 2-image context and comparison phrasing. Spatial bounding-box localization will not be produced for the change.
   - *Recommendation for M6*: Present task selection in the UI with an optional manual dropdown override so users can choose "Change Detection" vs "Grounding" if their prompt is inherently multi-objective.
2. **Implicit Optical-SAR Fusion without Specifying Types**:
   - If a user uploads an Optical image and a SAR image but leaves `image_types` blank and asks a generic question without mentioning "SAR" or "radar" (e.g. *"What is in these two images?"*), the controller will default to `change_detection`.
   - *Recommendation for M6*: Ensure the frontend file upload widget tags each uploaded slot with its modality (Optical vs SAR) and passes `image_types=['optical', 'sar']`.
3. **Complex Multi-Step Decompositions**:
   - The current controller operates as a single-turn deterministic router. Queries asking to perform multiple disjoint tasks in sequence (e.g. *"First describe the scene, then locate all buildings, then tell me if it rained"*) will route to the highest precedence task (`grounding`).

---

### 4. Verification
Run the Phase 5 Agent Controller verification suite:
```bash
python tests/test_agent_controller.py
```

---

## Phase 6: Web Application, PDF Reporting & Final Integration (Completed)

### 1. What Was Built & Where It Lives
- **Interactive Streamlit Web Dashboard** (`frontend/app.py`):
  - Ingests 1 or 2 satellite rasters (PNG, JPG, TIFF/GeoTIFF).
  - Explicit modality selector per uploaded image (`optical` vs `sar`).
  - Integrated **1-Click Benchmark Demo Presets** in the sidebar preloading all 5 official problem statement queries with sample imagery for instant live evaluation.
  - Inline visual evidence rendering: raw observations, dynamic visual grounding bounding-box overlays, and side-by-side bi-temporal / cross-modal comparison views.
  - Prominent answer rendering with collapsible, auditable `execution_summary` JSON expander.
  - One-click PDF intelligence report compilation and download.
  - Resilient error handling preventing UI crashes when validation fails.
- **PDF Intelligence Report Compiler** (`frontend/report_generator.py`):
  - Function: `generate_pdf_report(query: str, result: dict, output_path: Optional[str] = None) -> str`
  - Uses `fpdf2` to construct a formal Earth Observation intelligence PDF summary containing executive query metadata, detailed findings narrative, confidence metrics, analyzed asset paths, grounding coordinates, change status, and technical execution trace.
- **Full End-to-End Demo Verification Suite** (`tests/test_full_demo.py`):
  - Runs all 5 official benchmark queries through the exact same path that `app.py` uses.
  - Automatically compiles and validates non-empty PDF intelligence reports for every query.
- **Top-Level Documentation** (`README.md`):
  - Complete project architecture, setup instructions, web app usage, capabilities overview, and team contribution credits (M1 through M6).
- **Dependency Manifest Update** (`requirements.txt`):
  - Added `streamlit>=1.30.0` and `fpdf2>=2.7.0`.

---

### 2. End-to-End Pipeline Verification Confirmation

The full 6-person pipeline is verified and operational end-to-end:
- **`tests/test_full_demo.py`**: **100% Passed (5/5 queries succeeded + 5 valid PDF reports generated)**.
- **`tests/test_agent_controller.py`**: **100% Passed**.
- **`tests/test_vqa_captioning.py`**: **100% Passed**.
- **`tests/test_change_and_fusion.py`**: **100% Passed**.

---

### 3. Live Demo Tips & Rough Edges to Keep in Mind During Presentation

1. **Use the 1-Click Sidebar Presets**:
   - For the hackathon judging session, use the sidebar dropdown: selecting any of the 5 presets automatically sets the prompt, preloads sample images, and tags modalities accurately. This guarantees a deterministic, sub-second demonstration with zero typing friction.
2. **Uploading Custom Multi-Gigabyte GeoTIFFs**:
   - The UI supports `.tif`/`.tiff` files. However, extremely large multi-gigabyte uncompressed GeoTIFF rasters will incur memory overhead when loaded via PIL. For live demos, use patches $\le 1024 \times 1024$ pixels.
3. **Modality Dropdowns on Custom Uploads**:
   - When demonstrating Cross-Modal Optical+SAR Fusion with custom files, remember to tag Image 1 as `optical` and Image 2 as `sar`. If both are tagged as `optical`, the input validator will reject the fusion request with a diagnostic warning.

---

### 4. Future Work & Architectural Defense for Judges

When judges ask about scaling, accuracy, or future iterations, highlight these key design points:
1. **True Cross-Attention Neural Pixel Fusion**:
   - *Current Design*: Text-level and radiometric feature synthesis combining optical spectral reflectance with SAR backscatter statistics.
   - *Future Work*: Train a dual-branch Vision Transformer encoder (e.g. Swin / ViT cross-attention) fusing multi-channel optical bands (10m B2-B4, B8) and SAR polarizations (VV+VH) in latent space.
2. **Spatial Bounding-Box Regression Heads**:
   - *Current Design*: Heuristic spectral index clustering (ExG for vegetation, NDWI approximation for water, variance for urban structures).
   - *Future Work*: Integrate an open-vocabulary object detector (e.g. Grounding DINO or OWL-ViT fine-tuned on DOTA / DIOR remote sensing benchmarks) for sub-meter object detection.
3. **Large-Scale Gigapixel Tiling**:
   - *Current Design*: Evaluates localized satellite tiles ($128 \times 128$ to $1024 \times 1024$).
   - *Future Work*: Implement a spatial quadtree windowing pipeline with sliding-window VLM inference for continental-scale Earth Observation rasters.

---

### 5. How to Run the Demo for Judges
```bash
# Activate environment
.venv\Scripts\activate

# Launch Web Application
streamlit run frontend/app.py
```
