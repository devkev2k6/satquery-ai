# SatQuery AI 🛰️
### Multimodal Satellite Question Answering, Grounding & Cross-Modal Radar Fusion Agent

**SatQuery AI** is an autonomous Earth Observation (EO) intelligence agent designed to answer complex natural language questions over high-resolution optical and Synthetic Aperture Radar (SAR) satellite imagery. 

The system features fine-tuned vision-language models, spectral heuristic localization, bi-temporal change detection, microwave radar dielectric fusion, an explainable rule-based agent orchestrator, and an interactive Streamlit mission-control dashboard with automated PDF intelligence reporting.

---

## 1. System Architecture & 6-Stage Pipeline

```mermaid
flowchart TD
    subgraph UI ["Phase 6: Frontend & Application Layer"]
        User([User / Analyst]) -->|Query + Images| StreamlitApp[Streamlit Dashboard / frontend/app.py]
        StreamlitApp -->|Download Request| PDFGen[PDF Report Generator / frontend/report_generator.py]
    end

    subgraph Agent ["Phase 5: Agentic Controller & Tool Orchestration"]
        StreamlitApp -->|process_query| Controller[Agent Controller / agent/controller.py]
        Controller --> Classifier[Task Classifier / agent/task_classifier.py]
        Controller --> Validator[Input Validator / agent/input_validator.py]
    end

    subgraph Backend ["Phase 3 & Phase 4: Analytical Tool Registry"]
        Controller -->|vqa| VQATool[VQA Tool / backend/vqa_tool.py]
        Controller -->|captioning| CaptionTool[Captioning Tool / backend/captioning_tool.py]
        Controller -->|grounding| GroundingTool[Grounding Tool / backend/grounding_tool.py]
        Controller -->|change_detection| CDTool[Change Detection Tool / backend/change_detection_tool.py]
        Controller -->|fusion| FusionTool[Optical+SAR Fusion Tool / backend/fusion_tool.py]
    end

    subgraph Models ["Phase 2: Vision-Language Intelligence Layer"]
        VQATool --> VLMInference[Adapted VLM / models/inference.py]
        CaptionTool --> VLMInference
        CDTool --> VLMInference
        FusionTool --> VLMInference
        VLMInference --> BLIPLoRA[(Salesforce/blip-vqa-base + RS-LoRA Adapter)]
    end

    subgraph Data ["Phase 1: Dataset & Ingestion Layer"]
        BLIPLoRA -.-> DataLoader[Dataset Loader / data/loader.py]
        DataLoader -.-> Datasets[(BigEarthNet | VRSBench | RSVQA | CDVQA | Synthetic)]
    end
```

---

## 2. Core Capabilities

| Capability | Modalities | Primary Backend Tool | Key Outputs |
| :--- | :--- | :--- | :--- |
| **Scene Captioning** | Optical | `backend.captioning_tool` | Fluent natural language land-cover and surface feature summaries |
| **Visual Grounding** | Optical | `backend.grounding_tool` | Precise `[x1, y1, x2, y2]` pixel bounding box localization |
| **Visual Question Answering (VQA)** | Optical / SAR | `backend.vqa_tool` | Targeted answers regarding terrain, infrastructure, and object presence |
| **Bi-Temporal Change Detection** | Multi-temporal Optical ($T_1, T_2$) | `backend.change_detection_tool` | Normalized pixel variance metrics, change detection flag (`true`/`false`), and transition narrative |
| **Cross-Modal Optical + SAR Fusion** | Optical + SAR (Radar) | `backend.fusion_tool` | Joint analysis combining multispectral reflectance with microwave backscatter dielectric roughness |

---

## 3. Five Representative Benchmark Demonstrations

SatQuery AI natively handles all five representative queries from the problem statement:

1. **Scene Captioning**:
   - *Query*: *"Describe the land-cover and major objects visible in this image."*
   - *Result*: Autonomously routed to `captioning`; produces comprehensive scene description (e.g. *"Satellite imagery showing grass and trees"*).
2. **Text-Guided Region Grounding**:
   - *Query*: *"Highlight the water body referred to in the query."*
   - *Result*: Autonomously routed to `grounding`; localizes coordinates `[3, 3, 125, 125]` with highlighted visual bounding box.
3. **Bi-Temporal Change Detection**:
   - *Query*: *"What changed between these two dates, and where did the change occur?"*
   - *Result*: Autonomously routed to `change_detection`; analyzes radiometric difference and outputs comparative transition assessment.
4. **Optical + SAR Cross-Modal Fusion**:
   - *Query*: *"Use the optical and SAR images together to identify built-up and water-covered regions."*
   - *Result*: Autonomously routed to `fusion`; synthesizes optical spectral boundaries with SAR double-bounce corner reflections.
5. **Change Trend Analysis (CDVQA)**:
   - *Query*: *"Has the built-up area increased, decreased, or remained unchanged?"*
   - *Result*: Autonomously routed to `change_detection`; inspects temporal state evolution across Time 1 and Time 2.

---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- 4GB+ RAM (CUDA GPU optional; auto-detects GPU with transparent CPU fallback)

### Step 1: Clone and Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/devkev2k6/satquery-ai.git
cd satquery-ai

# Windows Setup (Command Prompt or PowerShell)
scripts\setup_env.bat
.venv\Scripts\activate

# Or on Linux / macOS
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Interactive Streamlit Web Application
```bash
streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Web Application Features

- **⚡ 1-Click Demo Presets**: Sidebar selector that instantly preloads all 5 benchmark queries and sample assets for zero-latency live presentations.
- **📁 Multi-Format Image Ingestion**: Supports `.png`, `.jpg`, `.jpeg`, and `.tif`/`.tiff` files for single-image, bi-temporal pair ($T_1/T_2$), or optical+SAR multimodal queries.
- **🏷️ Sensor Modality Tagging**: Interactive dropdowns to tag images as `optical` or `sar` for cross-modal routing.
- **🖼️ Visual Evidence Display**: Inline rendering of raw observations alongside dynamic visual grounding bounding box overlays.
- **🛠️ Auditable Execution Summary**: Collapsible expander detailing selected task, tool dispatched, parameters, and UTC timestamp.
- **📑 Downloadable PDF Reports**: Automated one-click generation of professional Earth Observation intelligence PDF summaries.

---

## 6. Verification Test Suites

SatQuery AI includes automated verification suites for all stages of the pipeline:

```bash
# 1. Run the Full End-to-End Demo Suite (Simulates UI & PDF generation)
python tests/test_full_demo.py

# 2. Run the Agent Controller Verification Suite (All 5 queries + validation checks)
python tests/test_agent_controller.py

# 3. Run Baseline Tools Verification (VQA, Captioning, Grounding)
python tests/test_vqa_captioning.py

# 4. Run Multi-Image Tools Verification (Change Detection & SAR Fusion)
python tests/test_change_and_fusion.py
```

---

## 7. Team Contributions & Pipeline Ownership

SatQuery AI was developed across six integrated engineering phases:

- **Teammate M1 (Environment & Data Layer Lead)**:
  - Repository structure, virtual environment setup scripts (`setup_env.sh`, `setup_env.bat`).
  - Curation of sub-500MB benchmark subsets (`data/bigearthnet/`, `data/vrsbench/`, `data/rsvqa/`, `data/cdvqa/`, `data/sample/`).
  - Unified Python loader utility (`data/loader.py`).
- **Teammate M2 (Model Fine-Tuning & Adaptation Lead)**:
  - Architectural model selection (`Salesforce/blip-vqa-base` with Low-Rank Adaptation).
  - Parameter-efficient LoRA training pipeline (`models/finetune.py`) and cached inference module (`models/inference.py`).
  - Model checkpoint management and benchmark evaluation (`models/evaluate.py`).
- **Teammate M3 (Single-Image Baseline Features Lead)**:
  - Visual Question Answering tool (`backend/vqa_tool.py`) with confidence calculation.
  - Scene captioning and land-cover description engine (`backend/captioning_tool.py`).
  - Text-guided region grounding with remote sensing spectral heuristics (`backend/grounding_tool.py`).
- **Teammate M4 (Multi-Image Capabilities Lead)**:
  - Bi-temporal change detection tool with radiometric pixel variance metrics (`backend/change_detection_tool.py`).
  - Cross-modal Optical + SAR fusion tool leveraging radar backscatter statistics (`backend/fusion_tool.py`).
  - Multi-image test suite (`tests/test_change_and_fusion.py`).
- **Teammate M5 (Agent Controller & Orchestration Lead)**:
  - Rule-based task classifier (`agent/task_classifier.py`) with deterministic keyword and modality routing.
  - Input schema and modality validator (`agent/input_validator.py`).
  - Central agent controller (`agent/controller.py`) with auditable execution summary logging.
  - Agent test suite (`tests/test_agent_controller.py`) and developer documentation (`agent/README.md`).
- **Teammate M6 (Web Application & Final Integration Lead)**:
  - Streamlit interactive mission-control dashboard (`frontend/app.py`) with 1-click benchmark demo showcase.
  - Automated PDF intelligence report compiler (`frontend/report_generator.py`).
  - Full end-to-end demo test suite (`tests/test_full_demo.py`).
  - Top-level documentation and final handoff consolidation (`docs/HANDOFF.md`).
