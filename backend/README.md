# SatQuery AI - Backend Tool Suite (`backend/`)

This directory houses the single-image baseline tools, geospatial query utilities, and multimodal pipelines for **SatQuery AI**. Downstream teammates (M4 Multimodal Fusion/Change Detection, M5 Agent Controller, and M6 Frontend) consume these standardized functions directly.

---

## 1. Overview of Tools

| Module | Primary Function | Task Type | Description |
| :--- | :--- | :--- | :--- |
| `backend/vqa_tool.py` | `run_vqa(image_path, question)` | `"vqa"` | Answers natural language queries regarding single satellite images. |
| `backend/captioning_tool.py` | `run_captioning(image_path, prompt=None)` | `"captioning"` | Produces a natural-language description of land cover and major visible objects. |
| `backend/grounding_tool.py` | `run_grounding(image_path, query)` | `"grounding"` | Localizes described features/objects to approximate `[x1, y1, x2, y2]` bounding boxes. |
| `backend/change_detection_tool.py` | `run_change_detection(image_path_before, image_path_after, question=None)` | `"change_detection"` | Detects bi-temporal land cover alterations and answers change-based VQA queries. |
| `backend/fusion_tool.py` | `run_optical_sar_fusion(optical_path, sar_path, question=None)` | `"fusion"` | Extracts and synthesizes complementary information from co-registered optical and SAR pairs. |

---

## 2. API Contracts & Signatures

### 2.1 Visual Question Answering: `run_vqa`

```python
from backend.vqa_tool import run_vqa

result = run_vqa(
    image_path="data/sample/optical/sample_optical_001.png",
    question="What terrain type is depicted in this optical satellite observation?"
)
```

#### Function Signature:
```python
def run_vqa(image_path: str, question: str) -> Dict[str, Any]
```

#### Parameters:
- `image_path` (`str`): Absolute or relative filesystem path to the satellite image (`.png`, `.jpg`, `.tif`).
- `question` (`str`): Natural language question query.

#### Return Dictionary Format:
```json
{
  "task": "vqa",
  "answer": "flat land",
  "confidence": 0.90,
  "image_path": "data/sample/optical/sample_optical_001.png"
}
```

#### Confidence Heuristic:
- **0.00**: Missing image file, unreadable image, or internal inference exception.
- **0.10**: Fallback response (`"No discernible feature detected."`).
- **0.35**: Expressions of uncertainty (`"uncertain"`, `"unclear"`, `"unknown"`, `"maybe"`, `"not sure"`).
- **0.90 – 0.95**: Concise categorical answers (1–4 words like `"water"`, `"forest"`, `"urban"`, `"yes"`, `"no"`).
- **0.80 – 0.85**: Standard multi-word descriptive answers.
- **0.70**: Overly verbose answers (>25 words) prone to hallucinations.

---

### 2.2 Scene Captioning: `run_captioning`

```python
from backend.captioning_tool import run_captioning

result = run_captioning(
    image_path="data/sample/optical/sample_optical_001.png"
)
```

#### Function Signature:
```python
def run_captioning(image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]
```

#### Parameters:
- `image_path` (`str`): Absolute or relative path to the satellite image.
- `prompt` (`str`, optional): Custom instruction prompt for the model. Defaults to:
  `"Describe the land cover and major objects visible in this image."`

#### Return Dictionary Format:
```json
{
  "task": "captioning",
  "caption": "Satellite imagery showing trees and grass.",
  "confidence": 0.95,
  "image_path": "data/sample/optical/sample_optical_001.png"
}
```

#### Confidence Heuristic:
- **0.00**: Missing file or error.
- **0.85**: Baseline for successful generation.
- **0.90 – 0.96**: Descriptions referencing confirmed remote sensing features (`"water"`, `"forest"`, `"urban"`, `"buildings"`, `"roads"`).

---

### 2.3 Region Grounding: `run_grounding`

```python
from backend.grounding_tool import run_grounding

result = run_grounding(
    image_path="data/sample/optical/sample_optical_001.png",
    query="forest and green vegetation"
)
```

#### Function Signature:
```python
def run_grounding(image_path: str, query: str) -> Dict[str, Any]
```

#### Parameters:
- `image_path` (`str`): Absolute or relative path to the satellite image.
- `query` (`str`): Target object or land cover query (e.g. `"water body"`, `"forest"`, `"runway"`, `"buildings"`).

#### Return Dictionary Format:
```json
{
  "task": "grounding",
  "bbox": [3, 3, 125, 125],
  "confidence": 0.65,
  "image_path": "data/sample/optical/sample_optical_001.png"
}
```

#### Bounding Box Format:
- `bbox`: `[x1, y1, x2, y2]` in pixel coordinates, where `(x1, y1)` is top-left and `(x2, y2)` is bottom-right.
- Uses 3rd and 97th spatial percentile trimming to prevent single-pixel noise from expanding the bounding box to the entire image.

---

### 2.4 Bi-Temporal Change Detection: `run_change_detection`

```python
from backend.change_detection_tool import run_change_detection

# General change description
result = run_change_detection(
    image_path_before="data/sample/pairs/sample_pair_001_t1.png",
    image_path_after="data/sample/pairs/sample_pair_001_t2.png"
)

# Targeted change VQA query
qa_result = run_change_detection(
    image_path_before="data/cdvqa/image_pairs/cdvqa_001_t1.png",
    image_path_after="data/cdvqa/image_pairs/cdvqa_001_t2.png",
    question="Did new building structures appear between Time 1 and Time 2?"
)
```

#### Function Signature:
```python
def run_change_detection(
    image_path_before: str,
    image_path_after: str,
    question: Optional[str] = None
) -> Dict[str, Any]
```

#### Parameters:
- `image_path_before` (`str`): Filesystem path to the pre-change ($T_1$) satellite image.
- `image_path_after` (`str`): Filesystem path to the post-change ($T_2$) satellite image.
- `question` (`str`, optional): Specific question inquiring what changed between the two passes. Defaults to `None` for a general change description.

#### Return Dictionary Format:
```json
{
  "task": "change_detection",
  "description": "Surface alteration detected: While primary land cover category remains land and water, structural and textural variance (metric: 0.245) indicates localized activity between passes.",
  "change_detected": true,
  "confidence": 0.91,
  "before_path": "data/sample/pairs/sample_pair_001_t1.png",
  "after_path": "data/sample/pairs/sample_pair_001_t2.png"
}
```

#### Confidence Heuristic:
- **0.00**: Missing files, read error, or model failure.
- **0.35**: Expressions of uncertainty (`"uncertain"`, `"unclear"`, `"cannot tell"`).
- **0.90 – 0.96**: High agreement between radiometric pixel differences and semantic descriptions.
- **0.98**: Identical image negative controls (`change_detected == False`).
- **0.80 – 0.88**: Standard descriptive comparisons.

---

### 2.5 Cross-Modal Optical-SAR Fusion: `run_optical_sar_fusion`

```python
from backend.fusion_tool import run_optical_sar_fusion

# General cross-modal analysis
result = run_optical_sar_fusion(
    optical_path="data/sample/optical/sample_optical_001.png",
    sar_path="data/sample/sar/sample_sar_001.png"
)

# Targeted joint query
qa_result = run_optical_sar_fusion(
    optical_path="data/sample/optical/sample_optical_003.png",
    sar_path="data/sample/sar/sample_sar_003.png",
    question="Identify built-up and water-covered regions using both images together."
)
```

#### Function Signature:
```python
def run_optical_sar_fusion(
    optical_path: str,
    sar_path: str,
    question: Optional[str] = None
) -> Dict[str, Any]
```

#### Parameters:
- `optical_path` (`str`): Filesystem path to the optical satellite image (RGB / multi-spectral).
- `sar_path` (`str`): Filesystem path to the co-registered SAR radar backscatter image.
- `question` (`str`, optional): Query requiring joint cross-modal reasoning across both sensors. Defaults to `None` for a joint descriptive report.

#### Return Dictionary Format:
```json
{
  "task": "fusion",
  "answer": "Joint Optical-SAR Analysis: Optical imagery provides spectral delineation indicating temperate (squares). Co-registered SAR confirms physical dielectric properties with elevated radar backscatter with strong double-bounce reflections (characteristic of built-up urban structures or metallic targets) (water). Combining both sensors enables robust cross-modal verification...",
  "confidence": 0.89,
  "optical_path": "data/sample/optical/sample_optical_003.png",
  "sar_path": "data/sample/sar/sample_sar_003.png"
}
```

#### Confidence Heuristic:
- **0.00**: Missing file or runtime exception.
- **0.35**: Expressions of uncertainty.
- **0.88 – 0.95**: Strong physical corroboration between optical classifications and SAR radar backscatter properties (e.g., optical water verified by low specular radar return; optical urban structures verified by high double-bounce return).

---

## 3. Known Caveats & Limitations for M4 / M5 / M6

1. **Captioning Granularity on Cluttered Scenes**:
   - The underlying BLIP-VQA model produces high-level land cover summaries (e.g., `"trees and grass"`, `"urban"`). In complex, multi-class scenes (e.g., suburban areas with interspersed trees, roads, and residential homes), the model may report only the dominant class.
   - *Recommendation for M5 Agent Lead*: Formulate targeted VQA questions (e.g., `"Are there roads visible near the buildings?"`) to drill into details.

2. **Grounding via Spectral Baseline**:
   - Because `Salesforce/blip-vqa-base` is an autoregressive encoder-decoder VQA model without an explicit bounding box regression token head, `backend/grounding_tool.py` uses heuristic spectral band indices (Excess Green for vegetation, NDWI RGB approximation for water, variance/reflectance for urban/structures).
   - This provides reliable macro-scale localization for distinct surface covers, but is not suitable for microscopic detection of small occluded objects.

3. **Bi-Temporal Lighting & Seasonal Variance (Change Detection)**:
   - Bi-temporal change detection relies on optical reflectance and VLM semantic transitions. Variations in solar illumination angle, shadow extents, cloud shadows, or seasonal phenology (e.g., wet season greening vs dry season senescence) can produce localized pixel differences without physical infrastructure changes.
   - *Guidance for M5 (Agent)*: For borderline cases, evaluate whether the VLM identified category shifts or purely radiometric variance before concluding human infrastructure activity.

4. **Text-Level vs Pixel-Level Fusion (Optical-SAR)**:
   - As documented, `run_optical_sar_fusion` executes a text-level and radiometric feature fusion approach rather than pixel-level tensor cross-attention. It effectively synthesizes complementary optical spectral properties (color, vegetation greenness, boundary definitions) with radar physical properties (surface roughness, moisture, metallic double-bounce), but does not perform sub-pixel alignment.
   - *Guidance for M5 (Agent)*: The agent controller should describe results to users as synthesized multimodal reasoning based on complementary sensor physics.

5. **SAR Speckle Noise**:
   - Coherent radar speckle creates high pixel-level variance in homogeneous zones (such as agricultural fields). The tool applies aggregated area statistics (mean and percentile thresholds) to minimize susceptibility to speckle artifacts.

---

## 4. Verification

Run the verification test suites at any time:
```bash
# Phase 3 single-image baseline tools (VQA, captioning, grounding)
python tests/test_vqa_captioning.py

# Phase 4 multi-image tools (change detection, optical-SAR fusion)
python tests/test_change_and_fusion.py
```
