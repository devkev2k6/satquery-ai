# SatQuery AI - Backend Tool Suite (`backend/`)

This directory houses the single-image baseline tools, geospatial query utilities, and multimodal pipelines for **SatQuery AI**. Downstream teammates (M4 Multimodal Fusion/Change Detection, M5 Agent Controller, and M6 Frontend) consume these standardized functions directly.

---

## 1. Overview of Tools

| Module | Primary Function | Task Type | Description |
| :--- | :--- | :--- | :--- |
| `backend/vqa_tool.py` | `run_vqa(image_path, question)` | `"vqa"` | Answers natural language queries regarding single satellite images. |
| `backend/captioning_tool.py` | `run_captioning(image_path, prompt=None)` | `"captioning"` | Produces a natural-language description of land cover and major visible objects. |
| `backend/grounding_tool.py` | `run_grounding(image_path, query)` | `"grounding"` | Localizes described features/objects to approximate `[x1, y1, x2, y2]` bounding boxes. |

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

## 3. Known Caveats & Limitations for M4 / M5 / M6

1. **Captioning Granularity on Cluttered Scenes**:
   - The underlying BLIP-VQA model produces high-level land cover summaries (e.g., `"trees and grass"`, `"urban"`). In complex, multi-class scenes (e.g., suburban areas with interspersed trees, roads, and residential homes), the model may report only the dominant class.
   - *Recommendation for M5 Agent Lead*: Formulate targeted VQA questions (e.g., `"Are there roads visible near the buildings?"`) to drill into details.

2. **Grounding via Spectral Baseline**:
   - Because `Salesforce/blip-vqa-base` is an autoregressive encoder-decoder VQA model without an explicit bounding box regression token head, `backend/grounding_tool.py` uses heuristic spectral band indices (Excess Green for vegetation, NDWI RGB approximation for water, variance/reflectance for urban/structures).
   - This provides reliable macro-scale localization for distinct surface covers, but is not suitable for microscopic detection of small occluded objects.

3. **Single Image Scope**:
   - These tools process one image at a time. For multitemporal change detection (CDVQA pairs), M4 should either tile images or call `run_vqa` on both $T_1$ and $T_2$ and perform differential comparison.

---

## 4. Verification

Run the verification test suite at any time:
```bash
python tests/test_vqa_captioning.py
```
