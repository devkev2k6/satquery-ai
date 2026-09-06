# SatQuery AI - Vision-Language Adaptation Layer (`models/`)

This directory contains the adapted vision-language model (VLM), training pipeline, evaluation benchmarks, and standardized inference interface for **SatQuery AI**.

---

## 1. Base Model & Adaptation Strategy

- **Base Architecture**: [`Salesforce/blip-vqa-base`](https://huggingface.co/Salesforce/blip-vqa-base) (385M parameters)
  - **Vision Encoder**: ViT-B/16 (Vision Transformer Base) pre-trained on high-resolution image patches.
  - **Multimodal Text Decoder**: Multimodal End-to-End Decoder (MED) with cross-attention layers directly conditioning token generation on visual feature representations.
- **Adaptation Method**: Low-Rank Adaptation (**LoRA** via Hugging Face `peft`)
  - **Target Modules**: Cross-attention and self-attention projections (`query`, `value`).
  - **Rank ($r$)**: 16
  - **Scaling ($\alpha$)**: 32
  - **Dropout**: 0.05
  - **Trainable Parameters**: ~5.8 Million (< 1.5% of model weights)
  - **Checkpoint Location**: `models/checkpoints/blip_rs_lora/` (~15 MB)

---

## 2. Quickstart: Calling `answer_question()`

Downstream teammates (M3 Agent Controller, M4 Backend API, M5 Frontend) should import and call `answer_question` directly from `models.inference`:

```python
from models.inference import answer_question

# Example 1: Land cover query
img_path = "data/bigearthnet/images/S2A_MSIL2A_20170717T113321_01.png"
query = "What land cover classes are present in this satellite image?"
answer = answer_question(img_path, query)
print("Answer:", answer)
# Output: "land and water"

# Example 2: Presence query
rsvqa_img = "data/rsvqa/images/rsvqa_0101.png"
query = "Are there buildings present in this image?"
answer = answer_question(rsvqa_img, query)
print("Answer:", answer)
# Output: "no"
```

### CLI Testing
You can also run inference directly from your command line:
```bash
python models/inference.py --image data/bigearthnet/images/S2A_MSIL2A_20170717T113321_01.png --question "What land cover is visible?"
```

---

## 3. How to Re-Run Training

To re-run or extend the LoRA fine-tuning process (e.g. after downloading additional BigEarthNet or RSVQA patches):

```bash
# Standard 3-epoch training
python models/finetune.py --epochs 3 --batch-size 2 --lr 5e-5

# Custom configuration
python models/finetune.py \
    --base-model "Salesforce/blip-vqa-base" \
    --output-dir "models/checkpoints/blip_rs_lora" \
    --epochs 5 \
    --batch-size 4 \
    --lr 5e-5 \
    --device auto
```

Training automatically discovers training samples from `data.loader.load_bigearthnet(split="train")`, `data.loader.load_rsvqa()`, and `data.loader.load_sample_data()`.

---

## 4. Expected Inference Latency & Hardware Profile

| Hardware Environment | Precision | Memory Footprint | Avg Latency per Query |
| :--- | :--- | :--- | :--- |
| **NVIDIA GPU (e.g. RTX 2050 4GB / T4)** | fp32 / fp16 | ~1.2 GB – 1.5 GB VRAM | **0.25 – 0.45 seconds** |
| **Laptop CPU (x86_64 / Apple Silicon)** | fp32 | ~1.5 GB System RAM | **1.20 – 1.80 seconds** |

*Note*: The model and processor are cached in memory upon first invocation, so sequential calls inside a persistent server or agent loop execute with zero reload overhead.

---

## 5. Known Limitations & Failure Modes (Guidance for M3 & M4)

1. **Exact Dense Object Counting**:
   - *Limitation*: Like most vision-language models without explicit bounding box regression heads, BLIP-VQA struggles with precise counting of densely clustered objects (e.g. "How many small boats are moored in the harbor?").
   - *Recommendation for M3 (Agent)*: For queries requiring exact numeric counts, prompt the model for presence first ("Are there boats present?"), and route to an object detector or raster analysis tool if exact counts are required.

2. **Fine-Grained Bi-Temporal Pixel Change**:
   - *Limitation*: The model takes a single image per standard VQA prompt. When querying change detection pairs ($T_1$ and $T_2$ from CDVQA), feeding only $T_2$ will describe the final state rather than the differential delta.
   - *Recommendation for M3 (Agent)*: For change detection queries, concatenate or tile $T_1$ and $T_2$ horizontally or formulate comparative prompts ("Comparing pre and post-construction...").

3. **Multi-Label Recall Breadth**:
   - *Limitation*: On complex BigEarthNet patches containing 4+ mixed land cover categories, the model reliably identifies the primary dominant classes (e.g. "Coniferous forest, water") but may omit minor background classes (e.g. "Transitional woodland-shrub").
   - *Recommendation for M4 (Backend)*: Provide structured prompt hints if specific land cover taxonomies are expected.
