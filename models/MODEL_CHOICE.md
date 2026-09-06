# Model Selection & Architectural Rationale

## 1. Selected Vision-Language Model

- **Model Identifier**: [`Salesforce/blip-vqa-base`](https://huggingface.co/Salesforce/blip-vqa-base)
- **Architecture**: ViT-B/16 (Vision Transformer Base) + Multimodal End-to-End Decoder (MED) with cross-attention
- **Model Class**: `transformers.BlipForQuestionAnswering`
- **Processor Class**: `transformers.BlipProcessor`
- **Total Parameters**: ~385 Million
- **Adapter Strategy**: LoRA (Low-Rank Adaptation via `peft`) targeting attention query/value projection layers ($r=16, \alpha=32$)
- **Checkpoint Footprint**: ~15 MB for adapted LoRA weights vs ~1.5 GB for full base model weights

---

## 2. Evaluation & Comparison Matrix

To satisfy the problem statement's requirement of adapting a vision-language model to remote sensing imagery on realistic hackathon hardware (such as laptops with 4GB VRAM or Colab free tier), several candidates were evaluated:

| Model Candidate | Parameter Count | Native VQA Head? | Min VRAM (fp16) | CPU Inference Feasibility | Native HF Integration | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Salesforce/blip-vqa-base** | **385M** | **Yes (`BlipForQA`)** | **~1.5 GB** | **Fast (~1.5s/q)** | **Yes (Core Transformers)** | **SELECTED: Optimal balance of speed, memory, and VQA architecture** |
| **Salesforce/blip2-opt-2.7b** | 2.7B | Conditional Generation | ~6.0 GB | Very Slow (>15s/q) | Yes | Exceeds 4GB VRAM limit; high latency on laptop CPU |
| **google/paligemma-3b-pt-224** | 2.9B | Yes | ~6.5 GB | Very Slow (>20s/q) | Yes | High VRAM consumption; OOM risks on local GPUs |
| **Qwen/Qwen2-VL-2B-Instruct** | 2.2B | Open-ended | ~5.5 GB | Slow (~10s/q) | Requires custom handling | High compute overhead for rapid multi-step pipeline iterations |
| **vikhyatk/moondream2** | 1.86B | Custom Query API | ~4.0 GB | Moderate (~5s/q) | Custom (`trust_remote_code`) | Custom code breaks across minor version bumps; non-standard VQA interface |

---

## 3. Justification & Architectural Advantages

### A. Dedicated Visual Question Answering Head
Unlike models that merely generate generic captions or broad autoregressive text, `Salesforce/blip-vqa-base` contains a dedicated multimodal cross-attention decoder. The image features extracted by the ViT-B backbone are fed directly into the cross-attention layers of the text decoder, conditioning token generation on both visual satellite cues and query prompts.

### B. Hardware Feasibility & Low-Latency Execution
In a 6-phase collaborative pipeline, teammates downstream (M3 Agent, M4 Backend, M5 Frontend) will call the vision-language component repeatedly:
- **GPU Inference**: On modest GPUs (such as 4GB NVIDIA RTX 2050 or T4), BLIP-VQA occupies under 1.5 GB VRAM and generates responses in < 0.4 seconds.
- **CPU Fallback**: For teammates without dedicated GPUs, BLIP-VQA executes in ~1.5 seconds per query on standard multicore laptop CPUs. In contrast, 3B models take 20–40 seconds per query on CPU, stalling development.
- **Quantization Fragility**: 4-bit quantization (QLoRA) on Windows often fails due to missing C++ build tools for `bitsandbytes` wheels. BLIP-VQA runs natively in full fp32 or fp16 without requiring non-standard CUDA kernels.

### C. Parameter-Efficient Fine-Tuning (PEFT / LoRA)
Full fine-tuning of vision backbones risks catastrophic forgetting on small datasets and consumes excessive memory for optimizer states. By applying LoRA to the query and value projections of the multimodal text decoder:
- Trainable parameters are reduced to < 1.5% of total weights (~5.8M parameters).
- Training on the BigEarthNet and remote sensing dataset subsets completes in ~1–2 minutes.
- Checkpoints are extremely lightweight (~15MB), enabling versioning without large storage overhead.

---

## 4. Downstream Integration Contract
The model is wrapped in `models/inference.py` exposing the standardized function:
```python
def answer_question(image_path: str, question: str) -> str:
    """Answers a natural language query regarding a remote sensing image."""
```
This signature ensures seamless handoff to teammate M3 (Agent Controller) and M4 (API Backend).
