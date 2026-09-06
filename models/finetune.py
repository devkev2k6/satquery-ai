"""
models/finetune.py
==================
Fine-tunes a vision-language model (Salesforce/blip-vqa-base) on remote sensing
imagery using Low-Rank Adaptation (LoRA / PEFT).

Loads dataset samples via data.loader (load_bigearthnet and load_rsvqa / load_sample_data),
constructs multimodal training pairs, and saves adapted weights to models/checkpoints/.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_bigearthnet, load_rsvqa, load_sample_data


class RemoteSensingVQADataset(Dataset):
    """
    Multimodal dataset wrapping remote sensing image-question-answer pairs.
    """
    def __init__(self, samples: List[Dict[str, Any]], processor):
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image_path = item["image_path"]
        question = item.get("question", "What is visible in this satellite image?")
        answer = str(item.get("answer", "satellite image"))

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupt / missing images: blank tile
            print(f"[!] Warning: Could not open {image_path} ({e}), creating blank fallback")
            image = Image.new("RGB", (224, 224), color=(100, 100, 100))

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }


def collate_fn(batch, processor):
    images = [b["image"] for b in batch]
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]

    # Process multimodal inputs (images + questions)
    inputs = processor(
        images=images,
        text=questions,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Tokenize answers as target labels
    target_enc = processor.tokenizer(
        text=answers,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = target_enc.input_ids

    decoder_input_ids = labels.clone()
    # Replace padding token ids in loss labels with -100 so cross-entropy ignores them
    labels_for_loss = labels.clone()
    labels_for_loss[labels_for_loss == processor.tokenizer.pad_token_id] = -100

    inputs["decoder_input_ids"] = decoder_input_ids
    inputs["decoder_attention_mask"] = target_enc.attention_mask
    inputs["labels"] = labels_for_loss

    return inputs


def get_training_samples(include_rsvqa: bool = True, include_sample: bool = True) -> List[Dict[str, Any]]:
    """
    Aggregates training samples from BigEarthNet, RSVQA, and synthetic fallback data.
    """
    all_samples = []

    # 1. BigEarthNet (Optical & SAR multi-label land cover)
    try:
        ben_samples = load_bigearthnet(split="train")
        print(f"[+] Loaded {len(ben_samples)} training samples from BigEarthNet")
        all_samples.extend(ben_samples)
    except Exception as e:
        print(f"[!] Warning loading BigEarthNet: {e}")

    # 2. RSVQA (Presence, count, area queries)
    if include_rsvqa:
        try:
            rsvqa_samples = load_rsvqa()
            print(f"[+] Loaded {len(rsvqa_samples)} samples from RSVQA")
            all_samples.extend(rsvqa_samples)
        except Exception as e:
            print(f"[!] Warning loading RSVQA: {e}")

    # 3. Synthetic optical / SAR fallback if primary sets are small
    if include_sample or len(all_samples) == 0:
        try:
            sample_samples = load_sample_data(category="optical") + load_sample_data(category="sar")
            print(f"[+] Loaded {len(sample_samples)} samples from synthetic fallback data")
            all_samples.extend(sample_samples)
        except Exception as e:
            print(f"[!] Warning loading sample data: {e}")

    # Filter for valid existing image paths
    valid_samples = [s for s in all_samples if Path(s.get("image_path", "")).is_file()]
    print(f"[+] Total verified training samples with valid images: {len(valid_samples)}")
    return valid_samples


def train(
    base_model_name: str = "Salesforce/blip-vqa-base",
    output_dir: str = "models/checkpoints/blip_rs_lora",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    device: str = "auto",
):
    """
    Executes parameter-efficient LoRA fine-tuning on the remote sensing dataset.
    """
    from transformers import BlipProcessor, BlipForQuestionAnswering
    from peft import LoraConfig, get_peft_model, TaskType

    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine compute device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"[*] Training on compute device: {dev}")

    # Load processor and base model
    print(f"[*] Loading base model and processor from {base_model_name}...")
    processor = BlipProcessor.from_pretrained(base_model_name)
    base_model = BlipForQuestionAnswering.from_pretrained(base_model_name)

    # Configure LoRA parameter-efficient adaptation
    # Target query and value projections in the multimodal text decoder
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.to(dev)

    # Load dataset
    samples = get_training_samples()
    if not samples:
        raise RuntimeError("No training samples found in data/ directories!")

    dataset = RemoteSensingVQADataset(samples, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    print(f"\n[*] Starting training: {epochs} epochs, {len(dataloader)} batches per epoch...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader, 1):
            input_ids = batch["input_ids"].to(dev)
            pixel_values = batch["pixel_values"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            decoder_input_ids = batch["decoder_input_ids"].to(dev)
            decoder_attention_mask = batch["decoder_attention_mask"].to(dev)
            labels = batch["labels"].to(dev)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 2 == 0 or step == len(dataloader):
                print(f"    Epoch [{epoch}/{epochs}] Step [{step}/{len(dataloader)}] - Batch Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"[+] Epoch [{epoch}/{epochs}] Completed - Avg Loss: {avg_loss:.4f}\n")

    # Save adapted LoRA weights and processor config
    print(f"[*] Saving adapted LoRA model weights to {out_path}...")
    model.save_pretrained(str(out_path))
    processor.save_pretrained(str(out_path))
    print(f"[+] Adaptation complete. Weights saved successfully to {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP-VQA on satellite imagery using LoRA.")
    parser.add_argument("--base-model", type=str, default="Salesforce/blip-vqa-base", help="Hugging Face base model ID")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints/blip_rs_lora", help="Path to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")

    args = parser.parse_args()
    train(
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
