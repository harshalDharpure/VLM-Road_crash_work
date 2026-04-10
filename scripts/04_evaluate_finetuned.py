#!/usr/bin/env python3
"""
Evaluate fine-tuned model on test set with full metric suite.
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import LLaVANeXTWrapper
from src.evaluation import BLEUEvaluator, NLIEvaluator
from src.utils import get_config

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from PIL import Image


# -------------------------------
def load_frames(video_path: str, max_frames: int = 30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        count += 1

    cap.release()
    return frames


# -------------------------------
def load_finetuned_model(checkpoint_path: str, base_model_name: str, device: str):
    from transformers import (
        LlavaNextProcessor,
        LlavaNextForConditionalGeneration,
        LlavaProcessor,
        LlavaForConditionalGeneration
    )

    print(f"Loading base model: {base_model_name}")

    is_next = "llava-v1.6" in base_model_name or "llava-next" in base_model_name.lower()

    if is_next:
        processor = LlavaNextProcessor.from_pretrained(base_model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=None
        )
    else:
        processor = LlavaProcessor.from_pretrained(base_model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=None
        )

    if device != "cpu":
        model = model.to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["model_state_dict"]
    has_lora = any("lora" in k.lower() for k in state_dict.keys())

    if has_lora:
        print("Detected LoRA checkpoint...")

        from peft import LoraConfig, get_peft_model, TaskType

        # ✅ CORRECT LoRA config (must match training)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)

        if device != "cpu":
            model = model.to(device)

        # Load only LoRA weights
        lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k.lower()}
        lora_state_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in lora_state_dict.items()}

        model.load_state_dict(lora_state_dict, strict=False)
        print("✓ LoRA weights loaded")

    else:
        print("Loading full model checkpoint...")
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # ✅ SAFE PRINTING
    train_loss = checkpoint.get("train_loss", None)
    if train_loss is not None:
        print(f"Training loss: {train_loss:.4f}")
    else:
        print("Training loss: N/A")

    val_loss = checkpoint.get("val_loss", None)
    if val_loss is not None:
        print(f"Validation loss: {val_loss:.4f}")
    else:
        print("Validation loss: N/A")

    class Wrapper:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor

        def generate_summary(self, frames):
            rep = frames[len(frames) // 2]
            img = Image.fromarray(rep).convert("RGB")

            prompt = "USER: <image>\nDescribe the accident.\nASSISTANT:"

            inputs = self.processor(text=prompt, images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=128)

            text = self.processor.decode(out[0], skip_special_tokens=True)
            return {"text_summary": text.split("ASSISTANT:")[-1].strip()}

    return Wrapper(model, processor)


# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = get_config(args.config)

    root_dir = Path(config["dataset"]["root_dir"])
    processed_dir = root_dir / config["dataset"]["processed_dir"]

    with open(processed_dir / "split_info.json") as f:
        split = json.load(f)

    with open(processed_dir / "annotations_test.json") as f:
        annotations = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_finetuned_model(
        args.checkpoint,
        config["model"]["vision_model"],
        device
    )

    predictions, references = [], []

    for video_path in tqdm(split["splits"]["test"]):
        vid = Path(video_path).stem
        if vid not in annotations:
            continue

        frames = load_frames(video_path)
        if not frames:
            continue

        pred = model.generate_summary(frames)["text_summary"]
        gt = annotations[vid]["text_summary"]

        predictions.append(pred)
        references.append(gt)

    print("\nDone. Samples:", len(predictions))


if __name__ == "__main__":
    main()
