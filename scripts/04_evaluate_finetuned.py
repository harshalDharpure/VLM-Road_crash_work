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

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
def load_finetuned_model(checkpoint_path, base_model_name):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import BitsAndBytesConfig

    print(f"Loading base model: {base_model_name}")

    processor = LlavaNextProcessor.from_pretrained(base_model_name)

    # 🔥 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = LlavaNextForConditionalGeneration.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # ✅ CORRECT LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # load only LoRA weights
    lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k.lower()}
    model.load_state_dict(lora_state_dict, strict=False)

    model.eval()
    torch.cuda.empty_cache()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    train_loss = checkpoint.get("train_loss", None)
    print(f"Training loss: {train_loss:.4f}" if train_loss else "Training loss: N/A")

    val_loss = checkpoint.get("val_loss", None)
    print(f"Validation loss: {val_loss:.4f}" if val_loss else "Validation loss: N/A")

    # -------------------------------
    class Wrapper:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor

        def generate_summary(self, frames):
            if not frames:
                return {"text_summary": ""}

            rep = frames[len(frames) // 2]
            img = Image.fromarray(rep).convert("RGB")

            prompt = (
                "You are an expert traffic accident analyst. "
                "Describe clearly vehicles, crash, and outcome."
            )

            formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

            inputs = self.processor(
                text=formatted_prompt,
                images=img,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=False
                )

            text = self.processor.decode(outputs[0], skip_special_tokens=True)
            return {"text_summary": text.split("ASSISTANT:")[-1].strip()}

    return Wrapper(model, processor)


# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = get_config(args.config)

    root_dir = Path(config["dataset"]["root_dir"])
    processed_dir = root_dir / config["dataset"]["processed_dir"]

    with open(processed_dir / "split_info.json") as f:
        split_info = json.load(f)

    with open(processed_dir / f"annotations_{args.split}.json") as f:
        annotations = json.load(f)

    videos = split_info["splits"][args.split]

    model = load_finetuned_model(
        args.checkpoint,
        config["model"]["vision_model"]
    )

    # evaluators
    bleu_eval = BLEUEvaluator(max_order=4, smooth=True)
    nli_eval = NLIEvaluator(device="cpu")

    predictions, references = [], []

    print("\nRunning inference...")

    for video_path in tqdm(videos):
        vid = Path(video_path).stem
        if vid not in annotations:
            continue

        gt = annotations[vid]["text_summary"]
        frames = load_frames(video_path)

        if not frames or not gt.strip():
            continue

        pred = model.generate_summary(frames)["text_summary"]

        predictions.append(pred)
        references.append(gt)

    print("\nComputing metrics...")

    bleu_scores = bleu_eval.compute_bleu_batch(predictions, references)
    nli_scores = nli_eval.evaluate(predictions, references)

    meteor_vals = [meteor_score([r.split()], p.split()) for p, r in zip(predictions, references)]
    mean_meteor = float(np.mean(meteor_vals))

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []

    for p, r in zip(predictions, references):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    mean_rouge_1 = float(np.mean(r1))
    mean_rouge_2 = float(np.mean(r2))
    mean_rouge_l = float(np.mean(rl))

    _, _, F1 = bert_score(predictions, references, lang="en", device="cpu")
    mean_bertscore = float(F1.mean())

    from pycocoevalcap.cider.cider import Cider
    cider = Cider()
    cider_vals = [cider.compute_score({0: [r]}, {0: [p]})[0] for p, r in zip(predictions, references)]
    mean_cider = float(np.mean(cider_vals))

    print("\n===== RESULTS =====")
    print(f"BLEU-4: {bleu_scores['bleu_4']:.4f}")
    print(f"METEOR: {mean_meteor:.4f}")
    print(f"ROUGE-L: {mean_rouge_l:.4f}")
    print(f"BERTScore: {mean_bertscore:.4f}")
    print(f"CIDEr: {mean_cider:.4f}")
    print(f"NLI Entailment: {nli_scores['entailment_accuracy']:.4f}")

        # -------------------------------
    # Save metrics in required JSON format
    results = {
        "num_samples": len(predictions),
        "bleu_scores": {
            "bleu_1": float(bleu_scores.get("bleu_1", 0.0)),
            "bleu_2": float(bleu_scores.get("bleu_2", 0.0)),
            "bleu_3": float(bleu_scores.get("bleu_3", 0.0)),
            "bleu_4": float(bleu_scores.get("bleu_4", 0.0))
        },
        "meteor": float(mean_meteor),
        "rouge_1": float(mean_rouge_1),
        "rouge_2": float(mean_rouge_2),
        "rouge_l": float(mean_rouge_l),
        "bertscore": float(mean_bertscore),
        "cider": float(mean_cider),
        "nli_scores": {
            "entailment_accuracy": float(nli_scores.get("entailment_accuracy", 0.0)),
            "avg_entailment_prob": float(nli_scores.get("entailment_accuracy", 0.0)),
            "contradiction_rate": float(nli_scores.get("contradiction_rate", 0.0)),
            "neutral_rate": float(nli_scores.get("neutral_rate", 0.0))
        }
    }

    save_path = Path("results/finetuned/metrics.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to: {save_path}")


if __name__ == "__main__":
    main()
