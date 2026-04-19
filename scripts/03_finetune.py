#!/usr/bin/env python3

import sys
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config
from src.models import LLaVANeXTWrapper

torch.backends.cuda.matmul.allow_tf32 = True


# ✅ CHECKPOINT FUNCTION (self-contained)
def save_checkpoint(save_dir, epoch, model, optimizer, train_loss):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": float(train_loss)
    }

    torch.save(checkpoint, save_dir / "checkpoint.pt")


class VideoSummarizationDataset(Dataset):
    def __init__(self, video_paths, annotations, max_frames=8, processor=None, is_next=True):
        self.video_paths = video_paths
        self.annotations = annotations
        self.max_frames = max_frames
        self.processor = processor
        self.is_next = is_next

        self.valid_pairs = []
        for video_path in video_paths:
            video_id = Path(video_path).stem
            if video_id in annotations:
                text = annotations[video_id].get("text_summary", "").strip()
                if text:
                    self.valid_pairs.append((video_path, video_id, text))

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        video_path, video_id, text_summary = self.valid_pairs[idx]

        frames = self._load_frames(video_path, self.max_frames)

        if len(frames) == 0:
            rep_frame = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            rep_frame = frames[len(frames) // 2]
            rep_frame = cv2.resize(rep_frame, (112, 112), interpolation=cv2.INTER_AREA)

        rep_img = Image.fromarray(rep_frame).convert("RGB")

        prompt = (
            "You are an expert traffic accident analyst. "
            "Watch this car crash video and generate a factual explanation. "
            "Describe clearly vehicles, actions, crash and outcome."
        )

        user_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        formatted_prompt = f"{user_prompt} {text_summary}"

        inputs = self.processor(
            text=formatted_prompt,
            images=rep_img,
            return_tensors="pt",
            padding=True
        )

        if self.is_next and "image_sizes" not in inputs:
            w, h = rep_img.size
            inputs["image_sizes"] = torch.tensor([[h, w]])

        result = {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "user_prompt": user_prompt
        }

        if "image_sizes" in inputs:
            result["image_sizes"] = inputs["image_sizes"].squeeze(0)

        return result

    def _load_frames(self, video_path, max_frames):
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


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])

    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]

    max_len = max(len(x) for x in input_ids)

    padded_ids = []
    padded_mask = []

    for ids, mask in zip(input_ids, attention_mask):
        pad = max_len - len(ids)
        padded_ids.append(torch.cat([ids, torch.zeros(pad, dtype=ids.dtype)]))
        padded_mask.append(torch.cat([mask, torch.zeros(pad, dtype=mask.dtype)]))

    result = {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_mask),
        "user_prompts": [x["user_prompt"] for x in batch]
    }

    if "image_sizes" in batch[0]:
        result["image_sizes"] = torch.stack([x["image_sizes"] for x in batch])

    return result


def prepare_labels(input_ids, user_prompts, processor):
    labels = input_ids.clone()
    pad_id = processor.tokenizer.pad_token_id
    labels[input_ids == pad_id] = -100
    labels[:, : input_ids.shape[1] // 2] = -100
    return labels


def train_epoch(model, loader, optimizer, processor, device, grad_accum):
    model.train()
    total_loss = 0

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for step, batch in enumerate(tqdm(loader)):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = prepare_labels(input_ids, batch["user_prompts"], processor)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                image_sizes=batch.get("image_sizes", None)
            )
            loss = outputs.loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach().item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = get_config(args.config)

    root_dir = Path(config.config["dataset"]["root_dir"])
    processed_dir = root_dir / config.config["dataset"]["processed_dir"]

    with open(processed_dir / "split_info.json") as f:
        split = json.load(f)

    with open(processed_dir / "annotations_train.json") as f:
        train_ann = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = LLaVANeXTWrapper(
        model_name=config.config["model"]["vision_model"],
        device=device
    )

    model = wrapper.model
    processor = wrapper.processor

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "mm_projector" in name or "lm_head" in name:
            p.requires_grad = True

    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora)

    train_ds = VideoSummarizationDataset(
        split["splits"]["train"],
        train_ann,
        max_frames=4,
        processor=processor,
        is_next=True
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5
    )

    train_losses = []
    
    num_epochs = int(config.config["training"]["num_epochs"])

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, processor, device, grad_accum=2)
        print(f"Epoch {epoch} Train Loss:", train_loss)

        # Save single checkpoint file
        save_checkpoint(
            save_dir="results",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loss=train_loss
        )

        train_losses.append(train_loss)

    with open(Path("results") / "training_loss.json", "w") as f:
        json.dump(train_losses, f, indent=2)


if __name__ == "__main__":
    main()
