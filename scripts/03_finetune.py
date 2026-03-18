#!/usr/bin/env python3
"""
Fine-tuning script for LLaVA-NeXT on car crash video summarization task.
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Any, Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config
from src.models import LLaVANeXTWrapper
from src.training import LossTracker


# ------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------
class VideoSummarizationDataset(Dataset):
    """PyTorch Dataset for video-text summarization pairs."""
    
    def __init__(
        self,
        video_paths: List[str],
        annotations: Dict[str, Dict],
        max_frames: int = 30,
        processor=None,
        is_next: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            video_paths: List of video file paths
            annotations: Dictionary mapping video_id to annotation dict
            max_frames: Maximum number of frames to extract
            processor: LLaVA processor for image/text processing
            is_next: Whether using LLaVA-NeXT (True) or LLaVA-1.5 (False)
        """
        self.video_paths = video_paths
        self.annotations = annotations
        self.max_frames = max_frames
        self.processor = processor
        self.is_next = is_next
        
        # Filter out videos without annotations
        self.valid_pairs = []
        for video_path in video_paths:
            video_path = Path(video_path)
            video_id = video_path.stem
            
            if video_id in annotations:
                text_summary = annotations[video_id].get("text_summary", "").strip()
                if text_summary:
                    self.valid_pairs.append((str(video_path), video_id, text_summary))
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        video_path, video_id, text_summary = self.valid_pairs[idx]
        
        # Load frames and use middle frame (consistent with evaluation)
        frames = self._load_frames(video_path, self.max_frames)
        if len(frames) == 0:
            # Return a dummy frame if video loading fails
            rep_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            rep_frame = frames[len(frames) // 2]
        
        # Convert to PIL Image
        rep_img = Image.fromarray(rep_frame).convert("RGB")
        
        # Create prompt (same as in generate_summary)
        prompt = (
            "You are an expert traffic accident analyst. "
            "Watch this car crash video and generate a factual explanation. "
            "Describe clearly:\n"
            "- vehicles involved\n"
            "- number of vehicles\n"
            "- what each vehicle was doing before the crash\n"
            "- how the crash happened\n"
            "- where the impact occurred\n"
            "- outcome of the crash\n"
            "Write 4-6 coherent sentences like an incident report. "
            "Do NOT hallucinate unseen details."
        )
        
        # Format: USER: <image>\n{prompt}\nASSISTANT: {text_summary}
        user_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        formatted_prompt = f"{user_prompt} {text_summary}"
        
        # Process with LLaVA processor
        if self.processor is None:
            raise ValueError("Processor not initialized")
        
        if self.is_next:
            inputs = self.processor(
                text=formatted_prompt,
                images=rep_img,
                return_tensors="pt",
                padding=True
            )
            # Add image_sizes for LLaVA-NeXT if not present
            if "image_sizes" not in inputs:
                w, h = rep_img.size
                inputs["image_sizes"] = torch.tensor([[h, w]])
        else:
            inputs = self.processor(
                images=rep_img,
                text=formatted_prompt,
                return_tensors="pt",
                padding=True
            )
        
        result = {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # Remove batch dim
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "video_id": video_id,
            "text_summary": text_summary,
            "user_prompt": user_prompt  # Store for label masking
        }
        
        # Store image sizes for LLaVA-NeXT if present
        if "image_sizes" in inputs:
            result["image_sizes"] = inputs["image_sizes"].squeeze(0)
        
        return result
    
    def _load_frames(self, video_path: str, max_frames: int) -> List[np.ndarray]:
        """Load frames from video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 5th frame (as per config)
            if frame_count % 5 == 0:
                frames.append(frame)
                if len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames


# ------------------------------------------------------------------
# Collate Function
# ------------------------------------------------------------------
def collate_fn(batch):
    """Custom collate function for batching."""
    # Get processor to handle padding
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    # Pad sequences
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    user_prompts = [item["user_prompt"] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_mask = []
    
    for ids, mask in zip(input_ids, attention_mask):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)]))
        padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))
    
    result = {
        "pixel_values": pixel_values,
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "video_ids": [item["video_id"] for item in batch],
        "text_summaries": [item["text_summary"] for item in batch],
        "user_prompts": user_prompts
    }
    
    # Handle image_sizes if present
    if "image_sizes" in batch[0]:
        result["image_sizes"] = torch.stack([item["image_sizes"] for item in batch])
    
    return result


# ------------------------------------------------------------------
# Training Functions
# ------------------------------------------------------------------
def prepare_labels(input_ids, user_prompts, processor, model, device):
    """Prepare labels by masking prompt tokens, keeping only assistant response."""
    batch_size = input_ids.shape[0]
    labels = input_ids.clone()
    pad_token_id = processor.tokenizer.pad_token_id
    
    # Mask all padding tokens first
    labels[input_ids == pad_token_id] = -100
    
    # Tokenize user prompts to find where assistant response starts
    for i, user_prompt in enumerate(user_prompts):
        # Tokenize the user prompt (without image tokens)
        # We need to find where "ASSISTANT:" appears in the input_ids
        try:
            # Find the position of "ASSISTANT:" token sequence
            assistant_text = "ASSISTANT:"
            assistant_token = processor.tokenizer.encode(assistant_text, add_special_tokens=False)
            
            # Search for assistant token sequence in input_ids
            input_seq = input_ids[i].cpu().tolist()
            
            # Find where ASSISTANT starts
            assistant_pos = None
            for j in range(len(input_seq) - len(assistant_token) + 1):
                if input_seq[j:j+len(assistant_token)] == assistant_token:
                    assistant_pos = j + len(assistant_token)
                    break
            
            if assistant_pos is not None:
                # Mask everything before ASSISTANT response
                labels[i, :assistant_pos] = -100
            else:
                # Fallback: mask first 50% of tokens (rough approximation)
                # This should cover most of the prompt
                prompt_len = len(input_seq) // 2
                labels[i, :prompt_len] = -100
            
            # CRITICAL: Check if all labels are masked (would cause NaN loss)
            valid_labels = (labels[i] != -100).sum().item()
            if valid_labels == 0:
                # If all labels are masked, keep at least the last 10% of tokens
                seq_len = (input_ids[i] != pad_token_id).sum().item()
                if seq_len > 0:
                    keep_start = int(seq_len * 0.9)
                    labels[i, keep_start:] = input_ids[i, keep_start:]
                    print(f"WARNING: All labels were masked for sample {i}, keeping last 10% of tokens")
        except Exception as e:
            # Fallback: mask first 50% of tokens
            seq_len = (input_ids[i] != pad_token_id).sum().item()
            prompt_len = seq_len // 2
            labels[i, :prompt_len] = -100
            
            # Check if all labels are masked
            valid_labels = (labels[i] != -100).sum().item()
            if valid_labels == 0 and seq_len > 0:
                keep_start = int(seq_len * 0.9)
                labels[i, keep_start:] = input_ids[i, keep_start:]
    
    return labels


def train_epoch(
    model,
    dataloader,
    optimizer,
    processor,
    device,
    epoch: int,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Check if model is quantized (4-bit or 8-bit)
    # Quantized models don't work well with gradient scaling
    is_quantized = False
    try:
        # Check if model has quantization config
        if hasattr(model, 'hf_quantizer') or hasattr(model, 'quantization_config'):
            is_quantized = True
        # Also check for bitsandbytes quantized modules
        for name, module in model.named_modules():
            if 'Linear4bit' in str(type(module)) or 'Linear8bit' in str(type(module)):
                is_quantized = True
                break
    except:
        pass
    
    # Mixed precision training
    # Use scaler for FP16 models, skip for quantized models
    use_scaler = not is_quantized
    if use_scaler:
        # Check if model is actually in FP16 (not quantized)
        # If using FP16, we can use gradient scaler
        try:
            scaler = torch.amp.GradScaler('cuda')
            print("Using gradient scaler for FP16 training")
        except Exception as e:
            print(f"Gradient scaler setup failed: {e}, training without scaler")
            use_scaler = False
    else:
        print("Model is quantized, skipping gradient scaler")
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    optimizer.zero_grad()
    
    # Track consecutive NaN batches to stop training if model is corrupted
    consecutive_nan_count = 0
    max_consecutive_nan = 20  # Stop if 20 consecutive batches are NaN
    
    for step, batch in enumerate(pbar):
        # Move to device
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        user_prompts = batch["user_prompts"]
        
        # Prepare labels - mask prompt, keep only assistant response
        labels = prepare_labels(input_ids, user_prompts, processor, model, device)
        
        # CRITICAL: Verify labels are valid before forward pass
        # Check if all labels are masked (would cause NaN loss)
        for i in range(labels.shape[0]):
            valid_labels = (labels[i] != -100).sum().item()
            if valid_labels == 0:
                print(f"ERROR: All labels masked for batch item {i} at step {step}!")
                print(f"  Input sequence length: {input_ids[i].shape[0]}")
                print(f"  User prompt: {user_prompts[i][:100]}...")
                # Keep at least the last 20% of tokens as labels
                seq_len = (input_ids[i] != processor.tokenizer.pad_token_id).sum().item()
                if seq_len > 0:
                    keep_start = int(seq_len * 0.8)
                    labels[i, keep_start:] = input_ids[i, keep_start:]
                    print(f"  Fixed: Keeping last 20% of tokens ({seq_len - keep_start} tokens)")
                else:
                    print(f"  ERROR: Sequence is all padding! Skipping this batch.")
                    optimizer.zero_grad()
                    continue
        
        # Prepare model inputs
        model_inputs = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        # Add image_sizes if present (required for LLaVA-NeXT)
        if "image_sizes" in batch:
            model_inputs["image_sizes"] = batch["image_sizes"].to(device)
        
        # Forward pass with mixed precision
        from torch.amp import autocast
        
        for step, batch in enumerate(train_loader):
        
            with autocast(device_type='cuda', dtype=torch.float16, enabled=use_scaler):
                outputs = model(**model_inputs)
                loss = outputs.loss / gradient_accumulation_steps
        
            # Backward
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
            # Step
            if (step + 1) % gradient_accumulation_steps == 0:
        
                if max_grad_norm > 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
        
                optimizer.zero_grad(set_to_none=True)  # IMPORTANT (saves memory)
        
            # Store loss
            loss_value = loss.item() * gradient_accumulation_steps
        
        # Check for NaN or Inf loss
        import math
        if math.isnan(loss_value) or math.isinf(loss_value):
            consecutive_nan_count += 1
            print(f"\nWARNING: NaN/Inf loss detected at step {step}! Loss value: {loss_value}")
            print(f"  Consecutive NaN batches: {consecutive_nan_count}/{max_consecutive_nan}")
            
            # If too many consecutive NaN batches, model is likely corrupted
            if consecutive_nan_count >= max_consecutive_nan:
                print(f"\nERROR: {max_consecutive_nan} consecutive NaN batches detected!")
                print("  Model weights may be corrupted. Stopping training to prevent further damage.")
                print("  Consider:")
                print("    - Reducing learning rate further (try 1e-5)")
                print("    - Checking label masking (ensure not all labels are -100)")
                print("    - Using FP16 instead of quantization")
                raise RuntimeError(f"Training stopped due to {max_consecutive_nan} consecutive NaN losses")
            
            print(f"  Skipping this batch and continuing...")
            # Skip this batch but reset optimizer to avoid accumulating corrupted gradients
            optimizer.zero_grad()
            del loss, outputs, model_inputs, labels
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
        # Reset NaN counter on successful batch
        consecutive_nan_count = 0
        
        # Delete intermediate tensors to free memory
        del loss, outputs, model_inputs, labels
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        total_loss += loss_value  # Scale back for logging
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss_value:.4f}", "avg_loss": f"{total_loss/num_batches:.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # Check if average loss is NaN
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        print(f"\nWARNING: Average loss for epoch {epoch} is {avg_loss}!")
        avg_loss = float('nan')  # Return NaN so it can be logged
    return avg_loss


def validate_epoch(model, dataloader, processor, device, epoch: int):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch in pbar:
            # Move to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            user_prompts = batch["user_prompts"]
            
            # Prepare labels - mask prompt, keep only assistant response
            labels = prepare_labels(input_ids, user_prompts, processor, model, device)
            
            # Prepare model inputs
            model_inputs = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            # Add image_sizes if present (required for LLaVA-NeXT)
            if "image_sizes" in batch:
                model_inputs["image_sizes"] = batch["image_sizes"].to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(**model_inputs)
                loss = outputs.loss
            
            loss_value = loss.item()
            
            # Check for NaN or Inf loss
            import math
            if math.isnan(loss_value) or math.isinf(loss_value):
                print(f"\nWARNING: NaN/Inf validation loss detected! Loss value: {loss_value}")
                # Skip this batch
                del loss, outputs, model_inputs, labels
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            
            total_loss += loss_value
            num_batches += 1
            
            # Delete intermediate tensors to free memory
            del loss, outputs, model_inputs, labels
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_value:.4f}", "avg_loss": f"{total_loss/num_batches:.4f}"})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # Check if average loss is NaN
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        print(f"\nWARNING: Average validation loss for epoch {epoch} is {avg_loss}!")
        avg_loss = float('nan')  # Return NaN so it can be logged
    return avg_loss


# ------------------------------------------------------------------
# Main Training Function
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaVA-NeXT for video summarization"
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = get_config(args.config)

    # Setup paths
    root_dir = Path(
        config.get("dataset.root_dir")
        or config.config["dataset"]["root_dir"]
    )
    processed_dir = root_dir / (
        config.get("dataset.processed_dir")
        or config.config["dataset"]["processed_dir"]
    )

    split_info_file = processed_dir / "split_info.json"
    train_annotations_file = processed_dir / "annotations_train.json"
    val_annotations_file = processed_dir / "annotations_val.json"

    # Check required files
    if not split_info_file.exists():
        print("ERROR: Run scripts/01_process_data.py first.")
        sys.exit(1)

    if not train_annotations_file.exists() or not val_annotations_file.exists():
        print("ERROR: Training/validation annotations not found.")
        sys.exit(1)

    # Load split info
    with open(split_info_file) as f:
        split_info = json.load(f)

    # Load annotations
    with open(train_annotations_file) as f:
        train_annotations = json.load(f)

    with open(val_annotations_file) as f:
        val_annotations = json.load(f)

    train_videos = split_info["splits"]["train"]
    val_videos = split_info["splits"]["val"]

    print("=" * 60)
    print("FINE-TUNING LLaVA-NeXT FOR VIDEO SUMMARIZATION")
    print("=" * 60)
    print(f"Train videos: {len(train_videos)}")
    print(f"Val videos: {len(val_videos)}")

    # Device setup
    device = config.get("model.device") or config.config["model"]["device"]

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    print(f"Device: {device}")

    # Clear GPU cache before loading model
    if device != "cpu" and torch.cuda.is_available():
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        device_id = torch.cuda.current_device()
        print(
            f"GPU {device_id} memory before loading: "
            f"{torch.cuda.memory_allocated(device_id) / 1e9:.2f}GB allocated, "
            f"{torch.cuda.memory_reserved(device_id) / 1e9:.2f}GB reserved"
        )

    # Load model
    print("\nLoading model...")
    model_name = config.get("model.vision_model") or config.config["model"]["vision_model"]

    wrapper = LLaVANeXTWrapper(
        model_name=model_name,
        device=device
    )

    model = wrapper.model
    processor = wrapper.processor
    is_next = wrapper.is_next

    # Enable memory optimizations (ADD THIS)
    model.config.use_cache = False

    # Use LoRA/PEFT for parameter-efficient fine-tuning
    use_lora = True    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from peft import TaskType
            
            print("\nSetting up LoRA for parameter-efficient fine-tuning...")
            
            # Prepare model for k-bit training if quantized
            is_quantized = False
            for name, module in model.named_modules():
                if 'Linear4bit' in str(type(module)) or 'Linear8bit' in str(type(module)):
                    is_quantized = True
                    break
            
           if is_quantized:
                print("Preparing quantized model for LoRA training...")
                # Enable gradient checkpointing to save memory (works better with LoRA)
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                model.config.use_cache = False
                print("Enabled gradient checkpointing for memory efficiency")

                model.config.use_cache = False

                for name, param in model.named_parameters():
                    if "mm_projector" in name or "lm_head" in name:
                        param.requires_grad = True
            
            # Configure LoRA
            # Target language model layers (not vision encoder)
            # For Mistral-based models, target these attention and MLP layers
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            # Verify these modules exist in the model
            found_modules = []
            for name, module in model.named_modules():
                module_name = name.split('.')[-1]
                if module_name in target_modules and 'language_model' in name:
                    found_modules.append(module_name)
            
            if found_modules:
                target_modules = list(set(found_modules))
                print(f"Found LoRA target modules: {target_modules}")
            else:
                print(f"Using default LoRA target modules: {target_modules}")
            
            lora_config = LoraConfig(
                r=8,  # LoRA rank (reduced from 16 to save memory)
                lora_alpha=16,  # LoRA alpha (typically 2x rank)
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print("✓ LoRA setup complete - only training adapter layers")
            
        except ImportError:
            print("WARNING: PEFT library not available. Install with: pip install peft")
            print("Falling back to full fine-tuning (may cause NaN with quantization)")
            use_lora = False
        except Exception as e:
            print(f"WARNING: LoRA setup failed: {e}")
            print("Falling back to full fine-tuning")
            use_lora = False
    else:
        # Freeze vision encoder to save memory
        if hasattr(model, 'vision_tower'):
            for param in model.vision_tower.parameters():
                param.requires_grad = False
            print("Frozen vision encoder to save memory")
    
    # Gradient checkpointing is handled by LoRA setup if using LoRA
    # Otherwise it's disabled for numerical stability
    if not use_lora:
        print("Gradient checkpointing disabled for numerical stability")
    
    # Enable training mode
    model.train()
    
    # Create datasets
    print("\nCreating datasets...")
    max_frames = config.get("model.max_frames") or config.config["model"]["max_frames"]
    train_dataset = VideoSummarizationDataset(
        video_paths=train_videos,
        annotations=train_annotations,
        max_frames=max_frames,
        processor=processor,
        is_next=is_next
    )
    
    val_dataset = VideoSummarizationDataset(
        video_paths=val_videos,
        annotations=val_annotations,
        max_frames=max_frames,
        processor=processor,
        is_next=is_next
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders - reduce batch size for memory efficiency
    batch_size = config.get("model.batch_size") or config.config["model"]["batch_size"]
    # Reduce batch size to 1 to avoid OOM
    effective_batch_size = min(batch_size, 1)
    gradient_accumulation_steps = batch_size // effective_batch_size
    
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Reduce num_workers and disable pin_memory to save GPU memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to reduce memory usage
        pin_memory=False  # Disable to save GPU memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to reduce memory usage
        pin_memory=False  # Disable to save GPU memory
    )
    
    # Setup optimizer - ensure numeric values are floats
    learning_rate = config.get("training.learning_rate") or config.config["training"]["learning_rate"]
    weight_decay = config.get("training.weight_decay") or config.config["training"]["weight_decay"]
    
    # Convert to float if string (YAML might load scientific notation as string)
    if isinstance(learning_rate, str):
        learning_rate = float(learning_rate)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    
    # Only train language model parameters (freeze vision encoder)
    # Alternatively, you can train all parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup loss tracker
    loss_file = config.get("training.loss_file") or config.config["training"]["loss_file"]
    val_loss_file = config.get("training.val_loss_file") or config.config["training"]["val_loss_file"]
    loss_tracker = LossTracker(
        loss_file=loss_file,
        val_loss_file=val_loss_file
    )
    
    # Training loop
    num_epochs = config.get("training.num_epochs") or config.config["training"]["num_epochs"]
    if isinstance(num_epochs, str):
        num_epochs = int(num_epochs)
    
    save_dir = Path(config.get("training.save_dir") or config.config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_every = config.get("training.save_every") or config.config["training"]["save_every"]
    if isinstance(save_every, str):
        save_every = int(save_every)
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    # Clear GPU cache before training
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        device_id = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_id) / 1e9
        reserved = torch.cuda.memory_reserved(device_id) / 1e9
        total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        print(f"GPU {device_id} memory status:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Free: {total - reserved:.2f} GB")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Get max_grad_norm from config (default to 0.5 for stability)
        max_grad_norm = config.get("training.max_grad_norm") or config.config["training"].get("max_grad_norm", 0.5)
        if isinstance(max_grad_norm, str):
            max_grad_norm = float(max_grad_norm)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, processor, device, epoch,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Validate
        val_loss = validate_epoch(model, val_loader, processor, device, epoch)
        
        # Clear cache after validation
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log losses
        loss_tracker.log_training_loss(epoch, train_loss)
        loss_tracker.log_validation_loss(epoch, val_loss)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % save_every == 0 or val_loss < best_val_loss:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
            
            # Save model state
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": dict(config.config)  # Save config
            }, checkpoint_path)
            
            print(f"  Saved checkpoint: {checkpoint_path}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = save_dir / "best_checkpoint.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": dict(config.config)
                }, best_checkpoint)
                print(f"  Saved best checkpoint: {best_checkpoint} (val_loss: {val_loss:.4f})")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()

