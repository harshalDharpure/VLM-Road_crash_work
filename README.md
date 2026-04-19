# Optimize and Run VLM Road Crash Project from Scratch (Colab / Kaggle)

**For:** Tanay and Tripti  
**Goal:** Run the complete pipeline from scratch on **Google Colab** or **Kaggle** and produce reproducible results and a short report.

---

## Project Overview 

- **What:** Fine-tune a Vision-Language Model (LLaVA-NeXT) for **car crash video summarization**.
- **Pipeline:** Process videos → Zero-shot evaluation (baseline) → Fine-tune → Evaluate fine-tuned model → Compare baseline vs fine-tuned.
- **Platform:** Google Colab (free GPU) or Kaggle Notebooks (free GPU, ~30 hrs/week). Both support PyTorch and enough GPU memory if you use a **small subset** of the dataset for training/eval.

---

## Task Split: Tanay & Tripti

| Phase | Tanay | Tripti |
|-------|--------|--------|
| **1. Setup & data** | [x] Own notebook: env + data pipeline | [x] Same (or use shared processed data) |
| **2. Baseline** | [x] Run zero-shot evaluation (02) | — |
| **3. Training** | — | [x] Run fine-tuning (03) |
| **4. Evaluation** | — | [x] Run fine-tuned eval (04) + comparison (05) |
| **5. Report** | [x] Document: setup, data stats, zero-shot metrics | [x] Document: training curves, fine-tuned metrics, comparison |

**Outcome:** One shared Colab/Kaggle notebook (or two linked notebooks) that together run the **full pipeline from scratch**, plus a short joint report with tables and findings.

---

## Prerequisites (Tanay & Tripti)

1. **Accounts**
   - Google account (for Colab) and/or Kaggle account.
   - Hugging Face account (for model download): https://huggingface.co/join  
   - Optional: Hugging Face token if models are gated: https://huggingface.co/settings/tokens

2. **Dataset**
   - **Car Crash Dataset**: videos + text summaries.
   - Source: [CarCrashDataset (GitHub)](https://github.com/Cogito2012/CarCrashDataset) or any mirror you provide.
   - Needed:
     - **Videos:** MP4 files (e.g. in a folder `videos/`).
     - **Ground truth:** Excel file with columns like `Video_ID` and `Text_Summary` (e.g. `Car_Crash_Text_Dataset (3).xlsx`).
   - **Important for Colab/Kaggle:** Use a **subset** (e.g. 100–300 videos) to avoid disk limits and long runtimes. You can still use 70/15/15 split on this subset.

3. **Code**
   - Clone or upload the project repo into the notebook environment (e.g. `/content/vlm-road-crash` on Colab or `/kaggle/working/vlm-road-crash` on Kaggle).

---

## Phase 1: Environment Setup (Tanay & Tripti)

**Where:** First cells of the notebook (Colab or Kaggle).

### 1.1 Enable GPU

- **Colab:** Runtime → Change runtime type → Hardware accelerator → **GPU** (T4 or better).
- **Kaggle:** Settings → Accelerator → **GPU**.

### 1.2 Set project and data paths

- **Colab:** Project at `/content/vlm-road-crash`, data e.g. `/content/data` or from Google Drive.
- **Kaggle:** Project at `/kaggle/working/vlm-road-crash`, data in `/kaggle/input/...` (e.g. from a Kaggle dataset).

Update `config/config.yaml` so that:

- `dataset.root_dir` points to the folder that contains:
  - `videos/` (all or subset of MP4s)
  - `Car_Crash_Text_Dataset (3).xlsx` (or your GT file name)

Example for Colab:

```yaml
dataset:
  root_dir: "/content/data"   # folder containing videos/ and Excel
  videos_dir: "videos"
  ground_truth_file: "Car_Crash_Text_Dataset (3).xlsx"
  processed_dir: "data/processed"
  # ... rest unchanged
```

Example for Kaggle (if data is in an input dataset):

```yaml
dataset:
  root_dir: "/kaggle/input/your-car-crash-dataset"
  videos_dir: "videos"
  ground_truth_file: "Car_Crash_Text_Dataset (3).xlsx"
  processed_dir: "/kaggle/working/data/processed"
```

### 1.3 Install dependencies

Run in a cell:

```python
# Colab/Kaggle: go to project root
import sys
from pathlib import Path
project_root = Path("/content/vlm-road-crash")  # or /kaggle/working/vlm-road-crash
sys.path.insert(0, str(project_root))
%cd {project_root}

# Install PyTorch (Colab often has it; Kaggle has it)
# If needed: pip install torch torchvision torchaudio
pip install -r requirements.txt

# NLTK for evaluation
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
```

### 1.4 Verify GPU and paths

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# Check data exists
from pathlib import Path
root = Path(config["dataset"]["root_dir"])  # from config
videos_dir = root / config["dataset"]["videos_dir"]
gt_file = root / config["dataset"]["ground_truth_file"]
print("Videos dir exists:", videos_dir.exists())
print("Videos count:", len(list(videos_dir.glob("*.mp4"))) if videos_dir.exists() else 0)
print("GT file exists:", gt_file.exists())
```

---

## Phase 2: Data Pipeline – Run from Scratch (Both / Tanay Lead)

**Script:** `scripts/01_process_data.py`

**What it does:** Finds all MP4s, extracts frames (5-sec segment, every 5th frame), parses Excel GT, splits into train/val/test (70/15/15), writes:

- `data/processed/split_info.json`
- `data/processed/annotations_train.json`, `annotations_val.json`, `annotations_test.json`
- Optionally frames under `data/processed/frames/` (can disable or limit to save disk on Colab/Kaggle).

**Run:**

```python
%cd /content/vlm-road-crash  # or /kaggle/working/vlm-road-crash
!python scripts/01_process_data.py
```

**Tanay’s tasks:**

- Ensure config paths are correct and dataset is a subset if needed.
- Run the script and confirm no errors.
- Note in the report: number of videos, train/val/test sizes, and where processed outputs are stored (paths).

---

## Phase 3: Zero-Shot Evaluation – Baseline (Tanay)

**Script:** `scripts/02_evaluate_zero_shot.py`

**What it does:** Loads pretrained LLaVA-NeXT, runs on the **test** set, computes BLEU, METEOR, ROUGE, BERTScore, CIDEr, NLI, and writes:

- `results/zero_shot/metrics.json`
- `results/zero_shot/detailed_results.json`

**Run:**

```python
!python scripts/02_evaluate_zero_shot.py
```

**Tanay’s tasks:**

- Run and confirm `results/zero_shot/metrics.json` is generated.
- In the report: paste or summarize the main metrics (e.g. BLEU-1/4, METEOR, ROUGE-1/L, NLI) and mention that this is the **baseline (no fine-tuning)**.

---

## Phase 4: Fine-Tuning (Tripti)

**Script:** `scripts/03_finetune.py`

**What it does:** Fine-tunes LLaVA-NeXT (LoRA, 8-bit) for 5 epochs (or fewer if you reduce in config), saves checkpoints and loss files:

- `results/checkpoints/checkpoint_epoch_1.pt`, ..., `best_checkpoint.pt`
- `results/training_loss.json`
- `results/validation_loss.json`

**Colab/Kaggle notes:**

- Use a **small subset** of training data if needed (e.g. 50–100 videos) and/or reduce `num_epochs` (e.g. 2–3) to fit within session time.
- If OOM: reduce `batch_size` in `config/config.yaml` (e.g. to 2 or 1).

**Run:**

```python
!python scripts/03_finetune.py
# Optional: resume from checkpoint
# !python scripts/03_finetune.py --resume results/checkpoints/checkpoint_epoch_2.pt
```

**Tripti’s tasks:**

- Run training and confirm checkpoints and loss files appear.
- In the report: plot or table training/validation loss per epoch and mention best checkpoint path.

---

## Phase 5: Fine-Tuned Evaluation and Comparison (Tripti)

**Scripts:**  
- `scripts/04_evaluate_finetuned.py`  
- `scripts/05_compare_results.py`

**4. Evaluate fine-tuned model**

```python
!python scripts/04_evaluate_finetuned.py \
  --checkpoint results/checkpoints/best_checkpoint.pt \
  --split test
```

This writes e.g. `results/finetuned/best_checkpoint/metrics.json` and `detailed_results.json`.

**5. Compare zero-shot vs fine-tuned**

```python
!python scripts/05_compare_results.py \
  --zero_shot_metrics results/zero_shot/metrics.json \
  --finetuned_metrics results/finetuned/best_checkpoint/metrics.json \
  --output results/comparison_report.json
```

**Tripti’s tasks:**

- Run both steps and confirm output files.
- In the report: table of zero-shot vs fine-tuned metrics and % improvement (e.g. METEOR, ROUGE-L). Use `comparison_report.json` if it contains these numbers.

---

## Where Results Are Stored (Quick Reference)

| Output | Path |
|--------|------|
| Processed data & splits | `data/processed/` (split_info.json, annotations_*.json) |
| Zero-shot metrics | `results/zero_shot/metrics.json`, `detailed_results.json` |
| Checkpoints | `results/checkpoints/*.pt` |
| Training/validation loss | `results/training_loss.json`, `results/validation_loss.json` |
| Fine-tuned metrics | `results/finetuned/<checkpoint_name>/metrics.json` |
| Comparison | `results/comparison_report.json` |

On Colab/Kaggle, these paths are relative to the project root (e.g. `/content/vlm-road-crash` or `/kaggle/working/vlm-road-crash`). Save important result files to Drive or Kaggle output if the session ends.

---

## Checklist for "Run from Scratch"

- [ ] **Setup:** GPU on, paths in `config/config.yaml` set for Colab/Kaggle, dependencies installed, data (videos + Excel) present.
- [ ] **Data:** `01_process_data.py` run successfully; train/val/test counts and paths noted.
- [ ] **Zero-shot (Tanay):** `02_evaluate_zero_shot.py` run; `results/zero_shot/metrics.json` present; baseline metrics in report.
- [ ] **Fine-tuning (Tripti):** `03_finetune.py` run; checkpoints and loss files present; loss curve/summary in report.
- [ ] **Eval + comparison (Tripti):** `04_evaluate_finetuned.py` and `05_compare_results.py` run; comparison table and improvements in report.

---

## Common Issues (Colab / Kaggle)

| Issue | What to do |
|-------|------------|
| **CUDA out of memory** | Reduce `batch_size` in `config/config.yaml`; use smaller subset of videos. |
| **Session timeout** | On Kaggle: use “Save Version” and resume later; reduce epochs or data size. |
| **Dataset not found** | Double-check `dataset.root_dir`, `videos_dir`, and `ground_truth_file`; list dirs with `!ls` in notebook. |
| **Model download slow / gated** | Log in to Hugging Face and set `HF_TOKEN` or use `login()` in notebook; ensure model name matches (e.g. `llava-hf/llava-v1.6-mistral-7b-hf`). |
| **Paths wrong** | Use absolute paths in config for Colab/Kaggle; avoid hardcoded `/home/...` from another machine. |

---
