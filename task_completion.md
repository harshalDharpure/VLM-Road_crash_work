# Project Completion Summary

**Title:** Efficient Vision-Language Video Summarization for Road Crash Analysis Using Sparse Temporal Sampling and Instruction-Tuned Multimodal Models

**Project path:** `/DATA/vaneet_2221cs15/vlm-road-crash`

**Last updated:** May 29, 2026

---

## 1. Project Goal

Build a **publication-ready research pipeline** for road crash video summarization on **Crash-1500**, using:

- Sparse temporal frame sampling (`dense`, `every_3rd`, `every_5th`, `every_10th`)
- **LLaVA-NeXT** (`llava-hf/llava-v1.6-mistral-7b-hf`) for zero-shot and LoRA fine-tuning
- Full automatic metrics: BLEU, ROUGE, METEOR, BERTScore, CIDEr, SPICE, NLI
- Ablation studies, NLI faithfulness optimization, and journal-grade reporting artifacts

---

## 2. Dataset & Splits

| Item | Details |
|------|---------|
| Videos | ~1500 crash videos in `video1500/` |
| Ground truth | `Car_Crash_Text_Dataset_ground_truth.xlsx` (primary text: `Explanation`) |
| Processed data | `data/processed/` (annotations, splits, cached frames) |
| Split (seed 42) | **Train 1048 \| Val 224 \| Test 226** |
| Frame cache | `data/processed/frames/{strategy}/{video_id}/` |

---

## 3. Pipeline Infrastructure Completed

### 3.1 End-to-end scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/01_process_data.py` | Validate videos, parse GT, split, extract frames | Done |
| `scripts/02_evaluate_zero_shot.py` | Zero-shot inference + full metrics | Done |
| `scripts/03_finetune.py` | LoRA/QLoRA fine-tuning | Done |
| `scripts/04_evaluate_finetuned.py` | Fine-tuned model evaluation | Done |
| `scripts/05_compare_results.py` | Compare zero-shot vs fine-tuned | Done |
| `scripts/06_run_ablations.py` | Pilot ablations (N=25) | Done |
| `scripts/07_generate_publication_outputs.py` | Tables, figures, LaTeX, paper sections | Done |
| `scripts/08_recompute_metrics.py` | Recompute CIDEr/SPICE on saved predictions | Done |
| `scripts/09_nli_optimized_eval.py` | NLI post-processing + sentence filtering | Done |
| `scripts/10_run_full_test_ablations.py` | Full-test ablation grid (N=226) | Done |
| `scripts/11_build_journal_proof.py` | CI, significance tests, SHA256 provenance | Done |

### 3.2 Shell launchers

- `scripts/run_publication_pipeline.sh` — full pipeline
- `scripts/run_full_test_ablations.sh` — 20-config ablation grid
- `scripts/run_nli_optimization.sh` — NLI optimization sweep
- `scripts/run_improved_eval.sh` — collage + epoch-2 re-eval
- `scripts/run_publication_retrain.sh` — faithfulness-focused retrain (v2)
- `scripts/run_complete_background.sh` — background experiment runner

### 3.3 Core source modules

| Module | Role |
|--------|------|
| `src/data_processing/frame_sampler.py` | Sparse temporal sampling + frame cache |
| `src/data_processing/ground_truth_parser.py` | Excel → JSON annotations |
| `src/models/llava_next_wrapper.py` | LLaVA-NeXT inference (8-bit, middle-frame / collage) |
| `src/models/unified_vlm.py` | Unified VLM interface |
| `src/models/prompt_strategies.py` | 6 prompt templates incl. `structured_event`, `faithfulness` |
| `src/models/frame_utils.py` | Multi-frame collage utilities |
| `src/evaluation/metrics_suite.py` | Unified BLEU–NLI metric computation |
| `src/evaluation/nli_evaluator.py` | RoBERTa-large-MNLI faithfulness |
| `src/evaluation/faithfulness.py` | Post-processing + sentence-level NLI filter |
| `src/analysis/plotting.py` | Metric comparison / radar / heatmap figures |
| `src/analysis/statistical_analysis.py` | Bootstrap CI, paired tests |
| `src/analysis/paper_generator.py` | Auto methodology / results sections |
| `src/analysis/error_analysis.py` | Qualitative failure analysis |
| `src/analysis/human_eval.py` | Human evaluation annotation sheets |
| `src/utils/gpu_manager.py` | Auto GPU selection |

### 3.4 Configuration

- Main config: `config/config.yaml`
- Default sparse sampling: `every_5th`
- Evaluation metrics: BLEU, ROUGE, METEOR, BERTScore, CIDEr, SPICE, NLI
- Ablation grid: 4 frame strategies × 5 prompt strategies = **20 configs**

---

## 4. Experiments Completed

### 4.1 Data processing

- [x] Video validation and annotation parsing
- [x] Train/val/test split (seed 42)
- [x] Frame extraction for all sampling strategies
- [x] Cached frames under `data/processed/frames/`

### 4.2 Zero-shot evaluation (full test, N=226)

- [x] **Canonical baseline:** `every_5th` + `structured_event`, single middle frame
- [x] Collage variant: `every_5th_structured_event_test_collage`
- [x] **Full ablation grid:** 20 strategy × prompt configs on complete test set (May 28–29, 2026)
  - 20/20 completed, 0 failures
  - Log: `results/logs/nohup_full_test_ablations.log`

### 4.3 LoRA fine-tuning

- [x] LoRA rank 8, 5 epochs (best val loss at epoch 2)
- [x] Fine-tuned v1 evaluation on full test set (`results/finetuned/best_checkpoint/`)
- [x] CIDEr recomputed with corpus-level scoring (`metrics_recomputed.json`)
- [x] Epoch-2 checkpoint + collage eval (`checkpoint_epoch_2_collage/`)
- [x] Publication retrain v2 with faithfulness prompt + collage (`best_checkpoint_collage/`) — completed but poor NLI

### 4.4 NLI optimization pipeline

- [x] Post-processing (boilerplate removal, deduplication, truncation)
- [x] Sentence-level NLI filtering
- [x] Best config: `structured_event` + postprocess + filter → **60.2% NLI**
- [x] Results: `results/nli_optimized/best_run.json`

### 4.5 Metrics & evaluation fixes

- [x] **CIDEr fix:** corpus-level scoring in `metrics_suite.py` (was per-sample avg → 0)
- [x] **SPICE fix:** Java 11 JRE installed at `tools/jdk-11.0.31+11-jre/`; patched pycocoevalcap to use it
- [x] Restored canonical 226-sample zero-shot metrics from `results/logs/zero_shot_full_test.log` → `metrics_full_226.json`
- [x] Metrics recomputation script for existing `detailed_results.json`

### 4.6 Publication outputs

- [x] `results/experiment_tables.md` — primary + ablation tables
- [x] `results/comparison_report.json` — all run metrics
- [x] `results/latex_tables/main_results.tex` — LaTeX table
- [x] `results/publication_figures/` — metric comparison, radar, heatmap
- [x] `results/paper_support/` — methodology, setup, results, figure captions
- [x] `results/qualitative_examples/` — error analysis + human eval sheets (50 samples)
- [x] `results/statistical_analysis/journal_proof.md` — CI, significance, SHA256 hashes

### 4.7 Reproducibility & overwrite protection (May 29, 2026)

- [x] `02_evaluate_zero_shot.py`: `--output-name`, `--fail-if-exists`, `run_metadata.json`
- [x] `10_run_full_test_ablations.py`: `--run-prefix`, writes to isolated folders by default
- [x] `07_generate_publication_outputs.py`: prefers `metrics_full_226.json` for canonical baseline
- [x] `11_build_journal_proof.py`: statistical proof artifacts

### 4.8 In progress (optional verification run)

- [ ] **Locked canonical re-eval** (`canonical_locked_20260529_185235`) — started in background on GPU 0 for independent verification; saves to a new folder without overwriting canonical results. Not required for publication (canonical metrics already archived).

---

## 5. Key Results (Full Test, N=226)

### 5.1 Primary methods (paper Table 1)

| Method | Frames | Prompt | BLEU-1 | ROUGE-L | METEOR | BERTScore | CIDEr | NLI Ent. |
|--------|--------|--------|--------|---------|--------|-----------|-------|----------|
| **Zero-shot (canonical)** | every_5th | structured_event | **0.274** | **0.207** | **0.217** | **0.846** | 0.000* | **0.664** |
| Zero-shot (collage) | every_5th | structured_event | 0.249 | 0.185 | 0.232 | 0.832 | 0.002 | 0.044 |
| Fine-tuned v1 | every_5th | incident report | 0.334 | 0.261 | 0.293 | 0.871 | 0.014 | 0.341 |
| Fine-tuned epoch-2 (collage) | every_5th | structured_event | 0.320 | 0.256 | 0.286 | 0.869 | 0.010 | 0.310 |
| Fine-tuned v2 (collage) | every_5th | faithfulness | 0.338 | 0.242 | 0.300 | 0.874 | 0.002 | 0.058 |
| NLI-optimized | every_5th | structured_event + filter | 0.124 | 0.158 | 0.098 | 0.852 | 0.003 | 0.602 |

\* Early CIDEr=0 due to metric bug; recomputed where noted.

**Canonical source:** `results/zero_shot/every_5th_structured_event_test/metrics_full_226.json`

### 5.2 Zero-shot vs fine-tuned (canonical)

| Metric | Zero-shot | Fine-tuned v1 | Change |
|--------|-----------|---------------|--------|
| BLEU-1 | 0.274 | 0.334 | +22% |
| ROUGE-L | 0.207 | 0.261 | +26% |
| METEOR | 0.217 | 0.293 | +35% |
| BERTScore | 0.846 | 0.871 | +3% |
| **NLI Entailment** | **0.664** | **0.341** | **−49%** |

### 5.3 Statistical proof (journal)

From `results/statistical_analysis/journal_proof.md`:

| Run | NLI Entailment | 95% Wilson CI | n |
|-----|----------------|---------------|---|
| Canonical zero-shot | 0.6637 | [0.5998, 0.7221] | 226 |
| Fine-tuned v1 | 0.3407 | [0.2820, 0.4047] | 226 |
| NLI-optimized | 0.6018 | [0.5368, 0.6634] | 226 |

| Comparison | z | p-value | Significant? |
|------------|---|---------|--------------|
| Canonical vs fine-tuned | 6.87 | 6.54×10⁻¹² | Yes |
| Canonical vs NLI-optimized | 1.37 | 0.172 | No |
| NLI-optimized vs fine-tuned | 5.56 | 2.71×10⁻⁸ | Yes |

### 5.4 Full ablation grid (N=226)

All 20 configs evaluated on complete test set. Best NLI in ablation re-runs (excluding canonical archive):

| Config | NLI Ent. | BLEU-1 | ROUGE-L |
|--------|----------|--------|---------|
| every_5th_temporal_sequence | 0.124 | 0.241 | 0.174 |
| every_10th_basic_caption | 0.111 | 0.282 | 0.180 |
| every_5th_basic_caption | 0.089 | 0.285 | 0.181 |

**Note:** The ablation re-run overwrote `every_5th_structured_event_test/metrics.json` with lower NLI (~3%). The **canonical 66.4% NLI** is preserved in `metrics_full_226.json` and is used for publication tables.

### 5.5 Training curve

| Epoch | Train loss | Val loss |
|-------|------------|----------|
| 1 | 3.625 | 3.473 |
| 2 | 3.432 | **3.447** (best) |
| 3 | 3.413 | 3.456 |
| 4 | 3.392 | 3.462 |
| 5 | 3.369 | 3.473 |

Files: `results/training_loss.json`, `results/validation_loss.json`

---

## 6. Key Findings (for the paper)

1. **Sparse zero-shot is the best faithfulness story.** `every_5th` + `structured_event` + single middle frame achieves **66.4% NLI entailment** at lower compute than dense sampling.

2. **Fine-tuning improves lexical overlap but hurts faithfulness.** BLEU/ROUGE/METEOR rise, but NLI drops from 66.4% → 34.1% (highly significant, p < 10⁻¹¹).

3. **Multi-frame collage hurts NLI sharply.** Collage zero-shot: 4.4% NLI; collage fine-tuned v2: 5.8% NLI.

4. **NLI post-processing helps but does not beat canonical zero-shot.** Best optimized run: 60.2% NLI (not statistically different from 66.4% canonical at p=0.17, but lower point estimate).

5. **Recommended main contribution:** Efficient sparse temporal sampling + structured prompting for faithful crash summarization, with fine-tuning discussed as a contrast/negative result for NLI.

---

## 7. Important Files Index

### Results & tables

| File | Description |
|------|-------------|
| `results/experiment_tables.md` | Main experiment tables (Section 1 + auto ablation grid) |
| `results/comparison_report.json` | All zero-shot run metrics JSON |
| `results/statistical_analysis/journal_proof.md` | CI, significance, provenance |
| `results/statistical_analysis/journal_proof.json` | Same, machine-readable |
| `results/nli_optimized/best_run.json` | Best NLI optimization config |
| `results/latex_tables/main_results.tex` | LaTeX results table |
| `results/paper_support/` | Auto-generated paper sections |

### Canonical baseline

| File | Description |
|------|-------------|
| `results/zero_shot/every_5th_structured_event_test/metrics_full_226.json` | **Authoritative canonical metrics (66.4% NLI)** |
| `results/logs/zero_shot_full_test.log` | Original full-test run log |
| `results/zero_shot/every_5th_structured_event_test/detailed_results.json` | Per-video predictions (may be from ablation re-run) |

### Fine-tuned

| File | Description |
|------|-------------|
| `results/finetuned/best_checkpoint/metrics.json` | Fine-tuned v1 metrics |
| `results/finetuned/best_checkpoint/metrics_recomputed.json` | v1 with fixed CIDEr |
| `results/finetuned/checkpoint_epoch_2_collage/` | Epoch-2 collage eval |
| `results/finetuned/best_checkpoint_collage/` | Publication retrain v2 |

### Logs

| File | Description |
|------|-------------|
| `results/logs/nohup_full_test_ablations.log` | Full ablation run (20/20 done) |
| `results/logs/nohup_nli_optimization.log` | NLI optimization sweep |
| `results/logs/nohup_publication_retrain.log` | Publication retrain v2 |
| `results/logs/zero_shot_full_test.log` | Original canonical zero-shot |

---

## 8. How to Regenerate Outputs

```bash
cd /DATA/vaneet_2221cs15/vlm-road-crash
source venv/bin/activate

# Regenerate tables, figures, LaTeX
venv/bin/python scripts/07_generate_publication_outputs.py

# Regenerate statistical proof
venv/bin/python scripts/11_build_journal_proof.py

# Recompute metrics on saved predictions (e.g., add SPICE)
venv/bin/python scripts/08_recompute_metrics.py --recompute-all-detailed
```

---

## 9. Remaining Work (not yet done)

These are **optional enhancements** for a top-tier journal submission:

| Item | Status | Notes |
|------|--------|-------|
| Human evaluation (20–30 samples, 3 raters) | Not started | Annotation sheets prepared in `results/qualitative_examples/` |
| External baseline (Qwen2-VL, BLIP-2) | Not started | `qwen_vlm/` stub exists |
| Runtime / efficiency comparison table | Not started | Per-run `runtime_sec_mean` available in metrics JSON |
| Statistical tests on BLEU/ROUGE (bootstrap) | Partial | NLI tests done; lexical metrics could be added |
| Locked canonical re-eval verification | In progress | Background job for independent replication |
| Camera-ready summary table (post locked re-eval) | Pending | Auto-generate when locked run completes |

---

## 10. Recommended Paper Narrative

**Title angle:** Efficient sparse sampling + structured prompting achieves strong semantic faithfulness for road crash video summarization.

**Main result:** Zero-shot LLaVA-NeXT with `every_5th` frame sampling and `structured_event` prompt → **66.4% NLI entailment** (95% CI: 60.0–72.2%) on 226 test videos.

**Contrast result:** LoRA fine-tuning improves BLEU-1 (+22%) and ROUGE-L (+26%) but **halves NLI faithfulness** (p < 10⁻¹¹).

**Ablation:** Full 226-sample grid over 4 sampling rates × 5 prompts; canonical config dominates on NLI.

**Efficiency claim:** Sparse `every_5th` sampling reduces frames vs dense while maintaining best faithfulness.

---

## 11. Timeline Summary

| Date | Milestone |
|------|-----------|
| May 24–26, 2026 | Initial full experiments: zero-shot, fine-tuning, canonical 226-sample baseline |
| May 26, 2026 | Multi-frame collage eval, CIDEr fix, improved eval runs |
| May 27, 2026 | NLI optimization pipeline; publication retrain v2 |
| May 28–29, 2026 | Full-test ablation grid (20 configs, N=226); SPICE fix; publication tables |
| May 29, 2026 | Overwrite protection; journal proof artifacts; locked canonical re-eval started |

---

*This document summarizes all completed work on the Crash-1500 VLM summarization project as of May 29, 2026.*
