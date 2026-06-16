# Cross-Modal Car Crash Video Summarization with Vision-Language Models (LLaVA-NeXT)

## Overview
This project implements an end-to-end pipeline to generate natural language descriptions from short car crash videos using Vision-Language Models (VLMs). The system combines zero-shot inference with domain-specific fine-tuning to improve understanding of accident scenarios.

Videos are processed using sparse frame sampling and passed to a multimodal model to generate structured textual summaries describing vehicles, actions, collisions, and outcomes.

---

## Objectives
- Generate meaningful accident descriptions from video inputs  
- Compare zero-shot and fine-tuned model performance  
- Evaluate outputs using lexical, semantic, and logical metrics  
- Optimize large models for execution under limited GPU resources  

---

## Methodology

### Pipeline
1. Video → Frame Extraction (every 5th frame)  
2. Frame → Model Input (LLaVA-NeXT)  
3. Text Generation  
4. Evaluation using multi-metric framework  

---

## Model Details
- Base Model: LLaVA-NeXT  
- Fine-tuning Method: LoRA (Low-Rank Adaptation)  
- Quantization: 4-bit (BitsAndBytes)  

---

## Dataset
| Component        | Value |
|----------------|------|
| Total Videos    | 1275 |
| Train           | 893  |
| Validation      | 191  |
| Test            | 191  |
| Segment Length  | 5 sec |
| Frame Sampling  | Every 5th frame |

---

## Training Configuration
- Batch size: 1  
- Epochs: 2–5 (resource constrained)  
- Sparse frame input (2–4 frames per video)  

---

## Key Contributions

### Fine-tuning Pipeline
- Implemented LoRA-based fine-tuning for multimodal model adaptation  
- Designed preprocessing pipeline for video-to-text conversion  
- Ensured compatibility between visual and textual inputs  

### Evaluation Framework
- Implemented a complete evaluation pipeline using:
  - BLEU (1–4)
  - METEOR
  - ROUGE (1, 2, L)
  - BERTScore
  - CIDEr
  - NLI-based semantic consistency  
- Built comparison pipeline for zero-shot vs fine-tuned results  

### System Optimization
- Reduced effective model size to approximately 9GB  
- Applied 4-bit quantization to reduce GPU memory usage  
- Optimized generation parameters (token limits, cache disabling)  
- Fixed CUDA out-of-memory issues during inference  
- Ensured consistency across training and evaluation configurations  

---

## Results

### Training Performance
| Epoch | Training Loss | Validation Loss |
|------|-------------|----------------|
| 1    | 5.28        | 3.49           |
| 2    | 3.47        | 3.48           |
| 3    | 3.46        | 3.46           |
| 4    | 3.45        | 3.46           |
| 5    | 3.44        | 3.45           |

### Zero-shot vs Fine-tuned Comparison
| Metric   | Zero-shot | Fine-tuned | Improvement |
|----------|----------|-----------|-------------|
| METEOR   | 0.235    | 0.293     | +24.7%      |
| ROUGE-1  | 0.350    | 0.396     | +13.0%      |
| ROUGE-L  | 0.221    | 0.264     | +19.2%      |

Observations:
- Fine-tuning improves semantic and structural alignment  
- BERTScore indicates strong contextual understanding  
- Lexical metrics remain lower due to paraphrasing  

---

## Running the Project

### Install Dependencies
```bash
pip install bitsandbytes accelerate
