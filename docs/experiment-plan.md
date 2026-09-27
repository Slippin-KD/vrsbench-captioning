# VRSBench Training-Time Optimization & Experiment Plan

## Overview
Task 1 evaluation is centered on **training-time optimization, systematic ablations, and quantitative analysis**. This document tracks experimental configurations, telemetry (VRAM, step latency, throughput), and quantitative captioning metrics (BLEU-1 through BLEU-4, ROUGE-L, and BERTScore).

---

## 1. Quantitative Evaluation Metrics

| Metric | Target Dimension | Computation Method |
| :--- | :--- | :--- |
| **BLEU-1 to BLEU-4** | N-gram lexical precision (unigram to 4-gram) | Modified n-gram precision with brevity penalty & Chen-Cherry smoothing |
| **ROUGE-L** | Structural sequence overlap | Longest Common Subsequence (LCS) F1-score |
| **BERTScore F1** | Deep contextual / semantic similarity | Pairwise cosine similarity over contextual token embeddings |
| **BERT-BLEU4** | Primary combined score | Geometric mean of BERTScore F1 and BLEU-4 after both are scaled to 0-100 |
| **Peak VRAM (MB)** | Memory footprint efficiency | `torch.cuda.max_memory_allocated()` |
| **Throughput (samples/s)** | Wall-clock computational efficiency | Total processed dataset examples / total elapsed training time |

Evaluation is executed via:
```bash
python scripts/evaluate_metrics.py --candidate reports/optimized_predictions.jsonl --baseline reports/baseline_predictions.jsonl
```

### Current exploratory comparison: baseline versus LoRA

The following comparison uses the five fixed validation images in
`reports/baseline_predictions.jsonl` and `reports/lora_tuned_predictions.jsonl`.
It is an exploratory comparison for selecting the next full validation run;
it is not a final score over all 9,350 validation images.

| Metric | BLIP baseline (B0) | LoRA tuned | Change |
| :--- | ---: | ---: | ---: |
| BLEU-4 | 0.63 | 1.09 | +0.46 |
| BERTScore precision | 24.69 | 36.01 | +11.32 |
| BERTScore recall | 3.05 | 9.17 | +6.12 |
| BERTScore F1 | 13.67 | 22.31 | +8.64 |
| **BERT-BLEU4** | **2.93** | **4.93** | **+2.00** |
| Mean inference latency (seconds/image) | 2.51 | 0.45 | -2.06 |

`BERT-BLEU4 = sqrt(BERTScore F1 x BLEU-4)`, with both input metrics on a
0-100 scale. This project-defined composite balances semantic similarity with
four-gram overlap. The BERTScore values are retained from the original
evaluation run, and `scripts/evaluate_metrics.py` remains the reproducible
path for future BERTScore calculations.

---

## 2. Systematic Optimization Matrix

| Exp ID | Experiment Category | Key Configuration | Evaluated Hypotheses & Trade-offs |
| :--- | :--- | :--- | :--- |
| **B0** | **Pretrained Baseline** | Pretrained `Salesforce/blip-image-captioning-base` | Reference caption quality floor and zero-shot remote-sensing accuracy. |
| **OPT-1** | **Data I/O Pipeline** | `num_workers=2/4`, `pin_memory=True`, lazy zip handles | Resolves sequential CPU I/O bottleneck; eliminates GPU idle time. |
| **OPT-2** | **Precision Scaling** | Mixed Precision (FP16 / BF16 vs FP32) | Halves activation memory footprint; leverages Tensor Cores for ~2x speedup. |
| **OPT-3** | **Gradient Accumulation** | Effective Batch Size = `per_device_batch` $\times$ `grad_accum_steps` | Simulates large batch stability without triggering CUDA Out-Of-Memory (OOM). |
| **OPT-4** | **Convergence Scheduling** | Cosine Annealing with 10% linear warmup | Prevents catastrophic forgetting of pretrained weights during early steps. |
| **ABL-1** | **Encoder: Frozen** | Freeze entire ViT encoder (`model.vision_model`) | Minimizes backward pass compute; trains ~137M decoder parameters. |
| **ABL-2** | **Encoder: Partial** | Unfreeze top-2 ViT transformer blocks + post-LN | Allows higher-level aerial feature adaptation while keeping lower ViT features stable. |
| **ABL-3** | **Encoder: Full** | Full model fine-tuning (~223M parameters) | Evaluates if end-to-end gradient flow improves fine-grained domain adaptation. |
| **PEFT-1** | **LoRA Adapters** | LoRA on QKV attention projections ($r=16, \alpha=32$) | Cuts trainable parameter count by >95% (~2M params) while preserving performance. |

---

## 3. Benchmark Execution Commands

### Data Pipeline & Baseline Comparison
```bash
# 1. Baseline Evaluation
python scripts/run_baseline.py --limit 10

# 2. Advanced Training with DataLoader + AMP + Cosine Warmup
python scripts/train_advanced.py \
  --max-samples 100 \
  --epochs 1 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --precision fp16 \
  --lr-scheduler cosine \
  --warmup-ratio 0.1 \
  --num-workers 2 \
  --pin-memory \
  --output reports/exp_amp_cosine.json \
  --checkpoint checkpoints/exp_amp_cosine.pt

# 3. Partial Encoder Fine-tuning Ablation
python scripts/train_advanced.py \
  --max-samples 100 \
  --encoder-strategy partial \
  --unfreeze-layers 2 \
  --output reports/exp_partial_vit.json \
  --checkpoint checkpoints/exp_partial_vit.pt

# 4. LoRA Adapter Training (PEFT)
python scripts/train_advanced.py \
  --max-samples 100 \
  --use-lora \
  --lora-r 16 \
  --output reports/exp_lora.json \
  --checkpoint checkpoints/exp_lora.pt

# 5. Validation Generation
python scripts/run_validation.py \
  --checkpoint checkpoints/exp_amp_cosine.pt \
  --output reports/exp_amp_cosine_preds.jsonl

# 6. Quantitative Scoring (BERT-BLEU4, BLEU-1..4, ROUGE-L, BERTScore)
python scripts/evaluate_metrics.py \
  --candidate reports/exp_amp_cosine_preds.jsonl \
  --baseline reports/baseline_predictions.jsonl
```
