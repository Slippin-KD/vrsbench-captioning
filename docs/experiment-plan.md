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
| **BERT-BLEU₁–₄** | Primary combined score | `LP · exp((1/n) · Σ log Pₙ)` where `Pₙ = mean max cosine similarity of BERT n-gram embeddings`; `LP = exp(-α·\|Lc-Lr\|/Lr)`, α=0.5 |
| **Peak VRAM (MB)** | Memory footprint efficiency | `torch.cuda.max_memory_allocated()` |
| **Throughput (samples/s)** | Wall-clock computational efficiency | Total processed dataset examples / total elapsed training time |

Evaluation is executed via:
```bash
python scripts/compute_bert_bleu4.py
```

### Current exploratory comparison: baseline versus LoRA

The following comparison uses the five fixed validation images in
`reports/baseline_predictions.jsonl` and `reports/lora_tuned_predictions.jsonl`.
It is an exploratory comparison for selecting the next full validation run;
it is not a final score over all 9,350 validation images.

BERT-BLEU scores are computed with `bert-base-uncased` using the formula:

```
P_n  =  (1/|R_n|) · Σ_{r ∈ R_n}  max_{c ∈ C_n}  cos(E(c), E(r))
BERT-BLEUₙ  =  LP · exp( (1/n) · Σ_{i=1}^{n}  log Pᵢ )
LP  =  exp( −α · |L_C − L_R| / L_R ),   α = 0.5
```

| Metric | BLIP baseline (B0) | LoRA tuned | Change |
| :--- | ---: | ---: | ---: |
| BLEU-1 | 6.58 | 10.96 | +4.38 |
| BLEU-2 | 2.83 | 5.68 | +2.85 |
| BLEU-3 | 1.05 | 2.10 | +1.05 |
| BLEU-4 | 0.63 | 1.09 | +0.46 |
| ROUGE-L | 14.43 | 22.95 | +8.52 |
| BERTScore F1 | 13.67 | 22.31 | +8.64 |
| BERT-BLEU₁ | 56.28 | 58.93 | +2.64 |
| BERT-BLEU₂ | 50.61 | 53.60 | +2.99 |
| BERT-BLEU₃ | 48.10 | 51.15 | +3.05 |
| **BERT-BLEU₄** | **47.18** | **50.17** | **+2.99** |
| Mean inference latency (s/image) | 2.51 | 0.45 | −2.06 |

Full results saved in `reports/bert_bleu4_results.json`.

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
