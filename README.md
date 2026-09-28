# VRSBench Image Captioning: Training-Time Optimization Study

This repository is the standalone submission for **Task 01** of the Inter-IIT
internal hackathon: train an image-captioning system on VRSBench and document
how training optimizations affect efficiency and caption quality.

Task 02 has its own repository and is intentionally not included here.

## Project goals

1. Establish a pretrained BLIP captioning baseline on VRSBench.
2. Fine-tune BLIP on VRSBench captions.
3. Compare practical training optimizations using measured runtime, memory,
   loss behavior, and validation captions.
4. Keep the experiment reproducible under limited local disk space by reading
   image archives directly instead of extracting them.

## Dataset

VRSBench contains 29,614 remote-sensing images with one detailed caption per
image. The supplied split contains 20,264 training pairs and 9,350 validation
pairs.

Large dataset files are not committed. See [data/README.md](data/README.md)
for the required files and placement.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source project_env.sh
```

The project cache is stored under `.cache/huggingface`.

## Prepare captions

After downloading and extracting only `Annotations_train.zip`:

```bash
python scripts/prepare_train_manifest.py
python scripts/prepare_validation_manifest.py
```

These create two JSONL manifests. Images remain in their ZIP archives.

## Baseline inference

Download `Salesforce/blip-image-captioning-base` once into the project cache,
then run:

```bash
python scripts/run_baseline.py --limit 5
```

Predictions and per-image inference times are written to
`reports/baseline_predictions.jsonl`.

## Training pilot

Create a deterministic sample from the training manifest, then run the CPU
pilot:

```bash
python scripts/train_pilot.py --max-samples 25
```

The pilot freezes BLIP's pretrained visual encoder and trains the language
generation component. It records loss and runtime in `reports/pilot_training.json`
and saves the trainable parameter state under `checkpoints/` for validation.

## Experiment reporting

The experiment plan and the currently observed baseline are in
[docs/experiment-plan.md](docs/experiment-plan.md). Every reported comparison
will include its configuration, elapsed time, resource observations, training
loss, and generated validation captions.

### BERT-BLEU₄ metric

Captioning quality is evaluated with **BERT-BLEU₄**, defined as:

```
P_n  =  (1/|R_n|) · Σ_{r ∈ R_n}  max_{c ∈ C_n}  cos(E(c), E(r))

BERT-BLEU₄  =  LP · exp( ¼ · Σ_{n=1}^{4}  log P_n )

LP  =  exp( −α · |L_C − L_R| / L_R ),   α = 0.5
```

where `E(x)` is the mean-pooled contextual BERT embedding of n-gram `x`
(`bert-base-uncased`). This mirrors the standard BLEU formula but replaces
exact n-gram matching with cosine similarity in BERT embedding space.

To reproduce the scores:

```bash
python scripts/compute_bert_bleu4.py
```

Results are saved to `reports/bert_bleu4_results.json`.

#### Current exploratory results (5-image pilot, 0–100 scale)

| Metric        | Baseline | LoRA Tuned |
|:--------------|--------:|----------:|
| BERT-BLEU₁    |  56.28  |   58.93   |
| BERT-BLEU₂    |  50.61  |   53.60   |
| BERT-BLEU₃    |  48.10  |   51.15   |
| **BERT-BLEU₄**| **47.18**| **50.17** |

## Repository layout

```text
data/        Dataset instructions and local manifests
docs/        Experiment plan and observations
reports/     Generated predictions and run reports
scripts/     Data preparation, baseline, and training scripts
```

## Attribution

- VRSBench: Xiang Li, Jian Ding, and Mohamed Elhoseiny, 2024.
- BLIP: Salesforce Research, 2022.
