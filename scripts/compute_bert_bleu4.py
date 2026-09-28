#!/usr/bin/env python3
"""
Compute BERT-BLEU₄ exactly as defined in the metric specification:

  P_n  =  (1/|R_n|) · Σ_{r ∈ R_n}  max_{c ∈ C_n}  cos(E(c), E(r))

  BERT-BLEU₄  =  LP · exp( (1/4) · Σ_{n=1}^{4}  log P_n )

  LP  =  exp( -α · |L_C - L_R| / L_R ),   α = 0.5
"""

import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ALPHA = 0.5
BERT_MODEL = "bert-base-uncased"
DEVICE = "cpu"


# ── helpers ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def length_penalty(lc: int, lr: int, alpha: float = ALPHA) -> float:
    """LP = exp(-α · |L_C - L_R| / L_R)"""
    if lr == 0:
        return 0.0
    return math.exp(-alpha * abs(lc - lr) / lr)


# ── BERT embedder ─────────────────────────────────────────────────────────────

def build_embedder(model_name: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev)
    model.eval()

    def embed(grams: list[str]):
        """Return L2-normalised mean-pooled BERT embeddings, shape (len(grams), H)."""
        enc = tokenizer(
            grams, padding=True, truncation=True,
            return_tensors="pt", return_special_tokens_mask=True,
        )
        special_mask = enc.pop("special_tokens_mask").to(dev).bool()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state          # (B, T, H)
        usable = enc["attention_mask"].bool() & ~special_mask   # (B, T)
        pooled = (hidden * usable.unsqueeze(-1)).sum(1) / \
                 usable.sum(1, keepdim=True).clamp_min(1)        # (B, H)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    return embed


# ── per-pair BERT-BLEU computation ───────────────────────────────────────────

def bert_bleu4(reference: str, prediction: str, embed_fn) -> dict[str, float]:
    """
    Returns bert_bleu_1 .. bert_bleu_4 and bert_bleu4_final (the BERT-BLEU₄).

    BERT-BLEU₄ = LP · exp( (1/4) · Σ_{n=1..4} log P_n )

    Each individual bert_bleu_n is: LP · exp( (1/n) · Σ_{i=1..n} log P_i )
    i.e. the "running" BERT-BLEU score up to order n (same structure as classic BLEU).
    """
    ref_tok = tokenize(reference)
    cand_tok = tokenize(prediction)

    if not ref_tok or not cand_tok:
        return {f"bert_bleu_{n}": 0.0 for n in range(1, 5)}

    lp = length_penalty(len(cand_tok), len(ref_tok))

    log_pn_list: list[float] = []
    scores: dict[str, float] = {}

    for n in range(1, 5):
        ref_grams  = ngrams(ref_tok,  n)   # R_n
        cand_grams = ngrams(cand_tok, n)   # C_n

        if not ref_grams or not cand_grams:
            # No n-grams of this order → P_n = 0 (clamp to tiny for log stability)
            log_pn_list.append(math.log(1e-12))
        else:
            # Embed all reference and candidate n-grams
            ref_vecs  = embed_fn(ref_grams)   # (|R_n|, H)
            cand_vecs = embed_fn(cand_grams)  # (|C_n|, H)

            # cosine similarity matrix:  sim[r, c] = cos(E(r), E(c))
            sim = ref_vecs @ cand_vecs.T      # (|R_n|, |C_n|)

            # P_n = (1/|R_n|) · Σ_r  max_c  cos(E(r), E(c))
            p_n = float(sim.max(dim=1).values.mean().item())
            log_pn_list.append(math.log(max(p_n, 1e-12)))

        # Running BERT-BLEU_n = LP · exp( (1/n) · Σ_{i=1}^{n} log P_i )
        scores[f"bert_bleu_{n}"] = round(
            lp * math.exp(sum(log_pn_list) / n) * 100, 4
        )

    # The headline BERT-BLEU₄ is bert_bleu_4 (uses all four P_n with equal weight 1/4)
    scores["bert_bleu4_final"] = scores["bert_bleu_4"]
    return scores


# ── corpus-level averaging ────────────────────────────────────────────────────

def evaluate(records: list[dict], embed_fn) -> dict[str, float]:
    totals: dict[str, float] = {f"bert_bleu_{n}": 0.0 for n in range(1, 5)}
    totals["bert_bleu4_final"] = 0.0

    for rec in records:
        pair_scores = bert_bleu4(rec["reference"], rec["prediction"], embed_fn)
        for k, v in pair_scores.items():
            totals[k] += v

    n = len(records)
    return {k: round(v / n, 4) for k, v in totals.items()}


# ── main ──────────────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    root = PROJECT_ROOT
    candidate_path = root / "reports" / "lora_tuned_predictions.jsonl"
    baseline_path  = root / "reports" / "baseline_predictions.jsonl"
    output_path    = root / "reports" / "bert_bleu4_results.json"

    print(f"Loading BERT model: {BERT_MODEL}  device: {DEVICE}")
    embed = build_embedder(BERT_MODEL, DEVICE)

    candidate = load(candidate_path)
    baseline  = load(baseline_path)

    print(f"\nEvaluating candidate ({len(candidate)} samples)…")
    cand_scores = evaluate(candidate, embed)

    print(f"Evaluating baseline  ({len(baseline)} samples)…")
    base_scores = evaluate(baseline, embed)

    # ── print table ──────────────────────────────────────────────────────────
    header = f"\n{'Metric':<20} {'Baseline':>12} {'LoRA Tuned':>12}"
    print(header)
    print("-" * len(header.rstrip()))
    for key in ["bert_bleu_1", "bert_bleu_2", "bert_bleu_3", "bert_bleu_4"]:
        print(f"{key.upper():<20} {base_scores[key]:>12.4f} {cand_scores[key]:>12.4f}")
    print(f"{'BERT-BLEU4 (final)':<20} {base_scores['bert_bleu4_final']:>12.4f}"
          f" {cand_scores['bert_bleu4_final']:>12.4f}")

    # ── save JSON ─────────────────────────────────────────────────────────────
    result = {
        "formula": {
            "P_n": "mean over r in R_n of max_{c in C_n} cos(E(r), E(c))",
            "BERT_BLEU4": "LP * exp( (1/4) * sum_{n=1..4} log(P_n) )",
            "LP": "exp( -alpha * |L_C - L_R| / L_R )",
            "alpha": ALPHA,
            "bert_model": BERT_MODEL,
        },
        "candidate_file": str(candidate_path.name),
        "candidate_metrics": cand_scores,
        "baseline_file": str(baseline_path.name),
        "baseline_metrics": base_scores,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
