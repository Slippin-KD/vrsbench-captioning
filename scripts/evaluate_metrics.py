#!/usr/bin/env python3
"""Evaluate VRSBench caption runs with the supplied BERT-BLEU metric."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import evaluate_predictions


def load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BERT-BLEU1..4 for VRSBench captions.")
    parser.add_argument("--candidate", type=Path, default=Path("reports/lora_tuned_predictions.jsonl"))
    parser.add_argument("--baseline", type=Path, default=Path("reports/baseline_predictions.jsonl"))
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/metrics_summary.json"))
    args = parser.parse_args()

    candidate = load_predictions(args.candidate)
    baseline = load_predictions(args.baseline)
    candidate_metrics = evaluate_predictions(candidate, args.bert_model, args.device, args.alpha)
    baseline_metrics = evaluate_predictions(baseline, args.bert_model, args.device, args.alpha)
    summary = {
        "evaluation_examples": len(candidate),
        "candidate_file": str(args.candidate),
        "candidate_metrics": candidate_metrics,
        "candidate_mean_seconds": round(sum(row.get("seconds", 0) for row in candidate) / len(candidate), 3),
        "baseline_file": str(args.baseline),
        "baseline_metrics": baseline_metrics,
        "baseline_mean_seconds": round(sum(row.get("seconds", 0) for row in baseline) / len(baseline), 3),
    }
    print("\nVRSBench BERT-BLEU comparison (0-100 scale)")
    print(f"{'Metric':<18} {'Baseline':>12} {'LoRA tuned':>12}")
    for key in ["bert_bleu_1", "bert_bleu_2", "bert_bleu_3", "bert_bleu_4", "bleu_4", "rouge_l"]:
        print(f"{key.upper():<18} {baseline_metrics[key]:>12.2f} {candidate_metrics[key]:>12.2f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
