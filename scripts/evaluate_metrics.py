#!/usr/bin/env python3
"""Compute evaluation metrics (BLEU-1..4, ROUGE-L, BERTScore) comparing runs."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import evaluate_predictions

def load_predictions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Evaluate captioning predictions with BLEU-4 and BERTScore.")
    parser.add_argument("--candidate", type=Path, default=Path("reports/optimized_predictions.jsonl"),
                        help="Path to candidate predictions JSONL.")
    parser.add_argument("--baseline", type=Path, default=Path("reports/baseline_predictions.jsonl"),
                        help="Optional path to baseline predictions JSONL for direct comparison.")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore calculation if offline.")
    parser.add_argument("--output", type=Path, default=Path("reports/metrics_summary.json"),
                        help="Path to save JSON metrics summary.")
    args = parser.parse_args()

    cand_records = load_predictions(args.candidate)
    print(f"Loaded {len(cand_records)} candidate predictions from {args.candidate}")
    cand_metrics = evaluate_predictions(cand_records, use_bertscore=not args.no_bertscore)
    cand_time = sum(r.get("seconds", 0) for r in cand_records) / max(len(cand_records), 1)

    summary = {
        "candidate_file": str(args.candidate),
        "candidate_metrics": cand_metrics,
        "candidate_mean_seconds": round(cand_time, 3)
    }

    if args.baseline.exists():
        base_records = load_predictions(args.baseline)
        print(f"Loaded {len(base_records)} baseline predictions from {args.baseline}")
        base_metrics = evaluate_predictions(base_records, use_bertscore=not args.no_bertscore)
        base_time = sum(r.get("seconds", 0) for r in base_records) / max(len(base_records), 1)
        summary["baseline_file"] = str(args.baseline)
        summary["baseline_metrics"] = base_metrics
        summary["baseline_mean_seconds"] = round(base_time, 3)

        print("\n" + "=" * 65)
        print("               VRSBENCH CAPTION EVALUATION SUMMARY               ")
        print("=" * 65)
        print(f"{'Metric':<20} | {'Baseline (B0)':<18} | {'Optimized':<18}")
        print("-" * 65)
        for key in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "bert_score_f1"]:
            b_val = str(base_metrics.get(key, "N/A"))
            c_val = str(cand_metrics.get(key, "N/A"))
            print(f"{key.upper():<20} | {b_val:<18} | {c_val:<18}")
        print("-" * 65)
        print(f"{'Mean Latency (s)':<20} | {base_time:<18.2f} | {cand_time:<18.2f}")
        print("=" * 65)
    else:
        print("\nMetrics:", json.dumps(cand_metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics summary to {args.output}")

if __name__ == "__main__":
    main()
