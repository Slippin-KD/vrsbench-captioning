#!/usr/bin/env python3
"""Generate baseline BLIP captions from VRSBench validation images stored in a ZIP."""

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def local_model_directory() -> Path:
    """Return the existing BLIP snapshot stored in this project's HF cache."""
    snapshots = (
        PROJECT_ROOT
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Salesforce--blip-image-captioning-base"
        / "snapshots"
    )
    candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            "BLIP is not present in the project cache. Download it before running this script."
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/validation_captions.jsonl")
    )
    parser.add_argument(
        "--images", type=Path, default=Path("data/downloads/Images_val.zip")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/baseline_predictions.jsonl")
    )
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as source:
        examples = [json.loads(line) for line in source][: args.limit]
    if not examples:
        raise ValueError("The validation manifest is empty.")

    model_directory = local_model_directory()
    processor = AutoProcessor.from_pretrained(model_directory, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_directory, local_files_only=True
    )
    model.eval()

    results = []
    with zipfile.ZipFile(args.images) as archive, torch.inference_mode():
        for index, example in enumerate(examples, start=1):
            image_path = f"Images_val/{example['image']}"
            with archive.open(image_path) as source:
                image = Image.open(io.BytesIO(source.read())).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            started = time.perf_counter()
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            seconds = time.perf_counter() - started
            prediction = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            result = {
                "image": example["image"],
                "reference": example["caption"],
                "prediction": prediction,
                "seconds": round(seconds, 3),
            }
            results.append(result)
            print(f"[{index}/{len(examples)}] {example['image']} | {seconds:.2f}s | {prediction}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for result in results:
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
    average = sum(result["seconds"] for result in results) / len(results)
    print(f"Saved {len(results)} predictions to {args.output}; mean generation time: {average:.2f}s/image")


if __name__ == "__main__":
    main()
