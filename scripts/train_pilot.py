#!/usr/bin/env python3
"""Run a disk-conscious BLIP fine-tuning pilot on VRSBench captions."""

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
        raise FileNotFoundError("BLIP is not present in the project cache.")
    return candidates[-1]


def load_examples(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as source:
        examples = [json.loads(line) for line in source]
    return examples[:limit] if limit else examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/train_pilot_250.jsonl")
    )
    parser.add_argument(
        "--images", type=Path, default=Path("data/downloads/Images_train.zip")
    )
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/pilot_training.json")
    )
    args = parser.parse_args()

    examples = load_examples(args.manifest, args.max_samples)
    if not examples:
        raise ValueError("No training examples were found.")

    model_directory = local_model_directory()
    processor = AutoProcessor.from_pretrained(model_directory, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_directory, local_files_only=True
    )

    # The image encoder is already pretrained. Freezing it is the first memory/time
    # optimization we will compare against a later, less-frozen configuration.
    for parameter in model.vision_model.parameters():
        parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    model.train()

    losses = []
    started = time.perf_counter()
    with zipfile.ZipFile(args.images) as archive:
        for epoch in range(1, args.epochs + 1):
            for index, example in enumerate(examples, start=1):
                image_path = f"Images_train/{example['image']}"
                with archive.open(image_path) as source:
                    image = Image.open(io.BytesIO(source.read())).convert("RGB")

                batch = processor(
                    images=image,
                    text=example["caption"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                )
                labels = batch.input_ids.clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100
                optimizer.zero_grad(set_to_none=True)
                output = model(**batch, labels=labels)
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                losses.append(float(output.loss.detach()))

                if index == 1 or index % 5 == 0 or index == len(examples):
                    print(
                        f"epoch {epoch}/{args.epochs}, step {index}/{len(examples)}, "
                        f"loss {losses[-1]:.4f}"
                    )

    elapsed = time.perf_counter() - started
    report = {
        "examples": len(examples),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "vision_encoder_frozen": True,
        "trainable_parameters": trainable_count,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / len(losses),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
