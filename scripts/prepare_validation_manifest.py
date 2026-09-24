#!/usr/bin/env python3
"""Convert VRSBench's caption evaluation JSON to the training manifest format."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/downloads/VRSBench_EVAL_Cap.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/validation_captions.jsonl"),
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as source:
        examples = json.load(source)

    records = []
    for example in examples:
        image_name = example.get("image_id")
        caption = example.get("ground_truth")
        if not isinstance(image_name, str) or not isinstance(caption, str):
            raise ValueError(f"Invalid caption record: {example}")
        records.append({"image": image_name, "caption": caption})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records):,} validation image-caption pairs to {args.output}")


if __name__ == "__main__":
    main()
