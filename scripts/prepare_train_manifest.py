#!/usr/bin/env python3
"""Build a compact image-caption manifest from VRSBench training annotations."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/Annotations_train"),
        help="Directory containing the extracted per-image VRSBench JSON annotations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/train_captions.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    records = []
    for path in sorted(args.annotations.glob("*.json")):
        with path.open(encoding="utf-8") as source:
            annotation = json.load(source)
        image_name = annotation.get("image")
        caption = annotation.get("caption")
        if not isinstance(image_name, str) or not isinstance(caption, str):
            raise ValueError(f"Missing image or caption in {path}")
        records.append({"image": image_name, "caption": caption})

    if not records:
        raise FileNotFoundError(f"No annotation JSON files found in {args.annotations}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records):,} image-caption pairs to {args.output}")


if __name__ == "__main__":
    main()
