"""High-performance Dataset and DataLoader for VRSBench images stored in ZIP archives.
Supports:
- Worker-safe lazy ZipFile handles (avoids fork-safety issues across num_workers)
- Optional in-memory image caching
- Pinned memory & persistent workers
- Batched collation with Hugging Face AutoProcessor
"""

import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class VRSBenchZipDataset(Dataset):
    """PyTorch Dataset reading directly from a ZIP archive with worker safety and caching."""
    def __init__(
        self,
        manifest_path: Path,
        zip_path: Path,
        image_subfolder: str = "Images_train",
        max_samples: Optional[int] = None,
        cache_images: bool = False,
    ):
        self.zip_path = Path(zip_path)
        self.image_subfolder = image_subfolder
        self.cache_images = cache_images
        self._cache: Dict[str, Image.Image] = {}
        self._archive: Optional[zipfile.ZipFile] = None

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.examples = [json.loads(line) for line in f if line.strip()]

        if max_samples and max_samples > 0:
            self.examples = self.examples[:max_samples]

    def _get_archive(self) -> zipfile.ZipFile:
        """Lazily initialize ZipFile per worker process to ensure multiprocessing safety."""
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.zip_path, "r")
        return self._archive

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = self.examples[idx]
        image_name = item["image"]
        caption = item["caption"]

        if self.cache_images and image_name in self._cache:
            image = self._cache[image_name]
        else:
            archive = self._get_archive()
            entry_path = f"{self.image_subfolder}/{image_name}"
            with archive.open(entry_path) as source:
                image = Image.open(io.BytesIO(source.read())).convert("RGB")
            if self.cache_images:
                self._cache[image_name] = image

        return {
            "image": image,
            "image_name": image_name,
            "caption": caption,
        }

    def __del__(self):
        if self._archive is not None:
            try:
                self._archive.close()
            except Exception:
                pass

class VRSBenchCollateFn:
    """Collate function that batches PIL images and text prompts through AutoProcessor."""
    def __init__(self, processor, max_length: int = 96):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        images = [b["image"] for b in batch]
        captions = [b["caption"] for b in batch]

        inputs = self.processor(
            images=images,
            text=captions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs.input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

def create_dataloader(
    manifest_path: Path,
    zip_path: Path,
    processor,
    image_subfolder: str = "Images_train",
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    cache_images: bool = False,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    max_length: int = 96,
) -> DataLoader:
    """Builds an optimized DataLoader with worker parallelism and pinned memory."""
    dataset = VRSBenchZipDataset(
        manifest_path=manifest_path,
        zip_path=zip_path,
        image_subfolder=image_subfolder,
        max_samples=max_samples,
        cache_images=cache_images,
    )

    collate_fn = VRSBenchCollateFn(processor=processor, max_length=max_length)

    # Note: persistent_workers requires num_workers > 0
    use_persistent = persistent_workers if num_workers > 0 else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
        drop_last=False,
    )
