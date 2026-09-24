#!/usr/bin/env python3
"""Unified Training & Optimization Framework for VRSBench Image Captioning.

Supports all benchmark ablation dimensions:
1. Mixed Precision (FP16, BF16, FP32)
2. Gradient Accumulation & Effective Batch Size
3. Learning Rate Schedulers (Cosine with Warmup, Linear, Constant)
4. Data Pipeline: Multi-worker DataLoader, persistent_workers, pin_memory, caching
5. Encoder Strategies: Frozen vs Partial (unfreeze top N ViT blocks) vs Full
6. LoRA Adapters (PEFT): Low-Rank Adaptation of cross-attention and projection layers
7. Peak VRAM & Throughput telemetry for presentation benchmarking
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# Insert project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_dataloader

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
        raise FileNotFoundError("BLIP base model not found in local cache.")
    return candidates[-1]

def configure_encoder_freeze(model, strategy: str, unfreeze_layers: int = 2):
    """Configures freezing strategy: frozen, partial, or full."""
    if strategy == "frozen":
        print("[Encoder Strategy] Freezing entire Vision Encoder.")
        for p in model.vision_model.parameters():
            p.requires_grad = False
    elif strategy == "partial":
        print(f"[Encoder Strategy] Partial freeze: Freezing all except top {unfreeze_layers} ViT layers.")
        for p in model.vision_model.parameters():
            p.requires_grad = False
        # Unfreeze the last N layers of the Vision Transformer
        if hasattr(model.vision_model, "encoder") and hasattr(model.vision_model.encoder, "layers"):
            for layer in model.vision_model.encoder.layers[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        if hasattr(model.vision_model, "post_layernorm"):
            for p in model.vision_model.post_layernorm.parameters():
                p.requires_grad = True
    elif strategy == "full":
        print("[Encoder Strategy] Full fine-tuning: Vision and Text encoders are both trainable.")
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def apply_lora(model, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """Injects LoRA adapters into attention projections."""
    try:
        from peft import LoraConfig, get_peft_model
        print(f"[LoRA] Applying LoRA adapters: rank={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        # Target cross-attention and self-attention projections in BLIP text decoder
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        return model, True
    except ImportError:
        print("[LoRA Warning] 'peft' library is not installed. Falling back to standard fine-tuning.")
        return model, False

def main():
    parser = argparse.ArgumentParser(description="Advanced BLIP Fine-tuning on VRSBench.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/train_pilot_250.jsonl"))
    parser.add_argument("--images", type=Path, default=Path("data/downloads/Images_train.zip"))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    
    # 1. Batching & Gradient Accumulation
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    
    # 2. Mixed Precision
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    
    # 3. Learning Rate Schedule & Warmup
    parser.add_argument("--lr-scheduler", choices=["cosine", "linear", "constant"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    
    # 4. DataLoader Parallelism & Cache
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--persistent-workers", action="store_true", default=False)
    parser.add_argument("--cache-images", action="store_true", default=False)
    
    # 5. Encoder Strategy
    parser.add_argument("--encoder-strategy", choices=["frozen", "partial", "full"], default="frozen")
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    
    # 6. LoRA Adapters
    parser.add_argument("--use-lora", action="store_true", default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    
    # Output and naming
    parser.add_argument("--output", type=Path, default=Path("reports/advanced_training_report.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/advanced_trainable_state.pt"))
    args = parser.parse_args()

    effective_batch_size = args.batch_size * args.grad_accum_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device} | Per-device Batch: {args.batch_size} | Effective Batch: {effective_batch_size}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model_dir = local_model_directory()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, local_files_only=True)

    # Apply Encoder Strategy
    configure_encoder_freeze(model, args.encoder_strategy, args.unfreeze_layers)

    # Apply LoRA if requested
    lora_active = False
    if args.use_lora:
        model, lora_active = apply_lora(model, r=args.lora_r, lora_alpha=args.lora_alpha)

    model.to(device)

    # Create optimized DataLoader
    dataloader = create_dataloader(
        manifest_path=args.manifest,
        zip_path=args.images,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        cache_images=args.cache_images,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.2f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    # Schedule setup
    total_steps = (len(dataloader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif args.lr_scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = None

    # Mixed precision setup
    use_amp = args.precision in ["fp16", "bf16"]
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16" and device.type == "cuda"))

    model.train()
    losses = []
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    step_counter = 0
    for epoch in range(1, args.epochs + 1):
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if batch_idx % args.grad_accum_steps == 0 or batch_idx == len(dataloader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                step_counter += 1

            unscaled_loss = float(outputs.loss.detach())
            losses.append(unscaled_loss)

            if batch_idx % max(len(dataloader) // 5, 1) == 0 or batch_idx == len(dataloader):
                current_lr = scheduler.get_last_lr()[0] if scheduler else args.learning_rate
                print(f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {unscaled_loss:.4f} | LR: {current_lr:.2e}")

    total_time = time.perf_counter() - start_time
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 2) if device.type == "cuda" else 0.0

    # Save checkpoint
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if lora_active:
        model.save_pretrained(args.checkpoint.parent / "lora_adapter")
    else:
        trainable_state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(trainable_state, args.checkpoint)

    # Save detailed telemetry report
    report = {
        "experiment_name": args.output.stem,
        "device": str(device),
        "total_samples": len(dataloader.dataset),
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "precision": args.precision,
        "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "encoder_strategy": args.encoder_strategy,
        "unfreeze_layers": args.unfreeze_layers if args.encoder_strategy == "partial" else None,
        "use_lora": lora_active,
        "lora_r": args.lora_r if lora_active else None,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_percent": round(trainable_count / total_count * 100, 2),
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "mean_loss": round(sum(losses) / len(losses), 4),
        "total_time_seconds": round(total_time, 2),
        "throughput_samples_per_sec": round(len(dataloader.dataset) * args.epochs / total_time, 2),
        "peak_vram_mb": peak_vram_mb,
        "checkpoint": str(args.checkpoint),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("           TRAINING & OPTIMIZATION REPORT SUMMARY           ")
    print("=" * 60)
    for k, v in report.items():
        print(f"{k:<30}: {v}")
    print("=" * 60)
    print(f"Saved complete telemetry report to: {args.output}")

if __name__ == "__main__":
    main()
