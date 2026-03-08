"""Fine-tuning script for desert semantic segmentation.

This module loads an existing checkpoint and continues training for a specified
number of epochs with updated loss weights and augmentations.
"""

from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import DesertDataset, get_train_transforms, get_val_test_transforms
from loss import CombinedLoss
from model import forward_logits, load_model


@dataclass
class EpochMetrics:
    """Container for epoch-level metrics."""

    train_loss: float
    val_loss: float
    val_mean_iou: float


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Resolve the active training device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(batch_size: int, device: torch.device) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        batch_size: Batch size to use.
        device: Active torch device.

    Returns:
        Tuple of train and validation dataloaders.
    """
    pin_memory = config.PIN_MEMORY and device.type == "cuda"
    image_size = config.IMAGE_SIZE
    num_workers = config.NUM_WORKERS
    if device.type == "cpu" and config.CPU_QUICK_MODE:
        image_size = config.CPU_QUICK_IMAGE_SIZE
        num_workers = config.CPU_QUICK_NUM_WORKERS

    train_dataset = DesertDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        masks_dir=config.TRAIN_MASKS_DIR,
        mode="train",
        transform=get_train_transforms(image_size),
    )
    val_dataset = DesertDataset(
        images_dir=config.VAL_IMAGES_DIR,
        masks_dir=config.VAL_MASKS_DIR,
        mode="val",
        transform=get_val_test_transforms(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def update_confusion_matrix(
    confusion: Tensor,
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: int,
) -> Tensor:
    """Accumulate confusion matrix counts for one batch."""
    valid = targets != ignore_index
    if not torch.any(valid):
        return confusion

    target_flat = targets[valid].view(-1)
    pred_flat = preds[valid].view(-1)

    indices = target_flat * num_classes + pred_flat
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion += counts.view(num_classes, num_classes)
    return confusion


def compute_mean_iou(confusion: Tensor) -> tuple[float, list[float]]:
    """Compute mean IoU and per-class IoU values from confusion matrix."""
    tp = torch.diag(confusion).float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp

    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    mean_iou = iou.mean().item()
    return mean_iou, iou.tolist()


def run_validation(
    model: torch.nn.Module,
    criterion: CombinedLoss,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Run one validation epoch and return loss and mean IoU."""
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=torch.int64)

    autocast_enabled = use_amp and device.type == "cuda"

    with torch.no_grad():
        progress = tqdm(val_loader, desc="Validation", leave=False)
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            autocast_context = (
                torch.amp.autocast(device_type="cuda", enabled=autocast_enabled)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = forward_logits(model=model, images=images, backend=config.MODEL_BACKEND)
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = criterion(logits, masks)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            confusion = update_confusion_matrix(
                confusion=confusion,
                preds=preds.cpu(),
                targets=masks.cpu(),
                num_classes=config.NUM_CLASSES,
                ignore_index=config.IGNORE_INDEX,
            )

    val_loss = total_loss / max(len(val_loader), 1)
    val_iou, per_class_iou = compute_mean_iou(confusion)
    return val_loss, val_iou


def fine_tune_model(checkpoint_path: str, fine_tune_epochs: int, batch_size: int) -> float:
    """Run fine-tuning from existing checkpoint.

    Args:
        checkpoint_path: Path to existing checkpoint file.
        fine_tune_epochs: Number of fine-tuning epochs to run.
        batch_size: Effective batch size.

    Returns:
        Best validation mean IoU during fine-tuning.
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Model backend: {config.MODEL_BACKEND}")
    print(f"Fine-tuning epochs: {fine_tune_epochs}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    config.ensure_output_dirs()
    config.validate_dataset_dirs(require_masks_for_test=False)

    train_loader, val_loader = create_dataloaders(batch_size=batch_size, device=device)

    # Load model
    model = load_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,  # Don't use pretrained weights when loading checkpoint
        backend=config.MODEL_BACKEND,
        model_name=config.HF_MODEL_NAME,
    ).to(device)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} "
          f"with val mIoU {checkpoint.get('val_mean_iou', 'unknown'):.4f}")

    # Setup loss with new class weights
    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = CombinedLoss(class_weights=class_weights)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Load optimizer state if available
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded optimizer state from checkpoint")

    use_amp = config.USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    scheduler_t_max = min(config.SCHEDULER_T_MAX, fine_tune_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(scheduler_t_max, 1))
    
    # Load scheduler state if available
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("Loaded scheduler state from checkpoint")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mean_iou": [],
    }

    best_val_iou = checkpoint.get("val_mean_iou", -1.0)  # Start from checkpoint's best
    best_epoch = 0
    no_improve_count = 0

    print(f"Starting fine-tuning from best val mIoU: {best_val_iou:.4f}")

    for epoch in range(1, fine_tune_epochs + 1):
        model.train()
        running_train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}/{fine_tune_epochs}", leave=False)
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.amp.autocast(device_type="cuda", enabled=use_amp)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = forward_logits(model=model, images=images, backend=config.MODEL_BACKEND)
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_train_loss / max(len(train_loader), 1)
        val_loss, val_iou = run_validation(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mean_iou"].append(val_iou)

        metrics = EpochMetrics(train_loss=train_loss, val_loss=val_loss, val_mean_iou=val_iou)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            no_improve_count = 0
            checkpoint = {
                "epoch": checkpoint.get("epoch", 0) + epoch,  # Continue from original epoch
                "val_mean_iou": best_val_iou,
                "model_backend": config.MODEL_BACKEND,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint, config.BEST_MODEL_PATH)
            print(f"Saved new best checkpoint to: {config.BEST_MODEL_PATH}")
        else:
            no_improve_count += 1

        print(
            f"Fine-tune Epoch {epoch}/{fine_tune_epochs} | "
            f"Train Loss: {metrics.train_loss:.2f} | "
            f"Val Loss: {metrics.val_loss:.2f} | "
            f"Val mIoU: {metrics.val_mean_iou:.4f} | "
            f"Best: {best_val_iou:.4f} | "
            f"No improve: {no_improve_count}"
        )

        with config.HISTORY_PATH.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Fine-tuning complete. Best validation mIoU: {best_val_iou:.4f}")
    return best_val_iou


def main() -> None:
    """Entry point for model fine-tuning."""
    set_seed(config.SEED)

    try:
        fine_tune_model(
            checkpoint_path=str(config.BEST_MODEL_PATH),
            fine_tune_epochs=20,
            batch_size=config.BATCH_SIZE,
        )
    except RuntimeError as error:
        if "out of memory" in str(error).lower() and config.BATCH_SIZE != config.FALLBACK_BATCH_SIZE:
            print(
                "Encountered GPU memory issue. "
                f"Retrying with batch size={config.FALLBACK_BATCH_SIZE}."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fine_tune_model(
                checkpoint_path=str(config.BEST_MODEL_PATH),
                fine_tune_epochs=20,
                batch_size=config.FALLBACK_BATCH_SIZE,
            )
        else:
            raise


if __name__ == "__main__":
    main()
