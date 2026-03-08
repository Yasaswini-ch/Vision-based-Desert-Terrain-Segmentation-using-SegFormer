"""Evaluation script for desert semantic segmentation.

This module evaluates the best checkpoint on the validation set and reports
mean IoU, per-class IoU, and confusion matrix.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import DesertDataset, get_val_test_transforms
from loss import CombinedLoss
from model import forward_logits, load_model
from train import compute_mean_iou, get_device, set_seed, update_confusion_matrix

print("Loading model from:", config.BEST_MODEL_PATH)
print("Model backend:", config.MODEL_BACKEND)
print("Val images dir:", config.VAL_IMAGES_DIR)


def _status_label(iou: float) -> str:
    if iou >= 0.55:
        return "\x1b[92mGOOD\x1b[0m"
    if iou >= 0.35:
        return "\x1b[93mOK\x1b[0m"
    if iou >= 0.20:
        return "\x1b[33mWEAK\x1b[0m"
    return "\x1b[91mFAILING\x1b[0m"



def tta_predict(
    model: torch.nn.Module,
    images: torch.Tensor,
    backend: str,
    target_size: tuple[int, int],
    scales: list[float],
) -> torch.Tensor:
    """Multi-scale + horizontal-flip TTA. Returns argmax predictions [B, H, W].

    Runs 2*len(scales) forward passes (original + H-flip at each scale),
    averages softmax probabilities, then returns the argmax class map.
    """
    accumulated = torch.zeros(
        images.shape[0], config.NUM_CLASSES, *target_size,
        device=images.device, dtype=torch.float32,
    )
    n = 0
    for scale in scales:
        for flip in (False, True):
            x = torch.flip(images, dims=[-1]) if flip else images
            if scale != 1.0:
                h = int(target_size[0] * scale)
                w = int(target_size[1] * scale)
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            logits = forward_logits(model=model, images=x, backend=backend)
            logits = F.interpolate(logits.float(), size=target_size, mode="bilinear", align_corners=False)
            if flip:
                logits = torch.flip(logits, dims=[-1])
            accumulated += torch.softmax(logits, dim=1)
            n += 1
    return (accumulated / n).argmax(dim=1)


def evaluate() -> dict[str, object]:
    """Evaluate best model on validation data.

    Returns:
        Dictionary of evaluation metrics and confusion matrix values.
    """
    config.ensure_output_dirs()
    config.validate_dataset_dirs(require_masks_for_test=False)

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {config.BEST_MODEL_PATH}. Run train.py first."
        )

    device = get_device()
    image_size = (
        config.CPU_QUICK_IMAGE_SIZE
        if device.type == "cpu" and config.CPU_QUICK_MODE
        else config.IMAGE_SIZE
    )
    num_workers = (
        config.CPU_QUICK_NUM_WORKERS
        if device.type == "cpu" and config.CPU_QUICK_MODE
        else config.NUM_WORKERS
    )
    dataset = DesertDataset(
        images_dir=config.VAL_IMAGES_DIR,
        masks_dir=config.VAL_MASKS_DIR,
        mode="val",
        transform=get_val_test_transforms(image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY and device.type == "cuda",
    )

    model = load_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        backend=config.MODEL_BACKEND,
        model_name=config.HF_MODEL_NAME,
    )

    # --- Checkpoint weight verification ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[verify] Model parameter count: {total_params:,}  (~85M expected for SegFormer B2)")

    first_param_before = next(iter(model.parameters())).detach().float().sum().item()
    print(f"[verify] First-layer weight sum BEFORE loading: {first_param_before:.6f}")

    print(f"[verify] Checkpoint path: {config.BEST_MODEL_PATH}")
    print(f"[verify] Checkpoint file size: {config.BEST_MODEL_PATH.stat().st_size / 1e6:.1f} MB")

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=True)

    print(f"[verify] Checkpoint keys: {list(checkpoint.keys())}")
    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    checkpoint_iou = checkpoint.get("val_mean_iou", "unknown")
    print(f"[verify] Checkpoint epoch: {checkpoint_epoch}, best_val_mIoU: {checkpoint_iou}")

    checkpoint_backend = checkpoint.get("model_backend")
    print(f"[verify] Checkpoint model_backend: {checkpoint_backend}")
    if checkpoint_backend is not None and checkpoint_backend != config.MODEL_BACKEND:
        raise ValueError(
            f"Checkpoint backend '{checkpoint_backend}' does not match current "
            f"MODEL_BACKEND='{config.MODEL_BACKEND}'. Retrain or switch config."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"'model_state_dict' not found in checkpoint. Available keys: {list(checkpoint.keys())}"
        )

    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing:
        print(f"[verify] WARNING - missing keys in checkpoint: {missing[:5]}")
    if unexpected:
        print(f"[verify] WARNING - unexpected keys in checkpoint: {unexpected[:5]}")

    first_param_after = next(iter(model.parameters())).detach().float().sum().item()
    print(f"[verify] First-layer weight sum AFTER loading:  {first_param_after:.6f}")

    if abs(first_param_before - first_param_after) < 1e-6:
        print("[verify] ERROR: weights did NOT change after load_state_dict — checkpoint may be corrupt or wrong file!")
    else:
        print("[verify] Weights changed after loading — checkpoint loaded correctly.")

    model.to(device)
    model.eval()
    print("[verify] model.eval() called — BatchNorm/Dropout in inference mode.")

    class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = CombinedLoss(class_weights=class_weights)

    confusion = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=torch.int64)
    total_loss = 0.0

    with torch.no_grad():
        progress = tqdm(loader, desc="Evaluating", leave=False)
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = forward_logits(model=model, images=images, backend=config.MODEL_BACKEND)
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = criterion(logits, masks)
            total_loss += loss.item()

            if config.USE_TTA:
                preds = tta_predict(
                    model=model,
                    images=images,
                    backend=config.MODEL_BACKEND,
                    target_size=tuple(masks.shape[-2:]),
                    scales=config.TTA_SCALES,
                )
            else:
                preds = logits.argmax(dim=1)
            confusion = update_confusion_matrix(
                confusion=confusion,
                preds=preds.cpu(),
                targets=masks.cpu(),
                num_classes=config.NUM_CLASSES,
                ignore_index=config.IGNORE_INDEX,
            )

    mean_iou, per_class_iou = compute_mean_iou(confusion)
    mean_iou_excluding_dom = float(np.mean([per_class_iou[i] for i in range(config.NUM_CLASSES) if i not in {8, 9}]))
    avg_loss = total_loss / max(len(loader), 1)
    pixel_counts = confusion.sum(dim=1).float()
    total_pixels = float(pixel_counts.sum().item())
    pixel_percents = [
        (float(pixel_counts[i].item()) / total_pixels * 100.0) if total_pixels > 0 else 0.0
        for i in range(config.NUM_CLASSES)
    ]

    class_iou_named = {
        config.CLASS_NAMES[i]: per_class_iou[i] for i in range(config.NUM_CLASSES)
    }
    per_class_table = [
        {
            "class_name": config.CLASS_NAMES[i],
            "iou": per_class_iou[i],
            "pixel_percent": pixel_percents[i],
            "status": "GOOD" if per_class_iou[i] >= 0.55 else "OK" if per_class_iou[i] >= 0.35 else "WEAK" if per_class_iou[i] >= 0.20 else "FAILING",
        }
        for i in range(config.NUM_CLASSES)
    ]

    results: dict[str, object] = {
        "val_loss": avg_loss,
        "mean_iou": mean_iou,
        "mean_iou_excluding_sky_landscape": mean_iou_excluding_dom,
        "per_class_iou": class_iou_named,
        "per_class_table": per_class_table,
        "confusion_matrix": confusion.tolist(),
    }

    with config.EVAL_RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    per_class_path = config.LOG_DIR / "per_class_iou.json"
    with per_class_path.open("w", encoding="utf-8") as f:
        json.dump(per_class_table, f, indent=2)

    print(f"Validation loss: {avg_loss:.4f}")
    print("Class Name       | IoU    | Pixel% | Status")
    print("---------------------------------------------")
    for i in range(config.NUM_CLASSES):
        class_name = config.CLASS_NAMES[i]
        iou = per_class_iou[i]
        px = pixel_percents[i]
        status = _status_label(iou)
        print(f"{class_name:<16} | {iou:0.4f} | {px:6.2f} | {status}")
    print(f"Mean IoU (all classes): {mean_iou:.4f}")
    print(f"Mean IoU (excluding Sky and Landscape): {mean_iou_excluding_dom:.4f}")
    print(f"Saved evaluation results to: {config.EVAL_RESULTS_PATH}")
    print(f"Saved per-class IoU table to: {per_class_path}")

    return results



def main() -> None:
    """Entry point for validation evaluation."""
    set_seed(config.SEED)
    evaluate()


if __name__ == "__main__":
    main()
