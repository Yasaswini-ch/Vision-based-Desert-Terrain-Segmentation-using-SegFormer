"""Inference script for desert semantic segmentation test images.

This module loads the best checkpoint, runs inference on test images, and saves
colorized segmentation outputs. If test masks are available, it also computes IoU.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import DesertDataset, collate_test_batch, get_val_test_transforms
from model import forward_logits, load_model
from train import compute_mean_iou, get_device, set_seed, update_confusion_matrix

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _predict_probs(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Run one forward pass and return per-class probabilities."""
    logits = forward_logits(model=model, images=images, backend=config.MODEL_BACKEND)
    logits = F.interpolate(
        logits,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return torch.softmax(logits, dim=1)


def _apply_brightness(images: torch.Tensor, factor: float) -> torch.Tensor:
    """Apply brightness scaling in RGB space, then renormalize."""
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    rgb = torch.clamp(images * std + mean, 0.0, 1.0)
    bright = torch.clamp(rgb * factor, 0.0, 1.0)
    return (bright - mean) / std


def predict_with_tta(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Predict class probabilities with optional 4-pass TTA averaging."""
    if not config.USE_TTA:
        return _predict_probs(model=model, images=images)

    probs_original = _predict_probs(model=model, images=images)

    images_hflip = torch.flip(images, dims=[3])
    probs_hflip = _predict_probs(model=model, images=images_hflip)
    probs_hflip = torch.flip(probs_hflip, dims=[3])

    images_bright = _apply_brightness(images=images, factor=1.2)
    probs_bright = _predict_probs(model=model, images=images_bright)

    images_hflip_bright = torch.flip(images_bright, dims=[3])
    probs_hflip_bright = _predict_probs(model=model, images=images_hflip_bright)
    probs_hflip_bright = torch.flip(probs_hflip_bright, dims=[3])

    return (probs_original + probs_hflip + probs_bright + probs_hflip_bright) / 4.0



def colorize_mask(mask: np.ndarray, color_map: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Convert class-index mask to RGB color visualization."""
    h, w = mask.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        colorized[mask == class_idx] = color
    return colorized



def save_colorized_prediction(pred_mask: np.ndarray, output_path: Path) -> None:
    """Save one colorized prediction image to disk."""
    colorized = colorize_mask(pred_mask, config.CLASS_COLORS)
    bgr = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(output_path), bgr)
    if not success:
        raise RuntimeError(f"Failed to save prediction image: {output_path}")



def run_test_inference() -> None:
    """Run segmentation inference on test images and save visualized predictions."""
    config.ensure_output_dirs()

    if not config.TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Test images directory not found: {config.TEST_IMAGES_DIR}. "
            "Update config.py with the correct path."
        )

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

    test_dataset = DesertDataset(
        images_dir=config.TEST_IMAGES_DIR,
        masks_dir=None,
        mode="test",
        transform=get_val_test_transforms(image_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY and device.type == "cuda",
        collate_fn=collate_test_batch,
    )

    model = load_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        backend=config.MODEL_BACKEND,
        model_name=config.HF_MODEL_NAME,
    )
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=True)
    checkpoint_backend = checkpoint.get("model_backend")
    if checkpoint_backend is not None and checkpoint_backend != config.MODEL_BACKEND:
        raise ValueError(
            f"Checkpoint backend '{checkpoint_backend}' does not match current "
            f"MODEL_BACKEND='{config.MODEL_BACKEND}'. Retrain or switch config."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Running inference on {len(test_dataset)} test images...")
    if config.USE_TTA:
        print("TTA: enabled (4-pass averaging)")
    else:
        print("TTA: disabled (single-pass)")

    with torch.no_grad():
        progress = tqdm(test_loader, desc="Inference", leave=False)
        for images, filenames in progress:
            images = images.to(device, non_blocking=True)
            probs = predict_with_tta(model=model, images=images)
            preds = probs.argmax(dim=1).cpu().numpy()

            for pred_mask, filename in zip(preds, filenames):
                output_path = config.PREDICTIONS_DIR / filename
                save_colorized_prediction(pred_mask=pred_mask, output_path=output_path)

    print(f"Saved colorized predictions to: {config.PREDICTIONS_DIR}")

    has_test_masks = config.TEST_MASKS_DIR is not None and config.TEST_MASKS_DIR.exists()
    if not has_test_masks:
        print("Test masks not configured/found. Skipping IoU computation.")
        return

    print("Test masks detected. Computing mean IoU on test split...")
    eval_dataset = DesertDataset(
        images_dir=config.TEST_IMAGES_DIR,
        masks_dir=config.TEST_MASKS_DIR,
        mode="val",
        transform=get_val_test_transforms(image_size),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY and device.type == "cuda",
    )

    confusion = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=torch.int64)

    with torch.no_grad():
        progress = tqdm(eval_loader, desc="Test IoU", leave=False)
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            probs = predict_with_tta(model=model, images=images)
            preds = probs.argmax(dim=1).cpu()
            confusion = update_confusion_matrix(
                confusion=confusion,
                preds=preds,
                targets=masks,
                num_classes=config.NUM_CLASSES,
                ignore_index=config.IGNORE_INDEX,
            )

    mean_iou, per_class_iou = compute_mean_iou(confusion)
    print(f"Test mean IoU: {mean_iou:.4f}")
    for index, iou_value in enumerate(per_class_iou):
        print(f"  {config.CLASS_NAMES[index]:>16}: {iou_value:.4f}")



def main() -> None:
    """Entry point for test-time inference."""
    set_seed(config.SEED)
    run_test_inference()


if __name__ == "__main__":
    main()
