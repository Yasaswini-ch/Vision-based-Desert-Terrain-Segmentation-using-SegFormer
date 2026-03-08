"""Monte Carlo Dropout uncertainty estimation for desert segmentation.

Runs N stochastic forward passes with dropout enabled, computes per-pixel
predictive entropy as a measure of model uncertainty, and saves heatmaps.

High-entropy pixels (red) = model is uncertain about the class.
Low-entropy pixels (blue)  = model is confident.

This is useful for the report's Failure Case Analysis section — uncertain
regions directly map to where the model is most likely to misclassify.

Usage:
    python segmentation_project/uncertainty.py

Outputs (runs/uncertainty/):
    <name>_pred.png        — color-coded class prediction
    <name>_uncertainty.png — JET heatmap: red=uncertain, blue=confident
    uncertainty_summary.json — mean uncertainty per class + overall stats
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import DesertDataset, get_val_test_transforms
from model import forward_logits, load_model
from train import get_device, set_seed

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _enable_dropout(model: torch.nn.Module) -> None:
    """Keep BatchNorm/LayerNorm in eval mode but activate all Dropout layers."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert ImageNet-normalized CHW tensor to uint8 HWC numpy array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * IMAGENET_STD + IMAGENET_MEAN).clip(0.0, 1.0)
    return (img * 255).astype(np.uint8)


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Apply CLASS_COLORS palette to a class-index mask."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in config.CLASS_COLORS.items():
        out[mask == idx] = color
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _entropy_heatmap(entropy: np.ndarray) -> np.ndarray:
    """Normalize entropy [0,1] and apply JET colormap (BGR)."""
    norm = (entropy / (entropy.max() + 1e-8) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def run_uncertainty() -> None:
    config.ensure_output_dirs()
    config.validate_dataset_dirs(require_masks_for_test=False)

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {config.BEST_MODEL_PATH}. Run train.py first."
        )

    device = get_device()
    n_passes = config.MC_DROPOUT_PASSES
    print(f"Device: {device}")
    print(f"MC Dropout passes: {n_passes}")

    model = load_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        backend=config.MODEL_BACKEND,
        model_name=config.HF_MODEL_NAME,
    )
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    _enable_dropout(model)  # Activate dropout only, keep normalization layers in eval

    dataset = DesertDataset(
        images_dir=config.VAL_IMAGES_DIR,
        masks_dir=config.VAL_MASKS_DIR,
        mode="val",
        transform=get_val_test_transforms(config.IMAGE_SIZE),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,  # Process one image at a time for per-image heatmaps
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and device.type == "cuda",
    )

    # Accumulators for summary statistics
    class_entropy_sums = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    class_pixel_counts = np.zeros(config.NUM_CLASSES, dtype=np.int64)
    global_entropy_values: list[float] = []

    max_images = config.UNCERTAINTY_MAX_IMAGES
    total = min(len(dataset), max_images) if max_images else len(dataset)
    if max_images:
        print(f"Processing first {max_images} images (UNCERTAINTY_MAX_IMAGES). Set to None for all.")

    progress = tqdm(enumerate(loader), total=total, desc="Uncertainty", leave=True)
    for i, (images, masks) in progress:
        if max_images and i >= max_images:
            break
        images = images.to(device, non_blocking=True)
        target_size = tuple(images.shape[-2:])

        # Accumulate softmax probs over N stochastic passes
        stacked = torch.zeros(
            n_passes, 1, config.NUM_CLASSES, *target_size,
            device=device, dtype=torch.float32,
        )
        with torch.no_grad():
            for p in range(n_passes):
                logits = forward_logits(model=model, images=images, backend=config.MODEL_BACKEND)
                logits = F.interpolate(logits.float(), size=target_size, mode="bilinear", align_corners=False)
                stacked[p] = torch.softmax(logits, dim=1)

        # Mean prediction and entropy
        mean_probs = stacked.mean(dim=0)  # [1, C, H, W]
        entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=1)  # [1, H, W]

        pred_mask = mean_probs[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        entropy_np = entropy[0].cpu().numpy()

        # Accumulate per-class entropy
        gt_mask = masks[0].numpy()
        for cls in range(config.NUM_CLASSES):
            cls_pixels = gt_mask == cls
            if cls_pixels.any():
                class_entropy_sums[cls] += float(entropy_np[cls_pixels].mean())
                class_pixel_counts[cls] += 1

        global_entropy_values.append(float(entropy_np.mean()))

        # Save prediction overlay
        stem = dataset.image_paths[i].stem
        pred_bgr = _colorize_mask(pred_mask)
        cv2.imwrite(str(config.UNCERTAINTY_DIR / f"{stem}_pred.png"), pred_bgr)

        # Save uncertainty heatmap
        heatmap = _entropy_heatmap(entropy_np)
        cv2.imwrite(str(config.UNCERTAINTY_DIR / f"{stem}_uncertainty.png"), heatmap)

        # Save side-by-side: original | prediction | uncertainty
        original_rgb = _denormalize(images[0])
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        h, w = pred_mask.shape
        pred_bgr_r = cv2.resize(pred_bgr, (w, h))
        heatmap_r = cv2.resize(heatmap, (w, h))
        panel = np.concatenate([original_bgr, pred_bgr_r, heatmap_r], axis=1)
        cv2.imwrite(str(config.UNCERTAINTY_DIR / f"{stem}_panel.png"), panel)

    # Build summary
    per_class_mean_entropy = {
        config.CLASS_NAMES[i]: (
            float(class_entropy_sums[i] / class_pixel_counts[i])
            if class_pixel_counts[i] > 0 else None
        )
        for i in range(config.NUM_CLASSES)
    }
    summary = {
        "mc_dropout_passes": n_passes,
        "overall_mean_entropy": float(np.mean(global_entropy_values)),
        "per_class_mean_entropy": per_class_mean_entropy,
        "interpretation": {
            "high_entropy": "Model uncertain — likely misclassification",
            "low_entropy": "Model confident — likely correct prediction",
        },
    }

    out_path = config.LOG_DIR / "uncertainty_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean entropy across val set: {summary['overall_mean_entropy']:.4f}")
    print(f"\n{'Class':<16} {'Mean Entropy':>12}")
    print("-" * 30)
    for name, val in per_class_mean_entropy.items():
        display = f"{val:.4f}" if val is not None else "  n/a"
        print(f"{name:<16} {display:>12}")
    print(f"\nSaved heatmaps to: {config.UNCERTAINTY_DIR}")
    print(f"Saved summary to:  {out_path}")


def main() -> None:
    set_seed(config.SEED)
    run_uncertainty()


if __name__ == "__main__":
    main()
