"""Dense CRF post-processing to sharpen segmentation boundaries.

Applies a fully-connected CRF (Krähenbühl & Koltun, 2011) using the original
RGB image as bilateral (appearance + position) pairwise features.
This refines jagged edges around fine-detail classes like Rocks, Logs,
and Flowers without any model retraining.

Install dependency first:
    pip install pydensecrf

Usage:
    python segmentation_project/densecrf.py

Outputs (runs/densecrf/):
    <name>_before.png      — raw model prediction overlay
    <name>_after.png       — CRF-refined prediction overlay
    <name>_diff.png        — pixel-level diff (white = changed by CRF)
    densecrf_results.json  — per-class IoU before and after CRF
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
    _HAS_DENSECRF = True
except ImportError:
    _HAS_DENSECRF = False

import config
from dataset import DesertDataset, get_val_test_transforms
from model import forward_logits, load_model
from train import compute_mean_iou, get_device, set_seed, update_confusion_matrix

# CRF hyperparameters — tune these to trade smoothness vs sharpness
CRF_INFERENCE_STEPS: int = 5
CRF_GAUSSIAN_XY: float = 3.0
CRF_GAUSSIAN_COMPAT: float = 3.0
CRF_BILATERAL_XY: float = 80.0
CRF_BILATERAL_RGB: float = 13.0
CRF_BILATERAL_COMPAT: float = 10.0


def _apply_crf(
    rgb_image: np.ndarray,
    softmax_probs: np.ndarray,
) -> np.ndarray:
    """Apply Dense CRF to refine softmax predictions using RGB image guidance.

    Args:
        rgb_image: HxWx3 uint8 original RGB image (not normalized).
        softmax_probs: CxHxW float32 softmax probability map.

    Returns:
        HxW int32 refined class-index prediction.
    """
    h, w = rgb_image.shape[:2]
    n_classes = softmax_probs.shape[0]

    d = dcrf.DenseCRF2D(w, h, n_classes)

    # Unary energy from softmax
    probs_c = np.ascontiguousarray(softmax_probs, dtype=np.float32)
    unary = unary_from_softmax(probs_c)
    d.setUnaryEnergy(unary)

    # Pairwise Gaussian: encourages nearby pixels to share the same label
    d.addPairwiseGaussian(
        sxy=(CRF_GAUSSIAN_XY, CRF_GAUSSIAN_XY),
        compat=CRF_GAUSSIAN_COMPAT,
    )

    # Pairwise Bilateral: uses RGB appearance to refine boundaries
    rgb_c = np.ascontiguousarray(rgb_image, dtype=np.uint8)
    d.addPairwiseBilateral(
        sxy=(CRF_BILATERAL_XY, CRF_BILATERAL_XY),
        srgb=(CRF_BILATERAL_RGB, CRF_BILATERAL_RGB, CRF_BILATERAL_RGB),
        rgbim=rgb_c,
        compat=CRF_BILATERAL_COMPAT,
    )

    q = d.inference(CRF_INFERENCE_STEPS)
    return np.argmax(q, axis=0).reshape(h, w).astype(np.int32)


def _colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in config.CLASS_COLORS.items():
        out[mask == idx] = color
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _diff_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """White pixels where CRF changed the predicted class."""
    changed = (before != after).astype(np.uint8) * 255
    return cv2.cvtColor(changed, cv2.COLOR_GRAY2BGR)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert ImageNet-normalized CHW tensor to uint8 HWC RGB numpy array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * IMAGENET_STD + IMAGENET_MEAN).clip(0.0, 1.0)
    return (img * 255).astype(np.uint8)


def run_densecrf() -> dict[str, object]:
    if not _HAS_DENSECRF:
        raise ImportError(
            "pydensecrf is not installed.\n"
            "Install it with:  pip install pydensecrf\n"
            "Then re-run:      python segmentation_project/densecrf.py"
        )

    config.ensure_output_dirs()
    config.validate_dataset_dirs(require_masks_for_test=False)

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {config.BEST_MODEL_PATH}. Run train.py first."
        )

    device = get_device()
    print(f"Device: {device}")
    print(f"CRF inference steps: {CRF_INFERENCE_STEPS}")

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

    dataset = DesertDataset(
        images_dir=config.VAL_IMAGES_DIR,
        masks_dir=config.VAL_MASKS_DIR,
        mode="val",
        transform=get_val_test_transforms(config.IMAGE_SIZE),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and device.type == "cuda",
    )

    confusion_before = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=torch.int64)
    confusion_after = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=torch.int64)

    max_images = config.DENSECRF_MAX_IMAGES
    total = min(len(dataset), max_images) if max_images else len(dataset)
    if max_images:
        print(f"Processing first {max_images} images (DENSECRF_MAX_IMAGES). Set to None for all.")

    progress = tqdm(enumerate(loader), total=total, desc="DenseCRF", leave=True)
    for i, (images, masks) in progress:
        if max_images and i >= max_images:
            break
        images = images.to(device, non_blocking=True)
        target_size = tuple(images.shape[-2:])

        with torch.no_grad():
            logits = forward_logits(model=model, images=images, backend=config.MODEL_BACKEND)
            logits = F.interpolate(logits.float(), size=target_size, mode="bilinear", align_corners=False)
            probs = torch.softmax(logits, dim=1)  # [1, C, H, W]

        softmax_np = probs[0].cpu().numpy()         # [C, H, W]
        pred_before = softmax_np.argmax(axis=0).astype(np.int32)

        rgb_image = _tensor_to_rgb(images[0])       # HWC uint8
        pred_after = _apply_crf(rgb_image, softmax_np)

        # Update confusion matrices
        gt = masks[0]
        confusion_before = update_confusion_matrix(
            confusion_before,
            torch.from_numpy(pred_before).unsqueeze(0),
            gt.unsqueeze(0),
            config.NUM_CLASSES,
            config.IGNORE_INDEX,
        )
        confusion_after = update_confusion_matrix(
            confusion_after,
            torch.from_numpy(pred_after).unsqueeze(0),
            gt.unsqueeze(0),
            config.NUM_CLASSES,
            config.IGNORE_INDEX,
        )

        # Save visualizations
        stem = dataset.image_paths[i].stem
        cv2.imwrite(str(config.DENSECRF_DIR / f"{stem}_before.png"), _colorize(pred_before))
        cv2.imwrite(str(config.DENSECRF_DIR / f"{stem}_after.png"), _colorize(pred_after))
        cv2.imwrite(str(config.DENSECRF_DIR / f"{stem}_diff.png"), _diff_image(pred_before, pred_after))

    miou_before, per_class_before = compute_mean_iou(confusion_before)
    miou_after, per_class_after = compute_mean_iou(confusion_after)

    results: dict[str, object] = {
        "miou_before_crf": miou_before,
        "miou_after_crf": miou_after,
        "miou_delta": miou_after - miou_before,
        "crf_params": {
            "inference_steps": CRF_INFERENCE_STEPS,
            "gaussian_xy": CRF_GAUSSIAN_XY,
            "gaussian_compat": CRF_GAUSSIAN_COMPAT,
            "bilateral_xy": CRF_BILATERAL_XY,
            "bilateral_rgb": CRF_BILATERAL_RGB,
            "bilateral_compat": CRF_BILATERAL_COMPAT,
        },
        "per_class": {
            config.CLASS_NAMES[i]: {
                "before": per_class_before[i],
                "after": per_class_after[i],
                "delta": per_class_after[i] - per_class_before[i],
            }
            for i in range(config.NUM_CLASSES)
        },
    }

    out_path = config.LOG_DIR / "densecrf_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    delta = results["miou_delta"]
    sign = "+" if delta >= 0 else ""
    print(f"\nmIoU before CRF: {miou_before:.4f}")
    print(f"mIoU after  CRF: {miou_after:.4f}  ({sign}{delta:.4f})")
    print(f"\n{'Class':<16} {'Before':>7} {'After':>7} {'Delta':>7}")
    print("-" * 40)
    for i, name in enumerate(config.CLASS_NAMES):
        d = per_class_after[i] - per_class_before[i]
        sign_d = "+" if d >= 0 else ""
        print(f"{name:<16} {per_class_before[i]:>7.4f} {per_class_after[i]:>7.4f} {sign_d}{d:>6.4f}")
    print(f"\nSaved refined images to: {config.DENSECRF_DIR}")
    print(f"Saved results to:        {out_path}")

    return results


def main() -> None:
    set_seed(config.SEED)
    run_densecrf()


if __name__ == "__main__":
    main()
