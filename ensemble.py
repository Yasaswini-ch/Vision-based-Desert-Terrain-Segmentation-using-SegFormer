"""Ensemble evaluation: averages softmax predictions from multiple checkpoints.

Usage:
    1. Train a second model (different seed or backbone) and note its checkpoint path.
    2. Add both paths to ENSEMBLE_CHECKPOINT_PATHS in config.py.
    3. Run:  python segmentation_project/ensemble.py

With a single checkpoint (default) this is equivalent to evaluate.py.
With two or more models, logits are averaged before argmax, typically gaining
2-4 mIoU points over the best single model.

Results are saved to runs/logs/ensemble_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import DesertDataset, get_val_test_transforms
from model import forward_logits, load_model
from train import compute_mean_iou, get_device, set_seed, update_confusion_matrix


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    """Load model weights from a checkpoint, inferring backend from the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backend: str = checkpoint.get("model_backend", config.MODEL_BACKEND)
    model = load_model(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
        backend=backend,
        model_name=config.HF_MODEL_NAME,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    saved_iou = checkpoint.get("val_mean_iou", float("nan"))
    print(f"  [{checkpoint_path.name}]  backend={backend}  epoch={epoch}  saved_mIoU={saved_iou:.4f}")
    return model, backend


def ensemble_evaluate() -> dict[str, object]:
    """Run ensemble evaluation and return results dict."""
    config.ensure_output_dirs()
    config.validate_dataset_dirs(require_masks_for_test=False)

    checkpoint_paths = [Path(p) for p in config.ENSEMBLE_CHECKPOINT_PATHS]
    missing = [str(p) for p in checkpoint_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Ensemble checkpoint(s) not found:\n" + "\n".join(missing)
            + "\nTrain additional models and update ENSEMBLE_CHECKPOINT_PATHS in config.py."
        )

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading {len(checkpoint_paths)} checkpoint(s):")

    models: list[torch.nn.Module] = []
    backends: list[str] = []
    for path in checkpoint_paths:
        model, backend = _load_checkpoint_model(path, device)
        models.append(model)
        backends.append(backend)

    dataset = DesertDataset(
        images_dir=config.VAL_IMAGES_DIR,
        masks_dir=config.VAL_MASKS_DIR,
        mode="val",
        transform=get_val_test_transforms(config.IMAGE_SIZE),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and device.type == "cuda",
    )

    confusion = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=torch.int64)

    with torch.no_grad():
        progress = tqdm(loader, desc=f"Ensemble ({len(models)} model(s))", leave=True)
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            target_size = tuple(masks.shape[-2:])

            # Average softmax probabilities across all models
            avg_probs: torch.Tensor | None = None
            for model, backend in zip(models, backends):
                logits = forward_logits(model=model, images=images, backend=backend)
                logits = F.interpolate(
                    logits.float(), size=target_size, mode="bilinear", align_corners=False
                )
                probs = torch.softmax(logits, dim=1)
                avg_probs = probs if avg_probs is None else avg_probs + probs

            assert avg_probs is not None
            preds = (avg_probs / len(models)).argmax(dim=1)

            confusion = update_confusion_matrix(
                confusion=confusion,
                preds=preds.cpu(),
                targets=masks.cpu(),
                num_classes=config.NUM_CLASSES,
                ignore_index=config.IGNORE_INDEX,
            )

    mean_iou, per_class_iou = compute_mean_iou(confusion)
    results: dict[str, object] = {
        "ensemble_mean_iou": mean_iou,
        "num_models": len(models),
        "checkpoint_paths": [str(p) for p in checkpoint_paths],
        "per_class_iou": {
            config.CLASS_NAMES[i]: per_class_iou[i] for i in range(config.NUM_CLASSES)
        },
    }

    out_path = config.LOG_DIR / "ensemble_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEnsemble mIoU ({len(models)} model(s)): {mean_iou:.4f}")
    print(f"{'Class':<16} {'IoU':>6}")
    print("-" * 24)
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"{name:<16} {per_class_iou[i]:>6.4f}")
    print(f"\nSaved ensemble results to: {out_path}")

    return results


def main() -> None:
    """Entry point for ensemble evaluation."""
    set_seed(config.SEED)
    ensemble_evaluate()


if __name__ == "__main__":
    main()
