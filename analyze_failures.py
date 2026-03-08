"""Failure analysis for desert semantic segmentation.

Loads the best model checkpoint, runs inference on all test images with
confidence scores, ranks every image by difficulty (lowest confidence),
and saves the top 10 hardest cases with full visual and textual reports.

Outputs
-------
runs/failure_analysis/rank_NN_<stem>/
    original.png
    prediction.png
    confidence_heatmap.png
    failure_report.txt
runs/failure_analysis/summary.json
"""

from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure segmentation_project modules are importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from dataset import DesertDataset, get_val_test_transforms
from model import forward_logits, load_model
from train import get_device, set_seed

FAILURE_DIR: Path = config.RUNS_DIR / "failure_analysis"
CONFIDENCE_THRESHOLD: float = 0.6
TOP_N_FAILURES: int = 10


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def colorize_prediction(pred: np.ndarray) -> np.ndarray:
    """Convert class-index map (H, W) → RGB visualisation."""
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for cls_idx, color in config.CLASS_COLORS.items():
        rgb[pred == cls_idx] = color
    return rgb


def make_confidence_heatmap(conf_map: np.ndarray) -> np.ndarray:
    """Convert confidence map [0,1] (H, W) → BGR heatmap via COLORMAP_JET."""
    conf_uint8 = (conf_map * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)  # returns BGR


# ---------------------------------------------------------------------------
# Per-image analysis helpers
# ---------------------------------------------------------------------------

def get_confused_pairs(
    probs: np.ndarray,
    confusion_ratio_threshold: float = 0.6,
    top_k: int = 5,
) -> list[dict]:
    """Identify class pairs where the model is most uncertain.

    A pixel is "confused" when second-best probability ≥
    ``confusion_ratio_threshold`` × best probability.

    Args:
        probs: Softmax probabilities (C, H, W).
        confusion_ratio_threshold: Ratio that triggers confusion flag.
        top_k: Number of top pairs to return.

    Returns:
        List of dicts with ``class_a``, ``class_b``, ``confused_pixels``.
    """
    C, H, W = probs.shape
    flat = probs.reshape(C, -1)                    # (C, N)
    argsorted = np.argsort(flat, axis=0)[::-1]     # descending, (C, N)

    top1_idx = argsorted[0]    # (N,)
    top2_idx = argsorted[1]    # (N,)
    top1_val = flat[top1_idx, np.arange(flat.shape[1])]
    top2_val = flat[top2_idx, np.arange(flat.shape[1])]

    confused_mask = top2_val / (top1_val + 1e-8) >= confusion_ratio_threshold

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for n in np.where(confused_mask)[0]:
        c1, c2 = int(top1_idx[n]), int(top2_idx[n])
        pair = (min(c1, c2), max(c1, c2))
        pair_counts[pair] += 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:top_k]
    return [
        {
            "class_a": config.CLASS_NAMES[p[0]],
            "class_b": config.CLASS_NAMES[p[1]],
            "confused_pixels": int(cnt),
        }
        for p, cnt in sorted_pairs
    ]


def generate_failure_report(
    image_name: str,
    mean_conf: float,
    uncertain_pct: float,
    confused_pairs: list[dict],
    pred_distribution: dict[str, float],
) -> str:
    """Produce a human-readable analysis for one hard image."""
    lines: list[str] = [
        "FAILURE ANALYSIS REPORT",
        "=" * 50,
        f"Image: {image_name}",
        "",
        "METRICS",
        f"  Mean prediction confidence : {mean_conf:.4f}  ({mean_conf * 100:.1f}%)",
        f"  Uncertain pixels (conf<{CONFIDENCE_THRESHOLD}) : {uncertain_pct:.1f}%",
        "",
        "WHY THE MODEL STRUGGLED HERE",
    ]

    if uncertain_pct > 40:
        lines += [
            "  - Very high proportion of uncertain regions (>40%), suggesting",
            "    the scene contains visual patterns poorly represented in training.",
        ]
    elif uncertain_pct > 20:
        lines += [
            "  - Significant uncertainty in multiple regions, likely due to",
            "    ambiguous textures, unusual lighting, or mixed-class boundaries.",
        ]
    else:
        lines += [
            "  - Localised low-confidence regions, suggesting boundary ambiguity",
            "    between visually similar classes.",
        ]

    if mean_conf < 0.5:
        lines += [
            "  - Very low mean confidence: the model is unsure about nearly every",
            "    prediction in this scene — strong out-of-distribution signal.",
        ]

    lines += [
        "",
        "CONFUSED CLASS PAIRS (top confusions)",
    ]
    if confused_pairs:
        for pair in confused_pairs:
            lines.append(
                f"  - {pair['class_a']:<18} <-> {pair['class_b']:<18} "
                f": {pair['confused_pixels']:,} pixels"
            )
        c_a = confused_pairs[0]["class_a"]
        c_b = confused_pairs[0]["class_b"]
        lines += ["", f"  Primary confusion: {c_a} vs {c_b}"]
        if "Dry" in c_a or "Dry" in c_b:
            lines.append("  (Dry-terrain classes share similar warm, earthy tones.)")
        if "Ground" in c_a or "Ground" in c_b:
            lines.append("  (Ground clutter overlaps visually with rocks and dry bushes.)")
        if "Landscape" in c_a or "Landscape" in c_b:
            lines.append("  (Open landscape can resemble dry grass at distance.)")
        if "Rocks" in c_a or "Rocks" in c_b:
            lines.append("  (Rocks and logs share dark, low-saturation appearances.)")
    else:
        lines.append("  No dominant confusion pairs detected.")

    lines += [
        "",
        "PREDICTED CLASS DISTRIBUTION (top 5)",
    ]
    for cls_name, pct in sorted(pred_distribution.items(), key=lambda x: -x[1])[:5]:
        bar = "#" * int(pct / 2)
        lines.append(f"  {cls_name:<20}: {pct:5.1f}%  {bar}")

    lines += [
        "",
        "SUGGESTED FIX",
    ]
    if confused_pairs:
        c_a = confused_pairs[0]["class_a"]
        c_b = confused_pairs[0]["class_b"]
        lines += [
            f"  1. Collect more training samples that show {c_a} and {c_b} under",
            "     similar lighting and texture conditions to improve discrimination.",
            "  2. Apply stronger colour-jitter and brightness augmentation so the",
            "     model learns shape and texture cues rather than colour alone.",
        ]
    if uncertain_pct > 30:
        lines += [
            "  3. Add test-time augmentation (TTA) or CRF post-processing to",
            "     sharpen predictions in high-uncertainty regions.",
        ]
    lines += [
        "  4. Review per-class loss weights — underrepresented or confused classes",
        "     may need higher weight to receive stronger training signal.",
        "  5. Consider adding hard-example mining so the training loop focuses",
        "     more epochs on scenes similar to this one.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_failure_analysis() -> None:
    """Full failure-analysis pipeline: infer, rank, save top-10."""
    set_seed(config.SEED)
    config.ensure_output_dirs()
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {config.BEST_MODEL_PATH}\n"
            "Run train.py first to produce a checkpoint."
        )

    device = get_device()
    image_size = (
        config.CPU_QUICK_IMAGE_SIZE
        if device.type == "cpu" and config.CPU_QUICK_MODE
        else config.IMAGE_SIZE
    )

    print(f"Device        : {device}")
    print(f"Image size    : {image_size}")
    print(f"Test images   : {config.TEST_IMAGES_DIR}")
    print(f"Output dir    : {FAILURE_DIR}")

    # ------------------------------------------------------------------
    # Dataset (test mode – no masks required)
    # ------------------------------------------------------------------
    dataset = DesertDataset(
        images_dir=config.TEST_IMAGES_DIR,
        masks_dir=None,
        mode="test",
        transform=get_val_test_transforms(image_size),
    )
    print(f"Found {len(dataset)} test images")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
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
    print(
        f"Loaded checkpoint  epoch={checkpoint.get('epoch','?')}  "
        f"mIoU={checkpoint.get('val_mean_iou','?')}"
    )

    # ------------------------------------------------------------------
    # Inference pass – collect per-image metrics
    # ------------------------------------------------------------------
    results: list[dict] = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Inference"):
            image_tensor, filename = dataset[idx]
            images = image_tensor.unsqueeze(0).to(device)

            logits = forward_logits(
                model=model, images=images, backend=config.MODEL_BACKEND
            )
            logits = F.interpolate(
                logits,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )

            probs = F.softmax(logits, dim=1)[0].cpu().numpy()   # (C, H, W)
            conf_map = probs.max(axis=0)                         # (H, W)
            pred_map = probs.argmax(axis=0)                      # (H, W)

            mean_conf = float(conf_map.mean())
            uncertain_pct = float((conf_map < CONFIDENCE_THRESHOLD).mean() * 100)
            confused_pairs = get_confused_pairs(probs)

            n_pixels = pred_map.size
            pred_dist = {
                config.CLASS_NAMES[c]: float((pred_map == c).sum() / n_pixels * 100)
                for c in range(config.NUM_CLASSES)
            }

            # Difficulty: lower mean confidence + higher uncertainty = harder
            difficulty_score = mean_conf - (uncertain_pct / 100) * 0.3

            results.append(
                {
                    "filename": filename,
                    "mean_conf": mean_conf,
                    "uncertain_pct": uncertain_pct,
                    "confused_pairs": confused_pairs,
                    "pred_distribution": pred_dist,
                    "difficulty_score": difficulty_score,
                    # raw arrays kept in memory only for the top-10 save step
                    "_probs": probs,
                    "_pred_map": pred_map,
                    "_conf_map": conf_map,
                }
            )

    # ------------------------------------------------------------------
    # Rank by difficulty (ascending = hardest first)
    # ------------------------------------------------------------------
    results.sort(key=lambda x: x["difficulty_score"])

    print("\nTop 10 hardest images:")
    for i, r in enumerate(results[:TOP_N_FAILURES]):
        print(
            f"  {i + 1:2d}. {r['filename']:<20}  "
            f"conf={r['mean_conf']:.3f}  uncertain={r['uncertain_pct']:.1f}%"
        )
    # also prepare a list of strings for inclusion in the JSON summary
    top_list = [
        f"{i + 1}. {r['filename']}   conf={r['mean_conf']:.3f}  uncertain={r['uncertain_pct']:.1f}%"
        for i, r in enumerate(results[:TOP_N_FAILURES])
    ]

    # ------------------------------------------------------------------
    # Save artefacts for top-10
    # ------------------------------------------------------------------
    failure_records: list[dict] = []

    for rank, r in enumerate(results[:TOP_N_FAILURES]):
        stem = Path(r["filename"]).stem
        out_dir = FAILURE_DIR / f"rank_{rank + 1:02d}_{stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Original image (copy from source)
        orig_src = config.TEST_IMAGES_DIR / r["filename"]
        orig_bgr = cv2.imread(str(orig_src))
        cv2.imwrite(str(out_dir / "original.png"), orig_bgr)

        # Prediction overlay
        pred_rgb = colorize_prediction(r["_pred_map"])
        cv2.imwrite(
            str(out_dir / "prediction.png"),
            cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR),
        )

        # Confidence heatmap
        heatmap_bgr = make_confidence_heatmap(r["_conf_map"])
        cv2.imwrite(str(out_dir / "confidence_heatmap.png"), heatmap_bgr)

        # Failure report
        report_text = generate_failure_report(
            image_name=r["filename"],
            mean_conf=r["mean_conf"],
            uncertain_pct=r["uncertain_pct"],
            confused_pairs=r["confused_pairs"],
            pred_distribution=r["pred_distribution"],
        )
        (out_dir / "failure_report.txt").write_text(report_text, encoding="utf-8")

        # Record (no raw arrays)
        failure_records.append(
            {
                "rank": rank + 1,
                "filename": r["filename"],
                "difficulty_score": round(r["difficulty_score"], 6),
                "mean_conf": round(r["mean_conf"], 6),
                "uncertain_pct": round(r["uncertain_pct"], 2),
                "confused_pairs": r["confused_pairs"],
                "pred_distribution": {
                    k: round(v, 2) for k, v in r["pred_distribution"].items()
                },
                "output_dir": str(out_dir.relative_to(config.PROJECT_ROOT)),
            }
        )
        print(f"  Saved rank {rank + 1:02d} → {out_dir.name}/")

    # ------------------------------------------------------------------
    # Global failure patterns summary
    # ------------------------------------------------------------------
    all_pair_counts: dict[str, int] = defaultdict(int)
    total_uncertain = 0.0
    total_conf = 0.0

    for r in results:
        total_uncertain += r["uncertain_pct"]
        total_conf += r["mean_conf"]
        for pair in r["confused_pairs"][:3]:
            key = f"{pair['class_a']} <-> {pair['class_b']}"
            all_pair_counts[key] += pair["confused_pixels"]

    n = len(results)
    global_top_pairs = sorted(all_pair_counts.items(), key=lambda x: -x[1])[:5]

    summary = {
        "total_images_analyzed": n,
        "global_mean_confidence": round(total_conf / max(n, 1), 6),
        "global_uncertain_pct": round(total_uncertain / max(n, 1), 2),
        "global_top_confused_pairs": [
            {"pair": k, "total_confused_pixels": int(v)}
            for k, v in global_top_pairs
        ],
        "failure_patterns": {
            "high_uncertainty_images": sum(
                1 for r in results if r["uncertain_pct"] > 40
            ),
            "low_confidence_images": sum(
                1 for r in results if r["mean_conf"] < 0.5
            ),
            "moderate_difficulty": sum(
                1 for r in results if 0.5 <= r["mean_conf"] < 0.65
            ),
            "well_predicted": sum(1 for r in results if r["mean_conf"] >= 0.65),
        },
        "hardest_images": failure_records,
        # human-friendly list matching console output
        "hardest_images_list": top_list,
    }

    summary_path = FAILURE_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Failure analysis complete")
    print(f"  Output dir   : {FAILURE_DIR}")
    print(f"  Summary      : {summary_path}")
    print(f"  Total images : {n}")
    print(f"  Global mean confidence  : {summary['global_mean_confidence']:.4f}")
    print(f"  Global uncertain pixels : {summary['global_uncertain_pct']:.1f}%")
    print(
        f"  Top confused pairs: "
        f"{[p['pair'] for p in summary['global_top_confused_pairs'][:3]]}"
    )


if __name__ == "__main__":
    run_failure_analysis()
