"""Desert Segmentation — Kaggle Evaluation Notebook.

Run this file inside a Kaggle notebook (GPU T4 x2 or P100) after attaching:
  - Dataset "desert-segmentation"       (your image + mask data)
  - Dataset "desert-seg-code"           (this project's source files)
  - Yesterday's notebook output dataset (contains best_model.pth)

Runs: evaluate → test inference → failure analysis → training curves → zip.
Does NOT retrain — loads the existing checkpoint.
"""

# ─── 0. Install extra packages not pre-installed on Kaggle ──────────────────
import subprocess, sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

pip_install(
    "albumentations==1.4.24",
    "transformers==4.46.3",
    "timm",
    "scipy==1.17.1",
)

# ─── 1. Paths and sys.path setup ────────────────────────────────────────────
import os
from pathlib import Path

# Where Kaggle mounts the project source code dataset
CODE_ROOT = Path("/kaggle/input/desert-seg-code/segmentation_project")

# Where Kaggle mounts the image/mask data dataset
DATA_ROOT = Path("/kaggle/input/desert-segmentation")

# All outputs go here (persisted after the session)
WORKING_DIR = Path("/kaggle/working")
RUNS_DIR = WORKING_DIR / "runs"

# Add project source to Python path so we can import config, train, etc.
sys.path.insert(0, str(CODE_ROOT))

# ─── 2. Verify paths exist before continuing ────────────────────────────────
def _check_path(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(
            f"{label} not found at: {p}\n"
            "Check that you attached the correct Kaggle dataset and the folder "
            "structure matches what README_KAGGLE.md describes."
        )

_check_path(CODE_ROOT, "Project source code")
_check_path(DATA_ROOT, "Image dataset")

# ─── 3. Detect GPU and choose model backend ──────────────────────────────────
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
    MODEL_BACKEND = "segformer_b2"
    IMAGE_SIZE    = 512
    BATCH_SIZE    = 8
    EPOCHS        = 50
    NUM_WORKERS   = 2
    PIN_MEMORY    = True
    USE_AMP       = True
else:
    print("WARNING: No GPU detected. Falling back to CPU with reduced settings.")
    MODEL_BACKEND = "deeplabv3_mobilenet"
    IMAGE_SIZE    = 256
    BATCH_SIZE    = 4
    EPOCHS        = 10
    NUM_WORKERS   = 0
    PIN_MEMORY    = False
    USE_AMP       = False

print(f"Model backend : {MODEL_BACKEND}")
print(f"Image size    : {IMAGE_SIZE}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Epochs        : {EPOCHS}")

# ─── 4. Patch config before any project module imports it ───────────────────
import config  # noqa: E402 (must come after sys.path insert)

# Detect which dataset folder layout the user uploaded
_OFFROAD_ROOT = DATA_ROOT / "Offroad_Segmentation_Training_Dataset"
_SIMPLE_ROOT  = DATA_ROOT / "dataset"

if _OFFROAD_ROOT.exists():
    config.TRAIN_IMAGES_DIR = _OFFROAD_ROOT / "train" / "Color_Images"
    config.TRAIN_MASKS_DIR  = _OFFROAD_ROOT / "train" / "Segmentation"
    config.VAL_IMAGES_DIR   = _OFFROAD_ROOT / "val"   / "Color_Images"
    config.VAL_MASKS_DIR    = _OFFROAD_ROOT / "val"   / "Segmentation"
    _test_color = DATA_ROOT / "Offroad_Segmentation_testImages" / "Color_Images"
    config.TEST_IMAGES_DIR  = _test_color if _test_color.exists() else _OFFROAD_ROOT / "val" / "Color_Images"
    config.TEST_MASKS_DIR   = None
    print("Dataset layout: Offroad_Segmentation_Training_Dataset")
elif _SIMPLE_ROOT.exists():
    config.TRAIN_IMAGES_DIR = _SIMPLE_ROOT / "train" / "images"
    config.TRAIN_MASKS_DIR  = _SIMPLE_ROOT / "train" / "masks"
    config.VAL_IMAGES_DIR   = _SIMPLE_ROOT / "val"   / "images"
    config.VAL_MASKS_DIR    = _SIMPLE_ROOT / "val"   / "masks"
    config.TEST_IMAGES_DIR  = _SIMPLE_ROOT / "testImages" / "images"
    config.TEST_MASKS_DIR   = None
    print("Dataset layout: simple dataset/")
else:
    raise FileNotFoundError(
        f"Could not find dataset under {DATA_ROOT}.\n"
        "Expected either 'Offroad_Segmentation_Training_Dataset' or 'dataset' subfolder.\n"
        "See README_KAGGLE.md for the required folder structure."
    )

# Override output paths to Kaggle working dir
config.RUNS_DIR         = RUNS_DIR
config.CHECKPOINT_DIR   = RUNS_DIR / "checkpoints"
config.LOG_DIR          = RUNS_DIR / "logs"
config.PREDICTIONS_DIR  = RUNS_DIR / "predictions"
config.BEST_MODEL_PATH  = config.CHECKPOINT_DIR / "best_model.pth"
config.HISTORY_PATH     = config.LOG_DIR / "history.json"
config.EVAL_RESULTS_PATH = config.LOG_DIR / "evaluation_results.json"
config.CURVES_PATH      = config.LOG_DIR / "training_curves.png"

# Apply GPU-tuned hyperparameters
config.MODEL_BACKEND  = MODEL_BACKEND
config.IMAGE_SIZE     = IMAGE_SIZE
config.BATCH_SIZE     = BATCH_SIZE
config.FALLBACK_BATCH_SIZE = max(BATCH_SIZE // 2, 1)
config.EPOCHS         = EPOCHS
config.NUM_WORKERS    = NUM_WORKERS
config.PIN_MEMORY     = PIN_MEMORY
config.USE_AMP        = USE_AMP
config.CPU_QUICK_MODE = False          # disable quick-mode on Kaggle

# Cosine scheduler T_max should not exceed epochs
config.SCHEDULER_T_MAX = EPOCHS

config.ensure_output_dirs()

print("\nConfig summary:")
print(f"  TRAIN_IMAGES_DIR : {config.TRAIN_IMAGES_DIR}")
print(f"  VAL_IMAGES_DIR   : {config.VAL_IMAGES_DIR}")
print(f"  CHECKPOINT_DIR   : {config.CHECKPOINT_DIR}")
print(f"  MODEL_BACKEND    : {config.MODEL_BACKEND}")

# ─── 5. Locate and copy checkpoint from yesterday's saved output ─────────────
import shutil

# Kaggle attaches previous notebook outputs as input datasets.
# The output dataset is typically mounted at /kaggle/input/<notebook-slug>/
# Check common locations for best_model.pth
CHECKPOINT_SEARCH_DIRS = [
    Path("/kaggle/input/desert-seg-checkpoint"),          # if uploaded as standalone dataset
    Path("/kaggle/input/desert-segmentation-training"),   # if attached from notebook output
    Path("/kaggle/input/desert-seg-results"),             # another common name
]

def find_checkpoint() -> Path | None:
    for base in CHECKPOINT_SEARCH_DIRS:
        for candidate in base.rglob("best_model.pth"):
            return candidate
    return None

src_ckpt = find_checkpoint()
if src_ckpt is None:
    raise FileNotFoundError(
        "Could not find best_model.pth in any attached input dataset.\n"
        "Attach yesterday's notebook output as an input dataset, or upload\n"
        "best_model.pth as a new Kaggle dataset named 'desert-seg-checkpoint'.\n"
        "Searched: " + ", ".join(str(p) for p in CHECKPOINT_SEARCH_DIRS)
    )

dest_ckpt = config.BEST_MODEL_PATH
dest_ckpt.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src_ckpt, dest_ckpt)
print(f"\nCheckpoint loaded: {src_ckpt}")
print(f"Copied to        : {dest_ckpt}")

# Also copy history.json if available (for training curves)
src_history = src_ckpt.parent.parent / "logs" / "history.json"
if not src_history.exists():
    src_history = src_ckpt.with_name("history.json")
if src_history.exists():
    shutil.copy2(src_history, config.HISTORY_PATH)
    print(f"History log copied: {src_history}")

from train import set_seed  # noqa: E402
set_seed(config.SEED)

# ─── 6. Run evaluation ──────────────────────────────────────────────────────
# Enable TTA on GPU for better accuracy
config.USE_TTA = CUDA_AVAILABLE
config.TTA_SCALES = [0.75, 1.0, 1.25] if CUDA_AVAILABLE else [1.0]

from evaluate import evaluate  # noqa: E402

print("\n" + "=" * 60)
print("RUNNING EVALUATION")
print("=" * 60)

results = evaluate()
print(f"\nFinal mean IoU: {results['mean_iou']:.4f}")

# ─── 6b. Run test inference (feature 5) ─────────────────────────────────────
from test import main as test_main  # noqa: E402

print("\n" + "=" * 60)
print("RUNNING TEST INFERENCE")
print("=" * 60)

test_main()

# ─── 6c. Run failure analysis ────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(CODE_ROOT))
from analyze_failures import run_failure_analysis  # noqa: E402

# Point failure analysis output to kaggle working dir
import analyze_failures as _af
_af.FAILURE_DIR = RUNS_DIR / "failure_analysis"
_af.FAILURE_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("RUNNING FAILURE ANALYSIS")
print("=" * 60)

run_failure_analysis()

# ─── 7. Plot training curves ────────────────────────────────────────────────
import json
import matplotlib.pyplot as plt

history_path = config.HISTORY_PATH
if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)

    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, history["val_mean_iou"], color="green", label="Val mIoU")
    axes[1].set_title("Validation Mean IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    curves_path = config.CURVES_PATH
    fig.savefig(curves_path, dpi=150)
    plt.show()
    print(f"Training curves saved to: {curves_path}")

# ─── 8. Package outputs for download ────────────────────────────────────────
import zipfile, datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
zip_path = WORKING_DIR / f"desert_seg_results_{timestamp}.zip"

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    # Best model checkpoint
    if config.BEST_MODEL_PATH.exists():
        zf.write(config.BEST_MODEL_PATH, "checkpoints/best_model.pth")
        print(f"Added checkpoint: {config.BEST_MODEL_PATH.stat().st_size / 1e6:.1f} MB")

    # All log files (history, eval results, curves)
    for log_file in config.LOG_DIR.iterdir():
        if log_file.is_file():
            zf.write(log_file, f"logs/{log_file.name}")

    # Failure analysis outputs
    failure_dir = RUNS_DIR / "failure_analysis"
    if failure_dir.exists():
        for ffile in failure_dir.rglob("*"):
            if ffile.is_file():
                zf.write(ffile, f"failure_analysis/{ffile.relative_to(failure_dir)}")
        print(f"Added failure analysis outputs from: {failure_dir}")

print(f"\nOutput zip created: {zip_path}")
print(f"Zip size: {zip_path.stat().st_size / 1e6:.1f} MB")
print("\nDownload it from: Kaggle notebook Output tab > Files panel")
print("Done!")
