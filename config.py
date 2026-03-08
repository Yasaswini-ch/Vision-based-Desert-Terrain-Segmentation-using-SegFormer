"""Central configuration for the desert semantic segmentation project.

This module stores all dataset paths, model hyperparameters, and runtime settings
used across training, evaluation, and inference scripts.
"""




from __future__ import annotations

from pathlib import Path



# Project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Data paths
_OFFROAD_TRAINING_ROOT: Path = PROJECT_ROOT.parent / "Offroad_Segmentation_Training_Dataset"
_OFFROAD_TEST_ROOT: Path = PROJECT_ROOT.parent / "Offroad_Segmentation_testImages"
_DEFAULT_DATASET_ROOT: Path = PROJECT_ROOT.parent / "dataset"

if _OFFROAD_TRAINING_ROOT.exists():
    TRAIN_IMAGES_DIR: Path = _OFFROAD_TRAINING_ROOT / "train" / "Color_Images"
    TRAIN_MASKS_DIR: Path = _OFFROAD_TRAINING_ROOT / "train" / "Segmentation"
    VAL_IMAGES_DIR: Path = _OFFROAD_TRAINING_ROOT / "val" / "Color_Images"
    VAL_MASKS_DIR: Path = _OFFROAD_TRAINING_ROOT / "val" / "Segmentation"
    TEST_IMAGES_DIR: Path = _OFFROAD_TEST_ROOT / "Color_Images"
    TEST_MASKS_DIR: Path | None = None
else:
    DATASET_ROOT: Path = _DEFAULT_DATASET_ROOT
    TRAIN_IMAGES_DIR = DATASET_ROOT / "train" / "images"
    TRAIN_MASKS_DIR = DATASET_ROOT / "train" / "masks"
    VAL_IMAGES_DIR = DATASET_ROOT / "val" / "images"
    VAL_MASKS_DIR = DATASET_ROOT / "val" / "masks"
    TEST_IMAGES_DIR = DATASET_ROOT / "testImages" / "images"
    TEST_MASKS_DIR = None

# Output paths
RUNS_DIR: Path = PROJECT_ROOT / "runs"
CHECKPOINT_DIR: Path = RUNS_DIR / "checkpoints"
LOG_DIR: Path = RUNS_DIR / "logs"
PREDICTIONS_DIR: Path = RUNS_DIR / "predictions"
BEST_MODEL_PATH: Path = CHECKPOINT_DIR / "best_model.pth"
HISTORY_PATH: Path = LOG_DIR / "history.json"
EVAL_RESULTS_PATH: Path = LOG_DIR / "evaluation_results.json"
CURVES_PATH: Path = LOG_DIR / "training_curves.png"

# Class mapping: raw mask value -> model class index
RAW_TO_CLASS_INDEX: dict[int, int] = {
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    600: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9,
}

CLASS_NAMES: list[str] = [
    "Trees",
    "Lush Bushes",
    "Dry Grass",
    "Dry Bushes",
    "Ground Clutter",
    "Flowers",
    "Logs",
    "Rocks",
    "Landscape",
    "Sky",
]

NUM_CLASSES: int = 10
IGNORE_INDEX: int = 255

# Training setup
IMAGE_SIZE: int = 512
BATCH_SIZE: int = 4
FALLBACK_BATCH_SIZE: int = 4
NUM_WORKERS: int = 0
PIN_MEMORY: bool = True
EPOCHS: int = 80
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 1e-4
GRAD_CLIP_NORM: float = 1.0
USE_AMP: bool = True
EARLY_STOPPING_PATIENCE: int = 15
LABEL_SMOOTHING: float = 0.1
USE_TTA: bool = False  # Set True only on GPU; doubles inference time on CPU
# Model backend switch:
# MODEL_BACKEND = "deeplabv3_mobilenet"  # Phase 1 - fast CPU prototyping
# MODEL_BACKEND = "segformer_b2"         # Phase 2 - final submission
MODEL_BACKEND = "segformer_b2"
HF_MODEL_NAME: str = "nvidia/mit-b2"
CPU_QUICK_MODE: bool = False
CPU_QUICK_IMAGE_SIZE: int = 256
CPU_QUICK_EPOCHS: int = 5
CPU_QUICK_NUM_WORKERS: int = 0

# Loss weights
CE_WEIGHT: float = 0.7
DICE_WEIGHT: float = 0.3
CLASS_WEIGHTS: list[float] = [1.0, 3.5, 1.2, 1.3, 2.5, 4.5, 5.0, 2.0, 0.6, 0.4]

# Scheduler
SCHEDULER_T_MAX: int = 50

# Multi-scale TTA scales (used in evaluate.py when USE_TTA=True).
# Use [1.0] for fast flip-only TTA on CPU. Use [0.75, 1.0, 1.25] on GPU for full benefit.
TTA_SCALES: list[float] = [1.0]

# Copy-paste augmentation for rare classes during training
USE_COPY_PASTE: bool = True
COPY_PASTE_PROB: float = 0.5
COPY_PASTE_RARE_CLASSES: list[int] = [3, 5, 6]  # Dry Bushes, Flowers, Logs

# Ensemble: list of checkpoint paths to average during ensemble.py evaluation.
# Add more paths after training additional models (different seeds or backbones).
ENSEMBLE_CHECKPOINT_PATHS: list[Path] = [BEST_MODEL_PATH]

# MC Dropout uncertainty estimation (uncertainty.py)
# Use 3-5 passes on CPU for a quick demo; set to 20+ on GPU for publication-quality maps.
MC_DROPOUT_PASSES: int = 3
# Limit images processed locally (None = all). Set to 20 for fast CPU demo.
UNCERTAINTY_MAX_IMAGES: int | None = 20
DENSECRF_MAX_IMAGES: int | None = 20
UNCERTAINTY_DIR: Path = RUNS_DIR / "uncertainty"

# DenseCRF post-processing output directory (densecrf.py)
DENSECRF_DIR: Path = RUNS_DIR / "densecrf"

# Reproducibility
SEED: int = 42

# Inference colors (RGB) for visualization
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (34, 139, 34),
    1: (60, 179, 113),
    2: (189, 183, 107),
    3: (154, 205, 50),
    4: (255, 140, 0),
    5: (255, 20, 147),
    6: (139, 69, 19),
    7: (112, 128, 144),
    8: (210, 180, 140),
    9: (135, 206, 235),
}


def ensure_output_dirs() -> None:
    """Create all output directories if they do not already exist."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
    DENSECRF_DIR.mkdir(parents=True, exist_ok=True)


def validate_dataset_dirs(require_masks_for_test: bool = False) -> None:
    """Validate that required dataset directories exist.

    Args:
        require_masks_for_test: If True, also requires TEST_MASKS_DIR to exist.

    Raises:
        FileNotFoundError: If any required directory is missing.
    """
    required_dirs = [
        TRAIN_IMAGES_DIR,
        TRAIN_MASKS_DIR,
        VAL_IMAGES_DIR,
        VAL_MASKS_DIR,
        TEST_IMAGES_DIR,
    ]

    if require_masks_for_test and TEST_MASKS_DIR is not None:
        required_dirs.append(TEST_MASKS_DIR)

    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "Missing required dataset directories:\n"
            f"{joined}\n"
            "Update paths in config.py to match your local dataset layout."
        )
