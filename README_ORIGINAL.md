# Desert Offroad Semantic Segmentation

Semantic segmentation of desert and offroad terrain into 10 classes using a
two-phase strategy: a DeepLabV3 prototype for fast pipeline validation, then
SegFormer B2 for final accuracy. Trained on the Offroad Segmentation Dataset
in 1.8 hours over 40 epochs, reaching **mIoU 0.644**.

---

## Full Pipeline Diagram

```
 RAW DATA
 =========
 Color_Images/  +  Segmentation/
        |                |
        v                v
 +------+----------------+------+
 |        DesertDataset         |
 |  - raw pixel -> class index  |
 |  - resize to 512x512         |
 |  - ImageNet normalize        |
 |  - augment (train only):     |
 |    flip, color jitter,       |
 |    random crop, rotation     |
 +------------------------------+
              |
              v
 PHASE 1 - PROTOTYPE
 ====================
 DeepLabV3 + MobileNet V3 Large
   - torchvision pretrained
   - fast on CPU
   - validates data pipeline
   - establishes baseline mIoU
              |
              | (switch MODEL_BACKEND in config.py)
              v
 PHASE 2 - FINAL MODEL
 =======================
 SegFormer B2 (nvidia/mit-b2)
   - Hierarchical Transformer encoder
   - MLP decoder head (no positional encoding bias)
   - 85M parameters
   - ImageNet-22k pretrained weights
              |
              v
 TRAINING LOOP (train.py)
 ==========================
   Optimizer  : AdamW  (lr=3e-4, wd=1e-4)
   Scheduler  : CosineAnnealingLR  (T_max=50)
   Loss       : 0.7 x CrossEntropy + 0.3 x Dice
   Class wts  : [1.0, 3.5, 1.2, 1.3, 2.5, 4.5, 5.0, 2.0, 0.6, 0.4]
   AMP        : enabled (fp16 on CUDA)
   Grad clip  : norm=1.0
   Early stop : patience=15 epochs
   Max epochs : 80
              |
              v
 FINE-TUNING (fine_tune.py)
 ============================
   Loads best checkpoint
   Continues with tighter augmentation
   Tracks per-class IoU for re-weighting
              |
              v
 EVALUATION (evaluate.py)
 ==========================
   TTA (test-time augmentation)
   Confusion matrix -> per-class IoU
   Excludes ignore_index=255
   Saves: evaluation_results.json
          per_class_iou.json
              |
              v
 INFERENCE (test.py + visualize.py)
 ====================================
   Unlabeled test images
   Color-coded prediction overlays
   Saved to runs/predictions/
```

---

## Why Our Approach

### The Two-Phase Strategy

This project uses a deliberate two-phase methodology rather than jumping
directly to the heaviest model available.

**Phase 1 - DeepLabV3 MobileNet (Prototyping)**

The first phase uses `deeplabv3_mobilenet_v3_large` from torchvision.
This is a lightweight model that runs on CPU in minutes. The goal is not
accuracy — it is to:

- validate the full data pipeline end-to-end (loading, mask conversion,
  transforms, loss, metrics)
- catch label-mapping bugs early with a model that gives interpretable
  failure modes
- produce a concrete baseline mIoU to measure improvement against
- allow fast iteration on augmentation, class weighting, and loss design
  without waiting for GPU training runs

Switching between phases is a single line change in `config.py`:

```python
MODEL_BACKEND = "deeplabv3_mobilenet"  # Phase 1
MODEL_BACKEND = "segformer_b2"         # Phase 2
```

**Phase 2 - SegFormer B2 (Submission)**

Once the pipeline is validated, the backbone is swapped to
`SegformerForSemanticSegmentation` (nvidia/mit-b2). SegFormer was chosen
over alternatives for three reasons:

1. **No positional encoding in the decoder.** Desert scenes vary widely in
   camera angle and scale. Position-dependent decoders like SETR can
   overfit to spatial priors that do not generalize across terrain types.

2. **Hierarchical Mix Transformer encoder.** The B2 variant extracts
   multi-scale features at 1/4, 1/8, 1/16, and 1/32 resolution, giving
   the model sensitivity to both fine texture (rock surfaces, dry grass
   blades) and large regions (sky, landscape).

3. **Strong ImageNet-22k pretraining.** Transfer from a large, diverse
   pretraining corpus is critical when the desert dataset is relatively
   small. The mit-b2 weights provide robust low-level feature extraction
   from the first epoch.

**Class Weighting**

Desert imagery is dominated by Landscape and Sky pixels. Without
correction, models collapse to predicting these two classes. Per-class
weights in the combined loss penalize dominant classes less and rare
classes (Flowers: 4.5x, Logs: 5.0x) more, ensuring the model learns
meaningful boundaries throughout the scene.

---

## Key Innovations

**1. Two-phase backend switching via a single config flag**
The entire pipeline supports swapping between a lightweight CPU-friendly
DeepLab prototype and the full SegFormer B2 model without touching any
training code. This makes the methodology reproducible and the baseline
comparison honest.

**2. Combined loss with per-class weighting tuned to desert class imbalance**
The loss is a weighted sum of CrossEntropy (0.7) and Dice (0.3). Class
weights are tuned for the specific imbalance of desert/offroad imagery,
heavily upweighting rare classes like Logs (5.0x) and Flowers (4.5x)
that are visually important but statistically rare.

**3. Fine-tuning stage as a separate script**
After initial training converges, `fine_tune.py` resumes from the best
checkpoint with adjusted settings. This separates the exploration phase
from the refinement phase and avoids overwriting the initial best model.

**4. Test-time augmentation (TTA) at evaluation**
At inference, TTA applies horizontal flips and averages logits before
argmax. This consistently improves mIoU by 1-3 points on boundary-heavy
classes like Rocks and Logs with zero extra training cost.

**5. Checkpoint verification on load**
`evaluate.py` checks parameter sums before and after `load_state_dict`
and validates that the saved backend tag matches the current config.
This prevents silent evaluation bugs from mismatched checkpoints.

**6. Ignore-index-aware confusion matrix**
The confusion matrix excludes pixels labeled 255 (void/ignore) from
both precision and recall, so boundary regions and unlabeled pixels do
not artificially inflate or deflate any class's IoU.

---

## Environment & Dependency Requirements

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.9+
- CUDA-capable GPU recommended (training runs on CPU but will be slow)

### Setup

```bash
# Create and activate the conda environment
conda create -n EDU python=3.9 -y
conda activate EDU

# Install PyTorch (adjust CUDA version to match your driver)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install transformers accelerate
pip install numpy pillow matplotlib scikit-learn tqdm
pip install opencv-python
```

> **Windows users:** You can also run the `ENV_SETUP/setup_env.bat` script from an Anaconda Prompt if it was included with the dataset download.
>
> **Mac/Linux users:** Run the equivalent shell commands above or create a `setup_env.sh` with the same steps.

### Dataset Layout Expected

Place the downloaded dataset so the directories match this layout (already shown in Project Structure below):

```
desert segmentation/
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/  (Color_Images/ + Segmentation/)
│   └── val/    (Color_Images/ + Segmentation/)
└── Offroad_Segmentation_testImages/
    └── Color_Images/
```

---

## Reproducing Results in One Command

```bash
python segmentation_project/train.py
```

That single command:

1. Detects GPU/CPU automatically
2. Loads the Offroad training dataset from `Offroad_Segmentation_Training_Dataset/`
3. Downloads `nvidia/mit-b2` pretrained weights on first run
4. Trains for up to 80 epochs with early stopping
5. Saves the best checkpoint to `segmentation_project/runs/checkpoints/best_model.pth`
6. Writes training curves to `segmentation_project/runs/logs/training_curves.png`

To then evaluate and generate prediction overlays:

```bash
python segmentation_project/evaluate.py
python segmentation_project/visualize.py
```

To run on test images (no masks required):

```bash
python segmentation_project/test.py
```

---

## Results

### Summary

| Metric | Baseline (epoch 5) | Final (epoch 40) | Delta |
|---|---|---|---|
| Mean IoU (all classes) | 0.4103 | **0.6442** | +23.4 pts |
| Mean IoU (excl. Sky/Landscape) | ~0.35 | **0.6035** | +25 pts |
| Val Loss | 1.2103 | **1.0427** | -0.17 |
| Training time | — | **1.8 hrs** | 40 epochs |

### Per-Class IoU

| ID | Name | Val IoU | Status | Pixel% | Class Weight |
|---|---|---|---|---|---|
| 0 | Trees | **0.857** | GOOD | 4.1% | 1.0 |
| 1 | Lush Bushes | **0.684** | GOOD | 6.0% | 3.5 |
| 2 | Dry Grass | **0.702** | GOOD | 19.3% | 1.2 |
| 3 | Dry Bushes | 0.511 | OK | 1.1% | 1.3 |
| 4 | Ground Clutter | 0.402 | OK | 4.2% | 2.5 |
| 5 | Flowers | **0.621** | GOOD | 2.4% | 4.5 |
| 6 | Logs | 0.520 | OK | 0.07% | 5.0 |
| 7 | Rocks | 0.532 | OK | 1.2% | 2.0 |
| 8 | Landscape | **0.631** | GOOD | 23.7% | 0.6 |
| 9 | Sky | **0.982** | GOOD | 37.8% | 0.4 |
| — | **Mean** | **0.644** | — | — | — |
| — | Mean (excl. Sky+Landscape) | **0.604** | — | — | — |

### Comparison Slider Preview

```
[  GIF PLACEHOLDER  ]

Description: Side-by-side slider GIF showing the raw desert scene on the
left sliding into the color-coded segmentation overlay on the right.
Each of the 10 semantic classes appears in a distinct color:

  Trees          (dark green)     Lush Bushes   (medium green)
  Dry Grass      (khaki)          Dry Bushes    (yellow-green)
  Ground Clutter (orange)         Flowers       (hot pink)
  Logs           (brown)          Rocks         (slate gray)
  Landscape      (tan)            Sky           (light blue)

To generate this GIF:
  python segmentation_project/visualize.py

The script writes overlaid prediction PNGs to runs/predictions/.
Stitch them into a slider GIF with any tool (e.g. ffmpeg or ezgif).
```

---

## Expected Outputs & How to Interpret Them

### After `train.py`

| Output | Location | What it means |
|---|---|---|
| `best_model.pth` | `runs/checkpoints/` | Saved weights from the epoch with the lowest validation loss. Use this for evaluation and inference. |
| `history.json` | `runs/logs/` | Per-epoch train loss, val loss, and mIoU. Load with any JSON viewer or Python to plot custom curves. |
| `training_curves.png` | `runs/logs/` | Auto-generated plot of loss and mIoU over epochs. A healthy run shows loss decreasing and mIoU rising before plateau. |

**Reading the loss curve:**
- Loss steadily decreasing → model is learning normally.
- Loss plateauing high → possible underfitting; try more epochs or lower weight decay.
- Val loss rising while train loss falls → overfitting; reduce augmentation strength or add dropout.

### After `evaluate.py`

| Output | Location | What it means |
|---|---|---|
| `evaluation_results.json` | `runs/logs/` | Top-level summary: overall mIoU, mean pixel accuracy, per-class IoU dict. |
| `per_class_iou.json` | `runs/logs/` | Same per-class breakdown in a flat structure for easy comparison across runs. |
| Console confusion matrix | stdout | Rows = ground truth class, columns = predicted class. Off-diagonal entries reveal which classes are confused with each other. |

**Interpreting IoU scores:**
- IoU > 0.7 → strong class performance.
- IoU 0.4–0.7 → acceptable; room for improvement via augmentation or re-weighting.
- IoU < 0.4 → class is underperforming; check pixel frequency and consider raising its class weight in `config.py`.

### After `test.py` / `visualize.py`

| Output | Location | What it means |
|---|---|---|
| Color overlay PNGs | `runs/predictions/` | Each test image with predicted class colors overlaid. Use these to do visual failure-case analysis. |

**Color legend for overlays:**

| Class | Color |
|---|---|
| Trees | Dark green |
| Lush Bushes | Medium green |
| Dry Grass | Khaki |
| Dry Bushes | Yellow-green |
| Ground Clutter | Orange |
| Flowers | Hot pink |
| Logs | Brown |
| Rocks | Slate gray |
| Landscape | Tan |
| Sky | Light blue |

Regions shown in an unexpected color indicate misclassification — compare against the input RGB image to identify systematic failure modes (e.g., shadowed rocks classified as Dry Bushes).

---

## Project Structure

```
desert segmentation/
├── README.md
├── train_segmentation.py          # DINOv2 + ConvNeXt head experiment
├── visualize.py                   # Standalone quick visualizer
├── test_segmentation.py           # Quick inference script
├── segmentation_project/
│   ├── config.py                  # All hyperparameters and paths
│   ├── dataset.py                 # DesertDataset + transforms
│   ├── model.py                   # DeepLabV3 / SegFormer loader
│   ├── loss.py                    # CombinedLoss (CE + Dice)
│   ├── train.py                   # Main training loop
│   ├── fine_tune.py               # Resume + refine from checkpoint
│   ├── evaluate.py                # Validation metrics + confusion matrix
│   ├── test.py                    # Unlabeled test inference
│   ├── visualize.py               # Prediction overlay generator
│   ├── report_generator.py        # Auto-generate results report
│   ├── app.py                     # Interactive demo app
│   └── runs/
│       ├── checkpoints/           # best_model.pth
│       ├── logs/                  # history.json, evaluation_results.json
│       └── predictions/           # Color overlay PNGs
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
└── Offroad_Segmentation_testImages/
    └── Color_Images/
```
