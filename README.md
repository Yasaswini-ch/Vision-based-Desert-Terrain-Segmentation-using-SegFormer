# Desert Segmentation Project

Production-ready semantic segmentation pipeline for the Duality AI Offroad Hackathon.

## Training Performance Summary

**Model:** SegFormer B2  
**Epochs:** 40  
**Training Time:** 3.5 hours  
**Final Validation mIoU:** 0.6352  
**Final Validation Loss:** 1.05

### Training Metrics
- **Training Loss:** 1.27 → 0.99 (convergence across 40 epochs)
- **Validation Loss:** 1.15 → 1.05 (steady improvement)
- **Validation mIoU:** 0.5355 → 0.6352 (18.6% improvement)

See `runs/logs/training_curves.png` for detailed loss and mIoU curves.

## 1. Install Dependencies

```bash
cd segmentation_project
python -m pip install -r requirements.txt
```

## 2. Organize Dataset

Default expected layout:

```text
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── testImages/
    └── images/
```

If your folders differ, update paths in `config.py`.

## 3. Run Training

```bash
cd segmentation_project
python train.py
```

Model backend switch (`config.py`):

```python
MODEL_BACKEND = "deeplabv3_mobilenet"  # Phase 1 - fast CPU prototyping
# MODEL_BACKEND = "segformer_b2"       # Phase 2 - final submission
```

What training does:
- Loads the selected backend (`deeplabv3_mobilenet` or `segformer_b2`).
- Uses weighted `0.7 * CrossEntropy + 0.3 * DiceLoss`.
- Applies Albumentations augmentations on train split only.
- Tracks `train_loss`, `val_loss`, `val_mean_iou` every epoch.
- Saves best checkpoint by validation mean IoU.

Expected outputs:
- `runs/checkpoints/best_model.pth`
- `runs/logs/history.json`

## 4. Run Evaluation (Validation Set)

```bash
cd segmentation_project
python evaluate.py
```

Printed metrics:
- Overall mean IoU
- Per-class IoU (all 10 classes)
- Confusion matrix

Saved output:
- `runs/logs/evaluation_results.json`

## 5. Run Test Inference

```bash
cd segmentation_project
python test.py
```

What it does:
- Loads best checkpoint.
- Runs inference on `testImages/images`.
- Saves colorized segmentation predictions.
- If test masks are configured in `config.py`, also prints test mean IoU.

Saved output:
- `runs/predictions/*.png`

## 6. Plot Training Curves

```bash
cd segmentation_project
python visualize.py
```

Generates side-by-side visualization of:
- **Loss plot:** Training vs. Validation loss across epochs
- **Validation mIoU plot:** Model performance improvement over time

Saved output:
- `runs/logs/training_curves.png`

## 7. Training on Kaggle (Faster GPU)

For faster training with a free **NVIDIA T4 GPU** (16 GB VRAM):

```bash
cd segmentation_project
python kaggle_notebook.py
```

Or follow the manual Kaggle guide: [README_KAGGLE.md](README_KAGGLE.md)

**Expected Results on Kaggle:**
- Model: SegFormer B2 (50 epochs)
- Expected mIoU: **0.68 – 0.75**
- Training time: **1 – 2 hours** (vs 3.5 hours on laptop)
- Output: Best checkpoint + evaluation results

## 8. Launch Beautiful Inference UI (Optional)

```bash
cd segmentation_project
streamlit run app.py
```

UI features:
- Upload image and run segmentation inference.
- View input, prediction, and adjustable overlay side-by-side.
- Built-in class color legend for fast interpretation.

## IoU Interpretation

- `mIoU >= 0.70`: very strong domain generalization for off-road scenes.
- `0.55 <= mIoU < 0.70`: competitive baseline with room for improvement.
- `mIoU < 0.55`: likely underfitting, domain gap, or class imbalance issues.

Use per-class IoU to identify weak classes (for example `Flowers`, `Logs`) and adjust class weights/augmentation strategy.

## File Overview

- `config.py`: all paths and hyperparameters
- `dataset.py`: dataset loader + mask remapping + transforms
- `model.py`: model construction and classifier replacement
- `loss.py`: Dice + combined weighted loss
- `train.py`: full training loop with AMP and checkpointing
- `evaluate.py`: validation metrics and confusion matrix
- `test.py`: inference and colorized predictions
- `visualize.py`: training curve plotting
- `app.py`: Streamlit UI for interactive inference
- `requirements.txt`: pinned dependencies
