# Running Desert Segmentation on Kaggle Free GPU

This guide walks you through uploading your project and dataset to Kaggle and
running the training notebook on a free T4 GPU.

---

## What You Will Get

- SegFormer B2 model trained for 50 epochs on a T4 GPU (16 GB VRAM)
- Expected mIoU: **0.68 – 0.75**
- Training time: **1 – 2 hours** (vs 10+ hours on your laptop CPU)
- A downloadable zip with the best checkpoint and evaluation results

---

## Step 1 — Create a Free Kaggle Account

Go to [kaggle.com](https://www.kaggle.com) and sign up if you do not have an
account. Verify your phone number — this is required to enable GPU access.

---

## Step 2 — Upload Your Dataset

You need to upload your image/mask data as a Kaggle Dataset.

### 2a. Create the dataset folder on your PC

Your data folder must have one of these two layouts:

**Layout A (what you already have):**
```
desert-segmentation/
  Offroad_Segmentation_Training_Dataset/
    train/
      Color_Images/      <- training images (.png)
      Segmentation/      <- training masks  (.png)
    val/
      Color_Images/
      Segmentation/
  Offroad_Segmentation_testImages/
    Color_Images/
```

**Layout B (generic):**
```
desert-segmentation/
  dataset/
    train/
      images/
      masks/
    val/
      images/
      masks/
    testImages/
      images/
```

The notebook auto-detects which layout you used.

### 2b. Upload to Kaggle

1. Go to **kaggle.com > Datasets > New Dataset**
2. Name it exactly: `desert-segmentation`
3. Drag your `desert-segmentation/` folder into the upload area
4. Click **Create** and wait for it to finish processing

---

## Step 3 — Upload Your Project Code

1. Zip the `segmentation_project/` folder from your PC:
   - Right-click `segmentation_project/` > Send to > Compressed (zipped) folder
   - Name the zip: `segmentation_project.zip`

2. Go to **kaggle.com > Datasets > New Dataset**
3. Name it exactly: `desert-seg-code`
4. Upload `segmentation_project.zip`
5. Click **Create**

The zip should contain these files at the top level inside the zip:
```
segmentation_project/
  config.py
  train.py
  evaluate.py
  model.py
  dataset.py
  loss.py
  visualize.py
  test.py
  app.py
  kaggle_notebook.py
  requirements.txt
```

---

## Step 4 — Create a New Kaggle Notebook

1. Go to **kaggle.com > Code > New Notebook**
2. In the top-right, click **Edit** (pencil icon) to rename it to something like
   `desert-segmentation-training`

### 4a. Enable GPU

1. In the right sidebar, click **Session options** (or the Settings gear)
2. Under **Accelerator**, select **GPU T4 x2** or **GPU P100**
3. Click **Save**

### 4b. Attach your datasets

In the right sidebar, click **Add data**:

1. Search for `desert-segmentation` (your dataset from Step 2)
   - Click **Add**
   - Verify the mount path shows: `/kaggle/input/desert-segmentation`

2. Search for `desert-seg-code` (your code dataset from Step 3)
   - Click **Add**
   - Verify the mount path shows: `/kaggle/input/desert-seg-code`

---

## Step 5 — Paste the Notebook Code

1. In the notebook editor, delete the default empty cell
2. Click **+ Code** to add a new code cell
3. Open `kaggle_notebook.py` from your PC in a text editor
4. Select all (Ctrl+A), copy, and paste into the Kaggle code cell

---

## Step 6 — Run the Notebook

Click **Run All** (the double-play button at the top) or press Shift+Enter on
each cell.

You will see output like:
```
GPU detected: Tesla T4
Model backend : segformer_b2
Image size    : 512
Batch size    : 8
Epochs        : 50

STARTING TRAINING
Epoch 1/50 | Train Loss: 1.23 | Val Loss: 1.18 | Val mIoU: 0.2341 | ...
...
RUNNING EVALUATION
Mean IoU (all classes): 0.7134
Output zip created: /kaggle/working/desert_seg_results_20240307_1423.zip
```

---

## Step 7 — Download Your Results

1. When training finishes, click the **Output** tab in the right sidebar
2. Find the file named `desert_seg_results_YYYYMMDD_HHMM.zip`
3. Click the download icon next to it
4. Unzip on your PC — it contains:
   - `checkpoints/best_model.pth` — your trained model
   - `logs/history.json` — loss and IoU per epoch
   - `logs/evaluation_results.json` — final metrics
   - `logs/per_class_iou.json` — per-class breakdown
   - `logs/training_curves.png` — loss/IoU plot

---

## Step 8 — Use the Checkpoint Locally

Copy `best_model.pth` into your local project:
```
segmentation_project/runs/checkpoints/best_model.pth
```

Then update `config.py` to use SegFormer B2 before running evaluate or app:
```python
MODEL_BACKEND = "segformer_b2"
```

Run evaluation locally:
```bash
cd segmentation_project
python evaluate.py
```

---

## Troubleshooting

**"Project source code not found"**
The `desert-seg-code` dataset was not attached or the zip structure is wrong.
Make sure the zip extracts to `segmentation_project/config.py` (not a nested
extra folder).

**"Image dataset not found"**
The `desert-segmentation` dataset was not attached, or the folder layout does
not match either Layout A or Layout B described in Step 2.

**Out of memory error**
The notebook automatically retries with half the batch size. If it still fails,
change `BATCH_SIZE = 4` and `IMAGE_SIZE = 384` near the top of the notebook.

**GPU not available**
You may have hit the weekly GPU quota (30 hours free). Wait until Monday for it
to reset, or use a Colab T4 GPU as an alternative.

**Import error on transformers / timm**
The pip install at the top of the notebook should handle this. If it fails,
add a separate cell: `!pip install transformers timm albumentations` and run it
before the main cell.

---

## Kaggle Free GPU Limits

| Resource | Limit |
|---|---|
| GPU hours per week | 30 hours |
| GPU type | T4 (16 GB VRAM) or P100 (16 GB) |
| Session length | 12 hours max |
| Disk (working) | 20 GB |
| Internet access | Enabled (for HuggingFace model download) |

50 epochs of SegFormer B2 at IMAGE_SIZE=512 uses roughly 1.5 – 2 GPU hours,
well within the free quota.
