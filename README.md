# CV-AS-5-TASK2 · Assignment 5 Task 2 — NST Video Pipeline

## Part A: Human Matting Model

Train a U-Net (or MobileNetV2 decoder) from scratch on the
[AISegment Matting Human Dataset](https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets)
to predict a single-channel alpha matte ∈ [0, 1] from an RGB portrait image.

| Item | Detail |
|---|---|
| Architecture | U-Net (~31 M params) or MobileNetDecoder (~6.6 M params) |
| Input | RGB frame, 320 × 320 |
| Output | Alpha matte (1 × H × W), sigmoid ∈ [0, 1] |
| Loss | `0.5 × L1 + 0.5 × Dice` |
| Optimiser | Adam, lr = 1e-4 |
| Epochs | 25 (min 20 per spec) |
| Target | Val IoU ≥ 0.85 |

---

## Repository Layout

```
CV-AS-5-TASK2/
├── model.py          # UNet + MobileNetDecoder architectures
├── dataset.py        # AISegmentDataset + build_dataloaders
├── train.py          # Full training loop with CSV logging
├── evaluate.py       # Test-split metrics (IoU, F1, MAE, …)
├── predict.py        # Inference on images / directories
├── visualise.py      # Plot training curves + prediction grids
├── config.yaml       # All hyperparameters & paths
├── requirements.txt  # pip dependencies
└── kaggle_train.ipynb  # ← Run this on Kaggle
```

---

## Running on Kaggle (recommended — T4 GPU)

### Step 1 — Add the dataset
In your Kaggle notebook:
**Data → Add Dataset → search `aisegmentcom-matting-human-datasets`** (by laurentmih)

### Step 2 — Enable GPU
**Settings → Accelerator → GPU T4 x1**  
Enable **"Internet"** (needed to clone this repo).

### Step 3 — Open `kaggle_train.ipynb`
Upload `kaggle_train.ipynb` to your Kaggle notebook, **or** use
*File → Import Notebook* and point to this repo.

### Step 4 — Set your repo URL
In **Cell 3** of the notebook, change:
```python
GITHUB_REPO = "https://github.com/YOUR_USERNAME/CV-AS-5-TASK2"
```

### Step 5 — Run All Cells
**Run → Run All**

The notebook will:
1. Clone this repo to `/kaggle/working/repo`
2. Install `requirements.txt`
3. Verify the dataset is mounted at `/kaggle/input/aisegmentcom-matting-human-datasets`
4. Write a `kaggle_config.yaml` (overrides worker count to 4 for the T4)
5. Train via `train.py` — streams epoch logs in real-time
6. Evaluate on test split via `evaluate.py`
7. Plot training curves + sample predictions via `visualise.py`
8. Zip all outputs → `part_a_results.zip` in the Output panel

---

## Running Locally

```bash
# 1. Install
pip install -r requirements.txt

# 2. Edit config.yaml — change dataset_root to your local path
#    e.g.  dataset_root: "C:/data/AISegment"

# 3. Train
python train.py --config config.yaml

# 4. Evaluate
python evaluate.py --weights outputs/matting_weights.pth --config config.yaml

# 5. Predict on an image
python predict.py --weights outputs/matting_weights.pth --input photo.jpg --visualise

# 6. Plot training curves
python visualise.py curves --log outputs/matting_train_log.csv
```

---

## Loss Design

```
Total = l1_weight × L1(pred, gt)  +  dice_weight × Dice(pred, gt)
      =     0.5   × L1            +      0.5      × Dice
```

- **L1** — dense gradient at every pixel, preserves soft alpha values in
  semi-transparent transition regions (hair, fine edges).
- **Dice** — IoU surrogate; corrects foreground/background pixel imbalance;
  directly optimises the evaluation metric. Equal 0.5/0.5 weighting is a
  well-validated default; raise `dice_weight` to 0.7 if val IoU plateaus below 0.80.

---

## Smoke Test

```bash
python model.py
# Running model smoke test …
#   unet                  params=31.38M   [OK]
#   mobilenet_decoder     params=6.62M    [OK]
# All smoke tests passed.
```
