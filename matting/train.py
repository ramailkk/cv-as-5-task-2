"""
train.py
─────────────────────────────────────────────────────────────────────────────
Train the human matting model on the AISegment dataset (Part A).

Usage:
    python train.py                        # reads config.yaml in same folder
    python train.py --config config.yaml   # explicit config path

Dataset:
    https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets
    Layout:  clip_img/**/*_clip.jpg  ↔  matting/**/*.png

Loss (documented):
    Total = l1_weight × L1(pred, gt)  +  dice_weight × Dice(pred, gt)
    Default weights: l1_weight=0.5, dice_weight=0.5

    • L1  → dense gradient at every pixel; preserves soft alpha values in
             semi-transparent transition regions (hair, fine edges).
    • Dice → IoU surrogate; corrects foreground/background pixel imbalance;
             directly targets the evaluation metric.

Target: val IoU ≥ 0.85 on the AISegment validation split.
"""

from __future__ import annotations

import os
import csv
import random
import argparse
import time
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import sys
from pathlib import Path

# Allow running from project root or from within the matting folder
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from matting.model   import build_matting_model
from matting.dataset import build_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
#  Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss — differentiable proxy for (1 − Dice coefficient).
    Operates on continuous predictions in [0, 1].
    """
    pred   = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter  = (pred * target).sum(dim=1)
    denom  = pred.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def matting_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    l1_w:   float,
    dice_w: float,
) -> torch.Tensor:
    """Combined L1 + Dice loss (see module docstring for rationale)."""
    l1   = nn.functional.l1_loss(pred, target)
    dice = dice_loss(pred, target)
    return l1_w * l1 + dice_w * dice


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_iou(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Binary IoU at a fixed binarisation threshold, averaged over the batch."""
    pred_bin   = (pred   > threshold).float()
    target_bin = (target > threshold).float()
    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(1, 2, 3))
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
#  Training / evaluation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg:       dict,
    device:    torch.device,
    epoch:     int,
    total_ep:  int,
) -> float:
    model.train()
    l1_w       = cfg["matting"]["l1_weight"]
    dice_w     = cfg["matting"]["dice_weight"]
    total_loss = 0.0
    n_samples  = 0

    it = (tqdm(loader, desc=f"Train [{epoch}/{total_ep}]", leave=False)
          if HAS_TQDM else loader)

    for imgs, mattes in it:
        imgs   = imgs.to(device,   non_blocking=True)
        mattes = mattes.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = matting_loss(preds, mattes, l1_w, dice_w)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n_samples  += bs
        if HAS_TQDM:
            it.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_samples


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg:    dict,
    device: torch.device,
    label:  str = "Val",
) -> tuple[float, float]:
    """Returns (mean_loss, mean_iou) over the loader."""
    model.eval()
    l1_w       = cfg["matting"]["l1_weight"]
    dice_w     = cfg["matting"]["dice_weight"]
    total_loss = 0.0
    total_iou  = 0.0
    n_samples  = 0

    it = (tqdm(loader, desc=f"  {label}", leave=False) if HAS_TQDM else loader)

    for imgs, mattes in it:
        imgs   = imgs.to(device,   non_blocking=True)
        mattes = mattes.to(device, non_blocking=True)
        preds  = model(imgs)
        loss   = matting_loss(preds, mattes, l1_w, dice_w)
        iou    = compute_iou(preds, mattes)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_iou  += iou * bs
        n_samples  += bs

    return total_loss / n_samples, total_iou / n_samples


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train human matting model — Part A (Assignment 5 Task 2)"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found: {cfg_path.resolve()}\n"
            "Run from the project root: python train.py"
        )
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Model ─────────────────────────────────────────────────────────────
    arch  = cfg["matting"]["architecture"]
    model = build_matting_model(arch).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : {arch}   trainable params = {params / 1e6:.2f} M")
    print(f"Loss   : {cfg['matting']['l1_weight']}×L1  +  "
          f"{cfg['matting']['dice_weight']}×Dice")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── Optimiser + LR scheduler ──────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["matting"]["lr"],
        weight_decay=cfg["matting"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # ── Output directory + checkpoint path ────────────────────────────────
    weights_path = Path(cfg["matting"]["weights_path"])
    os.makedirs(weights_path.parent, exist_ok=True)
    out_dir = Path(cfg["data"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    total_epochs = cfg["matting"]["epochs"]
    patience     = cfg["matting"]["patience"]
    best_iou     = 0.0
    patience_ctr = 0
    log_rows: list[dict] = []

    print(f"\nTraining for up to {total_epochs} epochs  "
          f"(early-stop patience={patience})\n")

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, cfg, device, epoch, total_epochs
        )
        val_loss, val_iou = evaluate(
            model, val_loader, cfg, device, label="Val"
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch:3d}/{total_epochs}]  "
            f"train={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_IoU={val_iou:.4f}  "
            f"lr={lr_now:.2e}  "
            f"({elapsed:.1f}s)"
        )

        scheduler.step(val_iou)
        log_rows.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss, 6),
            "val_loss"  : round(val_loss,   6),
            "val_iou"   : round(val_iou,    6),
            "lr"        : lr_now,
        })

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_iou > best_iou:
            best_iou     = val_iou
            patience_ctr = 0
            torch.save(model.state_dict(), weights_path)
            print(f"  ✓  Saved best checkpoint  (IoU={best_iou:.4f}) → {weights_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(val IoU did not improve for {patience} epochs)."
                )
                break

    # ── Save CSV training log ─────────────────────────────────────────────
    log_path = out_dir / "matting_train_log.csv"
    with open(log_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log → {log_path}")

    # ── Final test evaluation (best checkpoint) ───────────────────────────
    print("\nLoading best checkpoint for test evaluation …")
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    test_loss, test_iou = evaluate(
        model, test_loader, cfg, device, label="Test"
    )

    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  Test loss = {test_loss:.4f}")
    print(f"  Test IoU  = {test_iou:.4f}   (target ≥ 0.85)")
    print(f"  Best val IoU = {best_iou:.4f}")
    print(f"{bar}")

    if test_iou >= 0.85:
        print("  ✓  IoU ≥ 0.85 — target achieved!")
    else:
        print(f"  ⚠  IoU = {test_iou * 100:.1f}% < 85%.")
        print("  Suggestions to close the gap:")
        print("    1. Increase 'epochs' to 30-40 in config.yaml")
        print("    2. Increase 'train_size' to 10000+ (more data)")
        print("    3. Try 'mobilenet_decoder' architecture instead of 'unet'")
        print("    4. Raise 'dice_weight' to 0.7 to boost IoU directly")
        print("    5. Lower 'lr' to 5e-5 after first plateau")
        print("    6. Use the full AISegment dataset (~34k pairs)")

    print("\nDone.")


if __name__ == "__main__":
    main()
