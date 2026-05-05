"""
visualise.py
─────────────────────────────────────────────────────────────────────────────
Visualisation utilities for Part A — Human Matting.

Two modes:
  1. Plot training curves from the CSV log saved by train.py
     python visualise.py --mode curves --log outputs/matting_train_log.csv

  2. Run the trained model on random val images and show a side-by-side grid
     python visualise.py --mode predictions \
         --weights outputs/matting_weights.pth \
         --config  config.yaml \
         --n 8

Both modes save a PNG to --out (default: outputs/).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── Matplotlib backend that works headlessly on Kaggle / servers ─────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
#  Mode 1: Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(log_csv: Path, out_path: Path) -> None:
    """Parse the CSV written by train.py and plot loss + IoU curves."""
    epochs, train_loss, val_loss, val_iou = [], [], [], []

    with open(log_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_iou.append(float(row["val_iou"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Human Matting — Training Curves", fontsize=14, fontweight="bold")

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train loss", color="#4C72B0", linewidth=2)
    ax.plot(epochs, val_loss,   label="Val loss",   color="#DD8452", linewidth=2)
    best_ep = epochs[int(np.argmin(val_loss))]
    ax.axvline(best_ep, linestyle="--", color="grey", linewidth=1, label=f"Best epoch={best_ep}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss  (L1 + Dice)")
    ax.set_title("Loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # ── IoU ───────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, val_iou, label="Val IoU", color="#55A868", linewidth=2)
    best_iou = max(val_iou)
    best_ep2 = epochs[val_iou.index(best_iou)]
    ax.axvline(best_ep2, linestyle="--", color="grey", linewidth=1,
               label=f"Best epoch={best_ep2}")
    ax.axhline(0.85, linestyle=":", color="red", linewidth=1.5, label="Target IoU=0.85")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU  (threshold=0.5)")
    ax.set_title("Validation IoU")
    ax.set_ylim(0, 1)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curve plot saved → {out_path}")
    print(f"  Best val IoU : {best_iou:.4f}  (epoch {best_ep2})")


# ─────────────────────────────────────────────────────────────────────────────
#  Mode 2: Sample predictions grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    weights_path: Path,
    cfg: dict,
    out_path: Path,
    n: int = 8,
    device: torch.device | None = None,
) -> None:
    """
    Load the trained model, grab n random val images, and show a 3-column grid:
        Column 1 — Input RGB
        Column 2 — Ground-truth matte
        Column 3 — Predicted matte
    """
    import yaml

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from model   import build_matting_model
    from dataset import AISegmentDataset

    arch  = cfg["matting"]["architecture"]
    model = build_matting_model(arch)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    size = tuple(cfg["matting"]["input_size"])  # (H, W)
    ds   = AISegmentDataset(cfg["data"]["dataset_root"], input_size=size, augment=False)

    indices = random.sample(range(len(ds)), min(n, len(ds)))

    fig = plt.figure(figsize=(10, n * 3))
    fig.suptitle(
        f"Human Matting — Sample Predictions  ({arch})",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(n, 3, figure=fig, wspace=0.05, hspace=0.15)
    col_titles = ["Input RGB", "Ground Truth α", "Predicted α"]

    to_pil = transforms.ToPILImage()

    for row, idx in enumerate(indices):
        img_t, matte_t = ds[idx]                    # (3,H,W) and (1,H,W)

        with torch.no_grad():
            pred_t = model(img_t.unsqueeze(0).to(device))   # (1,1,H,W)
        pred_t = pred_t.squeeze(0).cpu()             # (1,H,W)

        panels = [
            to_pil(img_t),                          # RGB image
            to_pil(matte_t),                        # ground-truth (L)
            to_pil(pred_t),                         # prediction (L)
        ]
        cmaps = [None, "gray", "gray"]

        for col, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(np.array(panel), cmap=cmap, vmin=0, vmax=255 if cmap else None)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction grid saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualisation for Part A matting model"
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # ── curves subcommand ──────────────────────────────────────────────────
    pc = sub.add_parser("curves", help="Plot training loss / IoU curves")
    pc.add_argument("--log", required=True,
                    help="Path to matting_train_log.csv from train.py")
    pc.add_argument("--out", default=None,
                    help="Output PNG path (default: <log_dir>/training_curves.png)")

    # ── predictions subcommand ─────────────────────────────────────────────
    pp = sub.add_parser("predictions", help="Show sample model predictions")
    pp.add_argument("--weights", required=True,
                    help="Path to trained checkpoint .pth")
    pp.add_argument("--config",  default="config.yaml",
                    help="Config YAML (default: config.yaml)")
    pp.add_argument("--n",       type=int, default=8,
                    help="Number of sample images to show (default: 8)")
    pp.add_argument("--out",     default=None,
                    help="Output PNG path (default: outputs/sample_predictions.png)")
    pp.add_argument("--device",  default=None,
                    help="Force device: 'cpu' | 'cuda'")

    args = ap.parse_args()

    if args.mode == "curves":
        log_path = Path(args.log)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        out_path = Path(args.out) if args.out \
                   else log_path.parent / "training_curves.png"
        plot_curves(log_path, out_path)

    elif args.mode == "predictions":
        import yaml
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)

        weights_path = Path(args.weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        out_path = Path(args.out) if args.out \
                   else Path(cfg["data"]["output_dir"]) / "sample_predictions.png"
        device = torch.device(
            args.device if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        plot_predictions(weights_path, cfg, out_path, n=args.n, device=device)


if __name__ == "__main__":
    main()
