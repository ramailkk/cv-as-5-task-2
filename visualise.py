"""
visualise.py
─────────────────────────────────────────────────────────────────────────────
Visualisation utilities for Part A — Human Matting.

Sub-commands
────────────
  curves      Plot training loss / IoU curves from the CSV log
              python visualise.py curves --log outputs/matting_train_log.csv

  predictions Show 5-sample frame grid  (4 columns per row):
                  Input RGB | Ground-truth α | Predicted α | Cutout
              python visualise.py predictions \\
                  --weights outputs/matting_weights.pth \\
                  --config  config.yaml \\
                  --n 5

Both sub-commands save a PNG to --out.
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

# Headless backend — works on Kaggle / servers without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cutout(
    img_arr:   np.ndarray,   # (H, W, 3)  uint8
    alpha_arr: np.ndarray,   # (H, W)     uint8  [0–255]
    bg_color:  tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Composite the subject over a solid background using the predicted alpha.
        cutout = α * RGB  +  (1−α) * bg_color
    Returns uint8 (H, W, 3).
    """
    a = alpha_arr.astype(np.float32) / 255.0          # (H, W) ∈ [0,1]
    a3 = a[:, :, np.newaxis]                           # (H, W, 1)
    bg = np.full_like(img_arr, bg_color, dtype=np.float32)
    out = a3 * img_arr.astype(np.float32) + (1.0 - a3) * bg
    return np.clip(out, 0, 255).astype(np.uint8)


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

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Human Matting — Training Curves  (Part A)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    PALETTE = {"train": "#4C72B0", "val": "#DD8452", "iou": "#55A868",
               "best": "#6B6B6B", "target": "#C44E52"}

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#FEFEFE")
    ax.plot(epochs, train_loss, label="Train loss", color=PALETTE["train"],
            linewidth=2, zorder=3)
    ax.plot(epochs, val_loss,   label="Val loss",   color=PALETTE["val"],
            linewidth=2, zorder=3)
    best_ep = epochs[int(np.argmin(val_loss))]
    ax.axvline(best_ep, linestyle="--", color=PALETTE["best"],
               linewidth=1.2, label=f"Best epoch = {best_ep}", zorder=2)
    ax.fill_between(epochs, train_loss, val_loss, alpha=0.07,
                    color=PALETTE["val"])
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss  (0.5·L1 + 0.5·Dice)", fontsize=10)
    ax.set_title("Loss", fontsize=11, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")

    # ── IoU ───────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#FEFEFE")
    ax.plot(epochs, val_iou, label="Val IoU", color=PALETTE["iou"],
            linewidth=2.5, zorder=3)
    best_iou = max(val_iou)
    best_ep2 = epochs[val_iou.index(best_iou)]
    ax.axvline(best_ep2, linestyle="--", color=PALETTE["best"],
               linewidth=1.2, label=f"Best epoch = {best_ep2}", zorder=2)
    ax.axhline(0.85, linestyle=":", color=PALETTE["target"],
               linewidth=2, label="Target IoU = 0.85", zorder=2)
    ax.fill_between(epochs, val_iou, 0, alpha=0.12, color=PALETTE["iou"])

    # Annotate the peak IoU
    ax.annotate(
        f"  {best_iou:.4f}",
        xy=(best_ep2, best_iou),
        fontsize=9, color=PALETTE["iou"], fontweight="bold",
        va="bottom",
    )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("IoU  (threshold = 0.5)", fontsize=10)
    ax.set_title("Validation IoU", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Training curve plot saved -> {out_path}")
    print(f"  Best val IoU : {best_iou:.4f}  (epoch {best_ep2})")


# ─────────────────────────────────────────────────────────────────────────────
#  Mode 2: 5-sample frame visualisation
#  Columns: Input RGB | Ground-Truth α | Predicted α | Cutout
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    weights_path: Path,
    cfg:          dict,
    out_path:     Path,
    n:            int = 5,
    device:       torch.device | None = None,
    test_iou:     float | None = None,
) -> None:
    """
    Load the trained model, pick n random dataset samples, and produce a grid:

        Row 0..n-1, Columns:
            0  Input RGB          — original portrait
            1  Ground-truth α     — dataset matte (grayscale)
            2  Predicted α        — model output  (grayscale)
            3  Cutout             — subject composited over white background

    Parameters
    ----------
    weights_path : path to saved .pth checkpoint
    cfg          : loaded config dict
    out_path     : where to save the PNG
    n            : number of sample rows (default 5)
    device       : torch device (auto-detected if None)
    test_iou     : if provided, annotates the figure with the test-split IoU
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from model   import build_matting_model
    from dataset import AISegmentDataset

    # ── Load model ────────────────────────────────────────────────────────
    arch  = cfg["matting"]["architecture"]
    model = build_matting_model(arch)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # ── Dataset ───────────────────────────────────────────────────────────
    size = tuple(cfg["matting"]["input_size"])   # (H, W)
    ds   = AISegmentDataset(
        cfg["data"]["dataset_root"], input_size=size, augment=False
    )
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    to_pil = transforms.ToPILImage()

    # ── Figure layout ─────────────────────────────────────────────────────
    N_COLS   = 4
    col_w    = 2.8                              # inches per column
    row_h    = 2.8                              # inches per row
    header_h = 0.55                            # space for column headers
    iou_h    = 0.50 if test_iou is not None else 0.0

    fig_w = col_w * N_COLS + 0.3
    fig_h = row_h * n + header_h + iou_h + 0.2

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#1A1A2E")

    COL_TITLES  = ["Input RGB", "Ground-truth α", "Predicted α", "Cutout"]
    COL_COLORS  = ["#E2E8F0",  "#A0AEC0",         "#68D391",      "#FBD38D"]

    # Title
    title_txt = f"Human Matting — 5 Sample Frames  ({arch})"
    if test_iou is not None:
        title_txt += f"          Test IoU = {test_iou:.4f}"

    fig.text(
        0.5, 1.0 - 0.01,
        title_txt,
        ha="center", va="top",
        fontsize=12, fontweight="bold", color="white",
        transform=fig.transFigure,
    )

    # Column headers
    for c, (title, color) in enumerate(zip(COL_TITLES, COL_COLORS)):
        x_center = (c + 0.5) / N_COLS
        fig.text(
            x_center,
            1.0 - (header_h / fig_h) - 0.01,
            title,
            ha="center", va="top",
            fontsize=10, fontweight="bold", color=color,
            transform=fig.transFigure,
        )

    # Grid of axes
    gs = gridspec.GridSpec(
        n, N_COLS,
        figure=fig,
        left=0.01, right=0.99,
        top=1.0 - (header_h + 0.04) / fig_h,
        bottom=(iou_h + 0.04) / fig_h if test_iou else 0.01,
        wspace=0.04,
        hspace=0.04,
    )

    for row, idx in enumerate(indices):
        img_t, matte_t = ds[idx]                        # (3,H,W), (1,H,W)

        with torch.no_grad():
            pred_t = model(img_t.unsqueeze(0).to(device))   # (1,1,H,W)
        pred_t = pred_t.squeeze(0).cpu()                 # (1,H,W)

        img_pil   = to_pil(img_t)                        # PIL RGB
        gt_pil    = to_pil(matte_t)                      # PIL L
        pred_pil  = to_pil(pred_t)                       # PIL L

        # Cutout: subject over white background using predicted alpha
        cutout_arr = _make_cutout(
            np.array(img_pil),
            np.array(pred_pil),
            bg_color=(255, 255, 255),
        )

        panels = [
            (np.array(img_pil),   None,   None, 255),    # RGB
            (np.array(gt_pil),    "gray",  0,   255),    # GT alpha
            (np.array(pred_pil),  "gray",  0,   255),    # Pred alpha
            (cutout_arr,          None,   None, 255),    # Cutout
        ]

        for col, (panel_arr, cmap, vmin, vmax) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            kw = {"cmap": cmap} if cmap else {}
            ax.imshow(panel_arr, vmin=vmin if vmin is not None else 0,
                      vmax=vmax, interpolation="bilinear", **kw)
            ax.axis("off")

            # Thin border
            for spine in ax.spines.values():
                spine.set_edgecolor("#4A5568")
                spine.set_linewidth(0.8)

    # ── IoU badge at the bottom ───────────────────────────────────────────
    if test_iou is not None:
        target_met = test_iou >= 0.85
        badge_txt  = (
            f"Test-split IoU = {test_iou:.4f}   "
            + ("Target >= 0.85  [ACHIEVED]" if target_met else "Target >= 0.85  [NOT YET]")
        )
        badge_color = "#276749" if target_met else "#9B2335"
        fig.text(
            0.5, 0.005,
            badge_txt,
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="white",
            transform=fig.transFigure,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=badge_color,
                edgecolor="white",
                linewidth=1.2,
                alpha=0.9,
            ),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path, dpi=160, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"Prediction grid saved -> {out_path}")
    if test_iou is not None:
        print(f"  Test IoU annotated: {test_iou:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualisation for Part A matting model"
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # ── curves ────────────────────────────────────────────────────────────
    pc = sub.add_parser("curves", help="Plot training loss / IoU curves")
    pc.add_argument("--log", required=True,
                    help="Path to matting_train_log.csv from train.py")
    pc.add_argument("--out", default=None,
                    help="Output PNG (default: <log_dir>/training_curves.png)")

    # ── predictions ───────────────────────────────────────────────────────
    pp = sub.add_parser(
        "predictions",
        help="Show n sample frames: Input | GT alpha | Predicted alpha | Cutout"
    )
    pp.add_argument("--weights", required=True,
                    help="Path to trained checkpoint .pth")
    pp.add_argument("--config",  default="config.yaml",
                    help="Config YAML (default: config.yaml)")
    pp.add_argument("--n",       type=int, default=5,
                    help="Number of sample rows (default: 5)")
    pp.add_argument("--out",     default=None,
                    help="Output PNG (default: outputs/sample_predictions.png)")
    pp.add_argument("--device",  default=None,
                    help="Force device: 'cpu' | 'cuda'")
    pp.add_argument("--iou",     type=float, default=None,
                    help="Test-split IoU to annotate on the figure (optional)")

    args = ap.parse_args()

    if args.mode == "curves":
        log_path = Path(args.log)
        if not log_path.exists():
            raise FileNotFoundError(f"Log not found: {log_path}")
        out_path = (Path(args.out) if args.out
                    else log_path.parent / "training_curves.png")
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

        out_path = (Path(args.out) if args.out
                    else Path(cfg["data"]["output_dir"]) / "sample_predictions.png")
        device = torch.device(
            args.device if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        plot_predictions(
            weights_path, cfg, out_path,
            n=args.n, device=device, test_iou=args.iou,
        )


if __name__ == "__main__":
    main()
