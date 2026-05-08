"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluate a trained matting model on the AISegment test split and print a
full metrics report (Part A — Assignment 5 Task 2).

Usage:
    python evaluate.py --weights outputs/matting_weights.pth --config config.yaml

Metrics reported:
    • Mean IoU      (threshold = 0.5)
    • Mean Precision, Recall, F1
    • Mean Absolute Error (MAE) on continuous alpha mattes
    • Mean Loss  (L1 + Dice, same weights as training)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import torch
import torch.nn as nn

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from matting.model   import build_matting_model
from matting.dataset import build_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
#  Loss (duplicated here to keep evaluate.py self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred   = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter  = (pred * target).sum(dim=1)
    denom  = pred.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def matting_loss(pred, target, l1_w, dice_w):
    return l1_w * nn.functional.l1_loss(pred, target) + dice_w * dice_loss(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute binary segmentation metrics for a batch.

    Returns dict with keys: iou, precision, recall, f1, mae
    """
    pb = (pred   > threshold).float()
    tb = (target > threshold).float()

    tp = (pb * tb).sum(dim=(1, 2, 3))
    fp = (pb * (1 - tb)).sum(dim=(1, 2, 3))
    fn = ((1 - pb) * tb).sum(dim=(1, 2, 3))

    iou       = ((tp + 1e-6) / (tp + fp + fn + 1e-6)).mean().item()
    precision = ((tp + 1e-6) / (tp + fp + 1e-6)).mean().item()
    recall    = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    mae       = (pred - target).abs().mean().item()

    return {"iou": iou, "precision": precision,
            "recall": recall, "f1": f1, "mae": mae}


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_full(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg:    dict,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    l1_w   = cfg["matting"]["l1_weight"]
    dice_w = cfg["matting"]["dice_weight"]

    totals: dict[str, float] = {
        "loss": 0.0, "iou": 0.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": 0.0,
    }
    n = 0

    it = tqdm(loader, desc="Evaluating") if HAS_TQDM else loader
    for imgs, mattes in it:
        imgs   = imgs.to(device,   non_blocking=True)
        mattes = mattes.to(device, non_blocking=True)
        preds  = model(imgs)

        bs = imgs.size(0)
        totals["loss"] += matting_loss(preds, mattes, l1_w, dice_w).item() * bs
        m = compute_metrics(preds, mattes)
        for k, v in m.items():
            totals[k] += v * bs
        n += bs

    return {k: v / n for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full metrics evaluation for trained matting model (Part A)"
    )
    ap.add_argument("--weights", required=True,
                    help="Path to trained checkpoint .pth file")
    ap.add_argument("--config", default="config.yaml",
                    help="Path to config YAML (default: config.yaml)")
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test"],
                    help="Dataset split to evaluate on (default: test)")
    ap.add_argument("--device", default=None,
                    help="Force device: 'cpu' | 'cuda' (auto-detect if omitted)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}\n")

    # Model
    arch  = cfg["matting"]["architecture"]
    model = build_matting_model(arch)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Model  : {arch}   loaded from {args.weights}\n")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[args.split]
    print(f"Evaluating on '{args.split}' split ({len(loader.dataset)} samples) …\n")

    results = evaluate_full(model, loader, cfg, device)

    bar = "=" * 50
    print(f"\n{bar}")
    print(f"  Split       : {args.split}")
    print(f"  Loss        : {results['loss']:.4f}")
    print(f"  IoU         : {results['iou']:.4f}  (target ≥ 0.85)")
    print(f"  Precision   : {results['precision']:.4f}")
    print(f"  Recall      : {results['recall']:.4f}")
    print(f"  F1          : {results['f1']:.4f}")
    print(f"  MAE (alpha) : {results['mae']:.4f}  (lower is better)")
    print(f"{bar}\n")

    if results["iou"] >= 0.85:
        print("  ✓  IoU ≥ 0.85 — Part A target achieved!")
    else:
        gap = 0.85 - results["iou"]
        print(f"  ⚠  IoU is {gap:.4f} below the 0.85 target.")
        print("     Consider: more epochs, larger train_size, or higher dice_weight.")


if __name__ == "__main__":
    main()
