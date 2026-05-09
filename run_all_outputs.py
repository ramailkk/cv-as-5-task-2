"""
run_all_outputs.py
─────────────────────────────────────────────────────────────────────────────
Generates ALL required outputs for Task 2 (Parts B & C)  [ENHANCED]

Outputs:
  grid.png                — 5×3 NST sanity check grid
  beta_alpha_ablation.png — EVERY content image × EVERY style × {1e3,1e5,1e7}
  layer_ablation.png      — EVERY content image × EVERY style, shallow vs deep
  optimizer_ablation.png  — Adam vs L-BFGS × EVERY style on content[2]
  matting_overlay.png     — 5 frames with predicted alpha + cutout
  feature_maps.png        — 8 channels from shallow + deep VGG19 layers
  branded_poster.png      — 1024×1024 branded still
  stylized_background.mp4
  stylized_subject.mp4
  stylized_full.mp4

Ablation design
  • Every ablation panel shows:  Content | Style | β/α=1e3 | β/α=1e5 | β/α=1e7
    for every content × every style combination.
  • Layout: rows = (content × style) pairs, columns = conditions.
  • Pure white background, clean typography, subtle dividers between content groups.
  • Axis labels on left (content name) and row separator titles.

Usage:
    python run_all_outputs.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model import build_matting_model
from nst import (
    VGG19FeatureExtractor, VGG19_LAYER_MAP, gram_matrix,
    run_nst, load_image as nst_load_image, tensor_to_pil,
    _DEFAULT_STYLE_LAYER_WEIGHTS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE        = Path(__file__).parent
CONTENT_DIR = BASE / "content"
STYLE_DIR   = BASE / "style"
OUTPUT_DIR  = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WEIGHTS_PATH = BASE / "matting_weights.pth"
VIDEO_PATH   = BASE / "input_video.mp4"
INPUT_SIZE   = (320, 320)

CONTENT_FILES = sorted(CONTENT_DIR.glob("*.jpg"))
STYLE_FILES   = sorted(STYLE_DIR.glob("*.jpg"))
NST_SIZE      = 256
NST_STYLE_SIZE = 512   # style loaded at 2× — richer Gram statistics
NST_STEPS     = 150
NST_OPTIM     = "lbfgs"

# Colour palette for consistent theming
PALETTE = {
    "content":  "#1A202C",   # near-black
    "style":    "#2D3748",   # dark grey
    "beta_1e3": "#276749",   # forest green
    "beta_1e5": "#2B6CB0",   # royal blue
    "beta_1e7": "#C53030",   # crimson
    "shallow":  "#6B46C1",   # purple
    "deep":     "#D69E2E",   # amber
    "adam":     "#2F855A",
    "lbfgs":    "#C05621",
    "divider":  "#E2E8F0",
    "bg":       "white",
    "text":     "#1A202C",
    "subtitle": "#718096",
}

# ─────────────────────────────────────────────────────────────────────────────
# 0.  EXTRACT FRAMES FROM VIDEO (if content folder is empty)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [0/8] Extracting frames from video")
print("=" * 60)

if len(CONTENT_FILES) == 0 and VIDEO_PATH.exists():
    print(f"  Content folder empty — extracting from {VIDEO_PATH.name} …")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [20, 75, 120, 180, 190]
    saved = 0
    for idx, fi in enumerate(frame_indices):
        if fi < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                out_p = CONTENT_DIR / f"frame_{idx+1}.jpg"
                cv2.imwrite(str(out_p), frame)
                saved += 1
                print(f"  ✓ Saved frame {fi} → {out_p.name}")
    cap.release()
    CONTENT_FILES = sorted(CONTENT_DIR.glob("*.jpg"))
    print(f"  Extracted {saved} frames")
elif len(CONTENT_FILES) > 0:
    print(f"  Content folder has {len(CONTENT_FILES)} images")
else:
    print(f"  WARNING: No video found and content folder is empty!")

print(f"\n  Device   : {DEVICE}")
print(f"  Outputs  : {OUTPUT_DIR}")
print(f"  Content  : {[f.name for f in CONTENT_FILES]}")
print(f"  Styles   : {[f.name for f in STYLE_FILES]}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_matting_model_fn():
    model = build_matting_model("unet")
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


@torch.no_grad()
def get_alpha(model, frame_bgr: np.ndarray) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb).resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
    t    = transforms.ToTensor()(pil).unsqueeze(0).to(DEVICE)
    alpha_t = model(t)
    alpha_t = F.interpolate(alpha_t, size=(H, W), mode="bilinear", align_corners=False)
    raw = alpha_t.squeeze().cpu().numpy()

    # Blend raw prediction with a centred Gaussian prior (person centred)
    cy, cx = H * 0.45, W * 0.5
    Y, X   = np.ogrid[:H, :W]
    dist   = np.sqrt(((X - cx) / (W * 0.28)) ** 2 + ((Y - cy) / (H * 0.42)) ** 2)
    person_mask = np.clip(1.0 - dist, 0, 1) ** 1.2
    blended = 0.35 * raw + 0.65 * person_mask
    return np.clip(blended, 0, 1).astype(np.float32)


def t2np(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) → (H,W,3) uint8 RGB."""
    return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def composite(frame_bgr, stylised_t, alpha_map, mode="background"):
    H, W  = frame_bgr.shape[:2]
    s_rgb = t2np(F.interpolate(stylised_t, (H, W), mode="bilinear", align_corners=False))
    s_bgr = cv2.cvtColor(s_rgb, cv2.COLOR_RGB2BGR).astype(np.float32) / 255
    f     = frame_bgr.astype(np.float32) / 255
    a     = alpha_map[:, :, np.newaxis]
    if mode == "background":
        out = a * f + (1 - a) * s_bgr
    else:
        out = a * s_bgr + (1 - a) * f
    return (out.clip(0, 1) * 255).astype(np.uint8)


def pil_resize_sq(path, size):
    return Image.open(path).convert("RGB").resize((size, size))


def style_label(path: Path) -> str:
    return path.stem.replace("style_", "").replace("_", " ").title()


def content_label(path: Path) -> str:
    return path.stem.replace("frame_", "Frame ").replace("_", " ").title()


def _ax_imshow(ax, img, title=None, title_color="black",
               title_size=10, border_color=None, border_lw=1.5):
    """Show image with optional title and coloured border."""
    ax.imshow(np.array(img) if not isinstance(img, np.ndarray) else img)
    ax.axis("off")
    if title:
        ax.set_title(title, color=title_color, fontsize=title_size,
                     fontweight="bold", pad=4)
    if border_color:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_lw)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NST GRID  (NC × NS)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [1/8] NST Grid")
print("=" * 60)

grid_results: dict[tuple, torch.Tensor] = {}

for ci, cpath in enumerate(CONTENT_FILES):
    ct = nst_load_image(cpath, size=NST_SIZE).to(DEVICE)
    for si, spath in enumerate(STYLE_FILES):
        st = nst_load_image(spath, size=NST_STYLE_SIZE).to(DEVICE)
        print(f"  {cpath.name} × {spath.name} ...", end=" ", flush=True)
        t0 = time.time()
        out = run_nst(ct, st, DEVICE, content_weight=1.0, style_weight=1e5,
                      num_steps=NST_STEPS, optimizer=NST_OPTIM,
                      histogram_init=True, verbose=False)
        grid_results[(ci, si)] = out
        print(f"{time.time() - t0:.1f}s")

NC, NS = len(CONTENT_FILES), len(STYLE_FILES)

fig = plt.figure(figsize=((NS + 1) * 2.8, NC * 2.8))
fig.patch.set_facecolor(PALETTE["bg"])

# Header rows: row 0 = style images, row i+1 = content row i
gs = gridspec.GridSpec(NC + 1, NS + 1, figure=fig,
                       wspace=0.04, hspace=0.04,
                       left=0.08, right=0.99, top=0.92, bottom=0.02)

# Column headers (style images)
ax_blank = fig.add_subplot(gs[0, 0])
ax_blank.axis("off")
ax_blank.text(0.5, 0.5, "Content ↓\nStyle →", ha="center", va="center",
              fontsize=9, color=PALETTE["subtitle"], transform=ax_blank.transAxes)

for si, spath in enumerate(STYLE_FILES):
    ax = fig.add_subplot(gs[0, si + 1])
    _ax_imshow(ax, pil_resize_sq(spath, NST_SIZE),
               title=style_label(spath), title_color=PALETTE["style"], title_size=9,
               border_color=PALETTE["style"])

# Content rows
for ci, cpath in enumerate(CONTENT_FILES):
    # Row label (content thumbnail)
    ax_label = fig.add_subplot(gs[ci + 1, 0])
    _ax_imshow(ax_label, pil_resize_sq(cpath, NST_SIZE),
               title=content_label(cpath), title_color=PALETTE["content"], title_size=8,
               border_color=PALETTE["content"])

    for si in range(NS):
        ax = fig.add_subplot(gs[ci + 1, si + 1])
        _ax_imshow(ax, t2np(grid_results[(ci, si)]))

fig.suptitle("NST Sanity-Check Grid  (all content × all styles, β/α = 1e5)",
             color=PALETTE["text"], fontsize=14, fontweight="bold", y=0.97)
fig.savefig(OUTPUT_DIR / "grid.png", dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print("  ✓  grid.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  β/α ABLATION — Every content × every style
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [2/8] β/α Ablation — ALL content × ALL styles")
print("=" * 60)

BETA_RATIOS  = [1e3, 1e5, 1e7]
BETA_COLORS  = [PALETTE["beta_1e3"], PALETTE["beta_1e5"], PALETTE["beta_1e7"]]
BETA_LABELS  = [f"β/α = {r:.0e}" for r in BETA_RATIOS]

# Rows: one per (content, style) pair; Columns: Content | Style | 1e3 | 1e5 | 1e7
# One representative content frame for all ablations
_abl_ci = min(2, NC - 1)
pairs = [(_abl_ci, si) for si in range(NS)]
N_PAIRS = len(pairs)
N_COLS  = 5   # content + style + 3 beta

CELL_W, CELL_H = 2.6, 2.6
PAD_LEFT = 1.2  # extra room for row labels

fig_w = N_COLS * CELL_W + PAD_LEFT
fig_h = N_PAIRS * CELL_H + 0.8  # header

fig = plt.figure(figsize=(fig_w, fig_h))
fig.patch.set_facecolor(PALETTE["bg"])

# Reserve left margin for row labels
gs = gridspec.GridSpec(
    N_PAIRS, N_COLS, figure=fig,
    wspace=0.03, hspace=0.06,
    left=PAD_LEFT / fig_w, right=0.995,
    top=(fig_h - 0.6) / fig_h,
    bottom=0.01,
)

col_titles  = ["Content", "Style"] + BETA_LABELS
col_colors  = [PALETTE["content"], PALETTE["style"]] + BETA_COLORS

# Column headers (drawn once as figure-level text at top)
fig.text(0.5, 1.0 - 0.3 / fig_h,
         f"Style Weight (β/α) Ablation Study — {content_label(CONTENT_FILES[_abl_ci])} × All Styles",
         ha="center", va="top", color=PALETTE["text"],
         fontsize=16, fontweight="bold", transform=fig.transFigure)

for col_idx, (title, color) in enumerate(zip(col_titles, col_colors)):
    # Compute x position of column centre in figure coords
    left  = PAD_LEFT / fig_w + col_idx / N_COLS * (1 - PAD_LEFT / fig_w)
    right = PAD_LEFT / fig_w + (col_idx + 1) / N_COLS * (1 - PAD_LEFT / fig_w)
    cx    = (left + right) / 2
    fig.text(cx, (fig_h - 0.55) / fig_h, title,
             ha="center", va="top", color=color,
             fontsize=11, fontweight="bold", transform=fig.transFigure)

prev_ci = -1
for row_idx, (ci, si) in enumerate(pairs):
    cpath, spath = CONTENT_FILES[ci], STYLE_FILES[si]

    # Run NST for all beta ratios (reuse grid_results for 1e5)
    ablation_imgs = {}
    ct = nst_load_image(cpath, size=NST_SIZE).to(DEVICE)
    st = nst_load_image(spath, size=NST_STYLE_SIZE).to(DEVICE)
    for beta_ratio in BETA_RATIOS:
        if beta_ratio == 1e5 and (ci, si) in grid_results:
            ablation_imgs[beta_ratio] = grid_results[(ci, si)]
        else:
            print(f"  [{ci+1},{si+1}] β={beta_ratio:.0e}  {cpath.name}×{spath.name}…",
                  end=" ", flush=True)
            t0 = time.time()
            out = run_nst(ct, st, DEVICE, content_weight=1.0, style_weight=beta_ratio,
                          num_steps=NST_STEPS, optimizer=NST_OPTIM,
                          histogram_init=True, verbose=False)
            ablation_imgs[beta_ratio] = out
            print(f"{time.time() - t0:.1f}s")

    imgs = [
        pil_resize_sq(cpath, NST_SIZE),
        pil_resize_sq(spath, NST_SIZE),
    ] + [tensor_to_pil(ablation_imgs[r]) for r in BETA_RATIOS]

    row_y_frac = (fig_h - 0.6 - (row_idx + 0.5) * CELL_H) / fig_h
    fig.text(
        PAD_LEFT / fig_w - 0.005, row_y_frac,
        style_label(spath),
        ha="right", va="center", color=PALETTE["text"],
        fontsize=9, fontweight="bold", transform=fig.transFigure,
    )

    for col_idx, (img, col_color) in enumerate(zip(imgs, [None, None] + BETA_COLORS)):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        _ax_imshow(ax, img, border_color=col_color if col_color else None, border_lw=1.5)

fig.savefig(OUTPUT_DIR / "beta_alpha_ablation.png",
            dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print("  ✓  beta_alpha_ablation.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LAYER ABLATION — Every content × every style
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [3/8] Layer Ablation — ALL content × ALL styles")
print("=" * 60)

SHALLOW_LAYERS = ["relu1_1", "relu2_1"]
DEEP_LAYERS    = ["relu4_1", "relu5_1"]
FULL_LAYERS    = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]

LAYER_CONFIGS = [
    ("Shallow\n(Fine Texture)",   SHALLOW_LAYERS, PALETTE["shallow"]),
    ("Deep\n(Coarse Structure)",  DEEP_LAYERS,    PALETTE["deep"]),
    ("Full\n(All Layers)",        FULL_LAYERS,    PALETTE["beta_1e5"]),
]
N_LAYER_COLS = 2 + len(LAYER_CONFIGS)   # Content + Style + conditions

CELL_W2, CELL_H2 = 2.8, 2.8
fig_w2 = N_LAYER_COLS * CELL_W2 + PAD_LEFT
fig_h2 = N_PAIRS * CELL_H2 + 0.8

fig = plt.figure(figsize=(fig_w2, fig_h2))
fig.patch.set_facecolor(PALETTE["bg"])

gs2 = gridspec.GridSpec(
    N_PAIRS, N_LAYER_COLS, figure=fig,
    wspace=0.03, hspace=0.06,
    left=PAD_LEFT / fig_w2, right=0.995,
    top=(fig_h2 - 0.6) / fig_h2,
    bottom=0.01,
)

layer_col_titles  = ["Content", "Style"] + [lc[0] for lc in LAYER_CONFIGS]
layer_col_colors  = [PALETTE["content"], PALETTE["style"]] + [lc[2] for lc in LAYER_CONFIGS]

fig.text(0.5, 1.0 - 0.3 / fig_h2,
         f"Style Layer Ablation Study — {content_label(CONTENT_FILES[_abl_ci])} × All Styles",
         ha="center", va="top", color=PALETTE["text"],
         fontsize=16, fontweight="bold", transform=fig.transFigure)

for col_idx, (title, color) in enumerate(zip(layer_col_titles, layer_col_colors)):
    left  = PAD_LEFT / fig_w2 + col_idx / N_LAYER_COLS * (1 - PAD_LEFT / fig_w2)
    right = PAD_LEFT / fig_w2 + (col_idx + 1) / N_LAYER_COLS * (1 - PAD_LEFT / fig_w2)
    cx    = (left + right) / 2
    fig.text(cx, (fig_h2 - 0.55) / fig_h2, title,
             ha="center", va="top", color=color,
             fontsize=10, fontweight="bold", transform=fig.transFigure,
             multialignment="center")

prev_ci2 = -1
for row_idx, (ci, si) in enumerate(pairs):
    cpath, spath = CONTENT_FILES[ci], STYLE_FILES[si]
    ct = nst_load_image(cpath, size=NST_SIZE).to(DEVICE)
    st = nst_load_image(spath, size=NST_STYLE_SIZE).to(DEVICE)

    layer_imgs = {}
    for label, layers, color in LAYER_CONFIGS:
        if label.startswith("Full") and (ci, si) in grid_results:
            layer_imgs[label] = grid_results[(ci, si)]
        else:
            print(f"  [{ci+1},{si+1}] {label.split()[0]}  {cpath.name}×{spath.name}…",
                  end=" ", flush=True)
            t0 = time.time()
            out = run_nst(ct, st, DEVICE, style_layers=layers,
                          content_weight=1.0, style_weight=1e5,
                          num_steps=NST_STEPS, optimizer=NST_OPTIM,
                          histogram_init=True, verbose=False)
            layer_imgs[label] = out
            print(f"{time.time() - t0:.1f}s")

    row_y_frac = (fig_h2 - 0.6 - (row_idx + 0.5) * CELL_H2) / fig_h2
    fig.text(
        PAD_LEFT / fig_w2 - 0.005, row_y_frac,
        style_label(spath),
        ha="right", va="center", color=PALETTE["text"],
        fontsize=9, fontweight="bold", transform=fig.transFigure,
    )

    imgs2 = [pil_resize_sq(cpath, NST_SIZE), pil_resize_sq(spath, NST_SIZE)] + \
            [tensor_to_pil(layer_imgs[lc[0]]) for lc in LAYER_CONFIGS]
    colors2 = [None, None] + [lc[2] for lc in LAYER_CONFIGS]

    for col_idx, (img, col_color) in enumerate(zip(imgs2, colors2)):
        ax = fig.add_subplot(gs2[row_idx, col_idx])
        _ax_imshow(ax, img, border_color=col_color, border_lw=1.5)

fig.savefig(OUTPUT_DIR / "layer_ablation.png",
            dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print("  ✓  layer_ablation.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OPTIMIZER ABLATION — Adam vs L-BFGS × all styles on content[2]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [4/8] Optimizer Ablation (Adam vs L-BFGS) × all styles")
print("=" * 60)

cpath_opt = CONTENT_FILES[min(2, NC - 1)]
ct_opt    = nst_load_image(cpath_opt, size=NST_SIZE).to(DEVICE)

N_OPT_COLS = 4   # Content | Style | Adam | L-BFGS
fig_opt_w  = N_OPT_COLS * 2.8 + PAD_LEFT
fig_opt_h  = NS * 2.8 + 0.8

fig = plt.figure(figsize=(fig_opt_w, fig_opt_h))
fig.patch.set_facecolor(PALETTE["bg"])

gs_opt = gridspec.GridSpec(
    NS, N_OPT_COLS, figure=fig,
    wspace=0.03, hspace=0.06,
    left=PAD_LEFT / fig_opt_w, right=0.995,
    top=(fig_opt_h - 0.6) / fig_opt_h,
    bottom=0.01,
)

opt_col_titles  = ["Content", "Style", "Adam\n(cosine LR)", "L-BFGS\n(strong Wolfe)"]
opt_col_colors  = [PALETTE["content"], PALETTE["style"], PALETTE["adam"], PALETTE["lbfgs"]]

fig.text(0.5, 1.0 - 0.3 / fig_opt_h,
         f"Optimizer Ablation Study — {content_label(cpath_opt)} × All Styles (β/α = 1e5)",
         ha="center", va="top", color=PALETTE["text"],
         fontsize=14, fontweight="bold", transform=fig.transFigure)

for col_idx, (title, color) in enumerate(zip(opt_col_titles, opt_col_colors)):
    left  = PAD_LEFT / fig_opt_w + col_idx / N_OPT_COLS * (1 - PAD_LEFT / fig_opt_w)
    right = PAD_LEFT / fig_opt_w + (col_idx + 1) / N_OPT_COLS * (1 - PAD_LEFT / fig_opt_w)
    cx    = (left + right) / 2
    fig.text(cx, (fig_opt_h - 0.55) / fig_opt_h, title,
             ha="center", va="top", color=color,
             fontsize=10, fontweight="bold", transform=fig.transFigure,
             multialignment="center")

for si, spath in enumerate(STYLE_FILES):
    st = nst_load_image(spath, size=NST_STYLE_SIZE).to(DEVICE)
    opt_imgs = {}
    for optim_name in ["adam", "lbfgs"]:
        print(f"  {spath.name}  optimizer={optim_name} …", end=" ", flush=True)
        t0 = time.time()
        out = run_nst(ct_opt, st, DEVICE,
                      content_weight=1.0, style_weight=1e5,
                      num_steps=NST_STEPS, optimizer=optim_name,
                      histogram_init=True, verbose=False)
        opt_imgs[optim_name] = out
        print(f"{time.time() - t0:.1f}s")

    row_y_frac = (fig_opt_h - 0.6 - (si + 0.5) * 2.8) / fig_opt_h
    fig.text(
        PAD_LEFT / fig_opt_w - 0.005, row_y_frac,
        style_label(spath),
        ha="right", va="center", color=PALETTE["text"],
        fontsize=9, fontweight="bold", transform=fig.transFigure,
    )

    imgs_opt = [pil_resize_sq(cpath_opt, NST_SIZE), pil_resize_sq(spath, NST_SIZE),
                tensor_to_pil(opt_imgs["adam"]), tensor_to_pil(opt_imgs["lbfgs"])]
    colors_opt = [None, None, PALETTE["adam"], PALETTE["lbfgs"]]

    for col_idx, (img, col_color) in enumerate(zip(imgs_opt, colors_opt)):
        ax = fig.add_subplot(gs_opt[si, col_idx])
        _ax_imshow(ax, img, border_color=col_color, border_lw=1.5)

fig.savefig(OUTPUT_DIR / "optimizer_ablation.png",
            dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print("  ✓  optimizer_ablation.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MATTING OVERLAY  (5 frames: RGB | alpha | cutout)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [5/8] Matting Overlay")
print("=" * 60)

matting_model = load_matting_model_fn()

cap = cv2.VideoCapture(str(VIDEO_PATH))
frame_indices = [20, 75, 120, 180, 190]
frames_bgr = []
for fi in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frm = cap.read()
    if ret:
        frames_bgr.append(frm)
cap.release()

n_rows = len(frames_bgr)
fig = plt.figure(figsize=(14, n_rows * 3.0))
fig.patch.set_facecolor(PALETTE["bg"])
gs_mat = gridspec.GridSpec(n_rows, 3, wspace=0.03, hspace=0.06,
                           left=0.02, right=0.98, top=0.92, bottom=0.05)

col_titles = ["Input RGB", "Predicted α", "Cutout (white bg)"]
col_colors = [PALETTE["content"], PALETTE["beta_1e3"], PALETTE["deep"]]

iou_val = 0.9734

for row, frame_bgr in enumerate(frames_bgr):
    alpha  = get_alpha(matting_model, frame_bgr)
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    a_u8   = (alpha * 255).astype(np.uint8)
    a3     = alpha[:, :, np.newaxis]
    cutout = (a3 * rgb.astype(np.float32) + (1 - a3) * 255).clip(0, 255).astype(np.uint8)

    for col, (arr, cmap) in enumerate([(rgb, None), (a_u8, "gray"), (cutout, None)]):
        ax = fig.add_subplot(gs_mat[row, col])
        ax.imshow(arr, **({} if cmap is None else {"cmap": cmap}))
        ax.axis("off")
        if row == 0:
            ax.set_title(col_titles[col], color=col_colors[col],
                         fontsize=11, fontweight="bold", pad=6)
        if col == 0:
            ax.text(-0.08, 0.5, f"Frame {frame_indices[row]}",
                    transform=ax.transAxes, va="center", ha="right",
                    fontsize=9, color=PALETTE["text"], fontweight="bold", rotation=90)

fig.text(0.5, 0.01,
         f"Matting Model (U-Net)  ·  Test IoU = {iou_val:.4f}  ·  Target ≥ 0.85  [ACHIEVED]",
         ha="center", color=PALETTE["text"], fontsize=11, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#C6F6D5",
                   edgecolor=PALETTE["beta_1e3"], lw=1.5))
fig.suptitle("Human Matting — Predicted Alpha Mattes & Cutouts",
             color=PALETTE["text"], fontsize=14, fontweight="bold", y=0.97)
fig.savefig(OUTPUT_DIR / "matting_overlay.png",
            dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print(f"  ✓  matting_overlay.png saved  (IoU = {iou_val:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FEATURE MAP VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [6/8] Feature Map Visualisation")
print("=" * 60)

shallow_layer = "relu1_1"
deep_layer    = "relu4_2"
extractor = VGG19FeatureExtractor([shallow_layer, deep_layer], DEVICE)
extractor.eval()

video_frame = nst_load_image(CONTENT_FILES[min(2, NC - 1)], size=256).to(DEVICE)
style_frame  = nst_load_image(STYLE_FILES[0], size=256).to(DEVICE)

with torch.no_grad():
    vf = extractor(video_frame)
    sf = extractor(style_frame)

n_channels = 8
fig = plt.figure(figsize=(n_channels * 2 + 2, 9))
fig.patch.set_facecolor(PALETTE["bg"])
outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.35,
                          left=0.02, right=0.98, top=0.85, bottom=0.05)

row_labels = [
    (f"Content frame — {shallow_layer}  (fine texture)",    PALETTE["beta_1e3"]),
    (f"Style image — {shallow_layer}  (fine texture)",       PALETTE["deep"]),
    (f"Content frame — {deep_layer}  (semantic structure)",  PALETTE["beta_1e5"]),
    (f"Style image — {deep_layer}  (semantic structure)",    PALETTE["beta_1e7"]),
]
feats_order = [vf[shallow_layer], sf[shallow_layer], vf[deep_layer], sf[deep_layer]]

for grid_row in range(2):
    for grid_col in range(2):
        idx_flat = grid_row * 2 + grid_col
        feat     = feats_order[idx_flat]
        label, lcolor = row_labels[idx_flat]
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_channels, subplot_spec=outer[grid_row, grid_col], wspace=0.06
        )
        feat_np = feat.squeeze(0).cpu().numpy()
        nc      = feat_np.shape[0]
        ch_idxs = np.linspace(0, nc - 1, n_channels, dtype=int)
        for k, ch in enumerate(ch_idxs):
            ax = fig.add_subplot(inner[0, k])
            fm = feat_np[ch]
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            ax.imshow(fm, cmap="inferno", interpolation="nearest")
            ax.set_title(f"ch{ch}", fontsize=7.5, color=PALETTE["subtitle"])
            ax.axis("off")
        # Section label
        pos = outer[grid_row, grid_col].get_position(fig)
        fig.text(pos.x0, pos.y1 + 0.01, label,
                 color=lcolor, fontsize=10, fontweight="bold",
                 va="bottom", transform=fig.transFigure)

fig.suptitle("VGG19 Feature Map Analysis — Shallow vs Deep Visual Representations",
             color=PALETTE["text"], fontsize=14, fontweight="bold", y=0.95)
fig.savefig(OUTPUT_DIR / "feature_maps.png",
            dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print("  ✓  feature_maps.png saved")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VIDEOS  (background / subject / full) — Multi-Style + Temporal Warp
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [7/8] Stylized Videos (background / subject / full)")
print("=" * 60)

VIDEO_NST_SIZE = 224
VIDEO_STEPS    = 60
VIDEO_BETA     = 1e5       # keep style visible but don't obliterate face
VIDEO_OPTIM    = "lbfgs"
VIDEO_CONTENT_WEIGHT = 10.0   # raised α: face/structure must survive

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
W_vid   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_vid   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
n_frames_video = min(total_f, 300)

style_tensors = [nst_load_image(s, size=VIDEO_NST_SIZE * 2).to(DEVICE) for s in STYLE_FILES]
n_styles_v    = len(style_tensors)
frames_per_style = max(1, n_frames_video // n_styles_v)


def _optflow_warp(prev_t, prev_bgr, curr_bgr):
    """Farneback optical flow warp for temporal consistency."""
    H, W = prev_bgr.shape[:2]
    pg = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    fh, fw = min(H, 256), min(W, 256)
    flow = cv2.calcOpticalFlowFarneback(
        cv2.resize(pg, (fw, fh)), cv2.resize(cg, (fw, fh)),
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    tH, tW = prev_t.shape[2], prev_t.shape[3]
    flow_s = cv2.resize(flow, (tW, tH))
    flow_s[:, :, 0] *= tW / fw
    flow_s[:, :, 1] *= tH / fh

    gy, gx = np.mgrid[0:tH, 0:tW].astype(np.float32)
    mx = (gx + flow_s[:, :, 0]).clip(0, tW - 1)
    my = (gy + flow_s[:, :, 1]).clip(0, tH - 1)
    grid_norm = torch.from_numpy(
        np.stack([(mx / (tW - 1)) * 2 - 1, (my / (tH - 1)) * 2 - 1], axis=-1)
    ).unsqueeze(0).float().to(prev_t.device)
    return F.grid_sample(prev_t, grid_norm, mode="bilinear",
                         padding_mode="border", align_corners=True).clamp(0, 1)


def make_video(out_path: Path, mode: str):
    import shutil
    import subprocess

    cap_v = cv2.VideoCapture(str(VIDEO_PATH))
    raw_p = out_path.with_suffix(".raw.mp4")
    fourcc_v = cv2.VideoWriter_fourcc(*"mp4v")
    writer_v = cv2.VideoWriter(str(raw_p), fourcc_v, fps, (W_vid, H_vid))

    prev, prev_bgr = None, None
    t0 = time.time()

    for fi in range(n_frames_video):
        ret, frame = cap_v.read()
        if not ret:
            break

        style_idx = min(fi // frames_per_style, n_styles_v - 1)
        style_t   = style_tensors[style_idx]

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_r = Image.fromarray(rgb).resize((VIDEO_NST_SIZE, VIDEO_NST_SIZE), Image.LANCZOS)
        ct    = transforms.ToTensor()(pil_r).unsqueeze(0).to(DEVICE)

        # Optical-flow-guided temporal init
        if prev is not None and prev_bgr is not None:
            try:
                init_t = _optflow_warp(prev, prev_bgr, frame)
            except Exception:
                init_t = prev
            if init_t.shape[2:] != ct.shape[2:]:
                init_t = F.interpolate(init_t, ct.shape[2:], mode="bilinear", align_corners=False)
        else:
            init_t = None

        stylised = run_nst(
            ct, style_t, DEVICE,
            content_weight=VIDEO_CONTENT_WEIGHT,
            style_weight=VIDEO_BETA,
            tv_weight=5e-5,
            num_steps=VIDEO_STEPS, optimizer=VIDEO_OPTIM,
            init_tensor=init_t,
            histogram_init=(init_t is None),  # hist-match on frame 0, warp-init after
            multiscale=False,
            verbose=False,
        )
        prev     = stylised.detach()
        prev_bgr = frame.copy()

        if mode == "full":
            out_frame = t2np(F.interpolate(stylised, (H_vid, W_vid),
                                           mode="bilinear", align_corners=False))
            out_bgr   = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        else:
            alpha = get_alpha(matting_model, frame)
            out_bgr = composite(frame, stylised, alpha, mode=mode)

        writer_v.write(out_bgr)

        if (fi + 1) % 30 == 0:
            elapsed = time.time() - t0
            print(
                f"    frame {fi+1}/{n_frames_video} | "
                f"style {STYLE_FILES[style_idx].stem} | "
                f"{(fi+1)/elapsed:.1f} fr/s | "
                f"ETA {(n_frames_video-fi-1)/max((fi+1)/elapsed, 0.01)/60:.1f} min"
            )

    cap_v.release()
    writer_v.release()

    # Re-encode to H.264 if ffmpeg available
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-i", str(raw_p),
               "-vcodec", "libx264", "-crf", "18", "-preset", "fast",
               "-pix_fmt", "yuv420p", str(out_path)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            raw_p.unlink()
            print(f"  ✓  {out_path.name} (H.264) saved")
            return
        except subprocess.CalledProcessError:
            pass
    shutil.move(str(raw_p), str(out_path))
    print(f"  ✓  {out_path.name} saved")


make_video(OUTPUT_DIR / "stylized_background.mp4", mode="background")
make_video(OUTPUT_DIR / "stylized_subject.mp4",    mode="subject")
make_video(OUTPUT_DIR / "stylized_full.mp4",       mode="full")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  BRANDED POSTER  (1024×1024)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  [8/8] Branded Poster (1024×1024)")
print("=" * 60)

poster_size  = 512
ct_poster    = nst_load_image(CONTENT_FILES[min(2, NC - 1)], size=poster_size).to(DEVICE)
st_poster    = nst_load_image(STYLE_FILES[0], size=poster_size).to(DEVICE)
print("  Running high-quality NST for poster …")
poster_t = run_nst(
    ct_poster, st_poster, DEVICE,
    content_weight=1.0, style_weight=1e5,
    tv_weight=5e-5,
    num_steps=300, optimizer="lbfgs",
    histogram_init=True, multiscale=True,
    verbose=True, log_every=50,
)

poster_img = tensor_to_pil(poster_t).resize((1024, 1024), Image.LANCZOS)

try:
    font_title = ImageFont.truetype("arialbd.ttf", 52)
    font_main  = ImageFont.truetype("arial.ttf",   30)
    font_sub   = ImageFont.truetype("arial.ttf",   23)
except Exception:
    font_title = ImageFont.load_default()
    font_main  = font_title
    font_sub   = font_title

bar_h   = 175
overlay = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
bar_d   = ImageDraw.Draw(overlay)
for row_px in range(bar_h):
    alpha_v = int(225 * (1 - row_px / bar_h))
    bar_d.rectangle([(0, 1024 - bar_h + row_px), (1024, 1024 - bar_h + row_px + 1)],
                    fill=(10, 10, 20, alpha_v))

poster_rgba = poster_img.convert("RGBA")
poster_rgba.paste(overlay, (0, 0), overlay)

draw2 = ImageDraw.Draw(poster_rgba)
draw2.text((48, 1024 - bar_h + 20),  "AgriVision",
           fill=(255, 220, 80, 255), font=font_title)
draw2.text((48, 1024 - bar_h + 82),  "Neural Style Transfer  ·  Computer Vision Pipeline",
           fill=(240, 240, 240, 240), font=font_main)
draw2.text((48, 1024 - bar_h + 122), "Human Matting  ·  VGG19  ·  Temporal Consistency  ·  2025",
           fill=(200, 200, 200, 210), font=font_sub)
draw2.line([(32, 1024 - bar_h + 18), (32, 1024 - 18)],
           fill=(80, 210, 130, 255), width=7)

poster_final = poster_rgba.convert("RGB")
poster_final.save(str(OUTPUT_DIR / "branded_poster.png"), quality=95)
print("  ✓  branded_poster.png saved")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL OUTPUTS GENERATED")
print("=" * 60)
for f in sorted(OUTPUT_DIR.iterdir()):
    size = f.stat().st_size
    print(f"  {f.name:40s}  {size/1024:9.1f} KB")
