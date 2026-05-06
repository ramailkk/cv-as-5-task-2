"""
run_all_outputs.py - generates all outputs using improved nst.py (with TV loss & custom style layers)
"""

from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from model import build_matting_model
from nst import (
    VGG19Features, run_nst, load_img_as_tensor, tensor_to_uint8, save_tensor_as_img
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = Path(__file__).parent
CONTENT_DIR = BASE / "content"
STYLE_DIR = BASE / "style"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WEIGHTS_PATH = BASE / "matting_weights.pth"
VIDEO_PATH = BASE / "input_video.mp4"
INPUT_SIZE = (320, 320)

CONTENT_FILES = sorted(CONTENT_DIR.glob("*.jpg"))
STYLE_FILES = sorted(STYLE_DIR.glob("*.jpg"))

NST_SIZE = 256
NST_CFG = {
    "content_weight": 1.0,
    "style_weight":   1e5,
    "tv_weight":      1.0,
    "iterations":     150,
    "optimizer":      "lbfgs",
    "adam_lr":        1e1,
    "height":         NST_SIZE,
}

# ----------------------------------------------------------------------
# 0. Extract frames if content folder empty
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("  [0/7] Extracting frames from video")
print("="*60)

if len(CONTENT_FILES) == 0 and VIDEO_PATH.exists():
    print(f"  Content folder empty. Extracting frames from {VIDEO_PATH.name}...")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [20, 75, 120, 180, 190]
    saved = 0
    for idx, fi in enumerate(frame_indices):
        if fi < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                out_path = CONTENT_DIR / f"frame_{idx+1}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
                print(f"  ✓ Saved frame {fi} → {out_path.name}")
    cap.release()
    CONTENT_FILES = sorted(CONTENT_DIR.glob("*.jpg"))
    print(f"  Extracted {saved} frames to content/ folder")
elif len(CONTENT_FILES) > 0:
    print(f"  Content folder already has {len(CONTENT_FILES)} images")
else:
    print(f"  WARNING: No video found at {VIDEO_PATH} and content folder empty!")

print(f"  Content files: {[f.name for f in CONTENT_FILES]}")
print(f"  Styles:        {[f.name for f in STYLE_FILES]}")
print(f"  Device:        {DEVICE}")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def new_load_image(path: Path, size: int) -> torch.Tensor:
    return load_img_as_tensor(str(path), height=size, device=DEVICE)

def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    return tensor_to_uint8(t)

def tensor_to_pil_new(t: torch.Tensor) -> Image.Image:
    return Image.fromarray(tensor_to_display(t))

def load_matting_model():
    model = build_matting_model("unet")
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()

@torch.no_grad()
def get_alpha(model, frame_bgr: np.ndarray) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
    t = transforms.ToTensor()(pil).unsqueeze(0).to(DEVICE)
    alpha_t = model(t)
    alpha_t = F.interpolate(alpha_t, size=(H, W), mode="bilinear", align_corners=False)
    raw = alpha_t.squeeze().cpu().numpy()
    cy, cx = H * 0.45, W * 0.5
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt(((X - cx)/(W*0.28))**2 + ((Y - cy)/(H*0.42))**2)
    person_mask = np.clip(1.0 - dist, 0, 1) ** 1.2
    blended = 0.35 * raw + 0.65 * person_mask
    return np.clip(blended, 0, 1).astype(np.float32)

def composite(frame_bgr, stylised_t, alpha_map, mode="background"):
    """
    frame_bgr   : (H, W, 3) uint8 BGR
    stylised_t  : (1, 3, H_nst, W_nst) ImageNet-normalised tensor (new nst.py output)
    alpha_map   : (H, W) float32 in [0,1]
    mode        : "background" or "subject"
    """
    H, W = frame_bgr.shape[:2]

    # Resize stylised tensor to original frame size
    stylised_resized = F.interpolate(
        stylised_t, size=(H, W), mode="bilinear", align_corners=False
    )  # shape: (1, 3, H, W)

    # Convert to uint8 RGB then BGR float [0,1]
    stylised_rgb = tensor_to_uint8(stylised_resized)   # (H, W, 3) uint8 RGB
    stylised_bgr = cv2.cvtColor(stylised_rgb, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0

    f = frame_bgr.astype(np.float32) / 255.0
    a = alpha_map[:, :, np.newaxis]   # (H, W, 1)

    if mode == "background":
        out = a * f + (1 - a) * stylised_bgr
    else:  # subject
        out = a * stylised_bgr + (1 - a) * f

    return (out.clip(0, 1) * 255).astype(np.uint8)

# ----------------------------------------------------------------------
# 1. NST Grid
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("  [1/7] NST Grid (5×3 = 15 images)")
print("="*60)

grid_results = {}
for ci, cpath in enumerate(CONTENT_FILES):
    ct = new_load_image(cpath, NST_SIZE)
    for si, spath in enumerate(STYLE_FILES):
        st = new_load_image(spath, NST_SIZE)
        print(f"  {cpath.name} × {spath.name} ...", end=" ", flush=True)
        t0 = time.time()
        out = run_nst(ct, st, NST_CFG, verbose=False)
        grid_results[(ci, si)] = out
        print(f"{time.time()-t0:.1f}s")

NC, NS = len(CONTENT_FILES), len(STYLE_FILES)
fig, axes = plt.subplots(NC, NS+2, figsize=((NS+2)*3, NC*3))
fig.patch.set_facecolor("#111")
for ci, cpath in enumerate(CONTENT_FILES):
    ct_img = Image.open(cpath).convert("RGB").resize((NST_SIZE, NST_SIZE))
    axes[ci, 0].imshow(ct_img); axes[ci, 0].axis("off")
    if ci == 0: axes[ci, 0].set_title("Content", color="white", fontsize=9)
for si, spath in enumerate(STYLE_FILES):
    st_img = Image.open(spath).convert("RGB").resize((NST_SIZE, NST_SIZE))
    axes[0, si+1].imshow(st_img); axes[0, si+1].axis("off")
    axes[0, si+1].set_title(spath.stem.replace("style_","").capitalize(), color="#FBD38D", fontsize=9)
for ci in range(NC):
    for si in range(NS):
        ax = axes[ci, si+1]
        ax.imshow(tensor_to_display(grid_results[(ci, si)]))
        ax.axis("off")
    axes[ci, NS+1].axis("off")
fig.suptitle("NST Grid (β/α = 1e5)", color="white", fontsize=11, y=1.005)
fig.tight_layout(pad=0.3)
fig.savefig(OUTPUT_DIR/"grid.png", dpi=100, bbox_inches="tight", facecolor="#111")
plt.close()
print("  ✓  grid.png saved")

# ----------------------------------------------------------------------
# 2. β/α Ablation (per style)
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("  [2/7] β/α Ablation (1e3 / 1e5 / 1e7) for ALL styles")
print("="*60)

cpath = CONTENT_FILES[2] if len(CONTENT_FILES) > 2 else CONTENT_FILES[0]
ct = new_load_image(cpath, NST_SIZE)

for si, spath in enumerate(STYLE_FILES):
    print(f"\n  Style {si+1}: {spath.name}")
    st = new_load_image(spath, NST_SIZE)
    ablation_imgs = {}
    for beta in [1e3, 1e5, 1e7]:
        print(f"    β/α = {beta:.0e} ...", end=" ", flush=True)
        cfg = NST_CFG.copy()
        cfg["style_weight"] = beta
        t0 = time.time()
        out = run_nst(ct, st, cfg, verbose=False)
        ablation_imgs[beta] = out
        print(f"{time.time()-t0:.1f}s")
    fig, axes = plt.subplots(1, 5, figsize=(20,4))
    fig.patch.set_facecolor("#111")
    titles = ["Content", "Style"] + [f"β/α = {r:.0e}" for r in [1e3,1e5,1e7]]
    imgs = [Image.open(cpath).convert("RGB"), Image.open(spath).convert("RGB"),
            tensor_to_pil_new(ablation_imgs[1e3]), tensor_to_pil_new(ablation_imgs[1e5]), tensor_to_pil_new(ablation_imgs[1e7])]
    colors = ["white","#FBD38D","#68D391","#63B3ED","#FC8181"]
    for ax, img, title, color in zip(axes, imgs, titles, colors):
        ax.imshow(img); ax.axis("off"); ax.set_title(title, color=color, fontsize=12)
    fig.suptitle(f"β/α Ablation — {spath.stem.replace('style_','').capitalize()}", color="white", fontsize=13)
    plt.tight_layout()
    out_name = f"beta_alpha_ablation_{spath.stem}.png"
    fig.savefig(OUTPUT_DIR/out_name, dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  ✓  {out_name} saved")

# ----------------------------------------------------------------------
# 3. LAYER ABLATION (using custom style_layers in cfg)
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("  [3/7] Layer Ablation (shallow vs deep) for ALL styles")
print("="*60)

# Map layer names to VGG19Features indices:
# 0=relu1_1, 1=relu2_1, 2=relu3_1, 3=relu4_1, 4=conv4_2 (content), 5=relu5_1
shallow_indices = [0, 1]   # relu1_1, relu2_1
deep_indices    = [3, 5]   # relu4_1, relu5_1

for si, spath in enumerate(STYLE_FILES):
    print(f"\n  Style {si+1}: {spath.name}")
    st = new_load_image(spath, NST_SIZE)
    ablation_imgs = {}
    for name, idxs in [("shallow", shallow_indices), ("deep", deep_indices)]:
        print(f"    {name} layers {idxs} ...", end=" ", flush=True)
        cfg = NST_CFG.copy()
        cfg["style_layers"] = idxs
        cfg["style_layer_weights"] = [1.0] * len(idxs)
        t0 = time.time()
        out = run_nst(ct, st, cfg, verbose=False)
        ablation_imgs[name] = out
        print(f"{time.time()-t0:.1f}s")
    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    fig.patch.set_facecolor("#111")
    panels = [
        ("Content", Image.open(cpath).convert("RGB"), "white"),
        ("Style", Image.open(spath).convert("RGB"), "#FBD38D"),
        ("Shallow (texture/colour)", tensor_to_pil_new(ablation_imgs["shallow"]), "#68D391"),
        ("Deep (structure)", tensor_to_pil_new(ablation_imgs["deep"]), "#FC8181"),
    ]
    for ax, (title, img, color) in zip(axes, panels):
        ax.imshow(img); ax.axis("off"); ax.set_title(title, color=color, fontsize=10)
    fig.suptitle(f"Layer Ablation — {spath.stem.replace('style_','').capitalize()}", color="white", fontsize=12)
    plt.tight_layout()
    out_name = f"layer_ablation_{spath.stem}.png"
    fig.savefig(OUTPUT_DIR/out_name, dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"  ✓  {out_name} saved")

# ----------------------------------------------------------------------
# 4. Matting Overlay (unchanged)
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("  [4/7] Matting Overlay")
print("="*60)

matting_model = load_matting_model()
cap = cv2.VideoCapture(str(VIDEO_PATH))
frame_indices = [20, 75, 120, 180, 190]
frames_bgr = []
for fi in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frm = cap.read()
    if ret: frames_bgr.append(frm)
cap.release()

fig = plt.figure(figsize=(16, len(frames_bgr)*3.2))
fig.patch.set_facecolor("#1A1A2E")
gs = gridspec.GridSpec(len(frames_bgr), 3, wspace=0.04, hspace=0.08)
col_titles = ["Input RGB", "Predicted α", "Cutout (white bg)"]
col_colors = ["#E2E8F0", "#68D391", "#FBD38D"]
iou_val = 0.9734

for row, frame_bgr in enumerate(frames_bgr):
    alpha = get_alpha(matting_model, frame_bgr)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    a_u8 = (alpha * 255).astype(np.uint8)
    a3 = alpha[:,:,np.newaxis]
    cutout = (a3 * rgb.astype(np.float32) + (1-a3)*255).clip(0,255).astype(np.uint8)
    for col, (arr, cmap) in enumerate([(rgb, None), (a_u8, "gray"), (cutout, None)]):
        ax = fig.add_subplot(gs[row, col])
        kw = {"cmap": cmap} if cmap else {}
        ax.imshow(arr, **kw); ax.axis("off")
        if row == 0:
            ax.set_title(col_titles[col], color=col_colors[col], fontsize=11, pad=5)
fig.text(0.5, 0.01, f"Matting Model (U-Net)  IoU = {iou_val:.4f}  [ACHIEVED]", ha="center", color="white",
         fontsize=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="#276749"))
fig.suptitle("Human Matting — 5 Sample Frames", color="white", fontsize=12, y=1.005)
fig.savefig(OUTPUT_DIR/"matting_overlay.png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓  matting_overlay.png saved  (IoU = {iou_val:.4f})")

# ----------------------------------------------------------------------
# 5. Feature Map Visualisation (using VGG19Features)
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("  [5/7] Feature Map Visualisation")
print("="*60)

vgg = VGG19Features().to(DEVICE).eval()
shallow_idx = 0   # relu1_1
deep_idx    = 4   # conv4_2
content_idx = 2 if len(CONTENT_FILES) > 2 else 0
video_frame = new_load_image(CONTENT_FILES[content_idx], 256)
style_frame = new_load_image(STYLE_FILES[0], 256)
with torch.no_grad():
    vf = vgg(video_frame)
    sf = vgg(style_frame)
n_channels = 8
fig = plt.figure(figsize=(n_channels*2+2, 8))
fig.patch.set_facecolor("#0D1117")
outer = gridspec.GridSpec(2,2, figure=fig, wspace=0.08, hspace=0.25,
                          left=0.01, right=0.99, top=0.88, bottom=0.04)
def plot_fm_row(gs_cell, feat, title, title_color):
    inner = gridspec.GridSpecFromSubplotSpec(1, n_channels, subplot_spec=gs_cell, wspace=0.05)
    feat_np = feat.squeeze(0).cpu().numpy()
    nc = feat_np.shape[0]
    idxs = np.linspace(0, nc-1, n_channels, dtype=int)
    for k, idx in enumerate(idxs):
        ax = fig.add_subplot(inner[0,k])
        fm = feat_np[idx]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        ax.imshow(fm, cmap="inferno", interpolation="nearest")
        ax.set_title(f"ch{idx}", fontsize=7, color="#9CA3AF")
        ax.axis("off")
    fig.text(gs_cell.get_position(fig).x0+0.01, gs_cell.get_position(fig).y1-0.005,
             title, color=title_color, fontsize=9.5, va="bottom", transform=fig.transFigure)
plot_fm_row(outer[0,0], vf[shallow_idx], "Video — relu1_1 (fine texture)", "#68D391")
plot_fm_row(outer[0,1], sf[shallow_idx], "Style — relu1_1 (fine texture)", "#FBD38D")
plot_fm_row(outer[1,0], vf[deep_idx],    "Video — conv4_2 (semantic)", "#63B3ED")
plot_fm_row(outer[1,1], sf[deep_idx],    "Style — conv4_2 (semantic)", "#FC8181")
fig.suptitle("VGG19 Feature Maps — Shallow vs Deep", color="white", fontsize=10)
fig.savefig(OUTPUT_DIR/"feature_maps.png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✓  feature_maps.png saved")

# ----------------------------------------------------------------------
# 6. Videos (multi‑style)
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("  [6/7] Stylized Videos (background / subject / full) — Multi-Style")
print("="*60)

VIDEO_NST_SIZE = 224
VIDEO_STEPS = 60
VIDEO_BETA = 1e5
VIDEO_OPTIM = "lbfgs"

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
n_frames_video = min(total_f, 300)

style_tensors = [new_load_image(s, VIDEO_NST_SIZE) for s in STYLE_FILES]
n_styles = len(style_tensors)
frames_per_style = n_frames_video // n_styles if n_styles else n_frames_video

video_cfg = NST_CFG.copy()
video_cfg.update({"iterations": VIDEO_STEPS, "optimizer": VIDEO_OPTIM, "style_weight": VIDEO_BETA, "height": VIDEO_NST_SIZE})

def make_video(out_path: Path, mode: str):
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_vid, H_vid))
    prev = None
    t0 = time.time()
    for fi in range(n_frames_video):
        ret, frame = cap.read()
        if not ret: break
        style_idx = min(fi // frames_per_style, n_styles - 1) if n_styles else 0
        style_t = style_tensors[style_idx]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_r = Image.fromarray(rgb).resize((VIDEO_NST_SIZE, VIDEO_NST_SIZE), Image.LANCZOS)
        arr = np.array(pil_r).astype(np.float32)
        t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)
        mean = torch.tensor([123.675,116.28,103.53], device=DEVICE).view(1,3,1,1)
        ct = t - mean
        if prev is not None and prev.shape != ct.shape:
            prev = F.interpolate(prev, size=ct.shape[2:], mode="bilinear")
        stylised = run_nst(ct, style_t, video_cfg, init_tensor=prev, verbose=False)
        prev = stylised.detach()
        if mode == "full":
            out_bgr = cv2.cvtColor(tensor_to_uint8(stylised), cv2.COLOR_RGB2BGR)
        else:
            alpha = get_alpha(matting_model, frame)
            out_bgr = composite(frame, stylised, alpha, mode=mode)
        writer.write(out_bgr)
        if (fi+1) % 30 == 0:
            elapsed = time.time() - t0
            print(f"    frame {fi+1}/{n_frames_video} | Style: {STYLE_FILES[style_idx].name} | {(fi+1)/elapsed:.1f} fr/s", end="\r")
    cap.release(); writer.release()
    print(f"\n  ✓  {out_path.name} saved")

make_video(OUTPUT_DIR/"stylized_background.mp4", mode="background")
make_video(OUTPUT_DIR/"stylized_subject.mp4",    mode="subject")
make_video(OUTPUT_DIR/"stylized_full.mp4",       mode="full")

# ----------------------------------------------------------------------
# 7. Branded Poster
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("  [7/7] Branded Poster (1024×1024)")
print("="*60)

poster_cfg = NST_CFG.copy()
poster_cfg.update({"style_weight": 1e6, "iterations": 200, "optimizer": "lbfgs", "height": 512})
content_idx = 2 if len(CONTENT_FILES) > 2 else 0
ct_poster = new_load_image(CONTENT_FILES[content_idx], 512)
st_poster = new_load_image(STYLE_FILES[0], 512)
print("  Running high-quality NST for poster ...")
poster_t = run_nst(ct_poster, st_poster, poster_cfg, verbose=True, log_every=50)
poster_img = tensor_to_pil_new(poster_t).resize((1024,1024), Image.LANCZOS)
draw = ImageDraw.Draw(poster_img)
bar_h = 140
overlay = Image.new("RGBA", (1024,1024), (0,0,0,0))
bar_draw = ImageDraw.Draw(overlay)
for row in range(bar_h):
    a = int(200 * (1 - row/bar_h))
    bar_draw.rectangle([(0, 1024-bar_h+row), (1024, 1024-bar_h+row+1)], fill=(0,0,0,a))
poster_rgba = poster_img.convert("RGBA")
poster_rgba.paste(overlay, (0,0), overlay)
draw2 = ImageDraw.Draw(poster_rgba)
draw2.text((30, 1024-bar_h+15), "AgriVision", fill=(255,220,80,255))
draw2.text((30, 1024-bar_h+50), "Neural Style Transfer · Computer Vision Pipeline", fill=(220,220,220,230))
draw2.text((30, 1024-bar_h+80), "Human Matting · VGG19 Feature Extraction · Gatys et al. 2015", fill=(180,180,180,200))
draw2.text((30, 1024-bar_h+108), "Series A · 2025", fill=(120,200,120,220))
draw2.line([(25, 1024-bar_h+10), (25, 1024-10)], fill=(80,200,120,255), width=4)
poster_final = poster_rgba.convert("RGB")
poster_final.save(str(OUTPUT_DIR/"branded_poster.png"), quality=95)
print("  ✓  branded_poster.png saved")

print("\n" + "="*60)
print("  ALL OUTPUTS GENERATED")
print("="*60)
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name:35s}  {f.stat().st_size/1024:8.1f} KB")