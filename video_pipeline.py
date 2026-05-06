"""
video_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Part C — Video Compositing Pipeline
Assignment 5 — Task 2

Pipeline:
  1. Decode input video into frames (via OpenCV or ffmpeg).
  2. For each frame t:
       a. Run matting model → alpha matte α_t  ∈ [0,1]
       b. Run NST           → stylised image S_t
       c. Composite per pixel:
            background-stylised: O_t = α_t · F_t + (1−α_t) · S_t
            subject-stylised:    O_t = α_t · S_t + (1−α_t) · F_t
  3. Re-encode composited frames → output video at original frame rate.

Temporal consistency (recommended, default ON):
  When stylising frame t, the NST is initialised from the stylised frame t−1
  rather than from the raw content frame.  This reduces flicker between frames
  at zero extra parameters.

Usage:
    python video_pipeline.py \\
        --video      my_video.mp4 \\
        --style      artwork.jpg \\
        --weights    matting_weights.pth \\
        --out        stylised_output.mp4 \\
        --mode       background \\
        --beta_ratio 1e5

    # Sweep all three β/α ratios (saves three output videos):
    python video_pipeline.py --video my_video.mp4 --style artwork.jpg \\
        --weights matting_weights.pth --out out_dir/ --sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── Import sibling modules ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from model import build_matting_model
from nst import (
    VGG19FeatureExtractor,
    VGG19_LAYER_MAP,
    gram_matrix,
    run_nst,
    load_image as nst_load_image,
    tensor_to_pil,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Frame I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_matting_model(weights_path: str | Path, arch: str, device: torch.device):
    """Load and return the matting model in eval mode."""
    model = build_matting_model(arch)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def bgr_frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert an OpenCV BGR frame (H,W,3) uint8 → (1,3,H,W) float [0,1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t   = transforms.ToTensor()(rgb)  # (3,H,W) float [0,1]
    return t.unsqueeze(0)             # (1,3,H,W)


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """Convert (1,3,H,W) or (3,H,W) float [0,1] → BGR uint8 ndarray."""
    arr_rgb = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def run_matting(
    model:      torch.nn.Module,
    frame_bgr:  np.ndarray,
    input_size: tuple[int, int],
    device:     torch.device,
) -> np.ndarray:
    """
    Run the matting model on a single BGR frame.

    Returns
    -------
    alpha_map : (H, W) float32 in [0, 1], at original frame resolution
    """
    H_orig, W_orig = frame_bgr.shape[:2]

    # Resize for model input
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil_resized = pil.resize((input_size[1], input_size[0]), Image.BILINEAR)

    tensor = transforms.ToTensor()(pil_resized).unsqueeze(0).to(device)
    alpha_t = model(tensor)  # (1, 1, H_in, W_in)

    # Resize alpha back to original frame size
    alpha_resized = F.interpolate(
        alpha_t, size=(H_orig, W_orig),
        mode="bilinear", align_corners=False,
    )
    return alpha_resized.squeeze().cpu().numpy()  # (H, W) float32


# ─────────────────────────────────────────────────────────────────────────────
#  Compositing
# ─────────────────────────────────────────────────────────────────────────────

def composite_frame(
    frame_bgr:   np.ndarray,
    stylised_t:  torch.Tensor,
    alpha_map:   np.ndarray,
    mode:        str = "background",
) -> np.ndarray:
    """
    Per-pixel composite.

    Modes
    -----
    "background" : O = α·F + (1−α)·S  — keep subject, stylise background
    "subject"    : O = α·S + (1−α)·F  — stylise subject, keep background

    Parameters
    ----------
    frame_bgr   : (H, W, 3) uint8 BGR  — original frame
    stylised_t  : (1, 3, H, W) float [0,1] — NST result (RGB)
    alpha_map   : (H, W) float32 [0,1] — foreground probability
    mode        : "background" | "subject"

    Returns
    -------
    composited  : (H, W, 3) uint8 BGR
    """
    H, W = frame_bgr.shape[:2]

    # Convert stylised tensor → float RGB ndarray (H,W,3) [0,1]
    s_np = tensor_to_bgr(
        F.interpolate(stylised_t, size=(H, W), mode="bilinear", align_corners=False)
    ).astype(np.float32) / 255.0  # BGR float [0,1]

    f_np = frame_bgr.astype(np.float32) / 255.0  # BGR float [0,1]

    # Expand alpha to (H, W, 1) for broadcasting
    a = alpha_map[:, :, np.newaxis]

    if mode == "background":
        # Subject stays natural, background gets stylised
        out = a * f_np + (1.0 - a) * s_np
    elif mode == "subject":
        # Subject gets stylised, background stays natural
        out = a * s_np + (1.0 - a) * f_np
    else:
        raise ValueError(f"Unknown composite mode: {mode!r}. Use 'background' or 'subject'.")

    return (out.clip(0, 1) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    video_path:      str | Path,
    style_path:      str | Path,
    weights_path:    str | Path,
    out_path:        str | Path,
    matting_arch:    str   = "unet",
    matting_size:    tuple[int, int] = (320, 320),
    content_layer:   str   = "relu4_2",
    style_layers:    list[str] = None,
    style_layer_weights: list[float] = None,
    content_weight:  float = 1.0,
    style_weight:    float = 1e5,
    nst_steps:       int   = 300,
    nst_optimizer:   str   = "lbfgs",
    nst_size:        int   = 512,
    composite_mode:  str   = "background",
    temporal_init:   bool  = True,
    device:          torch.device = None,
    max_frames:      int   = None,
    verbose:         bool  = True,
) -> Path:
    """
    Full Part C video compositing pipeline.

    Parameters
    ----------
    video_path      : input video file
    style_path      : style artwork image
    weights_path    : matting model checkpoint (.pth)
    out_path        : output video path (.mp4)
    matting_arch    : "unet" | "mobilenet_decoder"
    matting_size    : (H, W) model input size
    content_layer   : VGG19 layer for content loss
    style_layers    : VGG19 layers for style loss
    style_weight    : β — style loss weight
    content_weight  : α — content loss weight
    nst_steps       : optimisation steps per frame
    nst_optimizer   : "lbfgs" | "adam"
    nst_size        : shorter-edge resize for NST
    composite_mode  : "background" | "subject"
    temporal_init   : initialise NST from previous frame (temporal consistency)
    device          : torch device (auto-detect if None)
    max_frames      : cap frames for testing (None = process all)
    verbose         : progress printing

    Returns
    -------
    Path to output video
    """
    if style_layers is None:
        style_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
    if style_layer_weights is None:
        style_layer_weights = [1.0] * len(style_layers)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_path   = Path(video_path)
    style_path   = Path(style_path)
    weights_path = Path(weights_path)
    out_path     = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Video Pipeline — Part C")
    print(f"{'═'*60}")
    print(f"  Device       : {device}")
    print(f"  Video        : {video_path.name}")
    print(f"  Style        : {style_path.name}")
    print(f"  Mode         : {composite_mode}")
    print(f"  β/α          : {style_weight/content_weight:.0e}")
    print(f"  Temporal init: {temporal_init}")
    print(f"  Output       : {out_path}")
    print(f"{'─'*60}\n")

    # ── 1. Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W_vid  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames is not None:
        n_total = min(n_total, max_frames)

    print(f"  Resolution   : {W_vid}×{H_vid}  |  FPS: {fps:.2f}  |  Frames: {n_total}")

    # ── 2. Set up output video writer ────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W_vid, H_vid))

    # ── 3. Load models ───────────────────────────────────────────────────────
    print("\n  Loading matting model…")
    matting_model = load_matting_model(weights_path, matting_arch, device)

    print("  Loading style image and pre-computing style Grams…")
    style_tensor = nst_load_image(style_path, size=nst_size).to(device)

    # Pre-compute style Gram matrices (same for every frame)
    all_nst_layers = list({content_layer} | set(style_layers))
    extractor = VGG19FeatureExtractor(all_nst_layers, device)
    extractor.eval()

    with torch.no_grad():
        style_feats = extractor(style_tensor)
        style_grams = {
            name: gram_matrix(style_feats[name]).detach()
            for name in style_layers
        }

    # ── 4. Frame loop ────────────────────────────────────────────────────────
    prev_stylised: Optional[torch.Tensor] = None  # for temporal consistency
    t_start = time.time()

    for frame_idx in range(n_total):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # ── 4a. Alpha matting ────────────────────────────────────────────────
        alpha_map = run_matting(matting_model, frame_bgr, matting_size, device)

        # ── 4b. NST ──────────────────────────────────────────────────────────
        # Resize frame to NST size for optimisation
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame  = Image.fromarray(frame_rgb)
        w_vid, h_vid = pil_frame.size
        if h_vid < w_vid:
            new_h = nst_size
            new_w = int(w_vid * nst_size / h_vid)
        else:
            new_w = nst_size
            new_h = int(h_vid * nst_size / w_vid)
        pil_resized   = pil_frame.resize((new_w, new_h), Image.LANCZOS)
        content_tensor = transforms.ToTensor()(pil_resized).unsqueeze(0).to(device)

        # Temporal init: use previous stylised frame as starting point
        init_tensor = prev_stylised
        if init_tensor is not None and init_tensor.shape != content_tensor.shape:
            init_tensor = F.interpolate(
                init_tensor, size=content_tensor.shape[2:],
                mode="bilinear", align_corners=False,
            )

        stylised_t = run_nst(
            content_tensor=content_tensor,
            style_tensor=style_tensor,
            device=device,
            content_layer=content_layer,
            style_layers=style_layers,
            style_layer_weights=style_layer_weights,
            content_weight=content_weight,
            style_weight=style_weight,
            num_steps=nst_steps,
            optimizer=nst_optimizer,
            init_tensor=init_tensor,
            verbose=False,  # suppress per-step logs during video
        )

        if temporal_init:
            prev_stylised = stylised_t.detach()

        # ── 4c. Composite ────────────────────────────────────────────────────
        comp_bgr = composite_frame(frame_bgr, stylised_t, alpha_map, mode=composite_mode)
        writer.write(comp_bgr)

        # ── Progress ─────────────────────────────────────────────────────────
        if verbose:
            elapsed  = time.time() - t_start
            fps_proc = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            eta      = (n_total - frame_idx - 1) / fps_proc if fps_proc > 0 else 0
            print(
                f"\r  Frame {frame_idx+1:4d}/{n_total}  |  "
                f"{fps_proc:.2f} fr/s  |  ETA {eta/60:.1f} min",
                end="", flush=True,
            )

    cap.release()
    writer.release()
    print(f"\n\n✓  Output saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video Compositing Pipeline — Part C (Assignment 5, Task 2)"
    )
    p.add_argument("--video",      required=True,
                   help="Input video file (.mp4, .avi, ...)")
    p.add_argument("--style",      required=True,
                   help="Style artwork image path")
    p.add_argument("--weights",    required=True,
                   help="Matting model checkpoint (.pth)")
    p.add_argument("--out",        default="outputs/stylised_video.mp4",
                   help="Output video path or directory (for --sweep)")
    p.add_argument("--arch",       default="unet",
                   choices=["unet", "mobilenet_decoder"],
                   help="Matting architecture (default: unet)")
    p.add_argument("--matting_size", type=int, nargs=2, default=[320, 320],
                   metavar=("H", "W"),
                   help="Matting model input size (default: 320 320)")

    # NST options
    p.add_argument("--beta_ratio", type=float, default=1e5,
                   help="β/α style ratio (default: 1e5). Ignored in --sweep mode.")
    p.add_argument("--sweep",      action="store_true",
                   help="Produce three videos for β/α ∈ {1e3, 1e5, 1e7}")
    p.add_argument("--steps",      type=int, default=100,
                   help="NST optimisation steps per frame (default: 100 for video speed)")
    p.add_argument("--optim",      default="adam", choices=["lbfgs", "adam"],
                   help="NST optimizer (default: adam — faster per-frame)")
    p.add_argument("--nst_size",   type=int, default=256,
                   help="NST image shorter-edge size (default: 256 for video speed)")

    # Composite options
    p.add_argument("--mode",       default="background",
                   choices=["background", "subject"],
                   help="Composite mode (default: background)")
    p.add_argument("--no_temporal", action="store_true",
                   help="Disable temporal consistency initialisation")

    # General
    p.add_argument("--device",     default=None,
                   help="Force device: cpu | cuda | cuda:0 (auto if omitted)")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Process only the first N frames (for testing)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(
        video_path=args.video,
        style_path=args.style,
        weights_path=args.weights,
        matting_arch=args.arch,
        matting_size=tuple(args.matting_size),
        nst_steps=args.steps,
        nst_optimizer=args.optim,
        nst_size=args.nst_size,
        composite_mode=args.mode,
        temporal_init=not args.no_temporal,
        device=device,
        max_frames=args.max_frames,
    )

    if args.sweep:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ratio in [1e3, 1e5, 1e7]:
            out_vid = out_dir / f"stylised_beta_{ratio:.0e}.mp4"
            run_pipeline(
                out_path=out_vid,
                style_weight=ratio,
                **common_kwargs,
            )
        print(f"\n✓  Sweep complete. Videos saved to {out_dir}")
    else:
        run_pipeline(
            out_path=args.out,
            style_weight=args.beta_ratio,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
