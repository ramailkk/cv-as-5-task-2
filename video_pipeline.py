"""
video_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Part C — Video Compositing Pipeline  [ENHANCED]
Assignment 5 — Task 2

Key upgrades over baseline:
  • Optical-flow-guided temporal warping: warp the previous stylised frame
    to the current frame using dense Farneback optical flow before using it
    as the NST init.  This ensures the init is spatially aligned, giving
    dramatically less ghosting than a raw t−1 copy.
  • Multi-scale NST per frame: coarse→fine gives richer texture in less time.
  • Alpha feathering: erode+Gaussian-blur the alpha matte boundary to avoid
    sharp compositing seams.
  • Histogram warm-start disabled on video (uses warp init instead).
  • H.264 re-encoding via ffmpeg after writing mp4v raw — far smaller files
    and universally playable.
  • Progress bar with ETA and per-frame timing.

Pipeline:
  1. Decode input video into frames (via OpenCV).
  2. For each frame t:
       a. Run matting model → alpha matte α_t  ∈ [0,1], feathered at edges
       b. Run NST (multi-scale, optflow-warped init) → stylised image S_t
       c. Composite per pixel:
            background-stylised: O_t = α_t · F_t + (1−α_t) · S_t
            subject-stylised:    O_t = α_t · S_t + (1−α_t) · F_t
  3. Re-encode composited frames → output video at original frame rate.

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
import os
import shutil
import subprocess
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
    t   = transforms.ToTensor()(rgb)
    return t.unsqueeze(0)


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

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil_resized = pil.resize((input_size[1], input_size[0]), Image.BILINEAR)

    tensor   = transforms.ToTensor()(pil_resized).unsqueeze(0).to(device)
    alpha_t  = model(tensor)

    alpha_resized = F.interpolate(
        alpha_t, size=(H_orig, W_orig),
        mode="bilinear", align_corners=False,
    )
    return alpha_resized.squeeze().cpu().numpy().astype(np.float32)


def feather_alpha(alpha: np.ndarray, blur_ksize: int = 15, erode_px: int = 3) -> np.ndarray:
    """
    Feather alpha matte at the boundary:
      1. Erode slightly to pull edge inward.
      2. Apply Gaussian blur so the transition is soft.
    This eliminates sharp compositing seams.
    """
    a_u8 = (alpha * 255).astype(np.uint8)
    kernel = np.ones((erode_px, erode_px), dtype=np.uint8)
    a_eroded = cv2.erode(a_u8, kernel, iterations=1)
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    a_blurred = cv2.GaussianBlur(a_eroded, (ksize, ksize), 0)
    return (a_blurred.astype(np.float32) / 255.0).clip(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Optical-flow-guided temporal warping
# ─────────────────────────────────────────────────────────────────────────────

def warp_tensor_with_flow(
    prev_tensor: torch.Tensor,
    prev_bgr:    np.ndarray,
    curr_bgr:    np.ndarray,
) -> torch.Tensor:
    """
    Warp `prev_tensor` (1,3,H,W) from frame t−1 to frame t using dense
    Farneback optical flow estimated between `prev_bgr` and `curr_bgr`.

    The warp ensures the temporal init is spatially aligned with the current
    content, reducing ghosting artefacts.

    Returns
    -------
    warped : (1, 3, H, W) float tensor in [0, 1], same size as prev_tensor
    """
    H, W = prev_bgr.shape[:2]
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    # Resize to a fast flow-estimation size
    flow_h, flow_w = min(H, 256), min(W, 256)
    pg = cv2.resize(prev_gray, (flow_w, flow_h))
    cg = cv2.resize(curr_gray, (flow_w, flow_h))

    flow = cv2.calcOpticalFlowFarneback(
        pg, cg, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    # Scale flow back to the NST tensor size
    tH, tW = prev_tensor.shape[2], prev_tensor.shape[3]
    flow_scaled = cv2.resize(flow, (tW, tH))
    flow_scaled[:, :, 0] *= tW / flow_w
    flow_scaled[:, :, 1] *= tH / flow_h

    # Build sampling grid
    grid_y, grid_x = np.mgrid[0:tH, 0:tW].astype(np.float32)
    map_x = (grid_x + flow_scaled[:, :, 0]).clip(0, tW - 1)
    map_y = (grid_y + flow_scaled[:, :, 1]).clip(0, tH - 1)

    # Normalise to [-1, 1] for F.grid_sample
    grid_norm_x = (map_x / (tW - 1)) * 2 - 1
    grid_norm_y = (map_y / (tH - 1)) * 2 - 1
    grid = torch.from_numpy(
        np.stack([grid_norm_x, grid_norm_y], axis=-1)
    ).unsqueeze(0).float().to(prev_tensor.device)

    warped = F.grid_sample(prev_tensor, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)
    return warped.clamp(0, 1)


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
    Per-pixel composite with feathered alpha.

    Modes
    -----
    "background" : O = α·F + (1−α)·S  — keep subject, stylise background
    "subject"    : O = α·S + (1−α)·F  — stylise subject, keep background

    Parameters
    ----------
    frame_bgr   : (H, W, 3) uint8 BGR
    stylised_t  : (1, 3, H, W) float [0,1] (RGB)
    alpha_map   : (H, W) float32 [0,1]
    mode        : "background" | "subject"
    """
    H, W = frame_bgr.shape[:2]

    s_np = tensor_to_bgr(
        F.interpolate(stylised_t, size=(H, W), mode="bilinear", align_corners=False)
    ).astype(np.float32) / 255.0

    f_np = frame_bgr.astype(np.float32) / 255.0
    a    = feather_alpha(alpha_map)[:, :, np.newaxis]

    if mode == "background":
        out = a * f_np + (1.0 - a) * s_np
    elif mode == "subject":
        out = a * s_np + (1.0 - a) * f_np
    else:
        raise ValueError(f"Unknown composite mode: {mode!r}.")

    return (out.clip(0, 1) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  ffmpeg re-encode helper
# ─────────────────────────────────────────────────────────────────────────────

def reencode_h264(input_path: Path, output_path: Path, crf: int = 18) -> None:
    """
    Re-encode a raw mp4v video to H.264 using ffmpeg.
    Produces a much smaller, universally playable file.
    Falls back silently if ffmpeg is not available.
    """
    if not shutil.which("ffmpeg"):
        print("  [info] ffmpeg not found — skipping H.264 re-encode")
        if input_path != output_path:
            shutil.move(str(input_path), str(output_path))
        return

    tmp = output_path.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.move(str(tmp), str(output_path))
        if input_path != output_path and input_path.exists():
            input_path.unlink()
        print(f"  ✓  H.264 re-encoded → {output_path.name}")
    except subprocess.CalledProcessError:
        print("  [warn] ffmpeg re-encode failed — keeping raw mp4v output")
        if input_path != output_path:
            shutil.move(str(input_path), str(output_path))


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
    tv_weight:       float = 1e-4,
    nst_steps:       int   = 100,
    nst_optimizer:   str   = "lbfgs",
    nst_size:        int   = 256,
    multiscale:      bool  = True,
    composite_mode:  str   = "background",
    temporal_init:   bool  = True,
    optflow_warp:    bool  = True,
    alpha_feather:   bool  = True,
    device:          torch.device = None,
    max_frames:      int   = None,
    verbose:         bool  = True,
) -> Path:
    """
    Full Part C video compositing pipeline.

    New parameters vs baseline:
    tv_weight     : TV regularisation for NST (default 1e-4)
    multiscale    : coarse→fine NST per frame (default True)
    optflow_warp  : warp prev frame with optical flow for temporal init
    alpha_feather : feather alpha matte boundaries
    """
    if style_layers is None:
        style_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
    if style_layer_weights is None:
        style_layer_weights = None  # use adaptive defaults in nst.py

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_path   = Path(video_path)
    style_path   = Path(style_path)
    weights_path = Path(weights_path)
    out_path     = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Video Pipeline — Part C  [Enhanced]")
    print(f"{'═'*60}")
    print(f"  Device       : {device}")
    print(f"  Video        : {video_path.name}")
    print(f"  Style        : {style_path.name}")
    print(f"  Mode         : {composite_mode}")
    print(f"  β/α          : {style_weight/content_weight:.0e}")
    print(f"  TV weight    : {tv_weight:.0e}")
    print(f"  Multiscale   : {multiscale}")
    print(f"  OptFlow warp : {optflow_warp}")
    print(f"  Alpha feather: {alpha_feather}")
    print(f"  Temporal init: {temporal_init}")
    print(f"  Output       : {out_path}")
    print(f"{'─'*60}\n")

    # ── 1. Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W_vid   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames is not None:
        n_total = min(n_total, max_frames)

    print(f"  Resolution   : {W_vid}×{H_vid}  |  FPS: {fps:.2f}  |  Frames: {n_total}")

    # ── 2. Set up output video writer ────────────────────────────────────────
    raw_out = out_path.with_suffix(".raw.mp4")
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(raw_out), fourcc, fps, (W_vid, H_vid))

    # ── 3. Load models ───────────────────────────────────────────────────────
    print("\n  Loading matting model…")
    matting_model = load_matting_model(weights_path, matting_arch, device)

    print("  Loading style image and pre-computing style Grams…")
    style_tensor = nst_load_image(style_path, size=nst_size).to(device)

    # Pre-compute style Gram matrices (same for every frame)
    all_nst_layers = list(set([content_layer] + style_layers))
    extractor = VGG19FeatureExtractor(all_nst_layers, device)
    extractor.eval()

    with torch.no_grad():
        style_feats = extractor(style_tensor)
        style_grams_precomputed = {
            name: gram_matrix(style_feats[name]).detach()
            for name in style_layers
        }

    # ── 4. Frame loop ────────────────────────────────────────────────────────
    prev_stylised: Optional[torch.Tensor] = None
    prev_frame_bgr: Optional[np.ndarray]  = None
    t_start = time.time()

    for frame_idx in range(n_total):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # ── 4a. Alpha matting ────────────────────────────────────────────────
        alpha_map = run_matting(matting_model, frame_bgr, matting_size, device)
        if alpha_feather:
            alpha_map = feather_alpha(alpha_map)

        # ── 4b. Build content tensor at NST resolution ───────────────────────
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame  = Image.fromarray(frame_rgb)
        w_vid, h_vid = pil_frame.size
        if h_vid < w_vid:
            new_h, new_w = nst_size, int(w_vid * nst_size / h_vid)
        else:
            new_w, new_h = nst_size, int(h_vid * nst_size / w_vid)
        pil_resized    = pil_frame.resize((new_w, new_h), Image.LANCZOS)
        content_tensor = transforms.ToTensor()(pil_resized).unsqueeze(0).to(device)

        # ── 4c. Temporal init with optical-flow warping ──────────────────────
        init_tensor = None
        if temporal_init and prev_stylised is not None:
            if optflow_warp and prev_frame_bgr is not None:
                init_tensor = warp_tensor_with_flow(
                    prev_stylised, prev_frame_bgr, frame_bgr
                )
                # Resize to content_tensor size if needed
                if init_tensor.shape[2:] != content_tensor.shape[2:]:
                    init_tensor = F.interpolate(
                        init_tensor, content_tensor.shape[2:],
                        mode="bilinear", align_corners=False,
                    )
            else:
                init_tensor = prev_stylised
                if init_tensor.shape[2:] != content_tensor.shape[2:]:
                    init_tensor = F.interpolate(
                        init_tensor, content_tensor.shape[2:],
                        mode="bilinear", align_corners=False,
                    )

        # ── 4d. NST ──────────────────────────────────────────────────────────
        stylised_t = run_nst(
            content_tensor=content_tensor,
            style_tensor=style_tensor,
            device=device,
            content_layer=content_layer,
            style_layers=style_layers,
            style_layer_weights=style_layer_weights,
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight,
            num_steps=nst_steps,
            optimizer=nst_optimizer,
            init_tensor=init_tensor,
            histogram_init=(init_tensor is None),  # only on first frame
            multiscale=multiscale,
            verbose=False,
        )

        if temporal_init:
            prev_stylised  = stylised_t.detach()
            prev_frame_bgr = frame_bgr.copy()

        # ── 4e. Composite ────────────────────────────────────────────────────
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
    print(f"\n\n  Raw video written: {raw_out}")

    # ── 5. Re-encode to H.264 ────────────────────────────────────────────────
    reencode_h264(raw_out, out_path)
    print(f"✓  Output saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Video Compositing Pipeline — Part C (Assignment 5, Task 2) [Enhanced]"
    )
    p.add_argument("--video",      required=True)
    p.add_argument("--style",      required=True)
    p.add_argument("--weights",    required=True)
    p.add_argument("--out",        default="outputs/stylised_video.mp4")
    p.add_argument("--arch",       default="unet", choices=["unet", "mobilenet_decoder"])
    p.add_argument("--matting_size", type=int, nargs=2, default=[320, 320])

    p.add_argument("--beta_ratio", type=float, default=1e5)
    p.add_argument("--sweep",      action="store_true")
    p.add_argument("--steps",      type=int, default=100)
    p.add_argument("--optim",      default="lbfgs", choices=["lbfgs", "adam"])
    p.add_argument("--nst_size",   type=int, default=256)
    p.add_argument("--tv_weight",  type=float, default=1e-4)
    p.add_argument("--multiscale", action="store_true", default=True)
    p.add_argument("--no_multiscale", dest="multiscale", action="store_false")

    p.add_argument("--mode",       default="background", choices=["background", "subject"])
    p.add_argument("--no_temporal", action="store_true")
    p.add_argument("--no_optflow",  action="store_true")
    p.add_argument("--no_feather",  action="store_true")

    p.add_argument("--device",     default=None)
    p.add_argument("--max_frames", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(
        video_path=args.video,
        style_path=args.style,
        weights_path=args.weights,
        matting_arch=args.arch,
        matting_size=tuple(args.matting_size),
        nst_steps=args.steps,
        nst_optimizer=args.optim,
        nst_size=args.nst_size,
        tv_weight=args.tv_weight,
        multiscale=args.multiscale,
        composite_mode=args.mode,
        temporal_init=not args.no_temporal,
        optflow_warp=not args.no_optflow,
        alpha_feather=not args.no_feather,
        device=device,
        max_frames=args.max_frames,
    )

    if args.sweep:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ratio in [1e3, 1e5, 1e7]:
            out_vid = out_dir / f"stylised_beta_{ratio:.0e}.mp4"
            run_pipeline(out_path=out_vid, style_weight=ratio, **common_kwargs)
        print(f"\n✓  Sweep complete. Videos saved to {out_dir}")
    else:
        run_pipeline(out_path=args.out, style_weight=args.beta_ratio, **common_kwargs)


if __name__ == "__main__":
    main()
