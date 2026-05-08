"""
predict.py
─────────────────────────────────────────────────────────────────────────────
Run the trained human matting model on images or a directory of images
and save the resulting alpha mattes (Part A — Assignment 5 Task 2).

Usage:
    # Single image
    python predict.py --weights outputs/matting_weights.pth --input photo.jpg

    # Directory of images → saves mattes to --out_dir
    python predict.py --weights outputs/matting_weights.pth \
                      --input  /path/to/images/ \
                      --out_dir /path/to/mattes/

    # Override architecture from config.yaml
    python predict.py --weights outputs/matting_weights.pth \
                      --input photo.jpg --arch unet

    # Visualise as composite (green background)
    python predict.py --weights outputs/matting_weights.pth \
                      --input photo.jpg --visualise

Outputs for each input image:
    <stem>_matte.png   — single-channel alpha matte  (uint8, 0-255)
    <stem>_vis.png     — optional composite visualisation (with --visualise)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ── Allow running from project root without installing as a package ──────────
sys.path.insert(0, str(Path(__file__).parent))
from matting.model import build_matting_model


# ─────────────────────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path) -> Image.Image:
    """Load an image as RGB PIL Image."""
    return Image.open(path).convert("RGB")


def preprocess(img: Image.Image, input_size: tuple[int, int]) -> torch.Tensor:
    """
    Resize image to (H, W) and convert to a normalised float tensor.

    Returns shape: (1, 3, H, W)
    """
    pil_wh = (input_size[1], input_size[0])  # PIL wants (W, H)
    img_r  = img.resize(pil_wh, Image.BILINEAR)
    tensor = transforms.ToTensor()(img_r)     # [0, 1], (3, H, W)
    return tensor.unsqueeze(0)                # (1, 3, H, W)


def postprocess_matte(
    matte_t: torch.Tensor,
    original_size: tuple[int, int],
) -> Image.Image:
    """
    Convert model output tensor → PIL grayscale matte at original resolution.

    Parameters
    ----------
    matte_t      : (1, 1, H, W) float tensor in [0, 1]
    original_size: (W, H) original image size (PIL convention)

    Returns
    -------
    PIL Image mode "L", resized to original_size
    """
    matte_np = matte_t.squeeze().cpu().numpy()          # (H, W) float [0,1]
    matte_u8 = (matte_np * 255).clip(0, 255).astype(np.uint8)
    matte_pil = Image.fromarray(matte_u8, mode="L")
    return matte_pil.resize(original_size, Image.BILINEAR)


def make_composite(
    img: Image.Image,
    matte: Image.Image,
    bg_color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """
    Composite the subject over a solid background using the alpha matte.

    Parameters
    ----------
    img      : original RGB image (any size)
    matte    : single-channel L mask at same size as img
    bg_color : RGB tuple for background (default: chroma green)

    Returns
    -------
    RGB composite image
    """
    img_np   = np.array(img.convert("RGB"),   dtype=np.float32)
    alpha_np = np.array(matte,                dtype=np.float32) / 255.0
    alpha_3  = alpha_np[:, :, np.newaxis]

    bg_np    = np.full_like(img_np, bg_color, dtype=np.float32)
    comp_np  = alpha_3 * img_np + (1.0 - alpha_3) * bg_np
    return Image.fromarray(comp_np.clip(0, 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
#  Core prediction function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(
    model:      torch.nn.Module,
    img:        Image.Image,
    input_size: tuple[int, int],
    device:     torch.device,
) -> Image.Image:
    """
    Run the matting model on a single PIL Image.

    Parameters
    ----------
    model      : trained MattingModel (UNet or MobileNetDecoder)
    img        : PIL RGB image (any resolution)
    input_size : (H, W) the model was trained with
    device     : torch device

    Returns
    -------
    PIL Image mode "L" — alpha matte at original resolution
    """
    original_size = img.size          # (W, H) PIL convention
    tensor = preprocess(img, input_size).to(device)
    out    = model(tensor)            # (1, 1, H, W) in [0, 1]
    return postprocess_matte(out, original_size)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Human matting inference (Part A)"
    )
    p.add_argument(
        "--weights", required=True,
        help="Path to trained checkpoint (.pth)"
    )
    p.add_argument(
        "--input", required=True,
        help="Path to an image file or directory of images"
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: same as input image / input directory)"
    )
    p.add_argument(
        "--arch", default="unet",
        choices=["unet", "mobilenet_decoder"],
        help="Model architecture (must match the checkpoint, default: unet)"
    )
    p.add_argument(
        "--size", type=int, nargs=2, default=[320, 320],
        metavar=("H", "W"),
        help="Input size the model was trained on (default: 320 320)"
    )
    p.add_argument(
        "--visualise", action="store_true",
        help="Also save a green-screen composite for visual inspection"
    )
    p.add_argument(
        "--device", default=None,
        help="Force device: 'cpu' | 'cuda' | 'cuda:0' etc. (auto-detect if omitted)"
    )
    return p.parse_args()


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_inputs(input_path: Path) -> list[Path]:
    """Return a list of image paths from a file or directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        found = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in VALID_EXTS
        )
        if not found:
            raise FileNotFoundError(
                f"No images found in {input_path}\n"
                f"Accepted extensions: {VALID_EXTS}"
            )
        return found
    raise FileNotFoundError(f"Input not found: {input_path}")


def main() -> None:
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load model ────────────────────────────────────────────────────────
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = build_matting_model(args.arch)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Model  : {args.arch}   (loaded from {weights_path})")

    input_size = tuple(args.size)   # (H, W)
    print(f"Input size : {input_size[0]}×{input_size[1]}")

    # ── Collect inputs ────────────────────────────────────────────────────
    input_path = Path(args.input)
    img_paths  = collect_inputs(input_path)
    print(f"Images : {len(img_paths)}")

    # ── Output directory ─────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif input_path.is_dir():
        out_dir = input_path / "mattes"
    else:
        out_dir = input_path.parent / "mattes"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output : {out_dir}\n")

    # ── Predict ───────────────────────────────────────────────────────────
    for idx, img_path in enumerate(img_paths, 1):
        img   = load_image(img_path)
        matte = predict_image(model, img, input_size, device)

        # Save alpha matte
        matte_out = out_dir / (img_path.stem + "_matte.png")
        matte.save(matte_out)

        # Optional composite visualisation
        if args.visualise:
            # Resize original to matte size for composite
            img_resized = img.resize(matte.size, Image.BILINEAR)
            comp = make_composite(img_resized, matte)
            vis_out = out_dir / (img_path.stem + "_vis.png")
            comp.save(vis_out)
            extra = f"  | vis → {vis_out.name}"
        else:
            extra = ""

        print(f"  [{idx}/{len(img_paths)}]  {img_path.name}"
              f"  → {matte_out.name}{extra}")

    print(f"\n✓  Done.  Results saved to {out_dir}")


if __name__ == "__main__":
    main()
