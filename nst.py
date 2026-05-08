"""
nst.py
─────────────────────────────────────────────────────────────────────────────
Neural Style Transfer (Gatys, Ecker, Bethge — CVPR 2015)
Assignment 5 — Task 2  Part B  [ENHANCED]

Key upgrades over baseline:
  • Adaptive per-layer style weights (emphasise shallow + mid layers more)
  • Total Variation (TV) regularisation to reduce pixel noise
  • Histogram-matching warm-start: initialise generated image by colour-
    matching the content to the style palette before optimisation begins.
    This collapses the effective search space and produces richer blends.
  • Multi-scale optimisation (coarse→fine): first optimise at half resolution,
    then upsample and refine at full resolution.  Much better fine detail.
  • L-BFGS max_iter tuned per scale; Adam fallback with cosine LR schedule.
  • Clamping inside closure using in-place detach trick (avoids autograd graph
    corruption with L-BFGS).

Architecture
  • Backbone  : pretrained VGG19, FROZEN, eval() mode — never updated.
  • Content   : relu4_2  (mid-level structural features)
  • Style     : relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
                → Gram matrix of feature activations, normalised by H×W×C.
  • Optimise  : pixels of the generated image (initialised from content frame).
  • Loss      : L_total = α × L_content + β × L_style + γ × L_TV
  • Sweep     : three β/α ratios — 1e3, 1e5, 1e7.

Temporal consistency (Part C integration)
  • When `init_tensor` is supplied the optimisation starts from that tensor
    instead of the content frame, dramatically reducing inter-frame flicker.

Usage (standalone):
    python nst.py --content content.jpg --style style.jpg --out out.png
    python nst.py --content content.jpg --style style.jpg \\
                  --beta_ratio 1e5 --steps 300 --optim lbfgs --multiscale
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import VGG19_Weights

# ─────────────────────────────────────────────────────────────────────────────
#  VGG19 layer-name → module index mapping (features sequential)
# ─────────────────────────────────────────────────────────────────────────────

VGG19_LAYER_MAP: dict[str, int] = {
    "conv1_1": 0,  "relu1_1": 1,
    "conv1_2": 2,  "relu1_2": 3,
    "pool1"  : 4,
    "conv2_1": 5,  "relu2_1": 6,
    "conv2_2": 7,  "relu2_2": 8,
    "pool2"  : 9,
    "conv3_1": 10, "relu3_1": 11,
    "conv3_2": 12, "relu3_2": 13,
    "conv3_3": 14, "relu3_3": 15,
    "conv3_4": 16, "relu3_4": 17,
    "pool3"  : 18,
    "conv4_1": 19, "relu4_1": 20,
    "conv4_2": 21, "relu4_2": 22,
    "conv4_3": 23, "relu4_3": 24,
    "conv4_4": 25, "relu4_4": 26,
    "pool4"  : 27,
    "conv5_1": 28, "relu5_1": 29,
    "conv5_2": 30, "relu5_2": 31,
    "conv5_3": 32, "relu5_3": 33,
    "conv5_4": 34, "relu5_4": 35,
    "pool5"  : 36,
}

# Normalisation constants for VGG19 (ImageNet mean / std)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# Default adaptive layer weights: shallow layers carry more style signal
# (textures), deeper layers add structure.  Weights are then normalised inside
# the style loss so the absolute magnitudes remain stable as layers are added/
# removed.
_DEFAULT_STYLE_LAYER_WEIGHTS = {
    "relu1_1": 1.0,
    "relu2_1": 1.0,
    "relu3_1": 1.0,
    "relu4_1": 1.0,
    "relu5_1": 1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class VGG19FeatureExtractor(nn.Module):
    """
    Pretrained VGG19 feature extractor.

    • Always in eval() mode.
    • Parameters are frozen (requires_grad=False).
    • Returns a dict of named intermediate activations.
    • Applies ImageNet normalisation internally so callers pass [0,1] tensors.
    """

    def __init__(self, layer_names: list[str], device: torch.device):
        super().__init__()

        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features: nn.Sequential = vgg.features

        # Freeze ALL parameters — VGG is never updated
        for param in self.features.parameters():
            param.requires_grad_(False)
        self.features.eval()

        # Resolve requested layer names → feature indices
        self._layer_names = layer_names
        self._layer_indices: dict[str, int] = {}
        for name in layer_names:
            if name not in VGG19_LAYER_MAP:
                raise ValueError(
                    f"Unknown VGG19 layer: {name!r}. "
                    f"Valid options: {sorted(VGG19_LAYER_MAP.keys())}"
                )
            self._layer_indices[name] = VGG19_LAYER_MAP[name]

        self._max_idx = max(self._layer_indices.values())

        # ImageNet normalisation (registered as buffers — move with .to(device))
        self.register_buffer("mean", _IMAGENET_MEAN.view(1, 3, 1, 1))
        self.register_buffer("std",  _IMAGENET_STD.view(1, 3, 1, 1))

        self.to(device)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, H, W) tensor, values in [0, 1]

        Returns
        -------
        dict  {layer_name: feature_tensor}
        """
        # ImageNet normalisation
        x = (x - self.mean) / self.std

        outputs: dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            for name, target_idx in self._layer_indices.items():
                if idx == target_idx:
                    outputs[name] = x
            if idx >= self._max_idx:
                break

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
#  Gram matrix
# ─────────────────────────────────────────────────────────────────────────────

def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    Compute normalised Gram matrix of a feature map.

    Parameters
    ----------
    feat : (B, C, H, W)

    Returns
    -------
    G    : (B, C, C)  — G_ij = (1 / C·H·W) · Σ_hw feat_ih · feat_jh
    """
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (C * H * W)


# ─────────────────────────────────────────────────────────────────────────────
#  Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def content_loss(gen_feat: torch.Tensor, content_feat: torch.Tensor) -> torch.Tensor:
    """Mean-squared difference between generated and content feature maps."""
    return F.mse_loss(gen_feat, content_feat)


def style_loss(
    gen_feats:   dict[str, torch.Tensor],
    style_grams: dict[str, torch.Tensor],
    layer_weights: dict[str, float],
) -> torch.Tensor:
    """
    Weighted sum of self-normalised style loss across style layers.

    Each layer's loss is divided by the squared Frobenius norm of its own
    style Gram matrix.  This makes the loss dimensionless and ensures shallow
    layers (small spatial size, large gram values) and deep layers contribute
    equally — preventing deep layers from numerically drowning out the fine
    texture signal in relu1_1/relu2_1 that carries Van Gogh's swirls etc.
    """
    total_w = sum(layer_weights.get(n, 1.0) for n in style_grams)
    if total_w == 0:
        total_w = 1.0

    loss = torch.tensor(0.0, device=next(iter(gen_feats.values())).device)
    for name in style_grams:
        w       = layer_weights.get(name, 1.0) / total_w
        G_gen   = gram_matrix(gen_feats[name])
        G_style = style_grams[name]
        # Self-normalise by the style gram's own energy — key fix
        denom   = (2.0 * (G_style.norm() ** 2)).clamp(min=1e-8)
        loss    = loss + w * torch.sum((G_gen - G_style) ** 2) / denom
    return loss


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic Total Variation regularisation.
    Encourages spatial smoothness in the generated image.
    """
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


# ─────────────────────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Path | str, size: int | None = None) -> torch.Tensor:
    """
    Load a PIL image → (1, 3, H, W) float tensor in [0, 1].
    """
    img = Image.open(path).convert("RGB")
    if size is not None:
        w, h = img.size
        if h < w:
            new_h = size
            new_w = int(w * size / h)
        else:
            new_w = size
            new_h = int(h * size / w)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    t = transforms.ToTensor()(img)
    return t.unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert (1, 3, H, W) or (3, H, W) tensor in [0,1] → PIL RGB Image."""
    t = t.squeeze(0).clamp(0, 1).cpu()
    return transforms.ToPILImage()(t)


def histogram_match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Match the colour histogram of `source` to `reference`.
    Both are (1, 3, H, W) tensors in [0, 1].
    Returns a colour-adjusted version of `source` with matched palette.

    Uses channel-wise CDF matching (Reinhard-style).
    """
    out = source.clone()
    for c in range(3):
        src = source[0, c].cpu().numpy().flatten()
        ref = reference[0, c].cpu().numpy().flatten()

        # CDF of source
        src_sorted = np.sort(src)
        src_counts, src_bins = np.histogram(src, bins=256, range=(0, 1))
        src_cdf = np.cumsum(src_counts).astype(float) / src_counts.sum()
        src_bin_centers = (src_bins[:-1] + src_bins[1:]) / 2

        # CDF of reference
        ref_counts, ref_bins = np.histogram(ref, bins=256, range=(0, 1))
        ref_cdf = np.cumsum(ref_counts).astype(float) / ref_counts.sum()
        ref_bin_centers = (ref_bins[:-1] + ref_bins[1:]) / 2

        # Build lookup: for each source bin value, find matching ref bin
        lookup = np.interp(src_cdf, ref_cdf, ref_bin_centers)

        # Map source pixels
        src_mapped = np.interp(src, src_bin_centers, lookup)
        h, w = source.shape[2], source.shape[3]
        out[0, c] = torch.tensor(src_mapped.reshape(h, w), dtype=torch.float32)

    return out.clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Core NST optimisation
# ─────────────────────────────────────────────────────────────────────────────

def run_nst(
    content_tensor: torch.Tensor,
    style_tensor:   torch.Tensor,
    device:         torch.device,
    content_layer:  str  = "relu4_2",
    style_layers:   list[str] = None,
    style_layer_weights: list[float] = None,
    content_weight: float = 1.0,        # α
    style_weight:   float = 1e6,        # β  — raised default; loss is now self-normalised
    tv_weight:      float = 5e-5,       # γ — lowered: let brushstrokes come through
    num_steps:      int   = 400,        # more steps for deeper style penetration
    optimizer:      str   = "lbfgs",    # "lbfgs" | "adam"
    init_tensor:    Optional[torch.Tensor] = None,
    histogram_init: bool  = True,       # colour-match warm-start
    multiscale:     bool  = False,      # coarse→fine optimisation
    verbose:        bool  = True,
    log_every:      int   = 50,
) -> torch.Tensor:
    """
    Run Neural Style Transfer.

    Parameters
    ----------
    content_tensor       : (1, 3, H, W) in [0,1]  — the content image / frame
    style_tensor         : (1, 3, H, W) in [0,1]  — the style artwork
    device               : torch.device
    content_layer        : VGG19 layer name for content loss
    style_layers         : list of VGG19 layer names for style loss
    style_layer_weights  : per-layer weights (defaults to adaptive schedule)
    content_weight       : α — scaling factor for content loss
    style_weight         : β — scaling factor for style loss
    tv_weight            : γ — total variation regularisation weight
    num_steps            : number of optimisation steps
    optimizer            : "lbfgs" or "adam"
    init_tensor          : optional (1,3,H,W) tensor to initialise from
                           (supply previous stylised frame for temporal consistency)
    histogram_init       : apply histogram matching warm-start when init_tensor=None
    multiscale           : run coarse→fine two-stage optimisation
    verbose              : print loss every log_every steps
    log_every            : print interval

    Returns
    -------
    (1, 3, H, W) float tensor in [0, 1] — stylised image
    """
    if style_layers is None:
        style_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]

    # Build adaptive layer weight map
    if style_layer_weights is None:
        layer_weight_map = {
            name: _DEFAULT_STYLE_LAYER_WEIGHTS.get(name, 1.0)
            for name in style_layers
        }
    else:
        layer_weight_map = dict(zip(style_layers, style_layer_weights))

    # Multi-scale: first run at half-res, upsample, then refine at full-res
    if multiscale:
        H, W = content_tensor.shape[2], content_tensor.shape[3]
        H_half, W_half = max(H // 2, 64), max(W // 2, 64)

        # Coarse stage (half resolution, fewer steps)
        c_half = F.interpolate(content_tensor, (H_half, W_half), mode="bilinear", align_corners=False)
        s_half = F.interpolate(style_tensor,   (H_half, W_half), mode="bilinear", align_corners=False)
        init_half = None
        if init_tensor is not None:
            init_half = F.interpolate(init_tensor, (H_half, W_half), mode="bilinear", align_corners=False)

        coarse = _run_nst_single_scale(
            c_half, s_half, device,
            content_layer, style_layers, layer_weight_map,
            content_weight, style_weight, tv_weight,
            max(num_steps // 3, 50), optimizer,
            init_half, histogram_init, verbose=False,
        )

        # Upsample coarse result to full res for fine-stage init
        fine_init = F.interpolate(coarse, (H, W), mode="bilinear", align_corners=False)

        # Fine stage (full resolution, remaining steps)
        return _run_nst_single_scale(
            content_tensor, style_tensor, device,
            content_layer, style_layers, layer_weight_map,
            content_weight, style_weight, tv_weight,
            num_steps, optimizer,
            fine_init, histogram_init=False,  # already matched in coarse stage
            verbose=verbose, log_every=log_every,
        )
    else:
        return _run_nst_single_scale(
            content_tensor, style_tensor, device,
            content_layer, style_layers, layer_weight_map,
            content_weight, style_weight, tv_weight,
            num_steps, optimizer,
            init_tensor, histogram_init,
            verbose=verbose, log_every=log_every,
        )


def _run_nst_single_scale(
    content_tensor:  torch.Tensor,
    style_tensor:    torch.Tensor,
    device:          torch.device,
    content_layer:   str,
    style_layers:    list[str],
    layer_weight_map: dict[str, float],
    content_weight:  float,
    style_weight:    float,
    tv_weight:       float,
    num_steps:       int,
    optimizer:       str,
    init_tensor:     Optional[torch.Tensor],
    histogram_init:  bool,
    verbose:         bool,
    log_every:       int = 50,
) -> torch.Tensor:
    """Single-scale NST optimisation loop."""

    all_layers = list(set([content_layer] + style_layers))
    extractor = VGG19FeatureExtractor(all_layers, device)
    extractor.eval()

    content_tensor = content_tensor.to(device)
    style_tensor   = style_tensor.to(device)

    # NOTE: Do NOT resize style to match content here.
    # The Gram matrix is spatially invariant — style statistics are richer
    # when computed at the style image's own (higher) resolution.
    # We only need style and content at the same size for the *generated* image,
    # not for computing the target Gram matrices.

    # Pre-compute targets
    with torch.no_grad():
        content_feats  = extractor(content_tensor)
        content_target = content_feats[content_layer].detach()

        style_feats = extractor(style_tensor)
        style_grams = {
            name: gram_matrix(style_feats[name]).detach()
            for name in style_layers
        }

    # Initialise generated image
    if init_tensor is not None:
        init = init_tensor.clone().to(device)
        if init.shape[2:] != content_tensor.shape[2:]:
            init = F.interpolate(init, content_tensor.shape[2:], mode="bilinear", align_corners=False)
        gen = init.clamp(0, 1)
    elif style_weight >= 1e6 and histogram_init:
        # For high style weights, start from the style image resized to content dims.
        # This puts us in a region of pixel-space already rich with style colour/texture,
        # avoiding the local minimum where the optimiser keeps the content colours and
        # only partially transfers the style palette (causing "flat blue" artefacts).
        gen = F.interpolate(
            style_tensor.clone(), size=content_tensor.shape[2:],
            mode="bilinear", align_corners=False,
        ).clamp(0, 1)
    elif histogram_init:
        # Moderate style weights: colour-match warm-start is sufficient
        gen = histogram_match(content_tensor.cpu(), style_tensor.cpu()).to(device)
    else:
        gen = content_tensor.clone()

    gen = gen.requires_grad_(True)

    # Optimiser setup
    if optimizer.lower() == "lbfgs":
        optim = torch.optim.LBFGS([gen], lr=1.0, max_iter=20,
                                   line_search_fn="strong_wolfe")
    elif optimizer.lower() == "adam":
        optim = torch.optim.Adam([gen], lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_steps, eta_min=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}. Use 'lbfgs' or 'adam'.")

    step   = [0]

    def closure():
        # Clamp without breaking the autograd graph (detach + re-attach)
        with torch.no_grad():
            gen.data.clamp_(0, 1)
        optim.zero_grad()

        gen_feats = extractor(gen)

        c_loss = content_loss(gen_feats[content_layer], content_target)
        s_loss = style_loss(gen_feats, style_grams, layer_weight_map)
        tv_l   = total_variation_loss(gen)

        total = content_weight * c_loss + style_weight * s_loss + tv_weight * tv_l
        total.backward()

        step[0] += 1
        if verbose and step[0] % log_every == 0:
            print(
                f"  Step {step[0]:4d}/{num_steps} | "
                f"Total={total.item():.4e} | "
                f"Content={c_loss.item():.4e} | "
                f"Style={s_loss.item():.4e} | "
                f"TV={tv_l.item():.4e}"
            )
        return total

    if optimizer.lower() == "lbfgs":
        lbfgs_steps = max(1, num_steps // 20)
        for _ in range(lbfgs_steps):
            optim.step(closure)
    else:
        for _ in range(num_steps):
            optim.step(closure)
            scheduler.step()

    with torch.no_grad():
        gen.data.clamp_(0, 1)

    return gen.detach()


# ─────────────────────────────────────────────────────────────────────────────
#  Style-ratio sweep helper
# ─────────────────────────────────────────────────────────────────────────────

def style_ratio_sweep(
    content_tensor: torch.Tensor,
    style_tensor:   torch.Tensor,
    device:         torch.device,
    beta_ratios:    list[float] = (1e3, 1e5, 1e7),
    content_weight: float = 1.0,
    **nst_kwargs,
) -> dict[float, torch.Tensor]:
    """
    Run NST for each β/α ratio and return a dict {beta_ratio: stylised_tensor}.
    """
    results: dict[float, torch.Tensor] = {}
    for ratio in beta_ratios:
        beta = ratio * content_weight
        print(f"\n{'─'*60}")
        print(f"  β/α = {ratio:.0e}   (α={content_weight}, β={beta:.0e})")
        print(f"{'─'*60}")
        out = run_nst(
            content_tensor=content_tensor,
            style_tensor=style_tensor,
            device=device,
            content_weight=content_weight,
            style_weight=beta,
            **nst_kwargs,
        )
        results[ratio] = out
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI — standalone usage
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neural Style Transfer — Part B (Assignment 5, Task 2) [Enhanced]"
    )
    p.add_argument("--content",    required=True, help="Path to content image")
    p.add_argument("--style",      required=True, help="Path to style image")
    p.add_argument("--out",        default="nst_output.png",
                   help="Output path (single run) or directory prefix (sweep mode)")
    p.add_argument("--beta_ratio", type=float, default=1e5,
                   help="β/α ratio  (default: 1e5). Ignored in --sweep mode.")
    p.add_argument("--sweep",      action="store_true",
                   help="Run all three β/α ratios: 1e3, 1e5, 1e7")
    p.add_argument("--steps",      type=int, default=300,
                   help="Optimisation steps (default: 300)")
    p.add_argument("--optim",      default="lbfgs", choices=["lbfgs", "adam"],
                   help="Pixel optimizer (default: lbfgs)")
    p.add_argument("--size",       type=int, default=512,
                   help="Shorter-edge resize before NST (default: 512)")
    p.add_argument("--tv_weight",  type=float, default=1e-4,
                   help="TV regularisation weight (default: 1e-4)")
    p.add_argument("--multiscale", action="store_true",
                   help="Enable coarse→fine multi-scale optimisation")
    p.add_argument("--no_histogram", action="store_true",
                   help="Disable histogram-matching warm-start")
    p.add_argument("--content_layer", default="relu4_2",
                   help="VGG19 content layer (default: relu4_2)")
    p.add_argument("--style_layers",  nargs="+",
                   default=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
                   help="VGG19 style layers")
    p.add_argument("--device",     default=None,
                   help="Force device: cpu | cuda | cuda:0 (auto if omitted)")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress per-step logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    content = load_image(args.content, size=args.size).to(device)
    style   = load_image(args.style,   size=args.size).to(device)
    print(f"Content : {Path(args.content).name}  {tuple(content.shape)}")
    print(f"Style   : {Path(args.style).name}    {tuple(style.shape)}")

    nst_kwargs = dict(
        content_layer=args.content_layer,
        style_layers=args.style_layers,
        num_steps=args.steps,
        optimizer=args.optim,
        tv_weight=args.tv_weight,
        multiscale=args.multiscale,
        histogram_init=not args.no_histogram,
        verbose=not args.quiet,
    )

    if args.sweep:
        results = style_ratio_sweep(
            content, style, device,
            beta_ratios=[1e3, 1e5, 1e7],
            **nst_kwargs,
        )
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ratio, t in results.items():
            fname = out_dir / f"nst_beta_{ratio:.0e}.png"
            tensor_to_pil(t).save(fname)
            print(f"  Saved: {fname}")

        _save_sweep_comparison(
            content, style, results,
            out_path=out_dir / "nst_sweep_comparison.png",
        )
    else:
        out = run_nst(
            content, style, device,
            style_weight=args.beta_ratio,
            **nst_kwargs,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(out).save(out_path)
        print(f"\n✓  Saved: {out_path}")


def _save_sweep_comparison(
    content: torch.Tensor,
    style:   torch.Tensor,
    results: dict[float, torch.Tensor],
    out_path: Path,
) -> None:
    """Save a side-by-side comparison of content, style, and all three ratios."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        cols   = ["Content", "Style"] + [f"β/α = {r:.0e}" for r in results]
        images = [tensor_to_pil(content), tensor_to_pil(style)] + [
            tensor_to_pil(t) for t in results.values()
        ]
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        fig.patch.set_facecolor("white")
        for ax, img, title in zip(axes, images, cols):
            ax.imshow(np.array(img))
            ax.set_title(title, fontsize=12, fontweight="bold", color="black")
            ax.axis("off")

        plt.suptitle("NST Style Ratio Sweep (Gatys et al. 2015) — Enhanced", fontsize=14, color="black")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"\n✓  Sweep comparison saved: {out_path}")
    except ImportError:
        print("  matplotlib not available — sweep comparison figure skipped.")


if __name__ == "__main__":
    main()
