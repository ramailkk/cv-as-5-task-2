"""
nst.py - Neural Style Transfer (Gatys et al., 2015) using pretrained VGG19

Public API
----------
  run_nst(content_img, style_img, cfg) -> stylized_tensor

  content_img : (1, 3, H, W) float32 tensor in [0, 255] ImageNet-normalised
  style_img   : same format
  cfg         : dict (see config.yaml, task2.nst section)
  returns     : (1, 3, H, W) same format as input

Extra cfg keys (optional):
  style_layers       : list of int indices of VGG19 output layers to use for style
                       (default: [0,1,2,3,5]  → relu1_1, relu2_1, relu3_1, relu4_1, relu5_1)
  style_layer_weights: list of floats, same length as style_layers (default: all 1.0)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, LBFGS
import torchvision.models as tv_models
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# ImageNet statistics
# ---------------------------------------------------------------------------
IMAGENET_MEAN_255 = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
IMAGENET_STD_1    = torch.tensor([1.0,     1.0,    1.0   ]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Image I/O utilities
# ---------------------------------------------------------------------------

def load_img_as_tensor(img_path: str, height: int, device: torch.device) -> torch.Tensor:
    """
    Load an image from disk, resize to `height` (preserving aspect ratio),
    and return a (1, 3, H, W) float32 tensor in [0, 255], ImageNet-normalised.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    new_w = int(w * height / h)
    img = cv2.resize(img, (new_w, height), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)            # [0, 255]
    t   = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    t    = (t.to(device) - IMAGENET_MEAN_255.to(device))
    return t


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (1,3,H,W) ImageNet-normalised tensor → uint8 RGB."""
    img = (t + IMAGENET_MEAN_255.to(t.device)).squeeze(0).permute(1,2,0).detach().cpu().numpy()
    return np.clip(img, 0, 255).astype(np.uint8)


def save_tensor_as_img(t: torch.Tensor, path: str):
    img = tensor_to_uint8(t)
    cv2.imwrite(path, img[:, :, ::-1])  # RGB → BGR


# ---------------------------------------------------------------------------
# VGG19 feature extractor (returns 6 layers)
# ---------------------------------------------------------------------------

class VGG19Features(nn.Module):
    """
    Pretrained VGG19 with six intermediate outputs:
      index 0: relu1_1, 1: relu2_1, 2: relu3_1, 3: relu4_1, 4: conv4_2, 5: relu5_1
    """
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        cuts = [1, 6, 11, 20, 21, 29]   # inclusive end indices for each slice
        self.slices = nn.ModuleList()
        prev = 0
        for cut in cuts:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, cut + 1)]))
            prev = cut + 1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for s in self.slices:
            x = s(x)
            outs.append(x)
        return outs  # list of 6 tensors


# ---------------------------------------------------------------------------
# Gram matrix & TV loss
# ---------------------------------------------------------------------------

def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    feat = x.view(b, c, h * w)
    gram = feat.bmm(feat.transpose(1, 2))
    return gram / (c * h * w)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))


# ---------------------------------------------------------------------------
# Core NST routine (supports custom style layers)
# ---------------------------------------------------------------------------

def run_nst(
    content_tensor: torch.Tensor,
    style_tensor:   torch.Tensor,
    cfg:            dict,
    init_tensor:    torch.Tensor | None = None,
    verbose:        bool = False,
) -> torch.Tensor:
    """
    cfg can contain:
      content_weight (default 1e5)
      style_weight   (default 3e4)
      tv_weight      (default 1.0)
      iterations     (default 500)
      optimizer      ('lbfgs' or 'adam', default 'lbfgs')
      adam_lr        (default 1e1)
      height         (for loading, not used here)
      style_layers   : list of int indices from VGG19Features (default [0,1,2,3,5])
      style_layer_weights : list of floats (default all 1.0)
    """
    device = content_tensor.device
    vgg = VGG19Features().to(device).eval()

    # ---- extract target features ----
    with torch.no_grad():
        content_feats = vgg(content_tensor)
        style_feats   = vgg(style_tensor)

    target_content = content_feats[4].squeeze(0).detach()   # conv4_2 is index 4

    # ---- custom style layers ----
    style_idxs = cfg.get("style_layers", [0, 1, 2, 3, 5])   # default relu1_1 ... relu5_1
    style_weights = cfg.get("style_layer_weights", [1.0] * len(style_idxs))
    if len(style_weights) != len(style_idxs):
        style_weights = [1.0] * len(style_idxs)

    target_style = []
    for idx in style_idxs:
        gram = gram_matrix(style_feats[idx]).detach()
        target_style.append(gram)

    # ---- initialise optimisation image ----
    if init_tensor is not None:
        opt_img = init_tensor.clone().detach().requires_grad_(True).contiguous()
    else:
        opt_img = content_tensor.clone().detach().requires_grad_(True).contiguous()

    content_w = cfg.get("content_weight", 1e5)
    style_w   = cfg.get("style_weight",   3e4)
    tv_w      = cfg.get("tv_weight",      1.0)
    optimizer_name = cfg.get("optimizer", "lbfgs").lower()
    max_iter = cfg.get("iterations", 500)
    lr = cfg.get("adam_lr", 1e1)

    # ---- loss closure ----
    def compute_loss():
        feats = vgg(opt_img)

        # content loss (index 4)
        c_loss = F.mse_loss(feats[4].squeeze(0), target_content)

        # style loss (only requested layers)
        s_loss = 0.0
        for idx_out, idx in enumerate(style_idxs):
            cur_gram = gram_matrix(feats[idx])
            tgt_gram = target_style[idx_out]
            s_loss += style_weights[idx_out] * F.mse_loss(cur_gram, tgt_gram)
        s_loss = s_loss / len(style_idxs)

        tv_loss = total_variation(opt_img)
        total = content_w * c_loss + style_w * s_loss + tv_w * tv_loss
        return total, c_loss, s_loss, tv_loss

    # ---- optimisation ----
    if optimizer_name == "adam":
        opt = Adam([opt_img], lr=lr)
        for it in range(max_iter):
            opt.zero_grad()
            total, c, s, tv = compute_loss()
            total.backward()
            opt.step()
            if verbose and it % 50 == 0:
                print(f"  Adam iter {it:04d} | total={total.item():.2f} "
                      f"content={content_w*c.item():.2f} "
                      f"style={style_w*s.item():.2f} "
                      f"tv={tv_w*tv.item():.2f}")

    elif optimizer_name == "lbfgs":
        opt = LBFGS([opt_img], max_iter=max_iter, line_search_fn="strong_wolfe")
        cnt = [0]
        def closure():
            opt.zero_grad()
            total, c, s, tv = compute_loss()
            total.backward()
            if verbose and cnt[0] % 50 == 0:
                print(f"  L-BFGS iter {cnt[0]:04d} | total={total.item():.2f} "
                      f"content={content_w*c.item():.2f} "
                      f"style={style_w*s.item():.2f} "
                      f"tv={tv_w*tv.item():.2f}")
            cnt[0] += 1
            return total
        opt.step(closure)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return opt_img.detach()