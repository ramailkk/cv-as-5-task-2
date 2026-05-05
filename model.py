"""
model.py
─────────────────────────────────────────────────────────────────────────────
Human matting models for Assignment 5 – Task 2 (Part A).

Two options (set in config.yaml):
  • "unet"               — classic encoder-decoder with skip connections
  • "mobilenet_decoder"  — MobileNetV2 backbone (weights=None) + lightweight decoder

Both output a single-channel alpha matte ∈ [0, 1].
Input:  (B, 3, H, W)  — RGB frame, already resized to config input_size
Output: (B, 1, H, W)  — alpha matte
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """3×3 conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pad: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive ConvBnRelu blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
#  U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNetDown(nn.Module):
    """Encoder block: DoubleConv → MaxPool. Returns (pooled, skip)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class UNetUp(nn.Module):
    """Decoder block: bilinear upsample × 2 → concat skip → DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial dimensions (pad to match skip)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for human matting.

    Encoder:     3 → 64 → 128 → 256 → 512
    Bottleneck:  512 → 1024
    Decoder:     1024+512 → 512 → 256+256 → 256 → 128+128 → 128 →
                 64+64 → 64 → 1
    Output: sigmoid(head) ∈ [0, 1]    shape: (B, 1, H, W)
    ~31 M parameters — trained from scratch as required.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = UNetDown(3,   64)
        self.down2 = UNetDown(64,  128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        # Decoder (in_ch = upsampled + skip channels)
        self.up4 = UNetUp(1024 + 512, 512)
        self.up3 = UNetUp(512  + 256, 256)
        self.up2 = UNetUp(256  + 128, 128)
        self.up1 = UNetUp(128  +  64,  64)
        # Output head
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, s1 = self.down1(x)   # s1: (B,  64, H/2,  W/2)
        x, s2 = self.down2(x)   # s2: (B, 128, H/4,  W/4)
        x, s3 = self.down3(x)   # s3: (B, 256, H/8,  W/8)
        x, s4 = self.down4(x)   # s4: (B, 512, H/16, W/16)
        x     = self.bottleneck(x)  # (B,1024, H/32, W/32) — note: no pool before BN
        x     = self.up4(x, s4)
        x     = self.up3(x, s3)
        x     = self.up2(x, s2)
        x     = self.up1(x, s1)
        return torch.sigmoid(self.head(x))   # (B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  MobileNetV2 backbone + lightweight decoder
# ─────────────────────────────────────────────────────────────────────────────

class MobileNetDecoder(nn.Module):
    """
    Encoder: MobileNetV2 feature layers — weights=None (no pretrained weights).
    Decoder: four lightweight upsample + conv blocks with skip connections.

    Skip taps from MobileNetV2's features list:
        Index  Stride  Out-channels
         1       2       16
         3       4       24
         6       8       32
        13      16       96
        18      32     1280  (bottleneck, after final conv)

    The decoder upsamples step-by-step back to full resolution.
    A final bilinear upsample × 2 recovers the stride-2 → full-res gap.
    """

    # MobileNetV2 `features` layer indices → (stride_factor, channels)
    SKIP_INDICES = {
        "s2" : 1,   # stride ×2,  ch=16
        "s4" : 3,   # stride ×4,  ch=24
        "s8" : 6,   # stride ×8,  ch=32
        "s16": 13,  # stride ×16, ch=96
        "s32": 18,  # stride ×32, ch=1280 (last features layer)
    }

    def __init__(self):
        super().__init__()
        # Load MobileNetV2 WITHOUT pretrained weights (task requirement)
        backbone = mobilenet_v2(weights=None)
        self.enc_features = backbone.features   # ModuleList of 19 layers

        # Decoder blocks — each does: conv only (no upsample inside)
        # Upsampling is done via F.interpolate in forward() for exact size matching.
        self.dec4 = self._conv_block(1280 + 96,  256)   # s32+s16  → 256
        self.dec3 = self._conv_block(256  + 32,  128)   # +s8      → 128
        self.dec2 = self._conv_block(128  + 24,   64)   # +s4      →  64
        self.dec1 = self._conv_block(64   + 16,   32)   # +s2      →  32

        # Final upsample ×2 (stride-2 → full res) + head
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.head     = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Two ConvBnRelu layers — no spatial change."""
        return nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def _upsample_cat(self, feat: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Bilinear upsample feat to skip's spatial size, then concatenate."""
        feat = F.interpolate(feat, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return torch.cat([feat, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: dict = {}
        for i, layer in enumerate(self.enc_features):
            x = layer(x)
            if i == self.SKIP_INDICES["s2"]:
                skips["s2"] = x
            elif i == self.SKIP_INDICES["s4"]:
                skips["s4"] = x
            elif i == self.SKIP_INDICES["s8"]:
                skips["s8"] = x
            elif i == self.SKIP_INDICES["s16"]:
                skips["s16"] = x
        # x is now at stride-32 (bottleneck, ch=1280)

        x = self.dec4(self._upsample_cat(x,         skips["s16"]))  # → stride-16, ch=256
        x = self.dec3(self._upsample_cat(x,         skips["s8"]))   # → stride-8,  ch=128
        x = self.dec2(self._upsample_cat(x,         skips["s4"]))   # → stride-4,  ch=64
        x = self.dec1(self._upsample_cat(x,         skips["s2"]))   # → stride-2,  ch=32
        x = self.final_up(x)                                         # → stride-1 (full res)
        return torch.sigmoid(self.head(x))                           # (B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_matting_model(arch: str) -> nn.Module:
    """Instantiate and return the requested matting architecture."""
    arch = arch.lower().strip()
    if arch == "unet":
        return UNet()
    elif arch in ("mobilenet_decoder", "mobilenetv2"):
        return MobileNetDecoder()
    else:
        raise ValueError(
            f"Unknown matting architecture: {arch!r}. "
            "Choose 'unet' or 'mobilenet_decoder'."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test  (python model.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running model smoke test …")
    for arch in ("unet", "mobilenet_decoder"):
        model = build_matting_model(arch)
        model.eval()
        for H, W in [(256, 256), (320, 320)]:
            x   = torch.randn(2, 3, H, W)
            out = model(x)
            assert out.shape == (2, 1, H, W), \
                f"[{arch}] Expected (2,1,{H},{W}), got {tuple(out.shape)}"
            assert out.min() >= 0.0 and out.max() <= 1.0, \
                f"[{arch}] Alpha matte out of [0,1] range!"
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {arch:20s}  params={params/1e6:.2f}M   [OK]")
    print("All smoke tests passed.")
