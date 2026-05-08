"""
dataset.py
─────────────────────────────────────────────────────────────────────────────
Standalone AISegment dataset module (Part A — Human Matting).

Importing from here keeps train.py and predict.py DRY:
    from dataset import AISegmentDataset, build_dataloaders

AISegment Kaggle URL:
    https://www.kaggle.com/datasets/laurentmih/aisegmentcom-matting-human-datasets

Expected directory layout (dataset_root):
    clip_img/<subdir>/<subdir>/<name>_clip.jpg   ← RGB portrait
    matting/<subdir>/<subdir>/<name>.png          ← single-channel alpha matte

Augmentations (training only):
    • Random horizontal flip
    • Random crop  (90–100 % of shorter side, preserves aspect ratio)
    • Random vertical flip (20 % probability)
    • Colour jitter (brightness ±0.3, contrast ±0.3, saturation ±0.2, hue ±0.05)
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AISegmentDataset(Dataset):
    """
    Loads (RGB image, alpha matte) pairs from the AISegment dataset.

    Parameters
    ----------
    root      : str | Path  — dataset_root (contains clip_img/ and matting/)
    input_size: (H, W)      — spatial size after resize
    augment   : bool        — apply training augmentations
    """

    def __init__(
        self,
        root: str | Path,
        input_size: tuple[int, int] = (320, 320),
        augment: bool = False,
    ):
        self.root       = Path(root)
        self.input_size = input_size   # (H, W)
        self.augment    = augment

        # ── Discover clip images ───────────────────────────────────────────
        candidates = list(self.root.glob("clip_img/**/*_clip.jpg"))
        if not candidates:
            candidates = list(self.root.glob("clip_img/**/*.jpg"))
        if not candidates:
            candidates = list(self.root.glob("clip_img/**/*.png"))

        if not candidates:
            raise FileNotFoundError(
                f"No images found under {self.root / 'clip_img'}.\n"
                "Make sure 'dataset_root' in config.yaml points to the extracted "
                "AISegment folder containing clip_img/ and matting/ sub-directories."
            )

        # Keep only pairs where the matte also exists
        self.image_paths: list[Path] = []
        missing = 0
        for p in sorted(candidates):
            mp = self._get_matte_path(p)
            if mp.exists():
                self.image_paths.append(p)
            else:
                missing += 1

        if not self.image_paths:
            raise FileNotFoundError(
                f"Found {len(candidates)} clip images but 0 matching mattes under "
                f"{self.root / 'matting'}.\n"
                "Verify the dataset has been extracted correctly."
            )

        if missing:
            print(f"[AISegmentDataset] WARNING: {missing} images skipped "
                  "(no matching matte found).")

        print(f"[AISegmentDataset] {len(self.image_paths)} valid pairs | "
              f"size={input_size} | augment={augment}")

        self._jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

    # ── Path resolution ────────────────────────────────────────────────────

    def _get_matte_path(self, img_path: Path) -> Path:
        """
        Resolve the corresponding matte path for a given clip image.
        
        Example:
        clip_img/1803151818/clip_00000000/1803151818-00000003.jpg
        → matting/1803151818/matting_00000000/1803151818-00000003.png
        """
        parts = list(img_path.parts)
        
        # Find the clip_img component and replace it with matting
        try:
            ci_idx = parts.index("clip_img")
        except ValueError:
            # Fallback: string replacement on full path
            alt = Path(str(img_path).replace("clip_img", "matting", 1))
            alt = alt.with_suffix(".png")
            return alt
        
        parts[ci_idx] = "matting"
        
        # Change folder name from "clip_XXXXXXX" to "matting_XXXXXXX"
        # Find the folder that starts with "clip_"
        for i, part in enumerate(parts):
            if part.startswith("clip_"):
                parts[i] = part.replace("clip_", "matting_", 1)
                break
        
        # Change extension from .jpg to .png
        parts[-1] = img_path.stem + ".png"
        
        return Path(*parts)

    # ── Spatial augmentations (identical transform on image + matte) ───────

    def _joint_augment(
        self,
        img: Image.Image,
        matte: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        # Random horizontal flip (50 %)
        if random.random() > 0.5:
            img   = TF.hflip(img)
            matte = TF.hflip(matte)

        # Random crop — 90–100 % of the shorter side
        w, h = img.size
        scale = random.uniform(0.9, 1.0)
        cw = max(1, int(w * scale))
        ch = max(1, int(h * scale))
        i  = random.randint(0, max(0, h - ch))
        j  = random.randint(0, max(0, w - cw))
        img   = TF.crop(img,   i, j, ch, cw)
        matte = TF.crop(matte, i, j, ch, cw)

        # Random vertical flip (20 %)
        if random.random() > 0.8:
            img   = TF.vflip(img)
            matte = TF.vflip(matte)

        return img, matte

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path   = self.image_paths[idx]
        matte_path = self._get_matte_path(img_path)

        img   = Image.open(img_path).convert("RGB")
        matte = Image.open(matte_path)

        # Normalise matte to a single grayscale channel
        if matte.mode == "RGBA":
            matte = matte.split()[3]      # use alpha channel
        elif matte.mode == "RGB":
            matte = matte.split()[0]      # channels should be identical; take R
        else:
            matte = matte.convert("L")

        # Spatial augmentation (same transform on both)
        if self.augment:
            img, matte = self._joint_augment(img, matte)

        # Resize to target input_size
        pil_wh = (self.input_size[1], self.input_size[0])  # PIL uses (W, H)
        img   = img.resize(pil_wh,   Image.BILINEAR)
        matte = matte.resize(pil_wh, Image.BILINEAR)

        # Colour jitter on image only (after spatial ops)
        if self.augment:
            img = self._jitter(img)

        img_t   = transforms.ToTensor()(img)    # float32 [0,1], shape (3, H, W)
        matte_t = transforms.ToTensor()(matte)  # float32 [0,1], shape (1, H, W)
        return img_t, matte_t


# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from the config dict.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    root    = cfg["data"]["dataset_root"]
    size    = tuple(cfg["matting"]["input_size"])
    bs      = cfg["matting"]["batch_size"]
    nw      = cfg["matting"]["num_workers"]
    n_train = cfg["matting"]["train_size"]
    n_val   = cfg["matting"]["val_size"]
    n_test  = cfg["matting"]["test_size"]
    seed    = cfg.get("seed", 42)

    # Single scan to find all valid pairs
    probe_ds = AISegmentDataset(root, input_size=size, augment=False)
    total    = len(probe_ds)
    need     = n_train + n_val + n_test

    if total < need:
        raise RuntimeError(
            f"Dataset only has {total} valid pairs; config requests "
            f"{n_train}+{n_val}+{n_test}={need}. "
            "Reduce split sizes in config.yaml or point to the full dataset."
        )

    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val : n_train + n_val + n_test]

    aug_ds   = AISegmentDataset(root, input_size=size, augment=True)
    train_ds = Subset(aug_ds,   train_idx)
    val_ds   = Subset(probe_ds, val_idx)
    test_ds  = Subset(probe_ds, test_idx)

    print(f"Dataset split — train: {len(train_ds)}  "
          f"val: {len(val_ds)}  test: {len(test_ds)}")

    _kw = dict(
        batch_size=bs,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **_kw)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke-test  (python dataset.py --root <path>)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Path to AISegment dataset_root")
    ap.add_argument("--size", type=int, nargs=2, default=[320, 320],
                    metavar=("H", "W"))
    ap.add_argument("--n",    type=int, default=3,
                    help="Number of samples to load and print info for")
    args = ap.parse_args()

    ds = AISegmentDataset(args.root, input_size=tuple(args.size), augment=True)
    print(f"Total valid pairs: {len(ds)}")
    for i in range(min(args.n, len(ds))):
        img, matte = ds[i]
        assert img.shape   == (3, args.size[0], args.size[1]), f"Image shape mismatch at {i}"
        assert matte.shape == (1, args.size[0], args.size[1]), f"Matte shape mismatch at {i}"
        assert matte.min() >= 0.0 and matte.max() <= 1.0,      f"Matte out of [0,1] at {i}"
        print(f"  [{i}] img={tuple(img.shape)}  matte={tuple(matte.shape)}  "
              f"matte_range=[{matte.min():.3f}, {matte.max():.3f}]")
    print("Dataset smoke test passed.")
