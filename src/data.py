"""Image cache, Dataset and transforms for the image and fusion arms.

Decoding a 1224x960 PNG costs ~21 ms. Five folds x 30 epochs over 1000 images is
~0.9 hours of pure decoding, repeated identically every run. So decode once into
a uint8 array (1000 x 384 x 384 x 3 = 442 MB, comfortably RAM-resident), cache it
to disk, and let every fold and every arm read from that.

The cache is aligned to `master.parquet` row order -- row i of the dataframe is
image i of the array. `img_idx` carries that mapping through fold subsetting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from src.config import ARTIFACTS, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def cache_path(size: int = IMG_SIZE) -> Path:
    return ARTIFACTS / f"images_{size}.npy"


def build_image_cache(df: pd.DataFrame, size: int = IMG_SIZE, force: bool = False) -> np.ndarray:
    """Decode and resize every image once; return the uint8 array.

    Row order matches `df`. Re-run with force=True if you change `size` or the
    dataframe ordering, otherwise the cache and the labels silently desynchronise
    -- which would look like a model that cannot learn rather than a data bug.
    """
    path = cache_path(size)
    if path.exists() and not force:
        # Loaded fully into RAM, NOT mmap_mode="r". Memmapping looks tempting for
        # a 442 MB file, but randomised access from the DataLoader through the
        # torchvision v2 transforms segfaults the process (exit 139, no
        # traceback). It is also slower here: the array fits in RAM many times
        # over, so paging buys nothing.
        return np.load(path)

    out = np.zeros((len(df), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(df["img_path"]):
        out[i] = np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))
        if (i + 1) % 200 == 0:
            print(f"  cached {i + 1}/{len(df)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


class PerImageStandardize(torch.nn.Module):
    """Standardise each image by its own per-channel mean and std.

    Why this exists, measured on this dataset: six numbers per image -- the
    channel means and stds, no spatial information, no cells -- predict the label
    at **0.905 AUROC**. Worse, the same statistics computed over BACKGROUND pixels
    only (slide regions containing no cells, per the shipped `Binary_segmented`
    masks) predict at **0.833**, actually beating cell-region colour at 0.806.

    Background carries no information about a patient's red cells. That signal is
    staining batch, illumination and camera exposure -- an acquisition artifact
    that happens to correlate with the class. Any model free to use raw colour
    will take it, and will look excellent for the wrong reason.

    This transform makes every image have channel mean 0 and std 1, which drives
    those six features to constants and provably destroys that exact confound.
    Genuine hypochromia survives it, because "pale" is only meaningful as cell
    brightness RELATIVE to the slide: the stain-invariant cell-minus-background
    contrast still predicts at 0.734, and in the right direction (anemic cells sit
    closer to background brightness).

    Replaces ImageNet normalisation rather than following it.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return (x - mean) / std


def build_transforms(train: bool, stain_norm: bool = False):
    """Augmentation for blood smears.

    Flips and 90-degree rotations are free: a smear has no canonical orientation,
    so a flipped smear is still a perfectly valid smear.

    Colour jitter is kept deliberately mild. The label is literally "hypochromic"
    -- pale, under-coloured cells. Aggressive brightness/saturation jitter teaches
    the model to ignore exactly the cue it is supposed to detect. 0.1 is a nudge
    that covers stain-batch variation without erasing the signal. If you raise it,
    ablate it and check the score does not fall.
    """
    steps = [v2.ToDtype(torch.float32, scale=True)]
    if train:
        steps = [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]
    steps.append(
        PerImageStandardize() if stain_norm else v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    )
    return v2.Compose(steps)


class SmearDataset(Dataset):
    """Returns (image, tabular, y).

    One class serves both arms: pass `features=[]` for image-only, and the
    tabular tensor comes back empty.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images: np.ndarray,
        features: list[str],
        train: bool,
        tabular: np.ndarray | None = None,
        stain_norm: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.images = images
        self.features = features
        self.transform = build_transforms(train, stain_norm)
        self.train = train
        # Pre-imputed, pre-standardised tabular block, aligned to `df` rows.
        # Built by train.py from TRAIN-fold statistics only.
        self.tabular = tabular
        self.y = self.df["y"].to_numpy(dtype=np.float32)
        self.idx = self.df["img_idx"].to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        img = torch.from_numpy(np.ascontiguousarray(self.images[self.idx[i]])).permute(2, 0, 1)
        img = self.transform(img)

        if self.train:
            # Rotating by a random multiple of 90 degrees needs no interpolation
            # and introduces no padding, unlike an arbitrary-angle rotation.
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = torch.rot90(img, k, dims=(1, 2))

        tab = (
            torch.from_numpy(self.tabular[i].astype(np.float32))
            if self.tabular is not None
            else torch.zeros(0)
        )
        return img, tab, torch.tensor(self.y[i])


def make_loader(
    df,
    images,
    features,
    train: bool,
    tabular=None,
    stain_norm: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """One DataLoader. Each fold needs three: inner-train, early-stop, outer-val.

    `train=True` turns on augmentation, shuffling and drop_last; evaluation
    loaders keep the original order so predictions line up with the dataframe.

    num_workers defaults to 0 on purpose. The images already live in RAM, so
    workers buy little, and torch's fork-based workers interact badly with MPS in
    some versions -- silent hangs that look like a stuck training loop.
    """
    ds = SmearDataset(df, images, features, train=train, tabular=tabular, stain_norm=stain_norm)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,  # BatchNorm needs >1 sample; a trailing batch of 1 crashes
    )
