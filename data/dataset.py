# data/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SARFloodDataset(Dataset):
    def __init__(self, processed_dir, tile_size=256, augment=False):
        """
        Loads preprocessed .npy chips and sub-tiles them into tile_size x tile_size patches.
        Each 512x512 chip produces 4 non-overlapping 256x256 tiles.

        Args:
            processed_dir : path to data/processed/ (must contain images/ and masks/ subdirs)
            tile_size     : sub-tile size (default 256)
            augment       : enable SAR-safe augmentations (flips + 90° rotations)
        """
        self.tile_size = tile_size
        self.augment = augment
        self.tiles = []  # list of (img_path, mask_path, row_offset, col_offset)

        img_dir = os.path.join(processed_dir, "images")
        mask_dir = os.path.join(processed_dir, "masks")

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"images/ not found inside {processed_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"masks/ not found inside {processed_dir}")

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".npy"):
                continue

            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname)

            if not os.path.exists(mask_path):
                print(f"⚠️  Missing mask for {fname}, skipping.")
                continue

            # Memory-map to read shape without loading full array
            img = np.load(img_path, mmap_mode="r")  # (2, H, W)
            _, H, W = img.shape

            # Generate all non-overlapping tile positions
            for r in range(0, H - tile_size + 1, tile_size):
                for c in range(0, W - tile_size + 1, tile_size):
                    self.tiles.append((img_path, mask_path, r, c))

        print(
            f"✅ SARFloodDataset: {len(self.tiles)} tiles from "
            f"{len(os.listdir(img_dir))} chips "
            f"(tile_size={tile_size}, augment={augment})"
        )

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_path, mask_path, r, c = self.tiles[idx]
        t = self.tile_size

        img = np.load(img_path)[:, r : r + t, c : c + t].astype(np.float32)  # (2, t, t)
        mask = np.load(mask_path)[r : r + t, c : c + t].astype(np.float32)  # (t, t)

        if self.augment:
            img, mask = self._augment(img, mask)

        # mask gets a channel dim: (t,t) → (1,t,t) to match model output shape
        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0)

    def _augment(self, img, mask):
        """
        SAR-safe augmentations only.
        NO brightness/contrast/color jitter — dB values are physically meaningful.
        Only spatial transforms that preserve radar geometry.
        """
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # Vertical flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random 90° rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)
        img = np.rot90(img, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()

        return img, mask


def get_dataloaders(
    processed_dir, tile_size=256, batch_size=8, train_split=0.8, num_workers=2, seed=42
):
    """
    Returns (train_loader, val_loader).
    Augmentation is automatically enabled on train set only.
    Split is reproducible via seed.

    Args:
        processed_dir : path to data/processed/
        tile_size     : sub-tile size passed to SARFloodDataset
        batch_size    : samples per batch
        train_split   : fraction of tiles used for training
        num_workers   : DataLoader workers (use 2 on Kaggle T4)
        seed          : random seed for reproducible split
    """
    # Build full dataset WITHOUT augmentation first (augment flag set after split)
    full_ds = SARFloodDataset(processed_dir, tile_size=tile_size, augment=False)

    n_train = int(len(full_ds) * train_split)
    n_val = len(full_ds) - n_train

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    # Enable augmentation on training subset only
    # random_split wraps the dataset — we patch the underlying dataset's flag
    # by subclassing the Subset to toggle augment per-index
    train_ds.dataset.augment = False  # we handle it below via wrapper

    train_loader = DataLoader(
        _AugmentedSubset(train_ds, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # avoid single-sample batches that break BatchNorm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n📊 Split: {n_train} train tiles / {n_val} val tiles")
    print(f"   Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    return train_loader, val_loader


class _AugmentedSubset(Dataset):
    """
    Thin wrapper around a Subset that forces augmentation ON,
    without affecting the val split which uses the same underlying dataset.
    """

    def __init__(self, subset, augment=True):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, mask_path, r, c = self.subset.dataset.tiles[self.subset.indices[idx]]
        t = self.subset.dataset.tile_size

        img = np.load(img_path)[:, r : r + t, c : c + t].astype(np.float32)
        mask = np.load(mask_path)[r : r + t, c : c + t].astype(np.float32)

        if self.augment:
            img, mask = self.subset.dataset._augment(img, mask)

        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0)