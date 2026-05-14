import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def _load_manifest(processed_dir):
    manifest_path = Path(processed_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scan_processed_dir(processed_dir):
    processed_path = Path(processed_dir)
    chip_records = []

    for event_dir in sorted(processed_path.iterdir()):
        if not event_dir.is_dir():
            continue

        img_dir = event_dir / "images"
        mask_dir = event_dir / "masks"
        valid_dir = event_dir / "valid_masks"
        if not img_dir.exists() or not mask_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.npy")):
            chip_id = img_path.stem
            mask_path = mask_dir / f"{chip_id}.npy"
            if not mask_path.exists():
                continue

            valid_mask_path = valid_dir / f"{chip_id}.npy"
            img = np.load(img_path, mmap_mode="r")
            mask = np.load(mask_path, mmap_mode="r")
            valid_mask = (
                np.load(valid_mask_path, mmap_mode="r")
                if valid_mask_path.exists()
                else np.ones_like(mask, dtype=np.uint8)
            )

            valid_pixels = int(valid_mask.sum())
            flood_pixels = int(((mask == 1) & (valid_mask == 1)).sum())
            flood_ratio = flood_pixels / valid_pixels if valid_pixels else 0.0
            valid_ratio = valid_pixels / mask.size if mask.size else 0.0

            chip_records.append(
                {
                    "event_id": event_dir.name,
                    "chip_id": chip_id,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "valid_mask_path": str(valid_mask_path) if valid_mask_path.exists() else None,
                    "height": int(img.shape[1]),
                    "width": int(img.shape[2]),
                    "channels": int(img.shape[0]),
                    "flood_ratio": flood_ratio,
                    "valid_ratio": valid_ratio,
                }
            )

    return {
        "chips": chip_records,
        "derived_channels": [],
    }


def _resolve_chip_records(processed_dir):
    manifest = _load_manifest(processed_dir)
    if manifest is None:
        manifest = _scan_processed_dir(processed_dir)

    processed_path = Path(processed_dir)
    chip_records = []
    for chip in manifest.get("chips", []):
        record = dict(chip)
        for key in ("image_path", "mask_path", "valid_mask_path"):
            if record.get(key):
                path = Path(record[key])
                if not path.is_absolute():
                    record[key] = str(processed_path / path)
        chip_records.append(record)

    return chip_records, manifest


def _build_tile_samples(chip_records, tile_size):
    samples = []
    for chip in chip_records:
        height = int(chip["height"])
        width = int(chip["width"])
        for row in range(0, height - tile_size + 1, tile_size):
            for col in range(0, width - tile_size + 1, tile_size):
                samples.append(
                    {
                        **chip,
                        "row": row,
                        "col": col,
                        "tile_size": tile_size,
                        "tile_id": f"{chip['event_id']}::{chip['chip_id']}::{row}_{col}",
                    }
                )
    return samples


def _categorize_sample(sample, no_flood_max=0.0, low_flood_max=0.02):
    flood_ratio = float(sample.get("flood_ratio", 0.0))
    if flood_ratio <= no_flood_max:
        return "dry"
    if flood_ratio <= low_flood_max:
        return "low_flood"
    return "flood"


def _make_balanced_sampler(samples, category_thresholds):
    categories = [
        _categorize_sample(
            sample,
            no_flood_max=category_thresholds["no_flood_max"],
            low_flood_max=category_thresholds["low_flood_max"],
        )
        for sample in samples
    ]
    counts = Counter(categories)
    weights = [1.0 / counts[category] for category in categories]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def _summarize_split(split_name, samples):
    event_counts = Counter(sample["event_id"] for sample in samples)
    chip_counts = Counter((sample["event_id"], sample["chip_id"]) for sample in samples)
    summary = (
        f"{split_name}: {len(samples)} tiles from {len(chip_counts)} chips "
        f"across {len(event_counts)} events"
    )
    if event_counts:
        event_bits = ", ".join(f"{event}:{count}" for event, count in sorted(event_counts.items()))
        summary += f" [{event_bits}]"
    return summary


def _split_train_val(chip_records, val_split, split_scope, seed):
    rng = random.Random(seed)
    if split_scope == "event":
        event_ids = sorted({chip["event_id"] for chip in chip_records})
        if len(event_ids) <= 1:
            return chip_records, []
        rng.shuffle(event_ids)
        val_events = max(1, round(len(event_ids) * val_split))
        val_event_ids = set(event_ids[:val_events])
        train_chips = [chip for chip in chip_records if chip["event_id"] not in val_event_ids]
        val_chips = [chip for chip in chip_records if chip["event_id"] in val_event_ids]
        return train_chips, val_chips

    chip_keys = sorted((chip["event_id"], chip["chip_id"]) for chip in chip_records)
    if len(chip_keys) <= 1:
        return chip_records, []

    rng.shuffle(chip_keys)
    val_chip_count = max(1, round(len(chip_keys) * val_split))
    val_key_set = set(chip_keys[:val_chip_count])

    train_chips = []
    val_chips = []
    for chip in chip_records:
        key = (chip["event_id"], chip["chip_id"])
        if key in val_key_set:
            val_chips.append(chip)
        else:
            train_chips.append(chip)

    return train_chips, val_chips


class SARTileDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = list(samples)
        self.augment = augment
        self.in_channels = self.samples[0]["channels"] if self.samples else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tile_size = int(sample["tile_size"])
        row = int(sample["row"])
        col = int(sample["col"])

        image = np.load(sample["image_path"], mmap_mode="r")[
            :, row : row + tile_size, col : col + tile_size
        ].astype(np.float32)
        mask = np.load(sample["mask_path"], mmap_mode="r")[
            row : row + tile_size, col : col + tile_size
        ].astype(np.float32)

        if sample.get("valid_mask_path"):
            valid_mask = np.load(sample["valid_mask_path"], mmap_mode="r")[
                row : row + tile_size, col : col + tile_size
            ].astype(np.float32)
        else:
            valid_mask = np.ones_like(mask, dtype=np.float32)

        if self.augment:
            image, mask, valid_mask = self._augment(image, mask, valid_mask)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "valid_mask": torch.from_numpy(valid_mask).unsqueeze(0),
            "event_id": sample["event_id"],
            "chip_id": sample["chip_id"],
            "row": row,
            "col": col,
            "tile_id": sample["tile_id"],
            "flood_ratio": float(sample.get("flood_ratio", 0.0)),
            "valid_ratio": float(sample.get("valid_ratio", 1.0)),
        }

    def _augment(self, image, mask, valid_mask):
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
            valid_mask = np.flip(valid_mask, axis=1).copy()

        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
            valid_mask = np.flip(valid_mask, axis=0).copy()

        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
        valid_mask = np.rot90(valid_mask, k, axes=(0, 1)).copy()
        return image, mask, valid_mask


def get_dataloaders(
    processed_dir,
    tile_size=256,
    batch_size=8,
    train_split=None,
    val_split=None,
    num_workers=2,
    seed=42,
    split_scope="chip",
    train_event_ids=None,
    holdout_event_ids=None,
    augment_train=False,
    balance_train=False,
    return_test=False,
    no_flood_max=0.0,
    low_flood_max=0.02,
):
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")

    chip_records, manifest = _resolve_chip_records(processed_dir)
    if not chip_records:
        raise RuntimeError(f"No processed chips found under {processed_dir}")

    if val_split is None:
        if train_split is None:
            train_split = 0.8
        val_split = 1.0 - float(train_split)

    holdout_event_ids = set(holdout_event_ids or [])
    train_event_ids = set(train_event_ids or [])
    selected_events = train_event_ids | holdout_event_ids
    if selected_events:
        chip_records = [chip for chip in chip_records if chip["event_id"] in selected_events]

    if holdout_event_ids:
        test_chip_records = [chip for chip in chip_records if chip["event_id"] in holdout_event_ids]
    else:
        test_chip_records = []

    if train_event_ids:
        train_val_chip_records = [chip for chip in chip_records if chip["event_id"] in train_event_ids]
    else:
        train_val_chip_records = [chip for chip in chip_records if chip["event_id"] not in holdout_event_ids]

    train_chip_records, val_chip_records = _split_train_val(
        train_val_chip_records,
        val_split=val_split,
        split_scope=split_scope,
        seed=seed,
    )

    train_samples = _build_tile_samples(train_chip_records, tile_size)
    val_samples = _build_tile_samples(val_chip_records, tile_size)
    test_samples = _build_tile_samples(test_chip_records, tile_size)

    train_dataset = SARTileDataset(train_samples, augment=augment_train)
    val_dataset = SARTileDataset(val_samples, augment=False)
    test_dataset = SARTileDataset(test_samples, augment=False)

    sampler = None
    shuffle = True
    if balance_train and train_samples:
        sampler = _make_balanced_sampler(
            train_samples,
            category_thresholds={
                "no_flood_max": no_flood_max,
                "low_flood_max": low_flood_max,
            },
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) >= batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    derived_channels = manifest.get("derived_channels", [])
    in_channels = train_dataset.in_channels or val_dataset.in_channels or test_dataset.in_channels
    print(
        f"✅ SARTileDataset: {len(train_samples) + len(val_samples) + len(test_samples)} tiles "
        f"from {len(train_chip_records) + len(val_chip_records) + len(test_chip_records)} chips "
        f"(tile_size={tile_size}, split_scope={split_scope}, in_channels={in_channels})"
    )
    if derived_channels:
        print(f"   Derived channels: {', '.join(derived_channels)}")
    print(f"   {_summarize_split('train', train_samples)}")
    print(f"   {_summarize_split('val', val_samples)}")
    if test_samples:
        print(f"   {_summarize_split('test', test_samples)}")

    metadata = {
        "in_channels": in_channels,
        "derived_channels": derived_channels,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "split_scope": split_scope,
        "holdout_event_ids": sorted(holdout_event_ids),
        "train_event_ids": sorted(train_event_ids),
    }

    if return_test:
        return train_loader, val_loader, test_loader, metadata
    return train_loader, val_loader
