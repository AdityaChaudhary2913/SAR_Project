# Reads raw C2SMSFloods chips (VV.tif, VH.tif, LabelWater.tif)
# Normalizes, filters nodata chips, saves as .npy arrays
# LLM-assisted: structure and rasterio I/O written with AI help

import numpy as np
import rasterio
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_sar(chip: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    """Clip dB SAR values to [norm_min, norm_max] and scale to [0, 1]."""
    chip = np.clip(chip, norm_min, norm_max)
    return (chip - norm_min) / (norm_max - norm_min)


def is_valid_chip(
    mask: np.ndarray, max_nodata_ratio: float, min_flood_ratio: float
) -> bool:
    """Return True if chip has acceptable nodata and at least some flood pixels."""
    nodata_ratio = (mask == 255).mean()
    flood_ratio = (mask == 1).mean()
    return nodata_ratio < max_nodata_ratio and flood_ratio >= min_flood_ratio


def process_event(event_dir: Path, out_dir: Path, cfg: dict) -> dict:
    """Process all chips in one event folder."""
    norm_min = cfg["data"]["norm_min"]
    norm_max = cfg["data"]["norm_max"]
    max_nodata = cfg["data"]["max_nodata_ratio"]
    min_flood = cfg["data"]["min_flood_ratio"]

    img_out = out_dir / "images"
    mask_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    chip_dirs = sorted([d for d in event_dir.iterdir() if d.is_dir()])
    stats = {"total": 0, "saved": 0, "skipped_nodata": 0, "skipped_noflood": 0}

    for chip_dir in tqdm(chip_dirs, desc=f"{event_dir.parent.parent.name[:8]}"):
        vv_path = chip_dir / "VV.tif"
        vh_path = chip_dir / "VH.tif"
        label_path = chip_dir / "LabelWater.tif"

        if not all(p.exists() for p in [vv_path, vh_path, label_path]):
            continue

        stats["total"] += 1

        # Read SAR bands
        with rasterio.open(vv_path) as src:
            vv = src.read(1).astype(np.float32)  # (512, 512)
        with rasterio.open(vh_path) as src:
            vh = src.read(1).astype(np.float32)  # (512, 512)

        # Read label
        with rasterio.open(label_path) as src:
            mask = src.read(1).astype(np.uint8)  # (512, 512), values: 0, 1, 255

        # Filter: nodata check
        if (mask == 255).mean() >= max_nodata:
            stats["skipped_nodata"] += 1
            continue

        # Filter: must have some flood pixels
        if (mask == 1).mean() < min_flood:
            stats["skipped_noflood"] += 1
            continue

        # Normalize SAR
        vv_norm = normalize_sar(vv, norm_min, norm_max)
        vh_norm = normalize_sar(vh, norm_min, norm_max)

        # Stack to (2, 512, 512)
        img = np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)

        # Clean mask: set nodata → 0 (treat as non-flood)
        clean_mask = np.where(mask == 255, 0, mask).astype(np.uint8)

        # Save
        chip_id = chip_dir.name
        np.save(img_out / f"{chip_id}.npy", img)
        np.save(mask_out / f"{chip_id}.npy", clean_mask)
        stats["saved"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess C2SMSFloods chips")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_base = Path(cfg["data"]["raw_dir"]) / "chips"
    out_base = Path(cfg["data"]["processed_dir"])

    # Find all event s1 folders
    event_dirs = sorted(raw_base.glob("*/s1"))

    if not event_dirs:
        print(f"❌ No event folders found under {raw_base}")
        return

    print(f"Found {len(event_dirs)} event(s) to process\n")

    total_stats = {"total": 0, "saved": 0, "skipped_nodata": 0, "skipped_noflood": 0}

    for event_s1_dir in event_dirs:
        uuid = event_s1_dir.parent.name
        out_dir = out_base / uuid
        stats = process_event(event_s1_dir, out_dir, cfg)

        print(f"\n📦 {uuid[:8]}...")
        print(f"   Total     : {stats['total']}")
        print(f"   ✅ Saved  : {stats['saved']}")
        print(f"   ⚠️  Nodata : {stats['skipped_nodata']}")
        print(f"   ⚠️  NoFlood: {stats['skipped_noflood']}")

        for k in total_stats:
            total_stats[k] += stats[k]

    print(f"\n{'=' * 45}")
    print(f"🎯 Total saved: {total_stats['saved']} / {total_stats['total']} chips")
    print(f"   Output at  : {out_base}")
    print(f"{'=' * 45}")


if __name__ == "__main__":
    main()