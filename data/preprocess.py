# Reads raw C2SMSFloods chips (VV.tif, VH.tif, LabelWater.tif)
# Normalizes SAR, preserves nodata via valid masks, and saves a manifest.

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_sar(chip, norm_min, norm_max):
    chip = np.clip(chip, norm_min, norm_max)
    return (chip - norm_min) / (norm_max - norm_min)


def build_derived_channels(vv_norm, vh_norm, derived_channels, ratio_eps):
    channels = []
    channel_names = []

    for channel_name in derived_channels:
        if channel_name == "vv_minus_vh":
            derived = np.clip(vv_norm - vh_norm + 0.5, 0.0, 1.0)
        elif channel_name == "vv_div_vh":
            derived = vv_norm / np.clip(vh_norm, ratio_eps, None)
            derived = np.clip(derived, 0.0, 4.0) / 4.0
        else:
            raise ValueError(f"Unsupported derived channel: {channel_name}")

        channels.append(derived.astype(np.float32))
        channel_names.append(channel_name)

    return channels, channel_names


def process_event(event_dir, out_dir, cfg):
    data_cfg = cfg["data"]
    norm_min = data_cfg["norm_min"]
    norm_max = data_cfg["norm_max"]
    ratio_eps = data_cfg.get("ratio_eps", 1e-3)
    derived_channels = data_cfg.get("derived_channels", [])
    min_flood_ratio = float(data_cfg.get("min_flood_ratio", 0.0))

    img_out = out_dir / "images"
    mask_out = out_dir / "masks"
    valid_out = out_dir / "valid_masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    valid_out.mkdir(parents=True, exist_ok=True)

    chip_dirs = sorted([d for d in event_dir.iterdir() if d.is_dir()])
    stats = {
        "total": 0,
        "saved": 0,
        "skipped_all_nodata": 0,
        "skipped_low_flood": 0,
    }
    manifest_entries = []

    for chip_dir in tqdm(chip_dirs, desc=event_dir.parent.name[:8]):
        vv_path = chip_dir / "VV.tif"
        vh_path = chip_dir / "VH.tif"
        label_path = chip_dir / "LabelWater.tif"

        if not all(path.exists() for path in (vv_path, vh_path, label_path)):
            continue

        stats["total"] += 1

        with rasterio.open(vv_path) as src:
            vv = src.read(1).astype(np.float32)
        with rasterio.open(vh_path) as src:
            vh = src.read(1).astype(np.float32)
        with rasterio.open(label_path) as src:
            mask = src.read(1).astype(np.uint8)

        valid_mask = (mask != 255).astype(np.uint8)
        valid_pixels = int(valid_mask.sum())
        if valid_pixels == 0:
            stats["skipped_all_nodata"] += 1
            continue

        clean_mask = np.where(valid_mask == 1, mask, 0).astype(np.uint8)
        flood_pixels = int(((clean_mask == 1) & (valid_mask == 1)).sum())
        flood_ratio = flood_pixels / valid_pixels
        valid_ratio = valid_pixels / clean_mask.size

        if flood_ratio < min_flood_ratio:
            stats["skipped_low_flood"] += 1
            continue

        vv_norm = normalize_sar(vv, norm_min, norm_max)
        vh_norm = normalize_sar(vh, norm_min, norm_max)

        image_channels = [vv_norm.astype(np.float32), vh_norm.astype(np.float32)]
        extra_channels, channel_names = build_derived_channels(
            vv_norm,
            vh_norm,
            derived_channels=derived_channels,
            ratio_eps=ratio_eps,
        )
        image_channels.extend(extra_channels)
        image = np.stack(image_channels, axis=0).astype(np.float32)

        chip_id = chip_dir.name
        image_path = img_out / f"{chip_id}.npy"
        mask_path = mask_out / f"{chip_id}.npy"
        valid_mask_path = valid_out / f"{chip_id}.npy"

        np.save(image_path, image)
        np.save(mask_path, clean_mask)
        np.save(valid_mask_path, valid_mask)
        stats["saved"] += 1

        manifest_entries.append(
            {
                "event_id": event_dir.parent.name,
                "chip_id": chip_id,
                "image_path": str(image_path.relative_to(out_dir.parent)),
                "mask_path": str(mask_path.relative_to(out_dir.parent)),
                "valid_mask_path": str(valid_mask_path.relative_to(out_dir.parent)),
                "height": int(image.shape[1]),
                "width": int(image.shape[2]),
                "channels": int(image.shape[0]),
                "flood_ratio": flood_ratio,
                "valid_ratio": valid_ratio,
                "channel_names": ["vv", "vh", *channel_names],
            }
        )

    return stats, manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Preprocess C2SMSFloods chips")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    raw_base = Path(data_cfg["raw_dir"]) / "chips"
    out_base = Path(data_cfg["processed_dir"])
    out_base.mkdir(parents=True, exist_ok=True)

    selected_event_ids = set(data_cfg.get("train_event_ids", [])) | set(
        data_cfg.get("holdout_event_ids", [])
    )
    event_dirs = sorted(raw_base.glob("*/s1"))
    if selected_event_ids:
        event_dirs = [event_dir for event_dir in event_dirs if event_dir.parent.name in selected_event_ids]

    if not event_dirs:
        print(f"❌ No event folders found under {raw_base}")
        return

    print(f"Found {len(event_dirs)} event(s) to process\n")

    total_stats = {
        "total": 0,
        "saved": 0,
        "skipped_all_nodata": 0,
        "skipped_low_flood": 0,
    }
    full_manifest = []

    for event_s1_dir in event_dirs:
        event_id = event_s1_dir.parent.name
        out_dir = out_base / event_id
        stats, manifest_entries = process_event(event_s1_dir, out_dir, cfg)
        full_manifest.extend(manifest_entries)

        print(f"\n📦 {event_id[:8]}...")
        print(f"   Total          : {stats['total']}")
        print(f"   ✅ Saved       : {stats['saved']}")
        print(f"   ⚠️  All nodata : {stats['skipped_all_nodata']}")
        print(f"   ⚠️  Low flood  : {stats['skipped_low_flood']}")

        for key in total_stats:
            total_stats[key] += stats[key]

    manifest = {
        "chips": full_manifest,
        "derived_channels": data_cfg.get("derived_channels", []),
        "train_event_ids": data_cfg.get("train_event_ids", []),
        "holdout_event_ids": data_cfg.get("holdout_event_ids", []),
        "event_names": data_cfg.get("event_names", {}),
    }
    manifest_path = out_base / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 45}")
    print(f"🎯 Total saved: {total_stats['saved']} / {total_stats['total']} chips")
    print(f"   Output at  : {out_base}")
    print(f"   Manifest   : {manifest_path}")
    print(f"{'=' * 45}")


if __name__ == "__main__":
    main()
