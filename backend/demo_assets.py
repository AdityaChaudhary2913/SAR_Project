import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window, bounds as window_bounds
from rasterio.warp import transform_bounds


def _tile_bbox(geo_image_path, row, col, tile_size):
    image_path = Path(geo_image_path)
    if not image_path.exists():
        return None

    with rasterio.open(image_path) as src:
        window = Window(col_off=col, row_off=row, width=tile_size, height=tile_size)
        left, bottom, right, top = window_bounds(window, src.transform)
        if src.crs and src.crs.to_string() != "EPSG:4326":
            left, bottom, right, top = transform_bounds(
                src.crs,
                "EPSG:4326",
                left,
                bottom,
                right,
                top,
            )
    return [left, bottom, right, top]


def _save_overlay_png(mask, output_path):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask == 1] = [30, 120, 255, 180]
    Image.fromarray(rgba, mode="RGBA").save(output_path)


def _save_base_tile_png(image, valid_mask, output_path):
    vv = image[0]
    vv = np.where(valid_mask > 0.5, vv, 0.0)
    vv = np.clip(vv, 0.0, 1.0)
    rgb = np.stack([vv, vv, vv], axis=-1)
    rgb = (rgb * 255.0).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(output_path)


def export_demo_assets(
    split_records,
    threshold,
    registry_path,
    tiles_dir,
    event_names=None,
):
    event_names = event_names or {}
    tiles_dir = Path(tiles_dir)
    registry_path = Path(registry_path)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    for existing_png in tiles_dir.glob("*.png"):
        existing_png.unlink()

    registry = {}
    tile_index = 0

    for split_name, records in split_records.items():
        for record in records:
            event_id = record["event_id"]
            chip_id = record["chip_id"]
            row = int(record["row"])
            col = int(record["col"])
            tile_size = int(record["probability"].shape[0])

            tile_prefix = f"tile_{tile_index:04d}"
            base_name = f"{tile_prefix}_base.png"
            unet_name = f"{tile_prefix}_unet.png"
            rf_name = f"{tile_prefix}_rf.png"

            unet_binary_mask = (
                (record["probability"] >= threshold) & (record["valid_mask"] > 0.5)
            ).astype(np.uint8)
            _save_base_tile_png(record["image"], record["valid_mask"], tiles_dir / base_name)
            _save_overlay_png(unet_binary_mask, tiles_dir / unet_name)

            rf_record = record.get("rf_record")
            rf_binary_mask = None
            if rf_record is not None:
                rf_binary_mask = (
                    (rf_record["probability"] >= threshold) & (rf_record["valid_mask"] > 0.5)
                ).astype(np.uint8)
                _save_overlay_png(rf_binary_mask, tiles_dir / rf_name)

            bbox = _tile_bbox(record.get("geo_image_path"), row, col, tile_size)
            if bbox is None:
                continue

            flood_pct = float(unet_binary_mask.sum() / np.clip(record["valid_mask"].sum(), 1, None) * 100.0)
            rf_flood_pct = None
            if rf_binary_mask is not None:
                rf_flood_pct = float(rf_binary_mask.sum() / np.clip(record["valid_mask"].sum(), 1, None) * 100.0)
            registry[f"tile_{tile_index:04d}"] = {
                "base_file": base_name,
                "unet_file": unet_name,
                "rf_file": rf_name if rf_binary_mask is not None else None,
                "bbox": bbox,
                "split": split_name,
                "event_id": event_id,
                "event": event_names.get(event_id, event_id),
                "source": record.get("source", "unknown"),
                "source_event_id": record.get("source_event_id", event_id),
                "chip": chip_id,
                "offset": [row, col],
                "unet_tile_iou": round(float(record["iou"]), 4),
                "unet_tile_f1": round(float(record["f1"]), 4),
                "unet_precision": round(float(record["precision"]), 4),
                "unet_recall": round(float(record["recall"]), 4),
                "rf_tile_iou": round(float(rf_record["iou"]), 4) if rf_record is not None else None,
                "rf_tile_f1": round(float(rf_record["f1"]), 4) if rf_record is not None else None,
                "rf_precision": round(float(rf_record["precision"]), 4) if rf_record is not None else None,
                "rf_recall": round(float(rf_record["recall"]), 4) if rf_record is not None else None,
                "threshold": round(float(threshold), 4),
                "unet_flood_pct": round(flood_pct, 2),
                "rf_flood_pct": round(rf_flood_pct, 2) if rf_flood_pct is not None else None,
            }
            tile_index += 1

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return registry
