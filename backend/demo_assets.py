import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window, bounds as window_bounds
from rasterio.warp import transform_bounds


def _tile_bbox(raw_base, event_id, chip_id, row, col, tile_size):
    vv_path = Path(raw_base) / "chips" / event_id / "s1" / chip_id / "VV.tif"
    if not vv_path.exists():
        return None

    with rasterio.open(vv_path) as src:
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


def export_demo_assets(
    split_records,
    threshold,
    raw_base,
    registry_path,
    tiles_dir,
    event_names=None,
):
    event_names = event_names or {}
    tiles_dir = Path(tiles_dir)
    registry_path = Path(registry_path)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    registry = {}
    tile_index = 0

    for split_name, records in split_records.items():
        for record in records:
            event_id = record["event_id"]
            chip_id = record["chip_id"]
            row = int(record["row"])
            col = int(record["col"])
            tile_size = int(record["probability"].shape[0])

            binary_mask = (
                (record["probability"] >= threshold) & (record["valid_mask"] > 0.5)
            ).astype(np.uint8)
            output_name = f"tile_{tile_index:04d}_mask.png"
            output_path = tiles_dir / output_name
            _save_overlay_png(binary_mask, output_path)

            bbox = _tile_bbox(raw_base, event_id, chip_id, row, col, tile_size)
            if bbox is None:
                continue

            flood_pct = float(binary_mask.sum() / np.clip(record["valid_mask"].sum(), 1, None) * 100.0)
            registry[f"tile_{tile_index:04d}"] = {
                "file": output_name,
                "bbox": bbox,
                "split": split_name,
                "event_id": event_id,
                "event": event_names.get(event_id, event_id),
                "chip": chip_id,
                "offset": [row, col],
                "tile_iou": round(float(record["iou"]), 4),
                "tile_f1": round(float(record["f1"]), 4),
                "precision": round(float(record["precision"]), 4),
                "recall": round(float(record["recall"]), 4),
                "threshold": round(float(threshold), 4),
                "flood_pct": round(flood_pct, 2),
            }
            tile_index += 1

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return registry
