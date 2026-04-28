# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json, math
from pathlib import Path

app = FastAPI(title="SAR Flood Detection API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/tiles", StaticFiles(directory="tiles"), name="tiles")

with open("tile_registry.json") as f:
    REGISTRY = json.load(f)


class AOIRequest(BaseModel):
    bbox: list[float]  # [min_lon, min_lat, max_lon, max_lat]


def bbox_iou(a, b):
    """Intersection-over-union between two bboxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


@app.post("/predict")
def predict(req: AOIRequest):
    best_tile, best_score = None, 0.0
    for tile_id, meta in REGISTRY.items():
        if meta.get("bbox") is None:
            continue
        score = bbox_iou(req.bbox, meta["bbox"])
        if score > best_score:
            best_score, best_tile = score, tile_id

    if best_tile is None or best_score < 0.05:
        raise HTTPException(
            404,
            "No tile overlaps this AOI. Draw your box over a dashed coverage rectangle.",
        )

    meta = REGISTRY[best_tile]
    return {
        "tile_id": best_tile,
        "mask_url": f"/tiles/{meta['file']}",
        "bbox": meta["bbox"],
        "event": meta.get("event", meta.get("chip", best_tile)),
        "split": meta.get("split", "unknown"),
        "tile_iou": meta.get("tile_iou"),
        "overlap_score": round(best_score, 3),
    }


@app.get("/tiles_list")
def tiles_list():
    return [
        {
            "id": k,
            "bbox": v["bbox"],
            "event": v.get(
                "event", v.get("chip", f"tile_{k}")
            ),  # fallback to chip name
            "split": v.get("split", "unknown"),
        }
        for k, v in REGISTRY.items()
        if v.get("bbox") is not None  # skip tiles with no geo-bbox
    ]