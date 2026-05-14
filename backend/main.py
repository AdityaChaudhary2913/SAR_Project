import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
TILES_DIR = BASE_DIR / "tiles"
REGISTRY_PATH = BASE_DIR / "tile_registry.json"
METRICS_PATH = BASE_DIR.parent / "checkspots" / "metrics.json"

app = FastAPI(title="SAR Flood Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/tiles", StaticFiles(directory=str(TILES_DIR)), name="tiles")


def load_json(path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_registry():
    return load_json(REGISTRY_PATH, {})


def load_metrics():
    return load_json(METRICS_PATH, {})


class AOIRequest(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4)


def bbox_iou(a, b):
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
    registry = load_registry()
    best_tile = None
    best_score = 0.0

    for tile_id, meta in registry.items():
        if meta.get("bbox") is None:
            continue
        score = bbox_iou(req.bbox, meta["bbox"])
        if score > best_score:
            best_score = score
            best_tile = tile_id

    if best_tile is None or best_score < 0.05:
        raise HTTPException(
            status_code=404,
            detail="No tile overlaps this AOI. Draw your box over one of the coverage rectangles.",
        )

    meta = registry[best_tile]
    return {
        "tile_id": best_tile,
        "mask_url": f"/tiles/{meta['file']}",
        "bbox": meta["bbox"],
        "event": meta.get("event", meta.get("chip", best_tile)),
        "event_id": meta.get("event_id"),
        "split": meta.get("split", "unknown"),
        "tile_iou": meta.get("tile_iou"),
        "tile_f1": meta.get("tile_f1"),
        "precision": meta.get("precision"),
        "recall": meta.get("recall"),
        "threshold": meta.get("threshold"),
        "flood_pct": meta.get("flood_pct"),
        "overlap_score": round(best_score, 3),
    }


@app.get("/tiles_list")
def tiles_list():
    registry = load_registry()
    return [
        {
            "id": tile_id,
            "bbox": meta["bbox"],
            "event": meta.get("event", meta.get("chip", tile_id)),
            "event_id": meta.get("event_id"),
            "split": meta.get("split", "unknown"),
        }
        for tile_id, meta in registry.items()
        if meta.get("bbox") is not None
    ]


@app.get("/metrics")
def metrics():
    return load_metrics()
