# SAR Flood Detection (Sentinel-1)

Click on the image to watch the demo
[![SAR Flood Detection Demo](https://img.youtube.com/vi/y6T_L64VMVg/maxresdefault.jpg)](https://www.youtube.com/watch?v=y6T_L64VMVg)

A small end-to-end SAR flood segmentation project: data preprocessing, a Random Forest baseline, a UNet model, and a map UI for AOI-based inference using exported prediction tiles.

## Highlights

- Task: flood / water segmentation on Sentinel-1 SAR chips
- Models: Random Forest baseline + UNet
- Data source: C2SMSFloods v1 (Cloud to Street + Microsoft)
- Data split: chip-level train/validation split plus a dedicated holdout event for final testing
- Features: normalized VV, VH, VV-VH, and VV/VH
- Labels: nodata is preserved through a valid-mask and ignored in loss / metrics
- Web app: AOI selection on a map, match the best exported tile, overlay the prediction mask

## Project Structure

- backend/ - FastAPI inference service that matches AOI to precomputed tiles
- frontend/ - React + Leaflet web app
- data/ - preprocessing and dataset loader
- models/ - UNet and RF baseline
- notebooks/kaggle_pipeline.ipynb - end-to-end training pipeline for Kaggle
- docs/SAR_Technical_Note.md - SAR fundamentals and ML survey
- configs/config.yaml - training and preprocessing config

## Setup (Local)

### 1) Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API serves exported prediction masks from `backend/tiles` and exposes:

- POST /predict - body: { "bbox": [min_lon, min_lat, max_lon, max_lat] }
- GET /tiles_list - available coverage rectangles
- GET /metrics - latest saved evaluation metrics from `checkspots/metrics.json`

### 3) Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open the app at http://localhost:5173 and ensure the backend is running on port 8000.

## Training (Kaggle Notebook Only)

Training is intended to be run in Kaggle using the provided notebook.

1) Create a new Kaggle Notebook (GPU recommended).
2) Upload notebooks/kaggle_pipeline.ipynb to the notebook.
3) Enable Internet in Kaggle (the notebook downloads data from S3).
4) Run all cells in order.

The notebook will:

- Install dependencies (awscli, rasterio, segmentation-models-pytorch)
- Clone this repo into /kaggle/working
- Download the configured train events plus one holdout event
- Preprocess the raw chips into data/processed
- Train the RF baseline and the UNet
- Tune the decision threshold on validation data
- Evaluate the dedicated holdout event
- Export demo tiles and save plots / metrics

Outputs to expect:

- checkspots/unet_best.pth
- checkspots/history.npy
- checkspots/metrics.json
- results/predictions_val.png
- results/predictions_test.png
- results/training_curve.png
- backend/tile_registry.json
- backend/tiles/*.png
- rf_baseline.joblib (saved under /kaggle/working/trained_models)

If Internet is disabled, upload the raw data into Kaggle and update the download cell to point to your local dataset path.

## Web App Notes

- The backend does not run the model live; it matches the AOI against a registry of exported prediction tiles and serves the corresponding PNG overlay.
- Coverage rectangles are colored by split so the holdout event is visible in the demo.

## References and Notes

- Technical note: docs/SAR_Technical_Note.md
- Configs: configs/config.yaml
- Kaggle pipeline: notebooks/kaggle_pipeline.ipynb
