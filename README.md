# SAR Flood Detection (Sentinel-1)

![SAR Flood Detection Demo](https://www.youtube.com/watch?v=y6T_L64VMVg)

A small end-to-end SAR flood segmentation project: data preprocessing, a Random Forest baseline, a UNet model, and a minimal web map UI for AOI-based inference using precomputed masks.

## Highlights

- Task: flood / water segmentation on Sentinel-1 VV/VH SAR chips
- Models: Random Forest baseline + UNet
- Data source: C2SMSFloods v1 (Cloud to Street + Microsoft), two flood events
- Web app: AOI selection on a map, match the best tile, overlay precomputed mask

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

The API serves precomputed masks from backend/tiles and exposes:

- POST /predict - body: { "bbox": [min_lon, min_lat, max_lon, max_lat] }
- GET /tiles_list - available coverage rectangles

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
- Download C2SMSFloods chips for two flood events
- Preprocess the raw chips into data/processed
- Train the RF baseline and the UNet
- Run evaluation and save plots

Outputs to expect:

- checkspots/unet_best.pth
- checkspots/history.npy
- results/predictions.png
- results/training_curve.png
- rf_baseline.joblib (saved under /kaggle/working/trained_models)

If Internet is disabled, upload the raw data into Kaggle and update the download cell to point to your local dataset path.

## Web App Notes

- The backend does not run the model live; it matches the AOI against a registry of tiles and serves the corresponding precomputed PNG mask.
- The dashed rectangles on the map show where tiles exist. Draw your AOI inside those areas for a valid match.

## References and Notes

- Technical note: docs/SAR_Technical_Note.md
- Configs: configs/config.yaml
- Kaggle pipeline: notebooks/kaggle_pipeline.ipynb
