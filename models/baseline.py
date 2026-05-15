import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


def extract_features(loader):
    X, y = [], []
    for batch in loader:
        images = batch["image"].numpy()
        masks = batch["mask"].numpy()
        valid_masks = batch["valid_mask"].numpy()

        batch_size, channels, _, _ = images.shape
        for idx in range(batch_size):
            image = images[idx].reshape(channels, -1).T
            mask = masks[idx, 0].reshape(-1)
            valid = valid_masks[idx, 0].reshape(-1) > 0.5

            if not np.any(valid):
                continue

            X.append(image[valid])
            y.append(mask[valid])

    if not X:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.vstack(X), np.concatenate(y)

def extract_features_per_tile(loader):
    """Returns list of (X_tile, y_tile) tuples, one per valid sample."""
    tiles = []
    for batch in loader:
        images = batch["image"].numpy()
        masks = batch["mask"].numpy()
        valid_masks = batch["valid_mask"].numpy()

        batch_size, channels, _, _ = images.shape
        for idx in range(batch_size):
            image = images[idx].reshape(channels, -1).T  # (N_pixels, channels)
            mask = masks[idx, 0].reshape(-1)  # (N_pixels,)
            valid = valid_masks[idx, 0].reshape(-1) > 0.5  # (N_pixels,) bool

            if not np.any(valid):
                continue

            tiles.append((image[valid], mask[valid]))  # one entry per tile

    return tiles  # List of (X_tile, y_tile)


def train_rf(train_loader, n_estimators=100, max_samples=500000):
    print("📦 Extracting pixel features from train set...")
    X, y = extract_features(train_loader)
    if X.size == 0:
        raise RuntimeError("No valid pixels available for Random Forest training.")

    print(f"   Raw   : {X.shape[0]:,} pixels, {y.mean() * 100:.2f}% flood")
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    print(f"   Using : {X.shape[0]:,} pixels (subsampled), {y.mean() * 100:.2f}% flood")

    print("🌲 Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y)
    print("✅ Training complete.")
    return clf


def evaluate_rf(clf, loader, split_name="val"):
    print(f"📊 Evaluating RF on {split_name} split...")

    # --- Feature mismatch check (unchanged) ---
    X_eval, y_eval = extract_features(loader)
    if X_eval.size == 0:
        return None

    expected_features = getattr(clf, "n_features_in_", None)
    if expected_features is not None and X_eval.shape[1] != expected_features:
        print(f"⚠️  Skipping RF evaluation on {split_name}: ...")
        return {"skipped": True}

    # --- Global prediction (for F1, precision, recall) ---
    y_pred_global = clf.predict(X_eval)   # shape: (N_total_pixels,)

    # --- Per-tile IoU ---
    tiles = extract_features_per_tile(loader)   # list of (X_tile, y_tile)
    tile_ious = []
    for X_tile, y_tile in tiles:
        y_pred_tile = clf.predict(X_tile)
        intersection = ((y_pred_tile == 1) & (y_tile == 1)).sum()
        union        = ((y_pred_tile == 1) | (y_tile == 1)).sum()
        tile_ious.append(float(intersection / (union + 1e-6)))

    iou = float(np.mean(tile_ious)) if tile_ious else 0.0

    # --- Global metrics (use y_pred_global, not y_pred from loop) ---
    f1        = float(f1_score(y_eval, y_pred_global, zero_division=0))
    precision = float(precision_score(y_eval, y_pred_global, zero_division=0))
    recall    = float(recall_score(y_eval, y_pred_global, zero_division=0))

    print(f"\n{'=' * 35}")
    print(f"  Baseline RF Results ({split_name})")
    print(f"{'=' * 35}")
    print(f"Mean IoU    : {iou:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"{'=' * 35}")
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "num_pixels": int(len(y_eval)),
    }


def collect_rf_records(clf, loader):
    expected_features = getattr(clf, "n_features_in_", None)
    records = []

    for batch in loader:
        images = batch["image"].numpy()
        masks = batch["mask"].numpy()
        valid_masks = batch["valid_mask"].numpy()

        batch_size, channels, height, width = images.shape
        if expected_features is not None and channels != expected_features:
            return None, {
                "skipped": True,
                "reason": f"feature_mismatch:{expected_features}!={channels}",
                "expected_features": int(expected_features),
                "provided_features": int(channels),
            }

        for idx in range(batch_size):
            image = images[idx]
            mask = masks[idx, 0]
            valid_mask = valid_masks[idx, 0]
            pixel_features = image.reshape(channels, -1).T
            valid = valid_mask.reshape(-1) > 0.5

            probability = np.zeros(height * width, dtype=np.float32)
            if np.any(valid):
                valid_features = pixel_features[valid]
                if hasattr(clf, "predict_proba"):
                    probability[valid] = clf.predict_proba(valid_features)[:, 1].astype(np.float32)
                else:
                    probability[valid] = clf.predict(valid_features).astype(np.float32)

            records.append(
                {
                    "image": image,
                    "mask": mask,
                    "valid_mask": valid_mask,
                    "probability": probability.reshape(height, width),
                    "event_id": batch["event_id"][idx],
                    "source": batch["source"][idx],
                    "source_event_id": batch["source_event_id"][idx],
                    "chip_id": batch["chip_id"][idx],
                    "geo_image_path": batch["geo_image_path"][idx],
                    "row": int(batch["row"][idx]),
                    "col": int(batch["col"][idx]),
                    "tile_id": batch["tile_id"][idx],
                    "flood_ratio": float(batch["flood_ratio"][idx]),
                    "valid_ratio": float(batch["valid_ratio"][idx]),
                }
            )

    return records, None
