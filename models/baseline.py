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
    X_eval, y_eval = extract_features(loader)
    if X_eval.size == 0:
        return None

    y_pred = clf.predict(X_eval)
    intersection = ((y_pred == 1) & (y_eval == 1)).sum()
    union = ((y_pred == 1) | (y_eval == 1)).sum()
    iou = float(intersection / (union + 1e-6))
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))
    precision = float(precision_score(y_eval, y_pred, zero_division=0))
    recall = float(recall_score(y_eval, y_pred, zero_division=0))

    print(f"\n{'=' * 35}")
    print(f"  Baseline RF Results ({split_name})")
    print(f"{'=' * 35}")
    print(f"  IoU       : {iou:.4f}")
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
