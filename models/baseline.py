import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib, os


def extract_features(loader):
    """
    Flatten every tile in a DataLoader into per-pixel feature rows.
    Features: [VV, VH]  →  shape (N_pixels, 2)
    Labels  : flood=1/dry=0  →  shape (N_pixels,)
    """
    X, y = [], []
    for imgs, masks in loader:
        imgs = imgs.numpy()  # (B, 2, 256, 256)
        masks = masks.numpy()  # (B, 1, 256, 256)
        B = imgs.shape[0]
        for i in range(B):
            vv = imgs[i, 0].flatten()  # (65536,)
            vh = imgs[i, 1].flatten()  # (65536,)
            X.append(np.stack([vv, vh], axis=1))  # (65536, 2)
            y.append(masks[i, 0].flatten())  # (65536,)
    return np.vstack(X), np.concatenate(y)


def train_rf(
    train_loader,
    save_path="checkpoints/rf_baseline.joblib",
    n_estimators=100,
    max_samples=200_000,
):
    """
    Train Random Forest on pixel-level VV/VH features.
    Subsample to max_samples pixels to keep training fast.
    """
    print("📦 Extracting pixel features from train set...")
    X, y = extract_features(train_loader)
    print(f"   Raw   : {X.shape[0]:,} pixels, {y.mean() * 100:.1f}% flood")

    # Subsample — RF doesn't need all 3.5M pixels to learn
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    print(f"   Using : {X.shape[0]:,} pixels (subsampled), {y.mean() * 100:.1f}% flood")

    print("🌲 Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        class_weight="balanced",  # handles flood/dry imbalance
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y)
    print("✅ Training complete.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"💾 Saved to {save_path}")
    return clf


def evaluate_rf(clf, val_loader):
    """
    Evaluate RF on val set.
    Returns: pixel-level IoU and F1.
    """
    print("📊 Evaluating on val set...")
    X_val, y_val = extract_features(val_loader)

    y_pred = clf.predict(X_val)

    intersection = ((y_pred == 1) & (y_val == 1)).sum()
    union = ((y_pred == 1) | (y_val == 1)).sum()
    iou = intersection / (union + 1e-6)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    print(f"\n{'=' * 35}")
    print(f"  Baseline RF Results")
    print(f"{'=' * 35}")
    print(f"  IoU : {iou:.4f}")
    print(f"  F1  : {f1:.4f}")
    print(f"{'=' * 35}")
    print(f"  (UNet target IoU: >0.877)")
    return {"iou": iou, "f1": f1}
