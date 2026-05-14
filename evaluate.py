import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

from backend.demo_assets import export_demo_assets
from data.dataset import get_dataloaders
from models.baseline import collect_rf_records, evaluate_rf
from models.unet import get_model
from train import combined_loss


def compute_sample_metrics(probability, mask, valid_mask, threshold):
    valid = valid_mask > 0.5
    if not np.any(valid):
        return {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    pred = (probability >= threshold) & valid
    target = (mask >= 0.5) & valid

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    intersection = tp
    union = tp + fp + fn
    iou = float(intersection / (union + 1e-6))
    precision = float(tp / (tp + fp + 1e-6))
    recall = float(tp / (tp + fn + 1e-6))
    f1 = float((2 * tp) / (2 * tp + fp + fn + 1e-6))
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


@torch.no_grad()
def collect_records(model, loader, device):
    model.eval()
    records = []
    losses = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        valid_masks = batch["valid_mask"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        masks_np = masks.cpu().numpy()
        valid_np = valid_masks.cpu().numpy()
        images_np = batch["image"].numpy()

        loss = combined_loss(logits, masks, valid_masks, bce_weight=0.4, dice_weight=0.6).item()
        losses.append(loss)

        for idx in range(images.shape[0]):
            records.append(
                {
                    "image": images_np[idx],
                    "mask": masks_np[idx, 0],
                    "valid_mask": valid_np[idx, 0],
                    "probability": probs[idx, 0],
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

    return records, float(np.mean(losses)) if losses else 0.0


def summarize_records(records, threshold, loss_value):
    per_tile = []
    for record in records:
        metrics = compute_sample_metrics(
            record["probability"],
            record["mask"],
            record["valid_mask"],
            threshold=threshold,
        )
        per_tile.append({"tile_id": record["tile_id"], **metrics})

    if not per_tile:
        return {
            "threshold": threshold,
            "loss": loss_value,
            "mean_iou": 0.0,
            "mean_f1": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "num_tiles": 0,
            "per_tile": [],
        }

    return {
        "threshold": threshold,
        "loss": loss_value,
        "mean_iou": float(np.mean([item["iou"] for item in per_tile])),
        "mean_f1": float(np.mean([item["f1"] for item in per_tile])),
        "mean_precision": float(np.mean([item["precision"] for item in per_tile])),
        "mean_recall": float(np.mean([item["recall"] for item in per_tile])),
        "num_tiles": len(per_tile),
        "per_tile": per_tile,
    }


def attach_tile_metrics(records, threshold):
    metrics_by_tile = {}
    for record in records:
        metrics_by_tile[record["tile_id"]] = {
            "tile_id": record["tile_id"],
            **compute_sample_metrics(
                record["probability"],
                record["mask"],
                record["valid_mask"],
                threshold=threshold,
            ),
        }
    return metrics_by_tile


def tune_threshold(records, thresholds):
    best_summary = None
    for threshold in thresholds:
        summary = summarize_records(records, threshold=threshold, loss_value=0.0)
        if best_summary is None or summary["mean_iou"] > best_summary["mean_iou"]:
            best_summary = summary
    return best_summary


def print_summary(split_name, summary):
    print(f"\n{'=' * 44}")
    print(f"  UNet Evaluation Results ({split_name})")
    print(f"{'=' * 44}")
    print(f"  Threshold        : {summary['threshold']:.2f}")
    print(f"  Mean Loss        : {summary['loss']:.4f}")
    print(f"  Mean IoU         : {summary['mean_iou']:.4f}")
    print(f"  Mean F1          : {summary['mean_f1']:.4f}")
    print(f"  Mean Precision   : {summary['mean_precision']:.4f}")
    print(f"  Mean Recall      : {summary['mean_recall']:.4f}")
    print(f"  Tiles evaluated  : {summary['num_tiles']}")
    print(f"{'=' * 44}")


def plot_predictions(records, threshold, split_name, save_dir, n=4):
    if not records:
        return None

    os.makedirs(save_dir, exist_ok=True)
    n = min(n, len(records))
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        f"UNet SAR Flood Detection - {split_name.title()} Split",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    col_titles = ["VV Band", "Ground Truth", "Probability", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    for idx, record in enumerate(records[:n]):
        metrics = compute_sample_metrics(
            record["probability"],
            record["mask"],
            record["valid_mask"],
            threshold=threshold,
        )
        vv = record["image"][0]
        gt = np.where(record["valid_mask"] > 0.5, record["mask"], np.nan)
        prob = np.where(record["valid_mask"] > 0.5, record["probability"], np.nan)
        pred = (record["probability"] >= threshold).astype(np.uint8)

        overlay = np.zeros((*record["mask"].shape, 3), dtype=np.float32)
        valid = record["valid_mask"] > 0.5
        target = (record["mask"] >= 0.5) & valid
        pred_valid = (pred == 1) & valid
        tp = pred_valid & target
        fp = pred_valid & ~target
        fn = (~pred_valid) & target
        overlay[tp] = [0.0, 0.8, 0.0]
        overlay[fp] = [1.0, 0.4, 0.0]
        overlay[fn] = [1.0, 0.0, 0.0]

        axes[idx, 0].imshow(vv, cmap="gray")
        axes[idx, 0].set_ylabel(
            f"{record['event_id'][:8]}\nIoU={metrics['iou']:.3f}",
            fontsize=9,
        )
        axes[idx, 1].imshow(gt, cmap="Blues", vmin=0, vmax=1)
        axes[idx, 2].imshow(prob, cmap="Blues", vmin=0, vmax=1)
        axes[idx, 3].imshow(vv, cmap="gray", alpha=0.5)
        axes[idx, 3].imshow(overlay, alpha=0.65)

    legend = [
        mpatches.Patch(color=(0.0, 0.8, 0.0), label="True Positive"),
        mpatches.Patch(color=(1.0, 0.4, 0.0), label="False Positive"),
        mpatches.Patch(color=(1.0, 0.0, 0.0), label="False Negative"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"predictions_{split_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved to {out_path}")
    return out_path


def plot_training_curve(history_path, metrics_path, save_dir):
    if not os.path.exists(history_path):
        return None

    os.makedirs(save_dir, exist_ok=True)
    history = np.load(history_path, allow_pickle=True).item()
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    epochs = range(1, len(history["val_iou"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#e07b39")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#3a7ebf")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_iou"], label="Train IoU@0.50", color="#e07b39")
    ax2.plot(epochs, history["val_iou"], label="Val IoU@0.50", color="#3a7ebf")

    best_val_iou = max(history["val_iou"]) if history["val_iou"] else 0.0
    ax2.axhline(best_val_iou, color="#3a7ebf", linestyle=":", linewidth=1.2, label=f"Best Val IoU@0.50 ({best_val_iou:.4f})")

    rf_val = metrics.get("rf_baseline", {}).get("val") if metrics else None
    if rf_val and not rf_val.get("skipped") and "iou" in rf_val:
        rf_iou = rf_val["iou"]
        ax2.axhline(rf_iou, color="gray", linestyle="--", linewidth=1.2, label=f"RF Baseline ({rf_iou:.4f})")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("IoU Progress")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle("UNet Training Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Saved to {out_path}")
    return out_path


def build_metrics_payload(cfg, checkpoint, val_summary, test_summary, rf_baseline=None):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "checkpoint": cfg["paths"]["best_checkpoint"],
            "best_epoch": int(checkpoint.get("epoch", 0)),
            "in_channels": int(checkpoint.get("in_channels", 0)),
            "derived_channels": checkpoint.get("derived_channels", []),
            "best_threshold": float(val_summary["threshold"]),
            "selection_metric": "val_mean_iou",
        },
        "splits": {
            "val": val_summary,
            "test": test_summary,
        },
        "rf_baseline": rf_baseline or {},
    }


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, data_meta = get_dataloaders(
        processed_dir=cfg["data"]["processed_dir"],
        tile_size=cfg["data"]["tile_size"],
        batch_size=cfg["train"]["batch_size"],
        train_split=cfg["data"].get("train_split"),
        val_split=cfg["data"].get("val_split"),
        num_workers=cfg["train"].get("num_workers", 2),
        seed=cfg["train"].get("seed", 42),
        split_scope=cfg["data"].get("split_scope", "chip"),
        train_event_ids=cfg["data"].get("train_event_ids", []),
        holdout_event_ids=cfg["data"].get("holdout_event_ids", []),
        augment_train=False,
        balance_train=False,
        return_test=True,
    )

    checkpoint = torch.load(cfg["paths"]["best_checkpoint"], map_location=device)
    in_channels = int(checkpoint.get("in_channels") or data_meta["in_channels"])
    model = get_model(in_channels=in_channels, features=cfg["model"]["features"], device=device)
    model.load_state_dict(checkpoint["model_state"])
    print(
        f"✅ Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(Val IoU@0.50: {checkpoint.get('val_iou_05', 0.0):.4f})"
    )

    val_records, val_loss = collect_records(model, val_loader, device)
    thresholds = np.arange(0.1, 0.91, 0.05)
    tuned_val = tune_threshold(val_records, thresholds=thresholds)
    val_summary = summarize_records(val_records, threshold=tuned_val["threshold"], loss_value=val_loss)
    print_summary("val", val_summary)

    test_records, test_loss = collect_records(model, test_loader, device)
    test_summary = summarize_records(test_records, threshold=tuned_val["threshold"], loss_value=test_loss)
    if test_summary["num_tiles"] > 0:
        print_summary("test", test_summary)

    rf_baseline = {}
    rf_test_records = None
    rf_skip = None
    rf_path = cfg["paths"].get("rf_baseline")
    if rf_path and os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        rf_baseline["val"] = evaluate_rf(rf_model, val_loader, split_name="val")
        if test_summary["num_tiles"] > 0:
            rf_baseline["test"] = evaluate_rf(rf_model, test_loader, split_name="test")
            rf_test_records, rf_skip = collect_rf_records(rf_model, test_loader)
            if rf_skip:
                rf_baseline["test_export"] = rf_skip

    metrics_payload = build_metrics_payload(
        cfg,
        checkpoint=checkpoint,
        val_summary=val_summary,
        test_summary=test_summary,
        rf_baseline=rf_baseline,
    )

    metrics_path = cfg["paths"]["metrics"]
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"💾 Saved metrics to {metrics_path}")

    plot_predictions(
        val_records,
        threshold=val_summary["threshold"],
        split_name="val",
        save_dir=cfg["paths"]["results"],
        n=4,
    )
    if test_summary["num_tiles"] > 0:
        plot_predictions(
            test_records,
            threshold=val_summary["threshold"],
            split_name="test",
            save_dir=cfg["paths"]["results"],
            n=4,
        )

    plot_training_curve(
        history_path=cfg["paths"]["history"],
        metrics_path=metrics_path,
        save_dir=cfg["paths"]["results"],
    )

    demo_cfg = cfg.get("demo", {})
    if demo_cfg.get("export_assets", False) and test_summary["num_tiles"] > 0:
        metrics_by_tile = attach_tile_metrics(test_records, threshold=val_summary["threshold"])
        rf_metrics_by_tile = {}
        rf_records_by_tile = {}
        if rf_test_records:
            rf_metrics_by_tile = attach_tile_metrics(rf_test_records, threshold=val_summary["threshold"])
            rf_records_by_tile = {record["tile_id"]: record for record in rf_test_records}
        demo_records = []
        for record in test_records:
            tile_id = record["tile_id"]
            rf_record = rf_records_by_tile.get(tile_id)
            if rf_record is not None:
                rf_record = {**rf_record, **rf_metrics_by_tile[tile_id]}
            demo_records.append({**record, **metrics_by_tile[tile_id], "rf_record": rf_record})

        export_demo_assets(
            split_records={"test": demo_records},
            threshold=val_summary["threshold"],
            registry_path=demo_cfg["registry_path"],
            tiles_dir=demo_cfg["tiles_dir"],
            event_names=cfg["data"].get("event_names", {}),
        )
        print(f"💾 Exported demo assets for holdout test tiles to {demo_cfg['registry_path']}")


if __name__ == "__main__":
    main()
