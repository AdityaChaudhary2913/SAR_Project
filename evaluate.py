# evaluate.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data.dataset import get_dataloaders
from models.unet import get_model


def iou_score(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def f1_score_torch(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    return (2 * tp / (2 * tp + fp + fn + 1e-6)).item()


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    ious, f1s = [], []

    for imgs, masks in val_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        for i in range(imgs.shape[0]):
            ious.append(iou_score(preds[i : i + 1], masks[i : i + 1]))
            f1s.append(f1_score_torch(preds[i : i + 1], masks[i : i + 1]))

    print(f"\n{'=' * 40}")
    print(f"  UNet Evaluation Results")
    print(f"{'=' * 40}")
    print(f"  Val IoU (mean) : {np.mean(ious):.4f}")
    print(f"  Val IoU (best) : {np.max(ious):.4f}")
    print(f"  Val IoU (worst): {np.min(ious):.4f}")
    print(f"  Val F1  (mean) : {np.mean(f1s):.4f}")
    print(f"{'=' * 40}")
    print(f"  RF Baseline IoU: 0.6148")
    print(f"  UNet Gain      : +{np.mean(ious) - 0.6148:.4f}")
    print(f"{'=' * 40}\n")
    return ious, f1s


@torch.no_grad()
def plot_predictions(model, val_loader, device, n=4, save_dir="results"):
    """Plot n side-by-side: VV band | Ground Truth | Prediction | Overlay"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    imgs_batch, masks_batch = next(iter(val_loader))
    imgs_batch = imgs_batch.to(device)
    preds_batch = torch.sigmoid(model(imgs_batch)).cpu().numpy()
    imgs_np = imgs_batch.cpu().numpy()
    masks_np = masks_batch.numpy()

    fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))
    fig.suptitle(
        "UNet SAR Flood Detection — Qualitative Results",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    col_titles = ["VV Band (Input)", "Ground Truth", "Prediction", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    for i in range(n):
        vv = imgs_np[i, 0]  # VV band
        gt = masks_np[i, 0]  # ground truth
        pred = (preds_batch[i, 0] > 0.5).astype(float)  # binary prediction
        prob = preds_batch[i, 0]  # probability map

        tile_iou = iou_score(
            torch.tensor(preds_batch[i : i + 1]), torch.tensor(masks_np[i : i + 1])
        )

        # Col 0: VV band
        axes[i, 0].imshow(vv, cmap="gray")
        axes[i, 0].set_ylabel(f"Tile {i + 1}\nIoU={tile_iou:.3f}", fontsize=9)

        # Col 1: Ground truth
        axes[i, 1].imshow(gt, cmap="Blues", vmin=0, vmax=1)

        # Col 2: Prediction probability
        axes[i, 2].imshow(prob, cmap="Blues", vmin=0, vmax=1)

        # Col 3: Overlay — TP/FP/FN coloured
        overlay = np.zeros((*gt.shape, 3))
        tp = (pred == 1) & (gt == 1)
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)
        overlay[tp] = [0.0, 0.8, 0.0]  # green  = correct flood
        overlay[fp] = [1.0, 0.4, 0.0]  # orange = false alarm
        overlay[fn] = [1.0, 0.0, 0.0]  # red    = missed flood
        axes[i, 3].imshow(vv, cmap="gray", alpha=0.5)
        axes[i, 3].imshow(overlay, alpha=0.6)

    # Legend for overlay column
    legend = [
        mpatches.Patch(color=(0.0, 0.8, 0.0), label="True Positive"),
        mpatches.Patch(color=(1.0, 0.4, 0.0), label="False Positive"),
        mpatches.Patch(color=(1.0, 0.0, 0.0), label="False Negative"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "predictions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Saved to {out_path}")


def plot_training_curve(history_path="checkspots/history.npy", save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    history = np.load(history_path, allow_pickle=True).item()

    epochs = range(1, len(history["val_iou"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#e07b39")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#3a7ebf")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_iou"], label="Train IoU", color="#e07b39")
    ax2.plot(epochs, history["val_iou"], label="Val IoU", color="#3a7ebf")
    ax2.axhline(
        0.6240,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"RF Baseline (0.6240)",
    )
    ax2.axhline(
        0.877, color="green", linestyle="--", linewidth=1.2, label=f"Target (0.877)"
    )
    ax2.axhline(
        0.8075,
        color="#3a7ebf",
        linestyle=":",
        linewidth=1.2,
        label=f"Best UNet (0.8075)",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("IoU Progress")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle("UNet Training Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Saved to {out_path}")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_loader = get_dataloaders("data/processed", tile_size=256, batch_size=8)

    model = get_model(device=device)
    ckpt = torch.load("checkspots/unet_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(
        f"✅ Loaded checkpoint from epoch {ckpt['epoch']} "
        f"(Val IoU: {ckpt['val_iou']:.4f})"
    )

    ious, f1s = evaluate(model, val_loader, device)
    plot_predictions(model, val_loader, device, n=4, save_dir="results")
    plot_training_curve("checkspots/history.npy", save_dir="results")