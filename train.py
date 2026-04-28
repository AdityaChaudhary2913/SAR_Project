# train.py
import os
import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataset import get_dataloaders
from models.unet import get_model


# ── Loss Functions ─────────────────────────────────────────────────────────────


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target, bce_weight=0.4, dice_weight=0.6):
    """
    Weighted BCE + Dice.
    Dice weight is higher (0.6) because our flood class is only 7.9% of pixels.
    Dice is naturally robust to class imbalance; BCE alone would ignore flood pixels.
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dice


# ── Metrics ────────────────────────────────────────────────────────────────────


def iou_score(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


# ── Training Loop ──────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_iou = 0.0, 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)

        total_loss += combined_loss(preds, masks).item()
        total_iou += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_iou / n


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    # Load config
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        processed_dir=cfg["data"]["processed_dir"],
        tile_size=cfg["data"]["tile_size"],
        batch_size=cfg["train"]["batch_size"],
        train_split=cfg["data"]["train_split"],
    )

    # Model
    model = get_model(
        in_channels=cfg["model"]["in_channels"],
        features=cfg["model"]["features"],
        device=device,
    )

    optimizer = Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True
    )

    # Checkpointing
    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_iou, best_epoch = 0.0, 0

    print(f"\n{'=' * 55}")
    print(f"  Starting training — {cfg['train']['epochs']} epochs")
    print(f"{'=' * 55}\n")

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, device)

        scheduler.step(val_iou)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch:02d}/{cfg['train']['epochs']}  |  "
            f"Train Loss: {train_loss:.4f}  IoU: {train_iou:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  IoU: {val_iou:.4f}"
            + (" ⭐" if val_iou > best_iou else "")
        )

        # Save best checkpoint
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_loss": val_loss,
                },
                os.path.join(ckpt_dir, "unet_best.pth"),
            )

    print(f"\n{'=' * 55}")
    print(f"  Training complete!")
    print(f"  Best Val IoU : {best_iou:.4f} (epoch {best_epoch})")
    print(f"  RF Baseline  : 0.6148")
    print(f"  Target       : 0.877+")
    print(f"{'=' * 55}")

    # Save training history for evaluate.py
    np.save(os.path.join(ckpt_dir, "history.npy"), history)
    return model, history


if __name__ == "__main__":
    main()