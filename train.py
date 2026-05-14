import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataset import get_dataloaders
from models.unet import get_model


def masked_bce_with_logits(logits, target, valid_mask):
    losses = nn.BCEWithLogitsLoss(reduction="none")(logits, target)
    masked_losses = losses * valid_mask
    denom = valid_mask.sum().clamp_min(1.0)
    return masked_losses.sum() / denom


def dice_loss(logits, target, valid_mask, smooth=1.0):
    pred = torch.sigmoid(logits) * valid_mask
    target = target * valid_mask

    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3))
    dice = 1 - (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.mean()


def combined_loss(logits, target, valid_mask, bce_weight=0.4, dice_weight=0.6):
    bce = masked_bce_with_logits(logits, target, valid_mask)
    dice = dice_loss(logits, target, valid_mask)
    return bce_weight * bce + dice_weight * dice


def iou_score(logits, target, valid_mask, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float() * valid_mask
    target = target * valid_mask

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    valid_pixels = valid_mask.sum(dim=(1, 2, 3))
    valid_samples = valid_pixels > 0
    if not valid_samples.any():
        return 0.0

    scores = torch.where(union > 0, intersection / (union + 1e-6), torch.zeros_like(union))
    return scores[valid_samples].mean().item()


def train_one_epoch(model, loader, optimizer, device, loss_cfg):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    steps = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        valid_masks = batch["valid_mask"].to(device)
        if valid_masks.sum().item() == 0:
            continue

        optimizer.zero_grad()
        logits = model(images)
        loss = combined_loss(
            logits,
            masks,
            valid_masks,
            bce_weight=loss_cfg["bce_weight"],
            dice_weight=loss_cfg["dice_weight"],
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += iou_score(logits, masks, valid_masks)
        steps += 1

    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_iou / steps


@torch.no_grad()
def validate(model, loader, device, loss_cfg):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    steps = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        valid_masks = batch["valid_mask"].to(device)
        if valid_masks.sum().item() == 0:
            continue

        logits = model(images)
        total_loss += combined_loss(
            logits,
            masks,
            valid_masks,
            bce_weight=loss_cfg["bce_weight"],
            dice_weight=loss_cfg["dice_weight"],
        ).item()
        total_iou += iou_score(logits, masks, valid_masks)
        steps += 1

    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_iou / steps


def main():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    train_loader, val_loader, _, data_meta = get_dataloaders(
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
        augment_train=cfg["train"].get("augment_train", True),
        balance_train=cfg["train"].get("balance_train", True),
        return_test=True,
        no_flood_max=cfg["train"].get("sampler_no_flood_max", 0.0),
        low_flood_max=cfg["train"].get("sampler_low_flood_max", 0.02),
    )

    model = get_model(
        in_channels=data_meta["in_channels"],
        features=cfg["model"]["features"],
        device=device,
    )

    optimizer = Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    loss_cfg = cfg["train"].get("loss", {"bce_weight": 0.4, "dice_weight": 0.6})

    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_iou = 0.0
    best_epoch = 0

    print(f"\n{'=' * 55}")
    print(f"  Starting training - {cfg['train']['epochs']} epochs")
    print(f"{'=' * 55}\n")

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device, loss_cfg)
        val_loss, val_iou = validate(model, val_loader, device, loss_cfg)
        scheduler.step(val_iou)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch:02d}/{cfg['train']['epochs']}  |  "
            f"Train Loss: {train_loss:.4f}  IoU@0.50: {train_iou:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  IoU@0.50: {val_iou:.4f}"
            + (" *" if val_iou > best_iou else "")
        )

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_iou_05": val_iou,
                    "val_loss": val_loss,
                    "in_channels": data_meta["in_channels"],
                    "split_scope": data_meta["split_scope"],
                    "holdout_event_ids": data_meta["holdout_event_ids"],
                    "train_event_ids": data_meta["train_event_ids"],
                    "derived_channels": data_meta["derived_channels"],
                },
                os.path.join(ckpt_dir, "unet_best.pth"),
            )

    np.save(os.path.join(ckpt_dir, "history.npy"), history)
    with open(os.path.join(ckpt_dir, "train_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_iou_05": best_iou,
                "in_channels": data_meta["in_channels"],
                "derived_channels": data_meta["derived_channels"],
                "split_scope": data_meta["split_scope"],
                "train_event_ids": data_meta["train_event_ids"],
                "holdout_event_ids": data_meta["holdout_event_ids"],
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 55}")
    print("  Training complete!")
    print(f"  Best Val IoU@0.50 : {best_iou:.4f} (epoch {best_epoch})")
    if data_meta["holdout_event_ids"]:
        print(f"  Holdout test event(s): {', '.join(data_meta['holdout_event_ids'])}")
    print(f"{'=' * 55}")
    return model, history


if __name__ == "__main__":
    main()
