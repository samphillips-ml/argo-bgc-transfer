"""
train/train_encoder.py

Stage 1 — encoder/decoder training on T/S/O reconstruction.
Checkpoint and losses saved to results_dir.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    LOW_DRIFT_PATH, INTERP_PATH, DEPTH_GRID,
    ENCODER_LR, ENCODER_EPOCHS, BATCH_SIZE, LATENT_DIM,
    ENCODER_HIDDEN, DECODER_HIDDEN,
)
from utils.split import build_splits
from utils.datasets import ArgoProfileDataset
from models.autoencoder import Autoencoder
from utils.loss_logger import LossLogger


def masked_mse(pred, target, mask):
    mask = mask.to(pred.device)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1)


def train_encoder(results_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "autoencoder_best.pt")
    log_path  = os.path.join(results_dir, "encoder_losses.csv")

    depth_tensor = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)

    df, _ = build_splits(LOW_DRIFT_PATH, INTERP_PATH)

    train_ds = ArgoProfileDataset(df, split="train")
    val_ds   = ArgoProfileDataset(df, split="test", stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = Autoencoder(latent_dim=LATENT_DIM,
                            encoder_hidden=ENCODER_HIDDEN,
                            decoder_hidden=DECODER_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ENCODER_LR)

    logger        = LossLogger(log_path)
    best_val_loss = float("inf")

    for epoch in range(1, ENCODER_EPOCHS + 1):
        t_start = time.time()

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            profile = batch["profile"].to(device)
            mask    = batch["mask"].to(device)

            recon, p = model(profile, mask, depth_tensor)
            loss = masked_mse(recon, profile, mask) + 1e-3 * (p ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                profile = batch["profile"].to(device)
                mask    = batch["mask"].to(device)

                recon, _ = model(profile, mask, depth_tensor)
                val_loss += masked_mse(recon, profile, mask).item()

        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{ENCODER_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  time={elapsed:.1f}s")

        logger.log(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(ckpt_path, stats=train_ds.stats)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

    print(f"\nEncoder training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path


if __name__ == "__main__":
    train_encoder()
