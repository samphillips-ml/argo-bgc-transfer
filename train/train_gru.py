"""
train/train_gru.py

GRU dynamics baseline — discrete-time alternative to the Neural ODE.
Same setup as train_node.py; only the dynamics model changes.

Checkpoint and losses saved to results_dir.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ODE_LR, ODE_EPOCHS, BATCH_SIZE, LATENT_DIM, ODE_HIDDEN, WINDOW_SIZE, STRIDE
from utils.datasets import ArgoLatentDataset
from models.gru import GRUDynamics
from train.train_node import SlidingWindowDataset
from utils.loss_logger import LossLogger


def train_gru(
    latent_path="results/latent_cycles.pt",
    results_dir="results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "gru_best.pt")
    log_path  = os.path.join(results_dir, "gru_losses.csv")

    print(f"Loading latent cycles from {latent_path}")
    ckpt = torch.load(latent_path, map_location="cpu", weights_only=False)

    latent_train = ArgoLatentDataset(ckpt["train"])
    latent_val   = ArgoLatentDataset(ckpt["val"])

    print("Building train windows...")
    train_windows = SlidingWindowDataset(latent_train, WINDOW_SIZE, STRIDE)
    print(f"Train windows: {len(train_windows)}")

    print("Building val windows...")
    val_windows = SlidingWindowDataset(latent_val, WINDOW_SIZE, STRIDE)
    print(f"Val windows: {len(val_windows)}")

    train_loader = DataLoader(train_windows, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_windows,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = GRUDynamics(latent_dim=LATENT_DIM, hidden=ODE_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ODE_LR)
    loss_fn   = nn.MSELoss()

    logger        = LossLogger(log_path)
    best_val_loss = float("inf")
    n_steps       = WINDOW_SIZE - 1

    for epoch in range(1, ODE_EPOCHS + 1):
        t_start = time.time()

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            p   = batch["p"].to(device)
            lat = batch["lat"].to(device)
            lon = batch["lon"].to(device)

            p_traj = model(p[:, 0, :], lat[:, 0], lon[:, 0], n_steps)
            p_pred = p_traj.permute(1, 0, 2)
            loss   = loss_fn(p_pred, p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                p   = batch["p"].to(device)
                lat = batch["lat"].to(device)
                lon = batch["lon"].to(device)

                p_traj = model(p[:, 0, :], lat[:, 0], lon[:, 0], n_steps)
                p_pred = p_traj.permute(1, 0, 2)
                val_loss += loss_fn(p_pred, p).item()

        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{ODE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  time={elapsed:.1f}s")

        logger.log(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict()}, ckpt_path)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

    print(f"\nGRU training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path


if __name__ == "__main__":
    train_gru()
