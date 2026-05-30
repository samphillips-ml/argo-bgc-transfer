"""
train/train_probe_raw.py

Trains an MLP directly on raw T/S/O depth profiles to predict Chlorophyll.
No encoder, no latent space, no NODE. The key baseline: if this beats the
latent probe, compression is adding nothing. If latent beats this, the AE
is finding structure that raw inputs don't expose directly.

Same window/split setup as train_probe.py for a fair comparison.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    DEPTH_GRID, INPUT_VARS, TARGET_VARS,
    BATCH_SIZE, SEED,
    DECODER_HIDDEN, PROBE_EPOCHS, PROBE_LR,
)
from train.train_probe import (
    SlidingWindowProbeDataset, masked_mse, compute_oxy_stats,
)
from utils.loss_logger import LossLogger

WINDOW_SIZE = 5
STRIDE      = 2

N_IN  = len(INPUT_VARS)   # 3 (T/S/O)
N_OUT = len(TARGET_VARS)  # 1 (Chlorophyll)

torch.manual_seed(SEED)


class RawProfileProbe(nn.Module):
    """
    Per-depth MLP on raw T/S/O — mirrors the encoder+probe_head pipeline
    but without the compression step. Takes raw profile at each depth level
    and predicts target variable at that depth.

    Input:  (batch, depth, n_in)  — raw T/S/O at each depth
    Output: (batch, depth, n_out) — predicted Chlorophyll at each depth
    """

    def __init__(self, n_in=N_IN, n_out=N_OUT, hidden=DECODER_HIDDEN):
        super().__init__()

        layers = []
        in_dim = n_in + 1  # +1 for depth in meters, mirrors probe_decoder design
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_out)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, profile, depth_levels):
        """
        profile      : (batch, depth, n_in)
        depth_levels : (depth,) float tensor — DEPTH_GRID in meters
        returns      : (batch, depth, n_out)
        """
        batch = profile.shape[0]
        depth = depth_levels.shape[0]

        # append depth to each level — mirrors how probe_decoder appends depth to latent
        d   = depth_levels.view(1, -1, 1).expand(batch, -1, -1)  # (B, D, 1)
        inp = torch.cat([profile, d], dim=-1)                     # (B, D, n_in+1)

        return self.mlp(inp)                                       # (B, D, n_out)


def train_probe_raw(
    probe_dataset,
    results_dir="results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "probe_raw_best.pt")
    log_path  = os.path.join(results_dir, "probe_raw_losses.csv")

    depth_tensor = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)

    print("Computing target normalization stats...")
    oxy_mean, oxy_std = compute_oxy_stats(probe_dataset)
    print(f"Target stats — mean: {oxy_mean:.2f}, std: {oxy_std:.2f}")
    oxy_mean_t = torch.tensor(oxy_mean, dtype=torch.float32, device=device)
    oxy_std_t  = torch.tensor(oxy_std,  dtype=torch.float32, device=device)

    print("Building probe windows...")
    window_ds = SlidingWindowProbeDataset(probe_dataset, WINDOW_SIZE, STRIDE)
    print(f"Probe windows: {len(window_ds)}")

    n_val   = max(1, int(0.2 * len(window_ds)))
    n_train = len(window_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        window_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = RawProfileProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROBE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PROBE_EPOCHS, eta_min=1e-6
    )

    logger        = LossLogger(log_path)
    best_val_loss = float("inf")

    for epoch in range(1, PROBE_EPOCHS + 1):
        t_start = time.time()

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            profile = batch["profile"].to(device)  # (B, W, D, n_in)
            target  = batch["target"].to(device)   # (B, W, D, n_out)

            B, W, D, n_in = profile.shape
            profile_flat  = profile.reshape(B * W, D, n_in)
            target_flat   = target.reshape(B * W, D, target.shape[-1])
            target_norm   = (target_flat - oxy_mean_t) / oxy_std_t

            chl_pred = model(profile_flat, depth_tensor)
            loss     = masked_mse(chl_pred, target_norm)

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
                target  = batch["target"].to(device)

                B, W, D, n_in = profile.shape
                profile_flat  = profile.reshape(B * W, D, n_in)
                target_flat   = target.reshape(B * W, D, target.shape[-1])
                target_norm   = (target_flat - oxy_mean_t) / oxy_std_t

                chl_pred  = model(profile_flat, depth_tensor)
                val_loss += masked_mse(chl_pred, target_norm).item()

        val_loss /= len(val_loader)

        elapsed    = time.time() - t_start
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{PROBE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}  time={elapsed:.1f}s")

        logger.log(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "oxy_mean":    oxy_mean,
                "oxy_std":     oxy_std,
            }, ckpt_path)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

        scheduler.step()

    print(f"\nRaw probe training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path
