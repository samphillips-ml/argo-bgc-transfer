"""
train/train_probe_static.py

Static latent probe — frozen encoder output directly, no dynamics model.
Fair comparison against GRU probe since probe head is trained fresh.
Checkpoint and losses saved to results_dir.
"""

import os
import time
import torch
from torch.utils.data import DataLoader

from config import (
    LATENT_DIM, DEPTH_GRID,
    BATCH_SIZE, SEED,
    DECODER_HIDDEN, PROBE_EPOCHS, PROBE_LR,
)
from models.probe_decoder import OxygenDecoderHead
from utils.probe_utils import (
    SlidingWindowProbeDataset, masked_mse,
    encode_profiles, compute_oxy_stats,
)
from utils.loss_logger import LossLogger

WINDOW_SIZE = 5
STRIDE      = 2

torch.manual_seed(SEED)


def train_probe_static(
    probe_dataset,
    encoder,
    results_dir="results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "probe_static_best.pt")
    log_path  = os.path.join(results_dir, "probe_static_losses.csv")

    depth_tensor = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = encoder.to(device)

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

    probe_head = OxygenDecoderHead(latent_dim=LATENT_DIM, hidden=DECODER_HIDDEN).to(device)
    optimizer  = torch.optim.Adam(probe_head.parameters(), lr=PROBE_LR)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PROBE_EPOCHS, eta_min=1e-6
    )

    logger        = LossLogger(log_path)
    best_val_loss = float("inf")

    for epoch in range(1, PROBE_EPOCHS + 1):
        t_start = time.time()

        probe_head.train()
        train_loss = 0.0

        for batch in train_loader:
            profile = batch["profile"].to(device)
            mask    = batch["mask"].to(device)
            target  = batch["target"].to(device)

            B, W, D, n_in = profile.shape
            profile_flat  = profile.reshape(B * W, D, n_in)
            mask_flat     = mask.reshape(B * W, D, n_in)

            p_flat      = encode_profiles(encoder, profile_flat, mask_flat, device)
            target_flat = target.reshape(B * W, D, target.shape[-1])
            target_norm = (target_flat - oxy_mean_t) / oxy_std_t

            oxy_pred = probe_head(p_flat, depth_tensor)
            loss     = masked_mse(oxy_pred, target_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        probe_head.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                profile = batch["profile"].to(device)
                mask    = batch["mask"].to(device)
                target  = batch["target"].to(device)

                B, W, D, n_in = profile.shape
                profile_flat  = profile.reshape(B * W, D, n_in)
                mask_flat     = mask.reshape(B * W, D, n_in)

                p_flat      = encode_profiles(encoder, profile_flat, mask_flat, device)
                target_flat = target.reshape(B * W, D, target.shape[-1])
                target_norm = (target_flat - oxy_mean_t) / oxy_std_t

                oxy_pred  = probe_head(p_flat, depth_tensor)
                val_loss += masked_mse(oxy_pred, target_norm).item()

        val_loss /= len(val_loader)

        elapsed    = time.time() - t_start
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{PROBE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}  time={elapsed:.1f}s")

        logger.log(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": probe_head.state_dict(),
                "oxy_mean":    oxy_mean,
                "oxy_std":     oxy_std,
            }, ckpt_path)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

        scheduler.step()

    print(f"\nStatic probe training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path
