"""
train/train_probe.py

Stage 3 — frozen encoder + frozen ODE probe for TARGET_VARS.
Checkpoint and losses saved to results_dir.
"""

import os
import time
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

from config import (
    LATENT_DIM, DEPTH_GRID,
    BATCH_SIZE, SEED,
    DECODER_HIDDEN, PROBE_EPOCHS, PROBE_LR,
)
from models.probe_decoder import OxygenDecoderHead
from utils.loss_logger import LossLogger

WINDOW_SIZE = 5
STRIDE      = 2
ODE_RTOL    = 1e-3
ODE_ATOL    = 1e-4
T_GRID      = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0], dtype=torch.float32)

torch.manual_seed(SEED)


class SlidingWindowProbeDataset(Dataset):

    def __init__(self, probe_dataset, window_size=WINDOW_SIZE, stride=STRIDE):
        self.probe_dataset = probe_dataset
        self.window_size   = window_size
        self.windows       = []

        device_indices = defaultdict(list)
        for i in range(len(probe_dataset)):
            item = probe_dataset[i]
            device_indices[item["wmo_id"]].append((item["t"].item(), i))

        for wmo_id, t_idx_pairs in device_indices.items():
            t_idx_pairs = sorted(t_idx_pairs, key=lambda x: x[0])
            indices = [idx for _, idx in t_idx_pairs]
            times   = [t   for t,  _  in t_idx_pairs]

            n = len(indices)
            for start in range(0, n - window_size + 1, stride):
                window_idx = indices[start : start + window_size]
                window_t   = times[start : start + window_size]
                if all(window_t[i] < window_t[i+1] for i in range(len(window_t)-1)):
                    self.windows.append(window_idx)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        indices = self.windows[idx]
        items   = [self.probe_dataset[i] for i in indices]
        return {
            "profile": torch.stack([it["profile"] for it in items]),
            "mask":    torch.stack([it["mask"]    for it in items]),
            "target":  torch.stack([it["target"]  for it in items]),
            "lat":     torch.stack([it["lat"]     for it in items]),
            "lon":     torch.stack([it["lon"]     for it in items]),
        }


def masked_mse(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return ((pred - target)[mask] ** 2).mean()


def encode_profiles(encoder, profiles, mask, device):
    with torch.no_grad():
        return encoder(profiles.to(device), mask.to(device))


def compute_oxy_stats(probe_dataset):
    all_targets = []
    for i in range(len(probe_dataset)):
        t = probe_dataset[i]["target"]
        all_targets.append(t.flatten())
    all_targets = torch.cat(all_targets)
    valid    = all_targets[~torch.isnan(all_targets)]
    oxy_mean = valid.mean().item()
    oxy_std  = valid.std().item()
    oxy_std  = oxy_std if oxy_std > 1e-6 else 1.0
    return oxy_mean, oxy_std


def train_probe(
    probe_dataset,
    encoder,
    ode_func,
    results_dir="results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "probe_head_best.pt")
    log_path  = os.path.join(results_dir, "probe_losses.csv")

    depth_tensor = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)
    t_grid       = T_GRID.to(device)

    for model in (encoder, ode_func):
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    encoder  = encoder.to(device)
    ode_func = ode_func.to(device)

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
            lat     = batch["lat"].to(device)
            lon     = batch["lon"].to(device)

            B, W, D, n_in = profile.shape
            profile_flat  = profile.reshape(B * W, D, n_in)
            mask_flat     = mask.reshape(B * W, D, n_in)
            p_flat        = encode_profiles(encoder, profile_flat, mask_flat, device)
            p             = p_flat.reshape(B, W, LATENT_DIM)

            lat0 = lat[:, 0:1]
            lon0 = lon[:, 0:1]
            z0   = torch.cat([p[:, 0, :], lat0, lon0], dim=-1)

            z_pred      = odeint(ode_func, z0, t_grid, method="dopri5",
                                 rtol=ODE_RTOL, atol=ODE_ATOL)
            p_pred      = z_pred[:, :, :LATENT_DIM].permute(1, 0, 2)
            p_pred_flat = p_pred.reshape(B * W, LATENT_DIM)
            target_flat = target.reshape(B * W, D, target.shape[-1])
            target_norm = (target_flat - oxy_mean_t) / oxy_std_t

            oxy_pred = probe_head(p_pred_flat, depth_tensor)
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
                lat     = batch["lat"].to(device)
                lon     = batch["lon"].to(device)

                B, W, D, n_in = profile.shape
                profile_flat  = profile.reshape(B * W, D, n_in)
                mask_flat     = mask.reshape(B * W, D, n_in)
                p_flat        = encode_profiles(encoder, profile_flat, mask_flat, device)
                p             = p_flat.reshape(B, W, LATENT_DIM)

                lat0 = lat[:, 0:1]
                lon0 = lon[:, 0:1]
                z0   = torch.cat([p[:, 0, :], lat0, lon0], dim=-1)

                z_pred      = odeint(ode_func, z0, t_grid, method="dopri5",
                                     rtol=ODE_RTOL, atol=ODE_ATOL)
                p_pred      = z_pred[:, :, :LATENT_DIM].permute(1, 0, 2)
                p_pred_flat = p_pred.reshape(B * W, LATENT_DIM)
                target_flat = target.reshape(B * W, D, target.shape[-1])
                target_norm = (target_flat - oxy_mean_t) / oxy_std_t

                oxy_pred  = probe_head(p_pred_flat, depth_tensor)
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

    print(f"\nProbe training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Losses saved to: {log_path}")
    return ckpt_path
