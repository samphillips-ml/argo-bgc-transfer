"""
train/train_probe_baseline.py

Depth-only baseline for the target variable probe. If this beats the
latent probe, the latent space isn't contributing anything beyond the
mean target-depth relationship.

Checkpoint and losses saved to results_dir.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    DEPTH_GRID, DECODER_HIDDEN,
    BATCH_SIZE, SEED, TARGET_VARS,
)
from train.train_probe import (
    SlidingWindowProbeDataset,
    masked_mse,
    compute_oxy_stats,
)

PROBE_LR     = 1e-3
PROBE_EPOCHS = 100

torch.manual_seed(SEED)


class DepthOnlyDecoder(nn.Module):
    """
    Predicts TARGET_VARS from depth alone — no latent vector.
    Same hidden architecture as OxygenDecoderHead for a fair comparison.
    """

    def __init__(self, hidden=DECODER_HIDDEN):
        super().__init__()
        n_out = len(TARGET_VARS)

        layers = []
        in_dim = 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, n_out)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, depth_levels):
        d = depth_levels.view(-1, 1)
        return self.mlp(d)


def train_probe_baseline(
    probe_dataset,
    results_dir="results",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(results_dir, "probe_baseline_best.pt")

    depth_tensor = torch.tensor(DEPTH_GRID, dtype=torch.float32).to(device)

    print("Computing target normalization stats...")
    oxy_mean, oxy_std = compute_oxy_stats(probe_dataset)
    print(f"Target stats — mean: {oxy_mean:.2f}, std: {oxy_std:.2f}")
    oxy_mean_t = torch.tensor(oxy_mean, dtype=torch.float32, device=device)
    oxy_std_t  = torch.tensor(oxy_std,  dtype=torch.float32, device=device)

    print("Building probe windows...")
    window_ds = SlidingWindowProbeDataset(probe_dataset)
    print(f"Probe windows: {len(window_ds)}")

    n_val   = max(1, int(0.2 * len(window_ds)))
    n_train = len(window_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        window_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = DepthOnlyDecoder(hidden=DECODER_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROBE_LR)

    best_val_loss = float("inf")

    for epoch in range(1, PROBE_EPOCHS + 1):
        t_start = time.time()

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            target = batch["target"].to(device)
            B, W, D, n_tgt = target.shape

            oxy_pred    = model(depth_tensor).unsqueeze(0).expand(B * W, -1, -1)
            target_flat = target.reshape(B * W, D, n_tgt)
            target_norm = (target_flat - oxy_mean_t) / oxy_std_t

            loss = masked_mse(oxy_pred, target_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                target = batch["target"].to(device)
                B, W, D, n_tgt = target.shape

                oxy_pred    = model(depth_tensor).unsqueeze(0).expand(B * W, -1, -1)
                target_flat = target.reshape(B * W, D, n_tgt)
                target_norm = (target_flat - oxy_mean_t) / oxy_std_t

                val_loss += masked_mse(oxy_pred, target_norm).item()

        val_loss /= len(val_loader)

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{PROBE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  time={elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "oxy_mean":    oxy_mean,
                "oxy_std":     oxy_std,
            }, ckpt_path)
            print(f"  -> saved checkpoint (val={best_val_loss:.4f})")

    print(f"\nBaseline training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")
    return ckpt_path
