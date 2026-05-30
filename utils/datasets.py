import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from config import INPUT_VARS, TARGET_VARS, DEPTH_STRIDE

TIME_EPOCH = pd.Timestamp("2000-01-01")   # days since this date used as t


def time_to_days(timestamp):
    """Convert datetime64 timestamp to days since TIME_EPOCH as float."""
    ts = pd.Timestamp(timestamp)
    return float((ts - TIME_EPOCH).total_seconds() / 86400)


## ArgoProfileDataset ##
## One item = one cast (depth x vars profile matrix + mask + metadata)
## Used to train the encoder/decoder on INPUT_VARS
## If split="train", computes normalization stats from its own data.
## Pass train_ds.stats to val/probe datasets to ensure consistent scaling.

class ArgoProfileDataset(Dataset):

    def __init__(self, df, split, stats=None):
        """
        df    : full annotated dataframe from build_splits()
        split : one of "train", "test", "probe"
        stats : dict of {var: (mean, std)} — if None, computed from this split
        """
        self.split = split
        self.input_vars = INPUT_VARS

        split_df = df[df["split"] == split].copy()

        # group by cast — each cast is one profile
        self.casts = list(split_df.groupby("wod_unique_cast", sort=False))

        # compute or store normalization stats
        if stats is None:
            self.stats = self._compute_stats(split_df)
        else:
            self.stats = stats

    def _compute_stats(self, df):
        stats = {}
        for var in self.input_vars:
            col = df[var].astype(float)
            mean = col.mean()
            std  = col.std()
            std  = std if std > 1e-6 else 1.0
            stats[var] = (mean, std)
        return stats

    def _normalize(self, arr, var):
        mean, std = self.stats[var]
        return (arr - mean) / std

    def __len__(self):
        return len(self.casts)

    def __getitem__(self, idx):
        cast_id, cast_df = self.casts[idx]
        cast_df = cast_df.sort_values("z").reset_index(drop=True)
        cast_df = cast_df.iloc[::DEPTH_STRIDE].reset_index(drop=True)

        n_depths = len(cast_df)
        n_vars   = len(self.input_vars)

        profile = np.zeros((n_depths, n_vars), dtype=np.float32)
        mask    = np.zeros((n_depths, n_vars), dtype=bool)

        for j, var in enumerate(self.input_vars):
            col = cast_df[var].values.astype(float)
            valid = ~np.isnan(col)
            normalized = np.where(valid, self._normalize(col, var), 0.0)
            profile[:, j] = normalized
            mask[:, j]    = valid

        row = cast_df.iloc[0]

        return {
            "profile":  torch.tensor(profile),
            "mask":     torch.tensor(mask),
            "lat":      torch.tensor(row["lat"],      dtype=torch.float32),
            "lon":      torch.tensor(row["lon"],      dtype=torch.float32),
            "t":        torch.tensor(time_to_days(row["time"]), dtype=torch.float64),
            "wmo_id":   int(row["WMO_ID"]),
            "cast_id":  int(cast_id),
        }


## ArgoProbeDataset ##
## Inherits from ArgoProfileDataset, adds TARGET_VARS profiles to each item.
## TARGET_VARS are NOT normalized — we predict in original physical units.
## NaNs are preserved; masked MSE handles missing observations.

class ArgoProbeDataset(ArgoProfileDataset):

    def __init__(self, df, split, stats=None):
        super().__init__(df, split, stats=stats)
        self.target_vars = TARGET_VARS

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        cast_id, cast_df = self.casts[idx]
        cast_df = cast_df.sort_values("z").reset_index(drop=True)
        cast_df = cast_df.iloc[::DEPTH_STRIDE].reset_index(drop=True)

        n_depths    = len(cast_df)
        n_target    = len(self.target_vars)
        target      = np.full((n_depths, n_target), np.nan, dtype=np.float32)

        for j, var in enumerate(self.target_vars):
            if var in cast_df.columns:
                target[:, j] = cast_df[var].values.astype(float)
            # if column missing entirely, leave as NaN — masked MSE will ignore

        item["target"] = torch.tensor(target)   # (depth, n_target_vars), may contain NaN
        return item


## ArgoLatentDataset ##
## One item = one cast's pre-computed latent vector p + metadata
## Used to train the Neural ODE: dp/dt = f(p, lat, lon, t)

class ArgoLatentDataset(Dataset):

    def __init__(self, records):
        """
        records : list of dicts with keys:
                    p          - latent vector (latent_dim,)
                    lat        - float
                    lon        - float
                    t          - float
                    device_idx - int (integer-encoded WMO_ID)
                    cast_id    - int
        """
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "p":          torch.tensor(r["p"],   dtype=torch.float32),
            "lat":        torch.tensor(r["lat"], dtype=torch.float32),
            "lon":        torch.tensor(r["lon"], dtype=torch.float32),
            "t":          torch.tensor(r["t"],   dtype=torch.float64),
            "device_idx": int(r["device_idx"]),
            "cast_id":    int(r["cast_id"]),
        }

    @classmethod
    def from_encoder(cls, profile_dataset, encoder, device, wmo_to_idx):
        """
        Build an ArgoLatentDataset by running the encoder over a profile dataset.
        """
        encoder.eval()
        records = []

        with torch.no_grad():
            for item in profile_dataset:
                profile = item["profile"].unsqueeze(0).to(device)
                mask    = item["mask"].unsqueeze(0).to(device)

                p = encoder(profile, mask).squeeze(0).cpu().numpy()

                records.append({
                    "p":          p,
                    "lat":        item["lat"].item(),
                    "lon":        item["lon"].item(),
                    "t":          item["t"].item(),
                    "device_idx": wmo_to_idx[item["wmo_id"]],
                    "cast_id":    item["cast_id"],
                })

        return cls(records)
## ArgoJointWindowDataset ##
## Sliding window dataset for joint encoder + ODE + probe finetuning.
## Each item is a window of W consecutive casts from a single float,
## containing T/S profiles (encoder input), masks, oxygen targets,
## lat/lon, and time — everything the joint training loop needs.
##
## Mirrors SlidingWindowProbeDataset from train_probe.py but lives here
## for reuse across training scripts.

class ArgoJointWindowDataset(Dataset):

    def __init__(self, probe_dataset, window_size=5, stride=2):
        from collections import defaultdict

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
            times   = [t   for t, _  in t_idx_pairs]

            n = len(indices)
            for start in range(0, n - window_size + 1, stride):
                window_idx = indices[start : start + window_size]
                window_t   = times[start : start + window_size]
                if all(window_t[i] < window_t[i+1] for i in range(len(window_t) - 1)):
                    self.windows.append(window_idx)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        indices = self.windows[idx]
        items   = [self.probe_dataset[i] for i in indices]
        return {
            "profile": torch.stack([it["profile"] for it in items]),  # (W, D, n_vars)
            "mask":    torch.stack([it["mask"]    for it in items]),  # (W, D, n_vars)
            "target":  torch.stack([it["target"]  for it in items]),  # (W, D, n_target)
            "lat":     torch.stack([it["lat"]     for it in items]),  # (W,)
            "lon":     torch.stack([it["lon"]     for it in items]),  # (W,)
            "t":       torch.stack([it["t"]       for it in items]),  # (W,)
        }