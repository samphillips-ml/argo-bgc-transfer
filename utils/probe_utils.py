# utils/probe_utils.py
# shared utilities for probe and GRU training scripts

import torch
from collections import defaultdict
from torch.utils.data import Dataset

from config import SEED, WINDOW_SIZE, STRIDE

torch.manual_seed(SEED)


## SlidingWindowProbeDataset ##
## windows of W consecutive casts from the same float
## each item has profile, mask, target, lat, lon

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


## SlidingWindowDataset ##
## windows of W consecutive latent vectors from the same float
## used by train_gru.py

class SlidingWindowDataset(Dataset):

    def __init__(self, latent_dataset, window_size=WINDOW_SIZE, stride=STRIDE):
        self.records     = latent_dataset.records
        self.window_size = window_size
        self.windows     = []

        device_records = defaultdict(list)
        for r in self.records:
            device_records[r["device_idx"]].append(r)

        for device_idx, recs in device_records.items():
            recs = sorted(recs, key=lambda r: r["t"])
            n = len(recs)
            for start in range(0, n - window_size + 1, stride):
                window = recs[start : start + window_size]
                times  = [r["t"] for r in window]
                if all(times[i] < times[i+1] for i in range(len(times)-1)):
                    self.windows.append(window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        p   = torch.stack([torch.tensor(r["p"],   dtype=torch.float32) for r in window])
        lat = torch.tensor([r["lat"] for r in window], dtype=torch.float32)
        lon = torch.tensor([r["lon"] for r in window], dtype=torch.float32)
        return {"p": p, "lat": lat, "lon": lon}


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
