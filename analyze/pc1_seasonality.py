"""
analyze/pc1_seasonality.py

Verify that latent PC1 phase-locks to calendar month across multiple floats.
Produces three figures:
  1. Individual float lines (PC1 vs month + PC1 vs time)
  2. Aggregate mean ± std by calendar month, faceted by ocean region
  3. Scatter of PC1 colored by month

Run from project root:
    python analyze/pc1_seasonality.py --latent data/processed/latent_cycles.pt
"""

import argparse
import random
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from sklearn.decomposition import PCA
import torch

sys.path.insert(0, ".")
from utils.split import assign_ocean_region

TIME_EPOCH = pd.Timestamp("2000-01-01")


def days_to_timestamp(days):
    return TIME_EPOCH + pd.Timedelta(days=float(days))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent",   default="data/processed/latent_cycles.pt")
    parser.add_argument("--n_floats", type=int, default=20)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out",      default="results/pc1_seasonality.png")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading latent cycles from {args.latent}")
    ckpt = torch.load(args.latent, map_location="cpu", weights_only=False)
    train_records = ckpt["train"]
    print(f"Train records: {len(train_records)}")

    # fit PCA on all training latents — consistent PC1 axis across all floats
    all_p = np.stack([r["p"] for r in train_records])
    pca   = PCA(n_components=3)
    pca.fit(all_p)
    print(f"Variance explained — PC1: {pca.explained_variance_ratio_[0]:.3f}  "
          f"PC2: {pca.explained_variance_ratio_[1]:.3f}  "
          f"PC3: {pca.explained_variance_ratio_[2]:.3f}")

    # group records by float, infer region from lat/lon
    float_records = defaultdict(list)
    for r in train_records:
        float_records[r["device_idx"]].append(r)

    # assign region per float using mean lat/lon
    float_region = {}
    for device_idx, records in float_records.items():
        lat = np.mean([r["lat"] for r in records])
        lon = np.mean([r["lon"] for r in records])
        float_region[device_idx] = assign_ocean_region(lat, lon)

    all_device_ids = list(float_records.keys())
    print(f"Total floats in train: {len(all_device_ids)}")
    print("Region counts:", pd.Series(float_region).value_counts().to_dict())

    eligible = [d for d in all_device_ids if len(float_records[d]) >= 6]
    print(f"Eligible floats (>=6 obs): {len(eligible)}")

    n = min(args.n_floats, len(eligible))
    selected = random.sample(eligible, n)
    colors = cm.tab20(np.linspace(0, 1, n))

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    # collect all obs across selected floats with region
    all_months = []
    all_pc1    = []
    all_device = []
    all_region = []

    for i, device_idx in enumerate(selected):
        records = sorted(float_records[device_idx], key=lambda r: r["t"])
        p       = np.stack([r["p"] for r in records])
        ts      = [days_to_timestamp(r["t"]) for r in records]
        pc1     = pca.transform(p)[:, 0]
        months  = [t.month for t in ts]
        region  = float_region[device_idx]
        all_months.extend(months)
        all_pc1.extend(pc1.tolist())
        all_device.extend([i] * len(pc1))
        all_region.extend([region] * len(pc1))

    all_months = np.array(all_months)
    all_pc1    = np.array(all_pc1)
    all_device = np.array(all_device)
    all_region = np.array(all_region)

    # --- figure 1: individual float lines ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    ax_month, ax_time = axes1

    for i, device_idx in enumerate(selected):
        records = sorted(float_records[device_idx], key=lambda r: r["t"])
        p       = np.stack([r["p"] for r in records])
        ts      = [days_to_timestamp(r["t"]) for r in records]
        months  = [t.month for t in ts]
        pc1     = pca.transform(p)[:, 0]
        years   = [t.year + (t.month - 1) / 12 for t in ts]

        ax_month.plot(months, pc1, "o-", color=colors[i], alpha=0.6, linewidth=1, markersize=3)
        ax_time.plot(years,  pc1, "o-", color=colors[i], alpha=0.6, linewidth=1, markersize=3)

    for ax, xlabel, title in [
        (ax_month, "Calendar Month", "PC1 vs Calendar Month — individual floats"),
        (ax_time,  "Year",           "PC1 vs Time — individual floats"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PC1")
        ax.set_title(title)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

    ax_month.set_xticks(range(1, 13))
    ax_month.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"])
    fig1.suptitle(f"Latent PC1 — {n} floats  |  PC1 {pca.explained_variance_ratio_[0]*100:.1f}% variance", fontsize=12)
    fig1.tight_layout()
    out1 = args.out.replace(".png", "_individual.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")

    # --- figure 2: aggregate by month, faceted by region ---
    regions     = sorted(set(all_region))
    n_regions   = len(regions)
    ncols       = min(3, n_regions)
    nrows       = (n_regions + ncols - 1) // ncols
    months_x    = np.arange(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, region in enumerate(regions):
        ax  = axes2[idx // ncols][idx % ncols]
        mask = all_region == region

        means = np.array([all_pc1[mask & (all_months == m)].mean()
                          if (mask & (all_months == m)).sum() > 0 else np.nan
                          for m in months_x])
        stds  = np.array([all_pc1[mask & (all_months == m)].std()
                          if (mask & (all_months == m)).sum() > 0 else np.nan
                          for m in months_x])
        ns    = np.array([(mask & (all_months == m)).sum() for m in months_x])

        ax.plot(months_x, means, "o-", color="steelblue", linewidth=2, markersize=5)
        ax.fill_between(months_x, means - stds, means + stds, alpha=0.2, color="steelblue")
        for m, mn, cnt in zip(months_x, means, ns):
            if cnt > 0:
                ax.annotate(f"n={cnt}", (m, mn), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=6, color="gray")
        ax.set_title(region)
        ax.set_xticks(months_x)
        ax.set_xticklabels(month_labels, fontsize=7)
        ax.set_ylabel("PC1")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for idx in range(n_regions, nrows * ncols):
        axes2[idx // ncols][idx % ncols].set_visible(False)

    fig2.suptitle(f"Aggregate PC1 by Month & Region — {n} floats\n"
                  f"PC1 {pca.explained_variance_ratio_[0]*100:.1f}% variance", fontsize=12)
    fig2.tight_layout()
    out2 = args.out.replace(".png", "_aggregate_by_region.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

    # --- figure 3: scatter colored by month ---
    fig3, ax_scatter = plt.subplots(figsize=(8, 5))
    sc = ax_scatter.scatter(all_device, all_pc1, c=all_months, cmap="hsv",
                            vmin=1, vmax=12, alpha=0.6, s=15)
    cbar = plt.colorbar(sc, ax=ax_scatter, ticks=range(1, 13))
    cbar.set_ticklabels(month_labels)
    cbar.set_label("Month")
    ax_scatter.set_xlabel("Float index (arbitrary order)")
    ax_scatter.set_ylabel("PC1")
    ax_scatter.set_title(f"PC1 colored by month — {n} floats\n"
                         "seasonal signal = same-colored points cluster vertically")
    ax_scatter.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax_scatter.grid(True, alpha=0.3)
    fig3.tight_layout()
    out3 = args.out.replace(".png", "_month_scatter.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()