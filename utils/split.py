import numpy as np
import pandas as pd

from config import (
    LOW_DRIFT_PATH, INTERP_PATH,
    TRAIN_FRAC, TEST_FRAC, PROBE_FRAC, SEED,
    TARGET_VARS, MIN_TARGET_PROBE,
)


## Region assignment ##

def assign_ocean_region(lat, lon):
    if lat > 50:
        return "subpolar"
    elif lat > 35:
        return "northwest_atlantic" if lon < -40 else "northeast_atlantic"
    elif lat > 25:
        return "subtropical_west" if lon < -40 else "subtropical_east"
    else:
        return "tropics"


## Float-level metadata ##

def get_float_level_metadata(low_drift_df, pfl_df):
    float_meta = low_drift_df[["WMO_ID", "start_lat", "start_lon"]].copy()
    float_meta["region"] = float_meta.apply(
        lambda r: assign_ocean_region(r["start_lat"], r["start_lon"]), axis=1
    )

    target_cols = [v for v in TARGET_VARS if v in pfl_df.columns]
    target_wmo_ids = set(
        pfl_df[pfl_df[target_cols].notna().any(axis=1)]["WMO_ID"].unique()
    )
    float_meta["has_target"] = float_meta["WMO_ID"].isin(target_wmo_ids)

    return float_meta


## Split logic ##

def stratified_float_split(float_meta, train_frac=TRAIN_FRAC, test_frac=TEST_FRAC,
                           probe_frac=PROBE_FRAC, seed=SEED):
    assert abs(train_frac + test_frac + probe_frac - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    split_map = {"train": [], "test": [], "probe": []}

    target_floats = float_meta[float_meta["has_target"]].copy()

    reserved_probe = []
    if len(target_floats) >= MIN_TARGET_PROBE:
        per_region = max(1, MIN_TARGET_PROBE // target_floats["region"].nunique())
        for _, group in target_floats.groupby("region"):
            wmo_ids = group["WMO_ID"].values.copy()
            rng.shuffle(wmo_ids)
            take = min(per_region, len(wmo_ids))
            reserved_probe.extend(wmo_ids[:take])
            if len(reserved_probe) >= MIN_TARGET_PROBE:
                break
        reserved_probe = reserved_probe[:MIN_TARGET_PROBE]
    else:
        print(f"  Warning: only {len(target_floats)} target floats — using all for probe")
        reserved_probe = target_floats["WMO_ID"].tolist()

    split_map["probe"].extend(reserved_probe)
    reserved_set = set(reserved_probe)

    remaining = float_meta[~float_meta["WMO_ID"].isin(reserved_set)].copy()

    for region, group in remaining.groupby("region"):
        wmo_ids = group["WMO_ID"].values.copy()
        rng.shuffle(wmo_ids)
        n = len(wmo_ids)

        if n < 5:
            print(f"  Region '{region}' has only {n} remaining floats — assigning all to train")
            split_map["train"].extend(wmo_ids)
            continue

        i1 = max(1, int(n * train_frac))
        i2 = max(i1 + 1, int(n * (train_frac + test_frac)))
        i2 = min(i2, n - 1)

        split_map["train"].extend(wmo_ids[:i1])
        split_map["test"].extend(wmo_ids[i1:i2])
        split_map["probe"].extend(wmo_ids[i2:])

    return {k: np.array(v) for k, v in split_map.items()}


## Utilities ##

def assign_split(df, split_map):
    wmo_to_split = {wmo: split for split, wmos in split_map.items() for wmo in wmos}
    df = df.copy()
    df["split"] = df["WMO_ID"].map(wmo_to_split)
    return df


def verify_split(pfl_filtered, float_meta, split_map):
    for split_name, wmo_ids in split_map.items():
        other = np.concatenate([v for k, v in split_map.items() if k != split_name])
        assert len(set(wmo_ids) & set(other)) == 0, f"Overlap detected in {split_name}"

    print("\nFloat counts by split:")
    for split_name, wmo_ids in split_map.items():
        target_count = float_meta[
            float_meta["WMO_ID"].isin(wmo_ids) & float_meta["has_target"]
        ].shape[0]
        print(f"  {split_name:6s}: {len(wmo_ids):3d} floats  ({target_count} with target)")

    print("\nRow counts by split:")
    print(pfl_filtered["split"].value_counts())

    print("\nFloat counts by region and split:")
    fm = float_meta.copy()
    fm["split"] = fm["WMO_ID"].map(
        {wmo: split for split, wmos in split_map.items() for wmo in wmos}
    )
    print(fm.groupby(["region", "split"]).size().unstack(fill_value=0))


## Entry point ##

def build_splits(low_drift_path=LOW_DRIFT_PATH, interp_path=INTERP_PATH):
    low_drift_df = pd.read_csv(low_drift_path)
    pfl_df       = pd.read_csv(interp_path)

    low_drift_wmo_ids = low_drift_df["WMO_ID"].unique()
    pfl_filtered = pfl_df[pfl_df["WMO_ID"].isin(low_drift_wmo_ids)].copy()

    print(f"Low drift floats:    {len(low_drift_wmo_ids)}")
    print(f"Depth observations:  {len(pfl_filtered):,}")

    float_meta = get_float_level_metadata(low_drift_df, pfl_filtered)

    print("\nRegion distribution:")
    print(float_meta["region"].value_counts())
    print(f"\nTarget floats: {float_meta['has_target'].sum()} / {len(float_meta)}")

    split_map = stratified_float_split(float_meta)

    pfl_filtered = assign_split(pfl_filtered, split_map)
    verify_split(pfl_filtered, float_meta, split_map)

    return pfl_filtered, split_map