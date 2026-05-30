import os
import argparse
import torch

from config import (
    LOW_DRIFT_PATH, INTERP_PATH,
    LATENT_DIM, ENCODER_HIDDEN, DECODER_HIDDEN, ODE_HIDDEN,
)
from utils.split import build_splits
from utils.datasets import ArgoProfileDataset, ArgoLatentDataset, ArgoProbeDataset
from models.autoencoder import Autoencoder
from models.ode import ODEFunc
from models.gru import GRUDynamics
from train.train_encoder import train_encoder
from train.train_node import train_ode
from train.train_probe import train_probe
from train.train_probe_static import train_probe_static
from train.train_probe_raw import train_probe_raw
from train.train_probe_baseline import train_probe_baseline
from train.train_gru import train_gru
from train.train_gru_probe import train_gru_probe
from train.train_joint import train_joint
from extrapolation import run_extrapolation
from train.train_finetune import train_finetune
from train.train_node_curriculum import train_ode_curriculum
from utils.seeding import set_seed

set_seed()


def stage_split():
    print("=== Stage: split ===")
    df, split_map = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    print("Split complete.")
    return df, split_map


def stage_encoder(results_dir):
    print("=== Stage: encoder ===")
    return train_encoder(results_dir=results_dir)


def stage_encode(results_dir,
                 checkpoint_path=None,
                 latent_path=None):
    print("=== Stage: encode ===")
    checkpoint_path = checkpoint_path or f"{results_dir}/autoencoder_best.pt"
    latent_path     = latent_path     or f"{results_dir}/latent_cycles.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _  = build_splits(LOW_DRIFT_PATH, INTERP_PATH)

    train_ds = ArgoProfileDataset(df, split="train")
    val_ds   = ArgoProfileDataset(df, split="test",  stats=train_ds.stats)
    probe_ds = ArgoProfileDataset(df, split="probe", stats=train_ds.stats)

    model, _ = Autoencoder.load(checkpoint_path, device=device)

    all_wmo_ids = df["WMO_ID"].unique()
    wmo_to_idx  = {wmo: i for i, wmo in enumerate(sorted(all_wmo_ids))}

    latent_train = ArgoLatentDataset.from_encoder(train_ds, model.encoder, device, wmo_to_idx)
    latent_val   = ArgoLatentDataset.from_encoder(val_ds,   model.encoder, device, wmo_to_idx)
    latent_probe = ArgoLatentDataset.from_encoder(probe_ds, model.encoder, device, wmo_to_idx)

    print(f"Latent train: {len(latent_train)} casts")
    print(f"Latent val:   {len(latent_val)} casts")
    print(f"Latent probe: {len(latent_probe)} casts")

    torch.save({
        "train":      latent_train.records,
        "val":        latent_val.records,
        "probe":      latent_probe.records,
        "wmo_to_idx": wmo_to_idx,
    }, latent_path)
    print(f"Saved latent cycles to {latent_path}")

    return latent_train, latent_val, latent_probe, wmo_to_idx


def stage_ode(results_dir, latent_path=None):
    print("=== Stage: ode ===")
    latent_path = latent_path or f"{results_dir}/latent_cycles.pt"
    return train_ode(latent_path=latent_path, results_dir=results_dir)


def stage_ode_curriculum(results_dir, latent_path=None):
    print("=== Stage: ode_curriculum ===")
    latent_path = latent_path or f"{results_dir}/latent_cycles.pt"
    return train_ode_curriculum(latent_path=latent_path, results_dir=results_dir)


def stage_joint(results_dir):
    # joint AE + NODE training — trajectory smoothness prior backprops into encoder
    print("=== Stage: joint ===")
    return train_joint(results_dir=results_dir)


def _load_probe_dataset(checkpoint_path):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _    = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    train_ds = ArgoProfileDataset(df, split="train")
    probe_ds = ArgoProbeDataset(df, split="probe", stats=train_ds.stats)
    model, _ = Autoencoder.load(checkpoint_path, device=device)
    print(f"Probe casts: {len(probe_ds)}")
    return probe_ds, model.encoder, device


def _load_probe_dataset_no_encoder():
    # for raw probe — no encoder needed
    df, _    = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    train_ds = ArgoProfileDataset(df, split="train")
    probe_ds = ArgoProbeDataset(df, split="probe", stats=train_ds.stats)
    print(f"Probe casts: {len(probe_ds)}")
    return probe_ds


def stage_probe(results_dir, autoencoder_checkpoint=None, ode_checkpoint=None):
    print("=== Stage: probe ===")
    autoencoder_checkpoint = autoencoder_checkpoint or f"{results_dir}/autoencoder_best.pt"
    ode_checkpoint         = ode_checkpoint         or f"{results_dir}/ode_best.pt"

    probe_ds, encoder, device = _load_probe_dataset(autoencoder_checkpoint)

    ode_func = ODEFunc(latent_dim=LATENT_DIM, hidden=ODE_HIDDEN).to(device)
    ode_ckpt = torch.load(ode_checkpoint, map_location=device, weights_only=False)
    ode_func.load_state_dict(ode_ckpt["model_state"])

    return train_probe(probe_ds, encoder, ode_func, results_dir=results_dir)


def stage_probe_static(results_dir, autoencoder_checkpoint=None):
    # static latent probe — encoder output directly, no NODE
    print("=== Stage: probe_static ===")
    autoencoder_checkpoint = autoencoder_checkpoint or f"{results_dir}/autoencoder_best.pt"
    probe_ds, encoder, _ = _load_probe_dataset(autoencoder_checkpoint)
    return train_probe_static(probe_ds, encoder, results_dir=results_dir)


def stage_probe_raw(results_dir):
    # raw T/S/O probe — no encoder, no NODE
    print("=== Stage: probe_raw ===")
    probe_ds = _load_probe_dataset_no_encoder()
    return train_probe_raw(probe_ds, results_dir=results_dir)


def stage_probe_baseline(results_dir, autoencoder_checkpoint=None):
    print("=== Stage: probe_baseline ===")
    autoencoder_checkpoint = autoencoder_checkpoint or f"{results_dir}/autoencoder_best.pt"

    df, _    = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    train_ds = ArgoProfileDataset(df, split="train")
    probe_ds = ArgoProbeDataset(df, split="probe", stats=train_ds.stats)
    print(f"Probe casts: {len(probe_ds)}")
    return train_probe_baseline(probe_ds, results_dir=results_dir)


def stage_gru(results_dir, latent_path=None):
    print("=== Stage: gru ===")
    latent_path = latent_path or f"{results_dir}/latent_cycles.pt"
    return train_gru(latent_path=latent_path, results_dir=results_dir)


def stage_gru_probe(results_dir, autoencoder_checkpoint=None, gru_checkpoint=None):
    print("=== Stage: gru_probe ===")
    autoencoder_checkpoint = autoencoder_checkpoint or f"{results_dir}/autoencoder_best.pt"
    gru_checkpoint         = gru_checkpoint         or f"{results_dir}/gru_best.pt"

    probe_ds, encoder, device = _load_probe_dataset(autoencoder_checkpoint)

    gru      = GRUDynamics(latent_dim=LATENT_DIM, hidden=ODE_HIDDEN).to(device)
    gru_ckpt = torch.load(gru_checkpoint, map_location=device, weights_only=False)
    gru.load_state_dict(gru_ckpt["model_state"])

    return train_gru_probe(probe_ds, encoder, gru, results_dir=results_dir)


def stage_extrapolation(results_dir, latent_path=None):
    print("=== Stage: extrapolation ===")
    latent_path = latent_path or f"{results_dir}/latent_cycles.pt"
    return run_extrapolation(
        latent_path=latent_path,
        output_path=f"{results_dir}/extrapolation_results.csv",
    )


def stage_finetune(results_dir, autoencoder_checkpoint=None, ode_checkpoint=None, probe_checkpoint=None):
    print("=== Stage: finetune ===")
    autoencoder_checkpoint = autoencoder_checkpoint or f"{results_dir}/autoencoder_best.pt"
    ode_checkpoint         = ode_checkpoint         or f"{results_dir}/ode_best.pt"
    probe_checkpoint       = probe_checkpoint       or f"{results_dir}/probe_head_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _          = build_splits(LOW_DRIFT_PATH, INTERP_PATH)
    train_ds       = ArgoProfileDataset(df, split="train")
    train_probe_ds = ArgoProbeDataset(df, split="train", stats=train_ds.stats)
    probe_ds       = ArgoProbeDataset(df, split="probe", stats=train_ds.stats)
    combined       = torch.utils.data.ConcatDataset([train_probe_ds, probe_ds])
    model, _       = Autoencoder.load(autoencoder_checkpoint, device=device)

    return train_finetune(
        combined,
        autoencoder_checkpoint=autoencoder_checkpoint,
        ode_checkpoint=ode_checkpoint,
        probe_checkpoint=probe_checkpoint,
        results_dir=results_dir,
    )


STAGES = [
    "split", "encoder", "encode", "ode", "ode_curriculum",
    "joint",
    "probe", "probe_static", "probe_raw", "probe_baseline",
    "gru", "gru_probe", "extrapolation", "finetune", "all",
]


def main():
    parser = argparse.ArgumentParser(description="Ocean Dynamics Latent ODE pipeline")
    parser.add_argument("--stage",       type=str, choices=STAGES, default="all")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory for all outputs (checkpoints, CSVs, figures)")
    parser.add_argument("--checkpoint",       type=str, default=None)
    parser.add_argument("--ode_checkpoint",   type=str, default=None)
    parser.add_argument("--gru_checkpoint",   type=str, default=None)
    parser.add_argument("--latent",           type=str, default=None)
    parser.add_argument("--probe_checkpoint", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    rd = args.results_dir

    if args.stage == "split":
        stage_split()
    elif args.stage == "encoder":
        stage_encoder(rd)
    elif args.stage == "encode":
        stage_encode(rd, args.checkpoint, args.latent)
    elif args.stage == "ode":
        stage_ode(rd, args.latent)
    elif args.stage == "ode_curriculum":
        stage_ode_curriculum(rd, args.latent)
    elif args.stage == "joint":
        stage_joint(rd)
    elif args.stage == "probe":
        stage_probe(rd, args.checkpoint, args.ode_checkpoint)
    elif args.stage == "probe_static":
        stage_probe_static(rd, args.checkpoint)
    elif args.stage == "probe_raw":
        stage_probe_raw(rd)
    elif args.stage == "probe_baseline":
        stage_probe_baseline(rd, args.checkpoint)
    elif args.stage == "gru":
        stage_gru(rd, args.latent)
    elif args.stage == "gru_probe":
        stage_gru_probe(rd, args.checkpoint, args.gru_checkpoint)
    elif args.stage == "extrapolation":
        stage_extrapolation(rd, args.latent)
    elif args.stage == "finetune":
        stage_finetune(rd, args.checkpoint, args.ode_checkpoint, args.probe_checkpoint)
    elif args.stage == "all":
        stage_split()
        checkpoint_path = stage_encoder(rd)
        stage_encode(rd, checkpoint_path, args.latent)
        stage_ode_curriculum(rd, args.latent)
        stage_gru(rd, args.latent)
        stage_probe(rd, checkpoint_path, args.ode_checkpoint)
        stage_probe_static(rd, checkpoint_path)
        stage_probe_raw(rd)
        stage_probe_baseline(rd, checkpoint_path)
        stage_gru_probe(rd, checkpoint_path, args.gru_checkpoint)


if __name__ == "__main__":
    main()
