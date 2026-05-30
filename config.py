import numpy as np

## Data paths ##
INTERP_PATH    = "data/processed/PFL_all_interp72.csv"
LOW_DRIFT_PATH = "data/processed/all_low_drift_target_devices.csv"

## Variables ##
INPUT_VARS       = ['Temperature', 'Salinity', 'Oxygen']
TARGET_VARS      = ['Chlorophyll']
ALL_VARS         = ['Temperature', 'Salinity', 'Oxygen', 'Nitrate', 'pH', 'Chlorophyll']
MIN_TARGET_PROBE = 4

## Preprocess ##
MAX_AVG_DRIFT_KM  = 50
MIN_CASTS         = 5
MIN_TARGET_CYCLES = 5

## Data ##
DEPTH_STRIDE = 1
DEVICE       = "cuda"
RESULTS_DIR  = "results/vanilla"

## Interpolation grid (73 levels, 0–2000m) ##
DEPTH_GRID = np.concatenate([
    np.arange(0,    200,  10),
    np.arange(200,  1000, 25),
    np.arange(1000, 2001, 50),
])

## Split ##
TRAIN_FRAC = 0.70
TEST_FRAC  = 0.20
PROBE_FRAC = 0.10
SEED       = 42

## Model hyperparameters ##
LATENT_DIM     = 32
ENCODER_HIDDEN = [128, 128]
DECODER_HIDDEN = [64, 64]
ODE_HIDDEN     = [128, 128, 128]

## Training ##
ENCODER_TYPE = "cnn"   # "mlp" or "cnn"
ENCODER_LR     = 1e-3
ENCODER_EPOCHS = 80
ODE_LR         = 1e-4
ODE_EPOCHS     = 80
BATCH_SIZE     = 32
PROBE_LR       = 5e-4
PROBE_EPOCHS   = 200
WINDOW_SIZE    = 25
STRIDE         = 2
ODE_METHOD          = "rk4"
CURRICULUM_WINDOWS  = [10]
CURRICULUM_WEIGHTS  = [1.0]

## Joint training ##
JOINT_LR      = 5e-4   # single optimizer over AE + ODE func
JOINT_EPOCHS  = 80
LAMBDA_RECON  = 1.0    # weight on T/S/O reconstruction loss
LAMBDA_TRAJ   = 0.1    # weight on latent trajectory consistency loss
LAMBDA_ODE = 0.5 # we don't use it, but its too much of a refactor to remove them imo
LAMBDA_OXY = 0.5