# Example environment for running the E2E-WAVE pipeline end-to-end.
#
# Copy this file to env.sh (which is gitignored) and edit the paths to match
# your machine:
#
#     cp scripts/env.example.sh env.sh
#     $EDITOR env.sh
#     source env.sh
#
# All Python and shell scripts in this repo read these variables via
# os.environ.get(..., <sensible default>), so unset variables fall back to
# paths relative to the repo root (which usually won't exist — set them).

# --- external datasets (see DATA.md for download links) --------------------

# UVE-38K 10-second underwater video clips, expected to contain clear/ and
# turbid/ subdirectories.
export E2E_WAVE_UVE_DIR="$HOME/datasets/uve38k/10_sec_clips"

# Watermark benchmark channel recordings. Expected layout:
#   $E2E_WAVE_CHANNELS_DIR/NOF1/mat/NOF1_001.mat
#   $E2E_WAVE_CHANNELS_DIR/NCS1/mat/...
#   $E2E_WAVE_CHANNELS_DIR/BCH1/mat/...
#   $E2E_WAVE_CHANNELS_DIR/KAU1/mat/...
export E2E_WAVE_CHANNELS_DIR="$HOME/datasets/watermark/channels"

# Root of a corpus used for training a new VQ-VAE or wavebank (UVE-38K train
# split works).
export E2E_WAVE_VIDEO_TRAIN_DIR="$HOME/datasets/uve38k/train"

# --- checkpoints (shipped in this repo) ------------------------------------

# Repo-relative paths — these match the files committed under checkpoints/.
# Override only if you re-trained on your own.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export E2E_WAVE_VQVAE_CKPT="$REPO_ROOT/checkpoints/vqvae/vqvae_41616_model_best_128x128.pth_1024.tar"
export E2E_WAVE_TRAINED_BANKS_DIR="$REPO_ROOT/checkpoints/trained_banks"

# --- outputs ---------------------------------------------------------------

# Where new evaluation CSVs land (the shipped ones are under results/wavebank).
export E2E_WAVE_RESULTS_DIR="$REPO_ROOT/results/wavebank"

# Where training runs / tensorboard logs go.
export E2E_WAVE_RUNS_DIR="$REPO_ROOT/runs/watermark_videogpt"

# --- PYTHONPATH ------------------------------------------------------------
# All package imports (videogpt, python_replicate, pairwise_lambda, softcast)
# live under src/ and eval/baselines/.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/eval/baselines:${PYTHONPATH:-}"
