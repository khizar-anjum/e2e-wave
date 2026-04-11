#!/bin/bash
# Run E2E-WAVE PSNR/SSIM evaluation across channels and waveform lengths.
#
# Required env vars (see scripts/env.example.sh):
#   E2E_WAVE_UVE_DIR          Root of UVE-38K 10-second clips (with clear/ and turbid/)
#   E2E_WAVE_CHANNELS_DIR     Root of Watermark channels (NOF1/, NCS1/, BCH1/, ...)
#   E2E_WAVE_VQVAE_CKPT       Path to the VQ-VAE .pth.tar checkpoint
#   E2E_WAVE_TRAINED_BANKS_DIR  Dir with cross-entropy_waveform_len_{L}_... subdirs
#
# Optional env vars:
#   E2E_WAVE_RESULTS_DIR      Where to write CSV outputs (default: results/wavebank)
#   WAVEFORM_LENS             Space-separated waveform lengths (default: 5 7 9 10 13 15 30)
#   CHANNELS                  Space-separated channels (default: NOF1 BCH1 NCS1)

set -euo pipefail

: "${E2E_WAVE_UVE_DIR:?set E2E_WAVE_UVE_DIR to the UVE-38K root}"
: "${E2E_WAVE_CHANNELS_DIR:?set E2E_WAVE_CHANNELS_DIR to the Watermark channels dir}"
: "${E2E_WAVE_VQVAE_CKPT:?set E2E_WAVE_VQVAE_CKPT to the VQ-VAE checkpoint}"
: "${E2E_WAVE_TRAINED_BANKS_DIR:?set E2E_WAVE_TRAINED_BANKS_DIR to the trained banks dir}"

RESULTS_DIR="${E2E_WAVE_RESULTS_DIR:-results/wavebank}"
WAVEFORM_LENS="${WAVEFORM_LENS:-5 7 9 10 13 15 30}"
CHANNELS="${CHANNELS:-NOF1 BCH1 NCS1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/eval_wave_bank_watermark_videogpt_full.py"

mkdir -p "${RESULTS_DIR}"

for CHANNEL in ${CHANNELS}; do
    for WAVEFORM_LEN in ${WAVEFORM_LENS}; do
        echo "=========================================="
        echo "Eval: channel=${CHANNEL} waveform_len=${WAVEFORM_LEN}"
        echo "=========================================="

        BANK_CKPT="${E2E_WAVE_TRAINED_BANKS_DIR}/cross-entropy_waveform_len_${WAVEFORM_LEN}_videogpt_4_16_16_1024_train_NCS1_eval_NCS1_temperature_0.01_top_5_video_training_res_128/best_ssim_bank.pth"

        if [ ! -f "${BANK_CKPT}" ]; then
            echo "WARNING: missing bank checkpoint ${BANK_CKPT}, skipping."
            continue
        fi

        OUTPUT_CSV="${RESULTS_DIR}/${CHANNEL}_waveform_len_${WAVEFORM_LEN}.csv"
        OUTPUT_DIR="${RESULTS_DIR}/videos/${CHANNEL}_waveform_len_${WAVEFORM_LEN}_videos"
        mkdir -p "${OUTPUT_DIR}"

        python "${EVAL_PY}" \
            --video_dir "${E2E_WAVE_UVE_DIR}" \
            --vqvae_ckpt "${E2E_WAVE_VQVAE_CKPT}" \
            --bank_ckpt "${BANK_CKPT}" \
            --channel "${CHANNEL}" \
            --waveform_len "${WAVEFORM_LEN}" \
            --output_csv "${OUTPUT_CSV}" \
            --output_dir "${OUTPUT_DIR}" \
            --snr_min 0 --snr_max 30 --snr_step 5 \
            --batch_size 8 \
            --max_clips 4000
    done
done

echo "Done. CSVs under ${RESULTS_DIR}"
