#!/bin/bash
# Run the L2 token-relevance sweep (used by plot_l2_comparison.py).
#
# Required env vars (same set as run_command.sh).

set -euo pipefail

: "${E2E_WAVE_CHANNELS_DIR:?set E2E_WAVE_CHANNELS_DIR to the Watermark channels dir}"
: "${E2E_WAVE_VQVAE_CKPT:?set E2E_WAVE_VQVAE_CKPT to the VQ-VAE checkpoint}"
: "${E2E_WAVE_TRAINED_BANKS_DIR:?set E2E_WAVE_TRAINED_BANKS_DIR to the trained banks dir}"

RESULTS_DIR="${E2E_WAVE_RESULTS_DIR:-results/wavebank}"
WAVEFORM_LENS="${WAVEFORM_LENS:-5 7 9 10 13 15 30}"
CHANNELS="${CHANNELS:-NOF1 BCH1 NCS1}"
SNR_MIN="${SNR_MIN:-0}"
SNR_MAX="${SNR_MAX:-30}"
SNR_STEP="${SNR_STEP:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="${SCRIPT_DIR}/eval_wave_bank_watermark_videogpt_full_random_tokens.py"

mkdir -p "${RESULTS_DIR}"

for CHANNEL in ${CHANNELS}; do
    for WAVEFORM_LEN in ${WAVEFORM_LENS}; do
        echo "=========================================="
        echo "L2 relevance eval: channel=${CHANNEL} waveform_len=${WAVEFORM_LEN}"
        echo "=========================================="

        BANK_CKPT="${E2E_WAVE_TRAINED_BANKS_DIR}/cross-entropy_waveform_len_${WAVEFORM_LEN}_videogpt_4_16_16_1024_train_NCS1_eval_NCS1_temperature_0.01_top_5_video_training_res_128/best_ssim_bank.pth"
        if [ ! -f "${BANK_CKPT}" ]; then
            echo "WARNING: missing bank checkpoint ${BANK_CKPT}, skipping."
            continue
        fi

        OUTPUT_CSV="${RESULTS_DIR}/${CHANNEL}_waveform_len_${WAVEFORM_LEN}_l2_relevance.csv"

        python "${EVAL_PY}" \
            --vqvae_ckpt "${E2E_WAVE_VQVAE_CKPT}" \
            --bank_ckpt "${BANK_CKPT}" \
            --channel "${CHANNEL}" \
            --waveform_len "${WAVEFORM_LEN}" \
            --snr_min "${SNR_MIN}" --snr_max "${SNR_MAX}" --snr_step "${SNR_STEP}" \
            --num_clips 500 \
            --batch_size 4 \
            --output_csv "${OUTPUT_CSV}"
    done
done
