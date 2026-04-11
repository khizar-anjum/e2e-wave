#!/bin/bash
# Summarize the raw CSVs produced by run_command.sh into *_summary.csv files
# that plot_psnr_ssim.py / consolidate_runs.py can consume.
#
# Optional env vars:
#   E2E_WAVE_RESULTS_DIR   Where the raw CSVs live (default: results/wavebank)
#   WAVEFORM_LENS          Space-separated (default: 5 7 9 10 13 15 30)
#   CHANNELS               Space-separated (default: NOF1 BCH1 NCS1)

set -euo pipefail

RESULTS_DIR="${E2E_WAVE_RESULTS_DIR:-results/wavebank}"
WAVEFORM_LENS="${WAVEFORM_LENS:-5 7 9 10 13 15 30}"
CHANNELS="${CHANNELS:-NOF1 BCH1 NCS1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESS_PY="${SCRIPT_DIR}/process_eval_results.py"

for CHANNEL in ${CHANNELS}; do
    for WAVEFORM_LEN in ${WAVEFORM_LENS}; do
        OUTPUT_CSV="${RESULTS_DIR}/${CHANNEL}_waveform_len_${WAVEFORM_LEN}.csv"
        if [ ! -f "${OUTPUT_CSV}" ]; then
            echo "skip: ${OUTPUT_CSV} (not found)"
            continue
        fi
        echo "processing ${OUTPUT_CSV}"
        python "${PROCESS_PY}" --input_csv "${OUTPUT_CSV}" --save_plots
    done
done
