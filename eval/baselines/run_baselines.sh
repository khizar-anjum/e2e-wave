#!/bin/bash
# Run the baseline comparison sweeps (VQ-VAE + digital, SoftCast, H.265/MPEG4)
# across NOF1, NCS1, BCH1 for BPSK and QPSK, with and without LDPC FEC.
#
# Required env vars:
#   E2E_WAVE_UVE_DIR        Root of UVE-38K 10-second clips
#   E2E_WAVE_VQVAE_CKPT     Path to the VQ-VAE checkpoint (for the vqvae baseline only)
# Optional:
#   E2E_WAVE_RESULTS_DIR    Output root (default: results)
#   CHANNELS                Space-separated channels (default: NOF1 BCH1 NCS1)

set -euo pipefail

: "${E2E_WAVE_UVE_DIR:?set E2E_WAVE_UVE_DIR}"
: "${E2E_WAVE_VQVAE_CKPT:?set E2E_WAVE_VQVAE_CKPT}"

RESULTS_DIR="${E2E_WAVE_RESULTS_DIR:-results}"
CHANNELS="${CHANNELS:-NOF1 BCH1 NCS1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_vqvae() {
    local channel=$1 mod=$2 extra=${3:-}
    python "${SCRIPT_DIR}/eval_vqvae_uve.py" \
        --ckpt "${E2E_WAVE_VQVAE_CKPT}" \
        --uve-path "${E2E_WAVE_UVE_DIR}" \
        --channel-type uwa --channel "${channel}" \
        --modulation "${mod}" \
        --snr-min 0 --snr-max 30 --snr-step 5 \
        ${extra}
}

run_mpeg4() {
    local channel=$1 mod=$2 fps=$3 extra=${4:-}
    python "${SCRIPT_DIR}/eval_mpeg4_uve.py" \
        --uve-path "${E2E_WAVE_UVE_DIR}" \
        --channel-type uwa --channel "${channel}" \
        --modulation "${mod}" \
        --snr-min 0 --snr-max 30 --snr-step 5 \
        --fps "${fps}" \
        ${extra}
}

run_softcast() {
    local channel=$1
    python "${SCRIPT_DIR}/eval_softcast_uve.py" \
        --uve-path "${E2E_WAVE_UVE_DIR}" \
        --channel "${channel}" \
        --snr-min 0 --snr-max 30 --snr-step 5
}

for CH in ${CHANNELS}; do
    for MOD in BPSK QPSK; do
        # VQ-VAE + digital modulation (uncoded + both LDPC rates)
        run_vqvae "${CH}" "${MOD}"
        run_vqvae "${CH}" "${MOD}" "--fec dvbs2_ldpc --fec-rate 0.73"
        run_vqvae "${CH}" "${MOD}" "--fec dvbs2_ldpc --fec-rate 0.33"

        # H.265 at the three FPS tiers (matching FEC overhead)
        run_mpeg4  "${CH}" "${MOD}" 16
        run_mpeg4  "${CH}" "${MOD}" 11 "--fec dvbs2_ldpc --fec-rate 0.73"
        run_mpeg4  "${CH}" "${MOD}"  5 "--fec dvbs2_ldpc --fec-rate 0.33"
    done

    # SoftCast is analog — one run per channel
    run_softcast "${CH}"
done

echo "Done. Collect runs with: python figures/consolidate_runs.py --input-dir ${RESULTS_DIR} --output-dir ${RESULTS_DIR}/plots"
