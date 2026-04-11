#!/usr/bin/env python3
"""Plot BER vs SNR across Watermark channels for BPSK and QPSK modulation.

Reproduces figures/ber_comparison.pdf from the E2E-WAVE paper.

This script only handles plotting. The raw BER numbers have to come from
ber_simo_combined.py / ber_kau1.py (which run OFDM replay over the
Watermark .mat channel files) and be consolidated into a CSV with columns:

    channel,modulation,fec,snr_db,ber

Example rows:

    NOF1,BPSK,none,0,0.213
    NOF1,BPSK,none,5,0.164
    NOF1,BPSK,ldpc_r33,0,0.095
    NOF1,QPSK,none,0,0.287
    BCH1,BPSK,none,0,0.401
    ...

Allowed `fec` values: none, ldpc_r73, ldpc_r33 (matches the paper's
uncoded / LDPC 0.73 / LDPC 0.33 curves).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHANNELS = ["NOF1", "NCS1", "BCH1", "KAU1"]
FEC_STYLE = {
    "none":     {"linestyle": "-",  "marker": "o"},
    "ldpc_r73": {"linestyle": "--", "marker": "s"},
    "ldpc_r33": {"linestyle": ":",  "marker": "^"},
}
MOD_COLOR = {"BPSK": "#1f77b4", "QPSK": "#d62728"}
FEC_LABEL = {"none": "uncoded", "ldpc_r73": "LDPC r=0.73", "ldpc_r33": "LDPC r=0.33"}


def plot_channel(ax, df: pd.DataFrame, channel: str):
    sub = df[df["channel"] == channel]
    if sub.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
        ax.set_title(channel, fontweight="bold")
        return
    for (mod, fec), grp in sub.groupby(["modulation", "fec"]):
        grp = grp.sort_values("snr_db")
        style = FEC_STYLE.get(fec, {"linestyle": "-", "marker": "x"})
        ax.semilogy(
            grp["snr_db"], grp["ber"].clip(lower=1e-6),
            color=MOD_COLOR.get(mod, "black"),
            label=f"{mod} {FEC_LABEL.get(fec, fec)}",
            linewidth=1.5, markersize=5, **style,
        )
    ax.set_title(channel, fontweight="bold")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-4, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True,
                   help="CSV with columns channel,modulation,fec,snr_db,ber")
    p.add_argument("--out", type=Path, default=Path("figures/ber_comparison.pdf"))
    p.add_argument("--channels", nargs="+", default=None,
                   help="Subset/ordering of channels (default: whatever is in the CSV, "
                        "in the order NOF1, NCS1, BCH1, KAU1)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    required = {"channel", "modulation", "fec", "snr_db", "ber"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    channels = args.channels or [c for c in CHANNELS if c in df["channel"].unique().tolist()]
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        plot_channel(ax, df, ch)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
