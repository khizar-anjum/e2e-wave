#!/usr/bin/env python3
"""Plot average L2 distance between transmitted and received token embeddings vs. SNR.

Reproduces figures/l2_comparison_raw.pdf from the E2E-WAVE paper using the
{CHANNEL}_waveform_len_{L}_l2_relevance.csv files produced by
eval/e2e_wave/eval_wave_bank_watermark_videogpt_full_random_tokens.py.

CSV schema (per row): snr_db, n, l2_mean, l2_std, l2_ci95, l2_norm_mean,
l2_norm_std, l2_norm_ci95, token_acc_mean, token_acc_std, token_acc_ci95.

Digital baselines (BPSK / QPSK +/- LDPC) are not included: they need to be
produced separately by running eval/baselines/ scripts and dropped into
results/baselines/ in a matching l2_relevance.csv format, then merged in
here with --baseline-dir.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHANNELS = ["NOF1", "NCS1", "BCH1"]

# Map waveform length -> (modulation, equiv rate, human label). Lifted from
# consolidate_runs.WAVEBANK_LENGTH_CONFIG so the two stay consistent.
LEN_TO_CFG = {
    9:  ("BPSK", "none", "Ours L=9 (BPSK 16 fps)"),
    10: ("BPSK", "none", "Ours L=10 (BPSK uncoded-equiv)"),
    13: ("BPSK", "r73",  "Ours L=13 (BPSK LDPC r=0.73-equiv)"),
    30: ("BPSK", "r33",  "Ours L=30 (BPSK LDPC r=0.33-equiv)"),
    5:  ("QPSK", "none", "Ours L=5 (QPSK uncoded-equiv)"),
    7:  ("QPSK", "r73",  "Ours L=7 (QPSK LDPC r=0.73-equiv)"),
    15: ("QPSK", "r33",  "Ours L=15 (QPSK LDPC r=0.33-equiv)"),
}

FILENAME_RE = re.compile(r"^([A-Z0-9]+)_waveform_len_(\d+)_l2_relevance\.csv$")


def load_l2_runs(csv_dir: Path) -> List[dict]:
    runs = []
    for csv in sorted(csv_dir.glob("*_waveform_len_*_l2_relevance.csv")):
        m = FILENAME_RE.match(csv.name)
        if not m:
            continue
        channel = m.group(1)
        wlen = int(m.group(2))
        mod, equiv, label = LEN_TO_CFG.get(wlen, ("?", "?", f"Ours L={wlen}"))
        df = pd.read_csv(csv)
        runs.append({
            "channel": channel,
            "waveform_len": wlen,
            "modulation": mod,
            "equiv_rate": equiv,
            "label": label,
            "df": df,
        })
    return runs


def plot_panel(ax, runs: List[dict], channel: str, metric: str, ylabel: str):
    subset = [r for r in runs if r["channel"] == channel]
    # Sort by (modulation, waveform_len) for stable legend order.
    subset.sort(key=lambda r: (r["modulation"], r["waveform_len"]))
    for r in subset:
        df = r["df"].sort_values("snr_db")
        snr = df["snr_db"].to_numpy()
        y = df[f"{metric}_mean"].to_numpy()
        err = df[f"{metric}_ci95"].to_numpy() if f"{metric}_ci95" in df.columns else None
        ls = "-" if r["modulation"] == "BPSK" else "--"
        ax.errorbar(snr, y, yerr=err, label=r["label"], linestyle=ls,
                    marker="o", markersize=4, capsize=2, linewidth=1.4)
    ax.set_title(channel, fontweight="bold")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-dir", type=Path, default=Path("results/wavebank"),
                   help="Directory containing *_l2_relevance.csv files")
    p.add_argument("--out", type=Path, default=Path("figures/l2_comparison_raw.pdf"))
    p.add_argument("--metric", choices=["l2", "l2_norm"], default="l2",
                   help="Raw L2 distance or normalized L2 (per-embedding norm)")
    args = p.parse_args()

    runs = load_l2_runs(args.csv_dir)
    if not runs:
        raise SystemExit(f"No *_l2_relevance.csv files found under {args.csv_dir}")

    channels_present = [c for c in CHANNELS if any(r["channel"] == c for r in runs)]
    fig, axes = plt.subplots(1, len(channels_present),
                             figsize=(4.5 * len(channels_present), 4.2),
                             sharey=True)
    if len(channels_present) == 1:
        axes = [axes]
    ylabel = "Avg. L2 distance (raw)" if args.metric == "l2" else "Avg. normalized L2 distance"
    for ax, ch in zip(axes, channels_present):
        plot_panel(ax, runs, ch, args.metric, ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
