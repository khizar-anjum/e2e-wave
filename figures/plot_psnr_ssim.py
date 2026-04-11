#!/usr/bin/env python3
"""
Generate comparison plots for UVE evaluation results.

Creates PDF plots comparing PSNR and SSIM across methods (VQ-VAE, MPEG4, SoftCast, Wavebank)
with 95% confidence intervals, separated by:
- Category: clear, turbid
- Modulation: BPSK, QPSK
- Channel: NOF1, BCH1, NCS1 (as subplots in each figure)
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Plot configuration
CHANNELS = ["NOF1", "BCH1", "NCS1"]
CATEGORIES = ["clear", "turbid"]
MODULATIONS = ["BPSK", "QPSK"]
METRICS = ["psnr", "ssim"]

# Color scheme - one primary color per method family
METHOD_COLORS = {
    "vqvae": {"base": "#1f77b4", "shades": ["#1f77b4", "#4a9fd4", "#7ec8f3"]},  # Blues
    "mpeg4": {"base": "#d62728", "shades": ["#d62728", "#e85a5a", "#f58d8d"]},  # Reds
    "softcast": {"base": "#2ca02c", "shades": ["#2ca02c"]},  # Green
    "wavebank": {"base": "#9467bd", "shades": ["#9467bd", "#b794d4", "#d9c1eb"]},  # Purples
}

# Marker styles by FEC rate
FEC_MARKERS = {
    "none": "o",
    "r33": "s",
    "r73": "^",
}

# Line styles
LINE_STYLES = {
    "vqvae": "-",
    "mpeg4": "--",
    "softcast": "-.",
    "wavebank": ":",
}


def load_manifest(plots_dir: Path) -> dict:
    """Load the manifest.json file."""
    manifest_path = plots_dir / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def load_digital_method_data(run_path: Path, category: str) -> Dict[float, dict]:
    """
    Load data from VQ-VAE, MPEG4, or SoftCast sweep_summary.json.
    Returns dict mapping SNR -> {mean, std, n} for specified category.
    """
    summary_path = run_path / "sweep_summary.json"
    if not summary_path.exists():
        return {}

    with open(summary_path) as f:
        data = json.load(f)

    results = {}
    category_data = data.get("category_results_by_snr", {})

    for snr_str, cats in category_data.items():
        snr = float(snr_str)
        if category in cats:
            cat_data = cats[category]
            results[snr] = {
                "psnr_mean": cat_data.get("psnr_mean", 0),
                "psnr_std": cat_data.get("psnr_std", 0),
                "ssim_mean": cat_data.get("ssim_mean", 0),
                "ssim_std": cat_data.get("ssim_std", 0),
                "n": cat_data.get("num_videos", 1),
            }

    return results


def load_wavebank_data(run_path: Path, category: str) -> Dict[float, dict]:
    """
    Load data from wavebank summary CSV.
    Returns dict mapping SNR -> {mean, ci95} for specified category.
    """
    # Find the summary CSV
    csv_files = list(run_path.glob("*_summary.csv"))
    if not csv_files:
        return {}

    df = pd.read_csv(csv_files[0])
    df = df[df["group"] == category]

    results = {}
    for _, row in df.iterrows():
        snr = float(row["snr_db"])
        results[snr] = {
            "psnr_mean": row["psnr_mean"],
            "psnr_ci95": row["psnr_ci95"],
            "ssim_mean": row["ssim_mean"],
            "ssim_ci95": row["ssim_ci95"],
            "n": row.get("n", 1),
        }

    return results


def get_curve_data(
    run_entry: dict,
    plots_dir: Path,
    category: str,
    metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract curve data for plotting.
    Returns (snr_values, means, ci95_values).
    """
    method = run_entry["method"]
    run_path = plots_dir / run_entry["path"]

    if method == "wavebank":
        data = load_wavebank_data(run_path, category)
    else:
        data = load_digital_method_data(run_path, category)

    if not data:
        return np.array([]), np.array([]), np.array([])

    snrs = sorted(data.keys())
    means = []
    ci95s = []

    for snr in snrs:
        d = data[snr]
        mean_key = f"{metric}_mean"
        means.append(d.get(mean_key, 0))

        if method == "wavebank":
            ci95_key = f"{metric}_ci95"
            ci95s.append(d.get(ci95_key, 0))
        else:
            # Calculate 95% CI from std: CI = 1.96 * std / sqrt(n)
            std_key = f"{metric}_std"
            std = d.get(std_key, 0)
            n = d.get("n", 1)
            ci95 = 1.96 * std / np.sqrt(n) if n > 0 else 0
            ci95s.append(ci95)

    return np.array(snrs), np.array(means), np.array(ci95s)


def get_curve_label(run_entry: dict) -> str:
    """Generate legend label for a curve."""
    method = run_entry["method"]

    if method == "vqvae":
        fec = run_entry.get("fec", "none")
        rate_str = run_entry.get("fec_rate_str", "")
        if fec == "none":
            return "VQVAE (none)"
        elif rate_str == "r33":
            return "VQVAE (LDPC 0.33)"
        elif rate_str == "r73":
            return "VQVAE (LDPC 0.73)"
        return f"VQVAE ({rate_str})"

    elif method == "mpeg4":
        fec = run_entry.get("fec", "none")
        rate_str = run_entry.get("fec_rate_str", "")
        if fec == "none":
            return "H.265 (none)"
        elif rate_str == "r33":
            return "H.265 (LDPC 0.33)"
        elif rate_str == "r73":
            return "H.265 (LDPC 0.73)"
        return f"H.265 ({rate_str})"

    elif method == "softcast":
        return "SoftCast (perfect metadata)"

    elif method == "wavebank":
        wlen = run_entry.get("waveform_len", "?")
        return f"Ours (len {wlen})"

    return method


def get_curve_style(run_entry: dict, method_idx: Dict[str, int]) -> dict:
    """Get color, marker, and line style for a curve."""
    method = run_entry["method"]

    # Get FEC rate for marker selection
    if method == "wavebank":
        equiv_rate = run_entry.get("equiv_rate", "none")
        fec_key = equiv_rate if equiv_rate in FEC_MARKERS else "none"
    else:
        fec = run_entry.get("fec", "none")
        rate_str = run_entry.get("fec_rate_str", "")
        fec_key = rate_str if rate_str in FEC_MARKERS else ("none" if fec == "none" else "r73")

    # Get color shade based on index within method
    colors = METHOD_COLORS.get(method, {"shades": ["#333333"]})["shades"]
    idx = method_idx.get(method, 0)
    color = colors[idx % len(colors)]
    method_idx[method] = idx + 1

    return {
        "color": color,
        "marker": FEC_MARKERS.get(fec_key, "o"),
        "linestyle": LINE_STYLES.get(method, "-"),
        "markersize": 6,
        "linewidth": 1.5,
    }


def filter_runs_for_plot(
    manifest: dict,
    modulation: str,
    channel: str
) -> List[dict]:
    """Filter runs for a specific modulation and channel."""
    filtered = []

    for run in manifest["runs"]:
        run_channel = run.get("channel", "")
        run_mod = run.get("modulation", "")
        method = run.get("method", "")

        if run_channel != channel:
            continue

        # SoftCast is analog - include on all plots
        if method == "softcast":
            filtered.append(run)
            continue

        # Match modulation
        if run_mod and run_mod.upper() == modulation.upper():
            filtered.append(run)

    # Sort by method and FEC rate for consistent ordering
    def sort_key(r):
        method = r.get("method", "")
        fec_rate = r.get("fec_rate") or 0
        wlen = r.get("waveform_len") or 0
        return (method, fec_rate, wlen)

    return sorted(filtered, key=sort_key)


def create_comparison_plot(
    manifest: dict,
    plots_dir: Path,
    metric: str,
    category: str,
    modulation: str,
    output_path: Path
):
    """Create a single comparison plot (trio of 3 channel subplots)."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    # Track all legend handles/labels
    all_handles = []
    all_labels = []
    seen_labels = set()

    for ax_idx, channel in enumerate(CHANNELS):
        ax = axes[ax_idx]
        ax.set_title(f"{channel}", fontsize=12, fontweight="bold")
        ax.set_xlabel("SNR (dB)", fontsize=10)
        if ax_idx == 0:
            ylabel = "PSNR (dB)" if metric == "psnr" else "SSIM"
            ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Get runs for this channel/modulation
        runs = filter_runs_for_plot(manifest, modulation, channel)

        if not runs:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                   ha="center", va="center", fontsize=12, color="gray")
            continue

        # Track method index for color shading
        method_idx = {}

        for run in runs:
            snrs, means, ci95s = get_curve_data(run, plots_dir, category, metric)

            if len(snrs) == 0:
                continue

            # Skip if all zeros (failed runs)
            if np.all(means == 0):
                continue

            style = get_curve_style(run, method_idx)
            label = get_curve_label(run)

            # Plot line with markers and error bars
            line = ax.errorbar(snrs, means, yerr=ci95s, label=label,
                              capsize=3, capthick=1, elinewidth=1,
                              **style)

            # Track for shared legend
            if label not in seen_labels:
                all_handles.append(line)
                all_labels.append(label)
                seen_labels.add(label)

        ax.set_xlim(-2, 32)

    # Add shared legend below the plots
    if all_handles:
        # Sort legend by method type
        sorted_items = sorted(zip(all_labels, all_handles),
                             key=lambda x: (
                                 0 if "VQ-VAE" in x[0] else
                                 1 if "MPEG4" in x[0] else
                                 2 if "SoftCast" in x[0] else 3,
                                 x[0]
                             ))
        all_labels, all_handles = zip(*sorted_items)

        fig.legend(all_handles, all_labels,
                  loc="lower center",
                  bbox_to_anchor=(0.5, -0.02),
                  ncol=min(len(all_handles), 8),
                  fontsize=9,
                  frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate UVE comparison plots")
    parser.add_argument("--plots-dir", type=Path, default=Path("results/plots"),
                       help="Path to consolidated plots directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots/figures"),
                       help="Output directory for PDF figures")
    parser.add_argument("--metric", choices=["psnr", "ssim", "all"], default="all",
                       help="Which metric to plot")
    parser.add_argument("--category", choices=["clear", "turbid", "all"], default="all",
                       help="Which category to plot")
    parser.add_argument("--modulation", choices=["BPSK", "QPSK", "all"], default="all",
                       help="Which modulation to plot")

    args = parser.parse_args()

    # Load manifest
    manifest = load_manifest(args.plots_dir)
    print(f"Loaded manifest with {len(manifest['runs'])} runs")

    # Determine what to plot
    metrics = METRICS if args.metric == "all" else [args.metric]
    categories = CATEGORIES if args.category == "all" else [args.category]
    modulations = MODULATIONS if args.modulation == "all" else [args.modulation]

    # Generate plots
    for metric in metrics:
        for category in categories:
            for modulation in modulations:
                output_path = args.output_dir / f"{metric}_{category}_{modulation.lower()}.pdf"
                create_comparison_plot(
                    manifest=manifest,
                    plots_dir=args.plots_dir,
                    metric=metric,
                    category=category,
                    modulation=modulation,
                    output_path=output_path
                )

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
