#!/usr/bin/env python3
"""Summarize eval CSV by SNR and turbidity group, and plot results.

Input CSV is expected to have at least: snr_db, psnr, ssim, video_path.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


GroupStats = Dict[float, Dict[str, List[float]]]


def _detect_group(path: str, turbid_keys: List[str], clear_keys: List[str]) -> str:
    lower = path.lower()
    for key in turbid_keys:
        if key in lower:
            return "turbid"
    for key in clear_keys:
        if key in lower:
            return "clear"
    return "unknown"


def _accumulate(rows: List[dict], turbid_keys: List[str], clear_keys: List[str]) -> Dict[str, GroupStats]:
    grouped: Dict[str, GroupStats] = {"turbid": {}, "clear": {}, "unknown": {}}
    for row in rows:
        try:
            snr = float(row["snr_db"])
            psnr = float(row["psnr"])
            ssim = float(row["ssim"])
            path = row.get("video_path", "")
        except KeyError as exc:
            raise KeyError(f"Missing required column: {exc}")
        group = _detect_group(path, turbid_keys, clear_keys)
        bucket = grouped[group].setdefault(snr, {"psnr": [], "ssim": []})
        bucket["psnr"].append(psnr)
        bucket["ssim"].append(ssim)
    return grouped


def _summary_stats(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(arr.size) if arr.size > 0 else float("nan")
    return mean, std, ci95


def _write_summary(path: Path, grouped: Dict[str, GroupStats], include_unknown: bool) -> None:
    fields = [
        "group",
        "snr_db",
        "n",
        "psnr_mean",
        "psnr_std",
        "psnr_ci95",
        "ssim_mean",
        "ssim_std",
        "ssim_ci95",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group in ("turbid", "clear", "unknown"):
            if group == "unknown" and not include_unknown:
                continue
            for snr in sorted(grouped[group].keys()):
                psnr_vals = grouped[group][snr]["psnr"]
                ssim_vals = grouped[group][snr]["ssim"]
                psnr_mean, psnr_std, psnr_ci = _summary_stats(psnr_vals)
                ssim_mean, ssim_std, ssim_ci = _summary_stats(ssim_vals)
                writer.writerow(
                    {
                        "group": group,
                        "snr_db": snr,
                        "n": len(psnr_vals),
                        "psnr_mean": psnr_mean,
                        "psnr_std": psnr_std,
                        "psnr_ci95": psnr_ci,
                        "ssim_mean": ssim_mean,
                        "ssim_std": ssim_std,
                        "ssim_ci95": ssim_ci,
                    }
                )


def _plot_grouped(grouped: Dict[str, GroupStats], include_unknown: bool, save_path:str = None) -> None:
    groups = ["turbid", "clear"] + (["unknown"] if include_unknown else [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    ax_psnr, ax_ssim = axes
    for group in groups:
        snrs = sorted(grouped[group].keys())
        if not snrs:
            continue
        psnr_means = []
        psnr_ci = []
        ssim_means = []
        ssim_ci = []
        for snr in snrs:
            psnr_mean, _, psnr_ci_val = _summary_stats(grouped[group][snr]["psnr"])
            ssim_mean, _, ssim_ci_val = _summary_stats(grouped[group][snr]["ssim"])
            psnr_means.append(psnr_mean)
            psnr_ci.append(psnr_ci_val)
            ssim_means.append(ssim_mean)
            ssim_ci.append(ssim_ci_val)
        ax_psnr.errorbar(snrs, psnr_means, yerr=psnr_ci, marker="o", capsize=3, label=group)
        ax_ssim.errorbar(snrs, ssim_means, yerr=ssim_ci, marker="o", capsize=3, label=group)

    ax_psnr.set_title("PSNR vs SNR")
    ax_psnr.set_xlabel("SNR (dB)")
    ax_psnr.set_ylabel("PSNR")
    ax_psnr.grid(True, alpha=0.3)
    ax_psnr.legend()

    ax_ssim.set_title("SSIM vs SNR")
    ax_ssim.set_xlabel("SNR (dB)")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.grid(True, alpha=0.3)
    ax_ssim.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_grouped.png")
        print(f"Saved {save_path}_grouped.png")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval CSV by SNR and turbidity group.")
    parser.add_argument("--input_csv", required=True, help="Path to eval CSV.")
    parser.add_argument("--output_csv", default="", help="Path for summary CSV.")
    parser.add_argument(
        "--turbid_keys",
        default="turbid",
        help="Comma-separated keywords to detect turbid videos.",
    )
    parser.add_argument(
        "--clear_keys",
        default="clear",
        help="Comma-separated keywords to detect clear videos.",
    )
    parser.add_argument(
        "--include_unknown",
        action="store_true",
        help="Include rows where group cannot be detected.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save plots to files instead of displaying them.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_summary.csv")

    turbid_keys = [k.strip().lower() for k in args.turbid_keys.split(",") if k.strip()]
    clear_keys = [k.strip().lower() for k in args.clear_keys.split(",") if k.strip()]
    if not turbid_keys or not clear_keys:
        raise ValueError("Both --turbid_keys and --clear_keys must be non-empty.")

    with input_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    grouped = _accumulate(rows, turbid_keys, clear_keys)
    _write_summary(output_path, grouped, args.include_unknown)
    # If save plots, construct save path
    if args.save_plots:
        save_path = str(output_path.with_suffix(''))
        _plot_grouped(grouped, args.include_unknown, save_path=save_path)
    else:
        _plot_grouped(grouped, args.include_unknown)

    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
