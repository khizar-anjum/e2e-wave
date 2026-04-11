#!/usr/bin/env python3
"""
Consolidate completed UVE evaluation runs into a single directory for plotting.

This script scans the results directories for completed runs (identified by the
presence of sweep_summary.json) and copies them to results/plots/ organized by
method type (vqvae, mpeg4, softcast, wavebank).
"""

import os
import shutil
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List


# Wavebank waveform length configuration
# Maps waveform length to (fps, fec_type, fec_rate, equivalent_fec_rate_str, modulation)
# Note: QPSK lengths are half of BPSK lengths for equivalent rates (2 bits/symbol vs 1 bit/symbol)
WAVEBANK_LENGTH_CONFIG: Dict[int, dict] = {
    # BPSK configurations
    9:  {"fps": 16.0,   "fec": "none", "fec_rate": None, "equiv_rate": "none", "modulation": "BPSK", "description": "BPSK 16 fps, no FEC"},
    10: {"fps": 14.43,  "fec": "none", "fec_rate": None, "equiv_rate": "none", "modulation": "BPSK", "description": "BPSK 14.43 fps, no FEC"},
    13: {"fps": 10.53,  "fec": "dvbs2_ldpc", "fec_rate": 0.73, "equiv_rate": "r73", "modulation": "BPSK", "description": "BPSK 10.53 fps, equivalent to LDPC r73"},
    30: {"fps": 4.76,   "fec": "dvbs2_ldpc", "fec_rate": 0.33, "equiv_rate": "r33", "modulation": "BPSK", "description": "BPSK 4.76 fps, equivalent to LDPC r33"},
    # QPSK configurations (wavelengths halved for same equivalent rates)
    5:  {"fps": 14.43,  "fec": "none", "fec_rate": None, "equiv_rate": "none", "modulation": "QPSK", "description": "QPSK ~14-16 fps, no FEC"},
    7:  {"fps": 10.53,  "fec": "dvbs2_ldpc", "fec_rate": 0.73, "equiv_rate": "r73", "modulation": "QPSK", "description": "QPSK 10.53 fps, equivalent to LDPC r73"},
    15: {"fps": 4.76,   "fec": "dvbs2_ldpc", "fec_rate": 0.33, "equiv_rate": "r33", "modulation": "QPSK", "description": "QPSK 4.76 fps, equivalent to LDPC r33"},
}


def is_complete_run(run_path: Path) -> bool:
    """Check if a run directory contains a completed evaluation."""
    return (run_path / "sweep_summary.json").exists()


def read_fec_rate_from_summary(run_path: Path) -> Optional[float]:
    """Read the FEC rate from sweep_summary.json."""
    summary_path = run_path / "sweep_summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            data = json.load(f)
        # Handle nested config structure
        cc = data.get("config", data).get("channel_config", {})
        # Try different key names used across scripts
        rate = cc.get("fec_rate_target", cc.get("fec_rate"))
        if rate is not None:
            return float(rate)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def read_modulation_from_summary(run_path: Path) -> Optional[str]:
    """Read the modulation type from sweep_summary.json."""
    summary_path = run_path / "sweep_summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            data = json.load(f)
        # Handle nested config structure
        cc = data.get("config", data).get("channel_config", {})
        return cc.get("modulation")
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def format_fec_rate(rate: Optional[float]) -> str:
    """Format FEC rate as a string for use in filenames."""
    if rate is None:
        return ""
    # Common rates: 0.33 -> "r33", 0.5 -> "r50", 0.73 -> "r73"
    return f"r{int(rate * 100)}"


def parse_run_info(run_name: str) -> dict:
    """Parse run directory name to extract metadata."""
    # Patterns:
    # vqvae: uve_eval_{channel}_{fec}_snr_sweep_{timestamp}
    # mpeg4: uve_mpeg4_{channel}_{fec}_{codec}_{timestamp}
    # softcast: uve_softcast_{channel}_{equalizer}_{timestamp}

    info = {"name": run_name}
    parts = run_name.split("_")

    if run_name.startswith("uve_eval_"):
        # VQ-VAE run
        info["method"] = "vqvae"
        # uve_eval_NOF1_dvbs2_ldpc_snr_sweep_20251227_133804
        info["channel"] = parts[2]
        if "ldpc" in run_name:
            info["fec"] = "dvbs2_ldpc"
        elif "none" in parts:
            info["fec"] = "none"
        else:
            info["fec"] = "unknown"
    elif run_name.startswith("uve_mpeg4_"):
        # MPEG4 run
        info["method"] = "mpeg4"
        # uve_mpeg4_NOF1_none_h265_20251228_013733
        info["channel"] = parts[2]
        info["fec"] = parts[3]
        info["codec"] = parts[4]
    elif run_name.startswith("uve_softcast_"):
        # SoftCast run
        info["method"] = "softcast"
        # uve_softcast_NOF1_zf_20251227_220508
        info["channel"] = parts[2]
        info["equalizer"] = parts[3]
    else:
        info["method"] = "unknown"

    # Extract timestamp (last two parts usually)
    try:
        # Find date pattern YYYYMMDD
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():
                info["timestamp"] = f"{part}_{parts[i+1]}" if i+1 < len(parts) else part
                break
    except:
        info["timestamp"] = "unknown"

    return info


def get_canonical_name(info: dict) -> str:
    """Generate a canonical name for the run based on its parameters."""
    method = info.get("method", "unknown")
    channel = info.get("channel", "unknown")
    modulation = info.get("modulation", "").lower()  # bpsk or qpsk

    if method == "vqvae":
        fec = info.get("fec", "unknown")
        fec_rate_str = info.get("fec_rate_str", "")
        # Include modulation in canonical name
        mod_suffix = f"_{modulation}" if modulation else ""
        if fec == "dvbs2_ldpc" and fec_rate_str:
            return f"vqvae_{channel}_{fec}_{fec_rate_str}{mod_suffix}"
        return f"vqvae_{channel}_{fec}{mod_suffix}"
    elif method == "mpeg4":
        fec = info.get("fec", "unknown")
        codec = info.get("codec", "h265")
        fec_rate_str = info.get("fec_rate_str", "")
        mod_suffix = f"_{modulation}" if modulation else ""
        # For dvbs2_ldpc, fec is parsed as "dvbs2" from directory name
        if "dvbs2" in fec and fec_rate_str:
            return f"mpeg4_{channel}_dvbs2_ldpc_{fec_rate_str}_{codec}{mod_suffix}"
        return f"mpeg4_{channel}_{fec}_{codec}{mod_suffix}"
    elif method == "softcast":
        eq = info.get("equalizer", "zf")
        return f"softcast_{channel}_{eq}"
    else:
        return info["name"]


def find_completed_runs(results_dir: Path) -> list:
    """Find all completed UVE evaluation runs."""
    completed = []

    # Check vqvae directory for uve_* runs
    vqvae_dir = results_dir / "vqvae"
    if vqvae_dir.exists():
        for run_dir in vqvae_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("uve_eval_"):
                if is_complete_run(run_dir):
                    info = parse_run_info(run_dir.name)
                    info["source_path"] = run_dir
                    # Read modulation type
                    modulation = read_modulation_from_summary(run_dir)
                    info["modulation"] = modulation
                    # Read FEC rate from summary for dvbs2_ldpc runs
                    if info.get("fec") == "dvbs2_ldpc":
                        fec_rate = read_fec_rate_from_summary(run_dir)
                        info["fec_rate"] = fec_rate
                        info["fec_rate_str"] = format_fec_rate(fec_rate)
                    completed.append(info)

    # Check mpeg4_uve directory
    mpeg4_dir = results_dir / "mpeg4_uve"
    if mpeg4_dir.exists():
        for run_dir in mpeg4_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("uve_mpeg4_"):
                if is_complete_run(run_dir):
                    info = parse_run_info(run_dir.name)
                    info["source_path"] = run_dir
                    # Read modulation type
                    modulation = read_modulation_from_summary(run_dir)
                    info["modulation"] = modulation
                    # Read FEC rate from summary for dvbs2 runs
                    if "dvbs2" in info.get("fec", ""):
                        fec_rate = read_fec_rate_from_summary(run_dir)
                        info["fec_rate"] = fec_rate
                        info["fec_rate_str"] = format_fec_rate(fec_rate)
                    completed.append(info)

    # Check softcast_uve directory
    softcast_dir = results_dir / "softcast_uve"
    if softcast_dir.exists():
        for run_dir in softcast_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("uve_softcast_"):
                if is_complete_run(run_dir):
                    info = parse_run_info(run_dir.name)
                    info["source_path"] = run_dir
                    completed.append(info)

    return completed


def select_best_runs(completed_runs: list) -> dict:
    """Select the most recent run for each unique configuration."""
    best_runs = {}

    for run in completed_runs:
        canonical = get_canonical_name(run)
        timestamp = run.get("timestamp", "")

        if canonical not in best_runs:
            best_runs[canonical] = run
        else:
            # Keep the most recent one
            if timestamp > best_runs[canonical].get("timestamp", ""):
                best_runs[canonical] = run

    return best_runs


def find_wavebank_results(results_dir: Path) -> List[dict]:
    """Find all wavebank results with summary CSV files."""
    wavebank_results = []
    wavebank_dir = results_dir / "wavebank"

    if not wavebank_dir.exists():
        return wavebank_results

    # Pattern: {CHANNEL}_waveform_len_{LENGTH}_summary.csv
    pattern = re.compile(r"^([A-Z0-9]+)_waveform_len_(\d+)_summary\.csv$")

    for csv_file in wavebank_dir.glob("*_summary.csv"):
        match = pattern.match(csv_file.name)
        if match:
            channel = match.group(1)
            waveform_len = int(match.group(2))

            # Get configuration for this waveform length
            config = WAVEBANK_LENGTH_CONFIG.get(waveform_len, {})

            # Check for associated files
            base_name = csv_file.stem.replace("_summary", "")
            associated_files = {
                "summary_csv": csv_file,
                "grouped_png": wavebank_dir / f"{base_name}_summary_grouped.png",
                "detailed_csv": wavebank_dir / f"{base_name}.csv",
            }

            wavebank_results.append({
                "method": "wavebank",
                "channel": channel,
                "waveform_len": waveform_len,
                "fps": config.get("fps"),
                "fec": config.get("fec"),
                "fec_rate": config.get("fec_rate"),
                "equiv_rate": config.get("equiv_rate"),
                "modulation": config.get("modulation"),
                "description": config.get("description", f"len_{waveform_len}"),
                "source_files": associated_files,
                "canonical_name": f"wavebank_{channel}_len{waveform_len}",
            })

    return wavebank_results


def consolidate_wavebank(wavebank_results: List[dict], output_dir: Path,
                         copy: bool = True, verbose: bool = True) -> List[dict]:
    """Consolidate wavebank results into the output directory."""
    manifest_entries = []
    wavebank_out = output_dir / "wavebank"
    wavebank_out.mkdir(parents=True, exist_ok=True)

    action = "Copying" if copy else "Moving"

    for result in sorted(wavebank_results, key=lambda x: (x["channel"], x["waveform_len"])):
        canonical = result["canonical_name"]
        dest_dir = wavebank_out / canonical
        dest_dir.mkdir(exist_ok=True)

        if verbose:
            print(f"{action} wavebank: {result['channel']} len_{result['waveform_len']} -> {canonical}")

        # Copy/move associated files
        for file_type, src_path in result["source_files"].items():
            if src_path.exists():
                dest_path = dest_dir / src_path.name
                if copy:
                    shutil.copy2(src_path, dest_path)
                else:
                    shutil.move(src_path, dest_path)

        manifest_entries.append({
            "canonical_name": canonical,
            "method": "wavebank",
            "channel": result["channel"],
            "waveform_len": result["waveform_len"],
            "fps": result["fps"],
            "fec": result["fec"],
            "fec_rate": result["fec_rate"],
            "equiv_rate": result["equiv_rate"],
            "modulation": result["modulation"],
            "description": result["description"],
            "path": str(dest_dir.relative_to(output_dir)),
        })

    return manifest_entries


def consolidate_runs(results_dir: Path, output_dir: Path, copy: bool = True, verbose: bool = True):
    """Consolidate completed runs into the output directory."""

    # Find all completed runs
    if verbose:
        print("Scanning for completed UVE evaluation runs...")

    completed = find_completed_runs(results_dir)

    if verbose:
        print(f"Found {len(completed)} completed runs")

        # Show breakdown by method
        by_method = {}
        for run in completed:
            method = run.get("method", "unknown")
            by_method[method] = by_method.get(method, 0) + 1
        for method, count in sorted(by_method.items()):
            print(f"  - {method}: {count} runs")

    # Select best (most recent) run for each configuration
    best_runs = select_best_runs(completed)

    if verbose:
        print(f"\nSelected {len(best_runs)} unique configurations (most recent for each)")

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Organize by method
    methods = ["vqvae", "mpeg4", "softcast", "wavebank"]
    for method in methods:
        (output_dir / method).mkdir(exist_ok=True)

    # Copy/move runs
    manifest = []
    action = "Copying" if copy else "Moving"

    for canonical, run in sorted(best_runs.items()):
        source = run["source_path"]
        method = run.get("method", "unknown")
        dest = output_dir / method / canonical

        if verbose:
            print(f"{action}: {source.name} -> {dest.name}")

        if dest.exists():
            if verbose:
                print(f"  (removing existing {dest.name})")
            shutil.rmtree(dest)

        if copy:
            shutil.copytree(source, dest)
        else:
            shutil.move(source, dest)

        # Read sweep summary for manifest
        summary_path = dest / "sweep_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = {}

        manifest.append({
            "canonical_name": canonical,
            "method": method,
            "channel": run.get("channel"),
            "fec": run.get("fec"),
            "fec_rate": run.get("fec_rate"),
            "fec_rate_str": run.get("fec_rate_str"),
            "modulation": run.get("modulation"),
            "codec": run.get("codec"),
            "equalizer": run.get("equalizer"),
            "original_name": run["name"],
            "timestamp": run.get("timestamp"),
            "path": str(dest.relative_to(output_dir)),
        })

    # Process wavebank results
    if verbose:
        print("\nScanning for wavebank results...")
    wavebank_results = find_wavebank_results(results_dir)
    if wavebank_results:
        if verbose:
            print(f"Found {len(wavebank_results)} wavebank results")
        wavebank_manifest = consolidate_wavebank(wavebank_results, output_dir, copy, verbose)
        manifest.extend(wavebank_manifest)
    elif verbose:
        print("No wavebank results found")

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "created": datetime.now().isoformat(),
            "wavebank_length_config": WAVEBANK_LENGTH_CONFIG,
            "runs": manifest
        }, f, indent=2)

    if verbose:
        print(f"\nManifest written to {manifest_path}")
        total_runs = len(best_runs) + len(wavebank_results)
        print(f"\nConsolidated {total_runs} runs to {output_dir}")

    # Print summary table
    if verbose:
        print("\n" + "=" * 110)
        print("CONSOLIDATED RUNS SUMMARY")
        print("=" * 110)
        print(f"{'Method':<10} {'Channel':<8} {'Mod':<6} {'FEC/Config':<18} {'Rate':<8} {'Path':<58}")
        print("-" * 110)
        for entry in sorted(manifest, key=lambda x: (x["method"], x.get("channel", ""), x.get("modulation", ""), x["canonical_name"])):
            config = ""
            rate_str = ""
            mod_str = ""
            if entry["method"] == "vqvae":
                config = entry.get("fec", "")
                rate_str = entry.get("fec_rate_str", "") or "-"
                mod_str = entry.get("modulation", "-") or "-"
            elif entry["method"] == "mpeg4":
                fec = entry.get("fec", "")
                codec = entry.get("codec", "")
                config = f"{fec} {codec}" if fec != "dvbs2" else f"dvbs2_ldpc {codec}"
                rate_str = entry.get("fec_rate_str", "") or "-"
                mod_str = entry.get("modulation", "-") or "-"
            elif entry["method"] == "softcast":
                config = entry.get("equalizer", "")
                rate_str = "-"
                mod_str = "analog"
            elif entry["method"] == "wavebank":
                wlen = entry.get("waveform_len", "?")
                fps = entry.get("fps")
                config = f"len_{wlen}" + (f" ({fps:.1f}fps)" if fps else "")
                rate_str = entry.get("equiv_rate", "-") or "-"
                mod_str = entry.get("modulation", "-") or "-"
            print(f"{entry['method']:<10} {entry.get('channel', ''):<8} {mod_str:<6} {config:<18} {rate_str:<8} {entry['path']:<58}")
        print("=" * 110)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Consolidate completed UVE evaluation runs")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Path to results directory (default: results)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"),
                        help="Output directory for consolidated runs (default: results/plots)")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    consolidate_runs(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        copy=not args.move,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
