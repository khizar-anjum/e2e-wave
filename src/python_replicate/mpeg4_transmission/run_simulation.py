#!/usr/bin/env python3
"""
CLI entry point for MPEG4 video transmission simulation.

Usage:
    python -m python_replicate.mpeg4_transmission.run_simulation \
        --video input/test.mp4 \
        --snr 0 5 10 15 20 \
        --modulation QPSK \
        --fec repetition \
        --channel awgn
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List

from .config import Mpeg4SimConfig, SimulationResult
from .mpeg_pipeline import Mpeg4Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='MPEG4 video transmission simulation over underwater acoustic channels'
    )

    # Video settings
    parser.add_argument('--video', type=Path, required=True,
                        help='Path to input video file')
    parser.add_argument('--resolution', type=int, nargs=2, default=[64, 64],
                        help='Target video resolution (width height)')
    parser.add_argument('--bitrate', type=int, default=5000,
                        help='Target video bitrate in bps (default: 5000)')

    # Channel settings
    parser.add_argument('--snr', type=float, nargs='+', default=[0, 5, 10, 15, 20],
                        help='SNR values in dB for sweep')
    parser.add_argument('--channel', choices=['awgn', 'uwa'], default='awgn',
                        help='Channel type (default: awgn)')
    parser.add_argument('--channel-name', type=str, default='NOF1',
                        help='UWA channel name (default: NOF1)')
    parser.add_argument('--channel-dir', type=Path, default=Path('input/channels'),
                        help='Base directory for channel files')

    # Modulation settings
    parser.add_argument('--modulation', choices=['BPSK', 'QPSK'], default='QPSK',
                        help='Modulation scheme (default: QPSK)')

    # FEC settings
    parser.add_argument('--fec', choices=['none', 'repetition'], default='none',
                        help='FEC type (default: none)')
    parser.add_argument('--fec-repetitions', type=int, default=3,
                        help='Repetitions for repetition FEC (default: 3)')

    # OFDM settings
    parser.add_argument('--num-carriers', type=int, default=64,
                        help='Number of OFDM subcarriers (default: 64)')
    parser.add_argument('--num-symbols', type=int, default=16,
                        help='Number of OFDM symbols per frame (default: 16)')
    parser.add_argument('--cp-length', type=int, default=30,
                        help='Cyclic prefix length (default: 30)')
    parser.add_argument('--pilot-period', type=int, default=4,
                        help='Pilot period (default: 4)')

    # Output settings
    parser.add_argument('--output-dir', type=Path, default=Path('results/mpeg4'),
                        help='Output directory for results')
    parser.add_argument('--save-videos', action='store_true',
                        help='Save reconstructed videos')

    return parser.parse_args()


def save_results(
    results: List[SimulationResult],
    config: Mpeg4SimConfig,
    output_dir: Path,
):
    """Save simulation results to CSV and JSON files.

    Args:
        results: List of simulation results
        config: Simulation configuration
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save summary CSV
    csv_path = output_dir / f'summary_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'snr_db', 'modulation', 'fec', 'ber', 'bit_errors', 'total_bits',
            'psnr_mean', 'ssim_mean', 'playable', 'frames_decoded', 'frames_total',
            'transmission_time_sec'
        ])
        for r in results:
            writer.writerow([
                r.snr_db,
                config.modulation,
                config.fec_type,
                r.ber,
                r.bit_errors,
                r.total_bits,
                r.psnr_mean,
                r.ssim_mean,
                r.playable,
                r.frames_decoded,
                r.frames_total,
                r.transmission_time_sec,
            ])

    print(f"Saved summary to {csv_path}")

    # Save detailed JSON
    json_path = output_dir / f'results_{timestamp}.json'
    data = {
        'config': {
            'video_path': str(config.video_path),
            'resolution': config.resolution,
            'target_bitrate_bps': config.target_bitrate_bps,
            'channel_type': config.channel_type,
            'channel_name': config.channel_name,
            'modulation': config.modulation,
            'fec_type': config.fec_type,
            'fec_repetitions': config.fec_repetitions,
            'num_carriers': config.num_carriers,
            'num_ofdm_symbols': config.num_ofdm_symbols,
            'cp_length': config.cp_length,
            'pilot_period': config.pilot_period,
        },
        'results': [r.to_dict() for r in results],
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved detailed results to {json_path}")


def print_results_table(results: List[SimulationResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print(f"{'SNR (dB)':>10} {'BER':>12} {'PSNR (dB)':>12} {'SSIM':>10} {'Playable':>10}")
    print("-" * 80)

    for r in results:
        playable_str = 'Yes' if r.playable else 'No'
        print(f"{r.snr_db:>10.1f} {r.ber:>12.6f} {r.psnr_mean:>12.2f} {r.ssim_mean:>10.4f} {playable_str:>10}")

    print("=" * 80)


def main():
    args = parse_args()

    # Build configuration
    config = Mpeg4SimConfig(
        video_path=args.video,
        resolution=tuple(args.resolution),
        target_bitrate_bps=args.bitrate,
        channel_type=args.channel,
        channel_name=args.channel_name,
        channel_base_dir=args.channel_dir,
        modulation=args.modulation,
        fec_type=args.fec,
        fec_repetitions=args.fec_repetitions,
        num_carriers=args.num_carriers,
        num_ofdm_symbols=args.num_symbols,
        cp_length=args.cp_length,
        pilot_period=args.pilot_period,
        output_dir=args.output_dir,
    )

    print("=" * 80)
    print("MPEG4 VIDEO TRANSMISSION SIMULATION")
    print("=" * 80)
    print(f"Video:       {config.video_path}")
    print(f"Resolution:  {config.resolution[0]}x{config.resolution[1]}")
    print(f"Bitrate:     {config.target_bitrate_bps} bps")
    print(f"Channel:     {config.channel_type.upper()}" +
          (f" ({config.channel_name})" if config.channel_type == 'uwa' else ''))
    print(f"Modulation:  {config.modulation}")
    print(f"FEC:         {config.fec_type}" +
          (f" (rep={config.fec_repetitions})" if config.fec_type == 'repetition' else ''))
    print(f"SNR range:   {args.snr}")
    print("=" * 80)

    # Check video exists
    if not config.video_path.exists():
        print(f"Error: Video file not found: {config.video_path}")
        return 1

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = Mpeg4Pipeline(config)

    # Print channel throughput info
    bits_per_frame = config.bits_per_frame
    effective_bits = config.effective_bits_per_frame
    print(f"\nChannel info:")
    print(f"  Bits per frame:     {bits_per_frame}")
    print(f"  Effective (w/FEC):  {effective_bits}")
    print(f"  Data symbols/frame: {config.data_symbols_per_frame}")

    # Run SNR sweep
    print("\nRunning SNR sweep...")
    results = pipeline.run_snr_sweep(args.snr)

    # Print and save results
    print_results_table(results)
    save_results(results, config, args.output_dir)

    return 0


if __name__ == '__main__':
    exit(main())
