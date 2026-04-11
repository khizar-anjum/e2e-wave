#!/usr/bin/env python3
"""
MPEG4/H.265 Evaluation Script for UVE38K Dataset with Channel Simulation.

Evaluates digital video transmission quality across the UVE38K dataset
(clear and turbid underwater video categories) using MPEG4/H.265 compression
with OFDM channel simulation and configurable FEC.

Features:
- Processes videos by category (clear/turbid) and reports per-category statistics
- Records per-video metrics (PSNR, SSIM, BER)
- Supports multiple channel types (AWGN, UWA channels)
- Supports configurable FEC (repetition, LDPC, polar, turbo, DVB-S2 LDPC)
- Auto-calculates bitrate from channel capacity
- SNR sweep mode for performance curves

Usage:
    # AWGN channel, no FEC
    python eval_mpeg4_uve.py --channel-type awgn --snr-min 0 --snr-max 30 --snr-step 5

    # UWA channel with DVB-S2 LDPC
    python eval_mpeg4_uve.py --channel-type uwa --channel NOF1 \
        --fec dvbs2_ldpc --fec-rate 0.33 --snr-min 0 --snr-max 30

    # With specific bitrate and H.265 codec
    python eval_mpeg4_uve.py --channel-type awgn --codec h265 --bitrate 5000
"""

import argparse
import json
import math
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from python_replicate.mpeg4_transmission import Mpeg4SimConfig, Mpeg4Pipeline
from python_replicate.mpeg4_transmission.video_utils import (
    reencode_video, extract_frames, get_video_info, extract_frames_from_bytes
)
from python_replicate.mpeg4_transmission.metrics import img_psnr, img_ssim


# Default UVE38K dataset path
DEFAULT_UVE_PATH = "/home/khizar/Datasets/UVE38K/raw/10_sec_clips"

# Video extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}

# Channel recording settings for evaluation (consistent with eval_vqvae_uve.py)
EVAL_RECORDING_MODE = "random"  # "first", "random", or "fixed"
EVAL_RECORDING_SEED = 123


def find_video_files(input_dir: Path) -> List[Path]:
    """Find all video files in directory."""
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_dir.glob(f'*{ext}'))
        video_files.extend(input_dir.glob(f'*{ext.upper()}'))
    return sorted(video_files)


def discover_categories(uve_path: Path) -> Dict[str, List[Path]]:
    """Auto-discover categories from UVE38K directory structure.

    Expects structure like:
        uve_path/
        ├── clear/    (videos)
        └── turbid/   (videos)

    Returns:
        Dict mapping category name to list of video paths
    """
    category_videos = {}

    # Find all subdirectories that contain videos
    for subdir in sorted(uve_path.iterdir()):
        if subdir.is_dir():
            videos = find_video_files(subdir)
            if videos:
                category_videos[subdir.name] = videos

    return category_videos


def parse_resolution(res_str: str) -> Tuple[int, int]:
    """Parse resolution string like '64x64' or '128x128' into (width, height)."""
    if 'x' in res_str.lower():
        parts = res_str.lower().split('x')
        return (int(parts[0]), int(parts[1]))
    else:
        size = int(res_str)
        return (size, size)


def select_fec_params(fec_type: str, target_rate: float, manual_k: int = None, manual_n: int = None) -> dict:
    """Auto-select K/N parameters for a given FEC type and target rate."""
    if fec_type == 'none':
        return {'k': None, 'n': None, 'rate': 1.0}

    if fec_type == 'repetition':
        reps = max(1, round(1.0 / target_rate)) if target_rate else 3
        if reps > 1 and reps % 2 == 0:
            reps += 1
        return {'k': None, 'n': None, 'rate': 1.0/reps, 'repetitions': reps}

    if fec_type == 'dvbs2_ldpc':
        if manual_k and manual_n:
            return {'k': manual_k, 'n': manual_n, 'rate': manual_k/manual_n}
        from python_replicate.aff3ct_codecs import DVBS2LDPCCodec
        target = target_rate if target_rate else 0.33
        actual_rate, k, n = DVBS2LDPCCodec.find_code(target, frame='short')
        return {'k': k, 'n': n, 'rate': actual_rate}

    if fec_type == 'polar':
        if manual_k and manual_n:
            return {'k': manual_k, 'n': manual_n, 'rate': manual_k/manual_n}
        target = target_rate if target_rate else 0.5
        n = 1024
        k = max(32, min(n-1, round(target * n)))
        k = max(32, (k // 32) * 32)
        return {'k': k, 'n': n, 'rate': k/n}

    if fec_type == 'turbo':
        valid_k = [112, 128, 304, 408, 456, 680, 768, 800, 864, 880,
                   1304, 1360, 1504, 1728, 2384, 2664, 3008, 3504]
        if manual_k:
            k = manual_k
        else:
            k = 880
        k = min(valid_k, key=lambda x: abs(x - k))
        n = 3 * k
        return {'k': k, 'n': n, 'rate': 1/3}

    if fec_type == 'rsc':
        k = manual_k if manual_k else 64
        n = 2 * k + 4
        return {'k': k, 'n': n, 'rate': k/n}

    if fec_type == 'ldpc':
        return {'k': None, 'n': None, 'rate': 0.5}

    return {'k': None, 'n': None, 'rate': 1.0}


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"


def process_video(
    pipeline: Mpeg4Pipeline,
    video_path: Path,
    resolution: Tuple[int, int],
    bitrate_bps: int,
    codec: str,
    snr_db: float,
    fps: float = None,
) -> Dict[str, Any]:
    """Process a single video through the MPEG4 pipeline.

    Args:
        pipeline: Mpeg4Pipeline instance
        video_path: Path to video file
        resolution: Target resolution (width, height)
        bitrate_bps: Target video bitrate
        codec: FFmpeg codec name ('libx264', 'libx265')
        snr_db: SNR in dB
        fps: Target frame rate (None = keep original)

    Returns:
        Dict with metrics (psnr, ssim, ber, etc.)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # High-quality reference at target resolution (for PSNR/SSIM comparison)
        reference_path = tmpdir / 'reference.mp4'
        reencode_video(
            video_path, reference_path,
            resolution=resolution,
            bitrate_bps=1_000_000,  # High bitrate for quality reference
            codec=codec,
            fps=fps,
        )
        reference_frames = extract_frames(reference_path)

        # Low-bitrate encoding for transmission
        encoded_path = tmpdir / 'encoded.mp4'
        video_bytes = reencode_video(
            video_path, encoded_path,
            resolution=resolution,
            bitrate_bps=bitrate_bps,
            codec=codec,
            fps=fps,
        )
        encoded_frames = extract_frames(encoded_path)

    # Transmit through channel
    rx_bytes, ber, bit_errors, total_bits = pipeline.transmit_bytes(video_bytes, snr_db)

    # Try to extract frames from received bytes
    rx_frames, playable, frames_decoded = extract_frames_from_bytes(
        rx_bytes, resolution[0], resolution[1]
    )

    # Compute video quality against HIGH-QUALITY REFERENCE
    if len(rx_frames) > 0 and len(reference_frames) > 0:
        min_frames = min(len(rx_frames), len(reference_frames))
        psnr_values = [img_psnr(reference_frames[i], rx_frames[i], 255.0)
                      for i in range(min_frames)]
        ssim_values = [img_ssim(reference_frames[i], rx_frames[i], 255.0)
                      for i in range(min_frames)]
        psnr_mean = float(np.mean(psnr_values))
        psnr_std = float(np.std(psnr_values))
        ssim_mean = float(np.mean(ssim_values))
        ssim_std = float(np.std(ssim_values))
    else:
        psnr_mean = 0.0
        psnr_std = 0.0
        ssim_mean = 0.0
        ssim_std = 0.0

    return {
        'ber': ber,
        'bit_errors': bit_errors,
        'total_bits': total_bits,
        'psnr_mean': psnr_mean,
        'psnr_std': psnr_std,
        'ssim_mean': ssim_mean,
        'ssim_std': ssim_std,
        'playable': playable,
        'frames_decoded': frames_decoded,
        'frames_total': len(encoded_frames),
        'video_bytes': len(video_bytes),
    }


def create_snr_sweep_plot(
    snr_sweep_results: List[Dict],
    categories: List[str],
    output_path: Path
):
    """Create SNR sweep plot showing metrics vs SNR."""
    snr_values = [r['snr_db'] for r in snr_sweep_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'clear': 'steelblue', 'turbid': 'seagreen', 'overall': 'purple'}

    # PSNR vs SNR
    ax = axes[0, 0]
    overall_psnr = [r['overall_stats']['psnr_mean'] for r in snr_sweep_results]
    ax.plot(snr_values, overall_psnr, 'o-', color=colors['overall'], linewidth=2, markersize=8, label='Overall')
    for cat in categories:
        cat_psnr = [r['category_stats'].get(cat, {}).get('psnr_mean', 0) for r in snr_sweep_results]
        ax.plot(snr_values, cat_psnr, 's--', color=colors.get(cat, 'gray'), linewidth=1.5, markersize=6, label=cat.capitalize())
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SSIM vs SNR
    ax = axes[0, 1]
    overall_ssim = [r['overall_stats']['ssim_mean'] for r in snr_sweep_results]
    ax.plot(snr_values, overall_ssim, 'o-', color=colors['overall'], linewidth=2, markersize=8, label='Overall')
    for cat in categories:
        cat_ssim = [r['category_stats'].get(cat, {}).get('ssim_mean', 0) for r in snr_sweep_results]
        ax.plot(snr_values, cat_ssim, 's--', color=colors.get(cat, 'gray'), linewidth=1.5, markersize=6, label=cat.capitalize())
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Playable fraction vs SNR
    ax = axes[1, 0]
    overall_playable = [r['overall_stats'].get('playable_fraction', 0) for r in snr_sweep_results]
    ax.plot(snr_values, overall_playable, 'o-', color=colors['overall'], linewidth=2, markersize=8, label='Overall')
    for cat in categories:
        cat_playable = [r['category_stats'].get(cat, {}).get('playable_fraction', 0) for r in snr_sweep_results]
        ax.plot(snr_values, cat_playable, 's--', color=colors.get(cat, 'gray'), linewidth=1.5, markersize=6, label=cat.capitalize())
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Playable Fraction')
    ax.set_title('Playable Videos vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # BER vs SNR (log scale)
    ax = axes[1, 1]
    overall_ber = [r['overall_stats'].get('ber', 0) for r in snr_sweep_results]
    overall_ber_plot = [max(b, 1e-7) for b in overall_ber]
    ax.semilogy(snr_values, overall_ber_plot, 'o-', color=colors['overall'], linewidth=2, markersize=8, label='Overall')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title('Bit Error Rate vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_category_bar_chart(
    category_stats: Dict[str, Dict],
    output_path: Path,
):
    """Create bar chart comparing category averages."""
    categories = list(category_stats.keys())
    if not categories:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    colors = {'clear': 'steelblue', 'turbid': 'seagreen'}
    x = np.arange(len(categories))

    # PSNR
    psnr_vals = [category_stats[c]['psnr_mean'] for c in categories]
    psnr_stds = [category_stats[c]['psnr_std'] for c in categories]
    bars = axes[0].bar(x, psnr_vals, yerr=psnr_stds, capsize=5,
                       color=[colors.get(c, 'gray') for c in categories], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.capitalize() for c in categories])
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Average PSNR by Category')
    for bar, val in zip(bars, psnr_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.2f}', ha='center', va='bottom')

    # SSIM
    ssim_vals = [category_stats[c]['ssim_mean'] for c in categories]
    ssim_stds = [category_stats[c]['ssim_std'] for c in categories]
    bars = axes[1].bar(x, ssim_vals, yerr=ssim_stds, capsize=5,
                       color=[colors.get(c, 'gray') for c in categories], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.capitalize() for c in categories])
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('Average SSIM by Category')
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars, ssim_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.4f}', ha='center', va='bottom')

    # BER
    ber_vals = [category_stats[c].get('ber', 0) for c in categories]
    bars = axes[2].bar(x, ber_vals,
                       color=[colors.get(c, 'gray') for c in categories], alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([c.capitalize() for c in categories])
    axes[2].set_ylabel('BER')
    axes[2].set_title('Average BER by Category')
    for bar, val in zip(bars, ber_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.6f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='MPEG4/H.265 Evaluation on UVE38K Dataset with Channel Simulation'
    )

    # Dataset arguments
    parser.add_argument('--uve-path', type=str, default=DEFAULT_UVE_PATH,
                        help=f'Path to UVE38K dataset root (default: {DEFAULT_UVE_PATH})')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum videos per category (default: all)')

    # Video settings
    parser.add_argument('--resolution', type=str, default='128x128',
                        help='Video resolution WxH (default: 128x128)')
    parser.add_argument('--bitrate', type=int, default=0,
                        help='Target video bitrate in bps (0 = auto from channel capacity)')
    parser.add_argument('--bitrate-efficiency', type=float, default=1.0,
                        help='Fraction of channel capacity to use (default: 1.0)')
    parser.add_argument('--codec', type=str, default='h265',
                        choices=['h264', 'h265'],
                        help='Video codec (default: h265)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Video frame rate (default: keep original)')

    # Channel arguments
    parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'uwa'],
                        help='Channel type (default: awgn)')
    parser.add_argument('--channel', type=str, default='NOF1',
                        help='Channel name for UWA (e.g., NOF1, KAU1, BCH1)')
    parser.add_argument('--channel-base-dir', type=str, default='input/channels',
                        help='Base directory containing channel .mat files')

    # SNR arguments
    parser.add_argument('--snr-min', type=float, default=0,
                        help='Minimum SNR for sweep (default: 0)')
    parser.add_argument('--snr-max', type=float, default=30,
                        help='Maximum SNR for sweep (default: 30)')
    parser.add_argument('--snr-step', type=float, default=5,
                        help='SNR step size for sweep (default: 5)')

    # FEC arguments
    parser.add_argument('--fec', type=str, default='none',
                        choices=['none', 'repetition', 'ldpc', 'polar', 'turbo', 'dvbs2_ldpc', 'rsc'],
                        help='FEC codec type (default: none)')
    parser.add_argument('--fec-rate', type=float, default=None,
                        help='Target FEC code rate (0.2-1.0)')
    parser.add_argument('--fec-k', type=int, default=None,
                        help='Manual override for message length K')
    parser.add_argument('--fec-n', type=int, default=None,
                        help='Manual override for codeword length N')

    # Modulation arguments
    parser.add_argument('--modulation', type=str, default='QPSK', choices=['BPSK', 'QPSK'],
                        help='Modulation scheme (default: QPSK)')

    # OFDM arguments
    parser.add_argument('--num-carriers', type=int, default=64,
                        help='Number of OFDM subcarriers (default: 64)')
    parser.add_argument('--num-ofdm-symbols', type=int, default=16,
                        help='OFDM symbols per frame (default: 16)')
    parser.add_argument('--cp-length', type=int, default=30,
                        help='Cyclic prefix length (default: 30)')
    parser.add_argument('--pilot-period', type=int, default=4,
                        help='Pilot period (default: 4)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results/mpeg4_uve',
                        help='Base output directory (default: results/mpeg4_uve)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (default: cpu)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Verify dataset path
    uve_path = Path(args.uve_path)
    if not uve_path.exists():
        print(f"Error: UVE38K dataset path '{uve_path}' does not exist")
        return 1

    # Parse resolution
    resolution = parse_resolution(args.resolution)

    # SNR values
    snr_values = list(np.arange(args.snr_min, args.snr_max + args.snr_step/2, args.snr_step))
    print(f"SNR sweep: {snr_values} dB")

    # Auto-select FEC parameters
    fec_params = select_fec_params(
        args.fec,
        args.fec_rate,
        manual_k=args.fec_k,
        manual_n=args.fec_n
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    channel_str = args.channel if args.channel_type == 'uwa' else 'awgn'
    run_name = f"uve_mpeg4_{channel_str}_{args.fec}_{args.codec}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create configuration
    config = Mpeg4SimConfig(
        video_path=Path('dummy.mp4'),  # Will be overwritten per video
        resolution=resolution,
        target_bitrate_bps=args.bitrate,
        auto_bitrate=args.bitrate == 0,
        bitrate_efficiency=args.bitrate_efficiency,
        codec=args.codec,
        fps=args.fps,
        channel_type=args.channel_type,
        channel_name=args.channel,
        channel_base_dir=Path(args.channel_base_dir),
        channel_recording_mode=EVAL_RECORDING_MODE,
        channel_recording_seed=EVAL_RECORDING_SEED,
        modulation=args.modulation,
        fec_type=args.fec,
        fec_repetitions=fec_params.get('repetitions', 3),
        dvbs2_ldpc_k=fec_params['k'] if args.fec == 'dvbs2_ldpc' else 3240,
        dvbs2_ldpc_n=fec_params['n'] if args.fec == 'dvbs2_ldpc' else 16200,
        polar_k=fec_params['k'] if args.fec == 'polar' else 512,
        polar_n=fec_params['n'] if args.fec == 'polar' else 1024,
        turbo_k=fec_params['k'] if args.fec == 'turbo' else 880,
        rsc_k=fec_params['k'] if args.fec == 'rsc' else 64,
        num_carriers=args.num_carriers,
        num_ofdm_symbols=args.num_ofdm_symbols,
        cp_length=args.cp_length,
        pilot_period=args.pilot_period,
    )

    # Print configuration
    print(f"\n{'='*70}")
    print("MPEG4/H.265 UVE38K EVALUATION")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Resolution:        {config.resolution[0]}x{config.resolution[1]}")
    print(f"  Codec:             {config.codec.upper()} ({config.ffmpeg_codec})")
    print(f"  Channel:           {args.channel_type.upper()}" +
          (f" ({args.channel})" if args.channel_type == 'uwa' else ''))
    print(f"  Modulation:        {config.modulation}")
    print(f"  FEC:               {args.fec} (rate={fec_params['rate']:.3f})")
    print(f"  OFDM:              {config.num_carriers} carriers, {config.num_ofdm_symbols} symbols")
    print(f"  SNR sweep:         {args.snr_min} to {args.snr_max} dB (step {args.snr_step})")

    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = Mpeg4Pipeline(config, device=device)

    # Get actual parameters from pipeline (may differ from config due to auto-derivation)
    actual_cp_length = pipeline.frame_config.cp_length
    actual_bandwidth = pipeline.frame_config.bandwidth_hz

    # Recalculate effective bitrate with actual CP length
    # Must match Mpeg4SimConfig formula which includes preamble overhead
    preamble_samples = 628  # sync(500) + sc(128) + train(0)
    ofdm_frame_samples = (config.num_carriers + actual_cp_length) * config.num_ofdm_symbols
    frame_baseband_samples = preamble_samples + ofdm_frame_samples
    frame_duration = frame_baseband_samples / actual_bandwidth  # baseband duration

    data_symbols_per_frame = config.num_carriers * (config.num_ofdm_symbols - config.num_ofdm_symbols // config.pilot_period)
    bits_per_symbol = 2 if config.modulation == 'QPSK' else 1
    effective_bits_per_frame = data_symbols_per_frame * bits_per_symbol * config.fec_rate
    actual_effective_bitrate = effective_bits_per_frame / frame_duration
    actual_video_bitrate = int(actual_effective_bitrate * config.bitrate_efficiency)

    # Print throughput info
    print(f"\nChannel throughput:")
    print(f"  CP length:             {actual_cp_length} (auto-derived)" if actual_cp_length != config.cp_length else f"  CP length:             {actual_cp_length}")
    print(f"  FEC rate:              {config.fec_rate:.3f}")
    print(f"  Effective bitrate:     {actual_effective_bitrate:.0f} bps")
    print(f"  Video bitrate:         {actual_video_bitrate} bps" +
          (" (auto)" if config.auto_bitrate else " (manual)"))

    # Auto-discover categories from directory structure
    print(f"\nDiscovering categories from: {uve_path}")
    category_videos = discover_categories(uve_path)

    if not category_videos:
        print("Error: No video categories found in dataset path")
        return 1

    # Apply max_videos limit if specified
    if args.max_videos:
        category_videos = {cat: videos[:args.max_videos] for cat, videos in category_videos.items()}

    categories = list(category_videos.keys())
    for cat in categories:
        print(f"  {cat}: {len(category_videos[cat])} videos")

    # Process each SNR value
    snr_sweep_results = []
    total_start_time = time.time()

    for snr_idx, snr_db in enumerate(snr_values):
        print(f"\n{'='*70}")
        print(f"SNR = {snr_db:.1f} dB ({snr_idx+1}/{len(snr_values)})")
        print(f"{'='*70}")

        # Results for this SNR
        all_results = []
        category_results = {cat: [] for cat in categories}

        # Create SNR-specific output directory
        snr_output_dir = output_dir / f"snr_{snr_db:.0f}dB"
        snr_output_dir.mkdir(parents=True, exist_ok=True)

        for category, videos in category_videos.items():
            if not videos:
                continue

            cat_output_dir = snr_output_dir / category
            cat_output_dir.mkdir(parents=True, exist_ok=True)

            for video_path in tqdm(videos, desc=f"{category}@{snr_db:.0f}dB"):
                video_name = video_path.stem

                try:
                    video_start_time = time.time()

                    # Process video
                    metrics = process_video(
                        pipeline,
                        video_path,
                        resolution=resolution,
                        bitrate_bps=actual_video_bitrate,
                        codec=config.ffmpeg_codec,
                        snr_db=snr_db,
                        fps=config.fps,
                    )

                    video_end_time = time.time()
                    video_runtime = video_end_time - video_start_time

                    result = {
                        'video_name': video_name,
                        'video_path': str(video_path),
                        'category': category,
                        'snr_db': snr_db,
                        'psnr_mean': metrics['psnr_mean'],
                        'psnr_std': metrics['psnr_std'],
                        'ssim_mean': metrics['ssim_mean'],
                        'ssim_std': metrics['ssim_std'],
                        'ber': metrics['ber'],
                        'bit_errors': metrics['bit_errors'],
                        'total_bits': metrics['total_bits'],
                        'playable': metrics['playable'],
                        'frames_decoded': metrics['frames_decoded'],
                        'frames_total': metrics['frames_total'],
                        'video_bytes': metrics['video_bytes'],
                        'runtime_seconds': video_runtime,
                        'status': 'success',
                    }

                    all_results.append(result)
                    category_results[category].append(result)

                except Exception as e:
                    print(f"  ERROR processing {video_name}: {e}")
                    all_results.append({
                        'video_name': video_name,
                        'video_path': str(video_path),
                        'category': category,
                        'snr_db': snr_db,
                        'status': 'failed',
                        'error': str(e),
                    })

        # Compute category statistics for this SNR
        category_stats = {}
        for cat, results in category_results.items():
            successful = [r for r in results if r.get('status') == 'success']
            if successful:
                playable_count = sum(1 for r in successful if r.get('playable', False))
                stats = {
                    'num_videos': len(successful),
                    'playable_count': playable_count,
                    'playable_fraction': playable_count / len(successful),
                    'psnr_mean': float(np.mean([r['psnr_mean'] for r in successful])),
                    'psnr_std': float(np.std([r['psnr_mean'] for r in successful])),
                    'ssim_mean': float(np.mean([r['ssim_mean'] for r in successful])),
                    'ssim_std': float(np.std([r['ssim_mean'] for r in successful])),
                    'ber': float(np.mean([r.get('ber', 0.0) for r in successful])),
                }
                category_stats[cat] = stats

        # Compute overall stats for this SNR
        successful_results = [r for r in all_results if r.get('status') == 'success']
        playable_count = sum(1 for r in successful_results if r.get('playable', False))
        overall_stats = {
            'snr_db': snr_db,
            'videos_successful': len(successful_results),
            'playable_count': playable_count,
            'playable_fraction': playable_count / len(successful_results) if successful_results else 0,
            'psnr_mean': float(np.mean([r['psnr_mean'] for r in successful_results])) if successful_results else 0,
            'ssim_mean': float(np.mean([r['ssim_mean'] for r in successful_results])) if successful_results else 0,
            'ber': float(np.mean([r.get('ber', 0.0) for r in successful_results])) if successful_results else 0,
        }

        snr_sweep_results.append({
            'snr_db': snr_db,
            'category_stats': category_stats,
            'overall_stats': overall_stats,
            'per_video_results': all_results,
        })

        # Save per-SNR summary
        with open(snr_output_dir / 'summary.json', 'w') as f:
            json.dump({
                'snr_db': snr_db,
                'category_stats': category_stats,
                'overall_stats': overall_stats,
                'per_video_results': all_results,
            }, f, indent=2)

        # Save per-SNR CSV
        csv_path = snr_output_dir / 'results.csv'
        with open(csv_path, 'w') as f:
            f.write('category,video_name,psnr_mean,psnr_std,ssim_mean,ssim_std,ber,playable,frames_decoded,frames_total,status\n')
            for r in all_results:
                if r.get('status') == 'success':
                    f.write(f"{r['category']},{r['video_name']},{r['psnr_mean']:.4f},{r['psnr_std']:.4f},"
                            f"{r['ssim_mean']:.6f},{r['ssim_std']:.6f},{r['ber']:.6f},"
                            f"{r['playable']},{r['frames_decoded']},{r['frames_total']},success\n")
                else:
                    f.write(f"{r.get('category', 'unknown')},{r['video_name']},0,0,0,0,0,False,0,0,failed\n")

    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time

    # Create sweep summary
    sweep_summary = {
        'run_info': {
            'timestamp': timestamp,
            'uve_path': str(uve_path),
            'output_dir': str(output_dir),
            'resolution': list(resolution),
            'codec': args.codec,
            'categories': categories,
            'max_videos': args.max_videos,
        },
        'channel_config': {
            'channel_type': args.channel_type,
            'channel_name': args.channel if args.channel_type == 'uwa' else None,
            'snr_sweep': {'min': args.snr_min, 'max': args.snr_max, 'step': args.snr_step},
            'snr_values': snr_values,
            'fec_type': args.fec,
            'fec_rate': fec_params['rate'],
            'fec_params': fec_params,
            'modulation': args.modulation,
            'video_bitrate_bps': actual_video_bitrate,
            'effective_bitrate_bps': actual_effective_bitrate,
            'actual_cp_length': actual_cp_length,
        },
        'ofdm_config': {
            'num_carriers': config.num_carriers,
            'num_ofdm_symbols': config.num_ofdm_symbols,
            'cp_length': config.cp_length,
            'pilot_period': config.pilot_period,
        },
        'runtime': {
            'total_seconds': total_runtime,
            'total_formatted': format_duration(total_runtime),
        },
        'snr_results': [r['overall_stats'] for r in snr_sweep_results],
        'category_results_by_snr': {r['snr_db']: r['category_stats'] for r in snr_sweep_results},
    }

    # Save sweep summary
    with open(output_dir / 'sweep_summary.json', 'w') as f:
        json.dump(sweep_summary, f, indent=2)

    # Save sweep CSV
    csv_path = output_dir / 'sweep_summary.csv'
    with open(csv_path, 'w') as f:
        f.write('snr_db,psnr_mean,ssim_mean,ber,playable_fraction')
        for cat in categories:
            f.write(f',{cat}_psnr,{cat}_ssim,{cat}_ber,{cat}_playable')
        f.write('\n')
        for r in snr_sweep_results:
            stats = r['overall_stats']
            f.write(f"{stats['snr_db']:.1f},{stats['psnr_mean']:.4f},{stats['ssim_mean']:.6f},"
                    f"{stats['ber']:.6f},{stats['playable_fraction']:.4f}")
            for cat in categories:
                cat_stats = r['category_stats'].get(cat, {})
                f.write(f",{cat_stats.get('psnr_mean', 0):.4f},{cat_stats.get('ssim_mean', 0):.6f},"
                        f"{cat_stats.get('ber', 0):.6f},{cat_stats.get('playable_fraction', 0):.4f}")
            f.write('\n')

    # Create plots
    create_snr_sweep_plot(snr_sweep_results, categories, output_dir / 'snr_sweep_plot.png')

    # Create category bar chart for middle SNR
    mid_idx = len(snr_sweep_results) // 2
    if snr_sweep_results:
        create_category_bar_chart(
            snr_sweep_results[mid_idx]['category_stats'],
            output_dir / f'category_comparison_snr{snr_sweep_results[mid_idx]["snr_db"]:.0f}dB.png'
        )

    # Print summary
    print(f"\n{'='*90}")
    print("SNR SWEEP COMPLETE")
    print(f"{'='*90}")
    print(f"SNR range: {args.snr_min} to {args.snr_max} dB (step {args.snr_step})")
    print(f"Total runtime: {format_duration(total_runtime)}")

    print(f"\n{'SNR':<8} {'PSNR':<10} {'SSIM':<10} {'BER':<12} {'Playable':<10}")
    print(f"{'-'*50}")
    for r in snr_sweep_results:
        stats = r['overall_stats']
        print(f"{stats['snr_db']:<8.1f} {stats['psnr_mean']:<10.2f} {stats['ssim_mean']:<10.4f} "
              f"{stats['ber']:<12.6f} {stats['playable_fraction']*100:<10.1f}%")

    # Print per-category summary at best SNR
    best_snr_result = snr_sweep_results[-1]  # Highest SNR
    print(f"\nCategory Statistics at SNR={best_snr_result['snr_db']}dB:")
    for cat, stats in best_snr_result['category_stats'].items():
        print(f"\n  {cat.upper()} ({stats['num_videos']} videos):")
        print(f"    PSNR: {stats['psnr_mean']:.2f} +/- {stats['psnr_std']:.2f} dB")
        print(f"    SSIM: {stats['ssim_mean']:.4f} +/- {stats['ssim_std']:.4f}")
        print(f"    BER:  {stats['ber']:.6f}")
        print(f"    Playable: {stats['playable_count']}/{stats['num_videos']} ({stats['playable_fraction']*100:.1f}%)")

    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - sweep_summary.json      (full sweep results)")
    print(f"  - sweep_summary.csv       (tabular sweep data)")
    print(f"  - snr_sweep_plot.png      (SNR vs metrics plot)")
    print(f"  - snr_XdB/                (per-SNR detailed results)")
    print(f"{'='*90}")

    return 0


if __name__ == '__main__':
    exit(main())
