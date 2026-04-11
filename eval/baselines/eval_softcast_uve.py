#!/usr/bin/env python3
"""
SoftCast Evaluation Script for UVE38K Dataset with Channel Simulation.

Evaluates SoftCast analog video transmission quality across the UVE38K dataset
(clear and turbid underwater video categories) with OFDM channel simulation.

Features:
- Processes videos by category (clear/turbid) and reports per-category statistics
- Records per-video metrics (PSNR, SSIM)
- Uses analog SoftCast transmission (no digital modulation)
- Zero-forcing or MMSE equalization
- Metadata assumed perfectly recovered (ground truth)
- Supports AWGN and UWA channels

Usage:
    # AWGN channel (flat)
    python eval_softcast_uve.py --flat-channel --snr-min 0 --snr-max 30 --snr-step 5

    # UWA channel with ZF equalization
    python eval_softcast_uve.py --channel NOF1 --snr-min 0 --snr-max 30

    # UWA channel with MMSE equalization
    python eval_softcast_uve.py --channel NOF1 --equalizer mmse
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from softcast import SoftCast
from python_replicate.channel_dataset import ChannelCollection
from python_replicate.frame_preparation import FramePrepConfig
from python_replicate.ofdm_mapper import OFDMConfig
from python_replicate.softcast_integration import (
    SoftCastTransmitter,
    SoftCastTxConfig,
)
from python_replicate.softcast_pipeline import simulate_softcast_channel


# Default UVE38K dataset path
DEFAULT_UVE_PATH = "/home/khizar/Datasets/UVE38K/raw/10_sec_clips"

# Video extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}

# Channel collection settings for evaluation (consistent with eval_vqvae_uve.py)
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
    """Auto-discover categories from UVE38K directory structure."""
    category_videos = {}
    for subdir in sorted(uve_path.iterdir()):
        if subdir.is_dir():
            videos = find_video_files(subdir)
            if videos:
                category_videos[subdir.name] = videos
    return category_videos


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


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute PSNR between original and reconstructed."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute SSIM between original and reconstructed."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = np.mean(original)
    mu_y = np.mean(reconstructed)
    var_x = np.var(original)
    var_y = np.var(reconstructed)
    cov_xy = np.mean((original - mu_x) * (reconstructed - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim)


def center_crop_to_square(frames: np.ndarray) -> np.ndarray:
    """Center crop frames to square aspect ratio.

    Args:
        frames: Array of shape (T, H, W, C) or (T, H, W)

    Returns:
        Cropped array with H == W
    """
    if frames.ndim == 4:
        t, h, w, c = frames.shape
    else:
        t, h, w = frames.shape
        c = None

    crop_size = min(h, w)
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2

    if c is not None:
        return frames[:, top:top+crop_size, left:left+crop_size, :]
    else:
        return frames[:, top:top+crop_size, left:left+crop_size]


def load_video_frames(
    video_path: Path,
    resolution: int,
    max_frames: int = None,
) -> Tuple[np.ndarray, int, float]:
    """Load video and preprocess to grayscale frames in [0,1].

    Returns:
        frames: Array of shape (H, W, T) in [0, 1]
        total_frames: Total number of frames in video
        fps: Frame rate
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()

    if max_frames is not None:
        frames_to_load = min(total_frames, max_frames)
    else:
        frames_to_load = total_frames

    frame_indices = list(range(frames_to_load))
    frames = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

    # Center crop to square
    frames = center_crop_to_square(frames)

    # Resize to target resolution
    from PIL import Image
    resized_frames = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i])
        img = img.resize((resolution, resolution), Image.BILINEAR)
        resized_frames.append(np.array(img))
    frames = np.stack(resized_frames, axis=0)  # (T, H, W, C)

    # Convert to grayscale and normalize to [0, 1]
    if frames.ndim == 4 and frames.shape[3] == 3:
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        frames_gray = (0.299 * frames[:, :, :, 0] +
                       0.587 * frames[:, :, :, 1] +
                       0.114 * frames[:, :, :, 2])
    else:
        frames_gray = frames.squeeze()

    frames_gray = frames_gray.astype(np.float32) / 255.0

    # Convert from (T, H, W) to (H, W, T) for SoftCast
    frames_hwt = np.transpose(frames_gray, (1, 2, 0))

    return frames_hwt, total_frames, fps


def complex_to_real_iq(complex_data: np.ndarray, scale: float) -> np.ndarray:
    """Convert complex back to real via I/Q demod, restoring scale."""
    complex_data = complex_data * scale
    real_part = np.real(complex_data)
    imag_part = np.imag(complex_data)
    real_data = np.column_stack([real_part, imag_part]).flatten()
    return real_data


def complex_to_real_iq_overlap(complex_data: np.ndarray, scale: float, num_real_samples: int) -> np.ndarray:
    """Convert complex back to real via overlapping I/Q demod with averaging."""
    complex_data = complex_data * scale
    real_parts = np.real(complex_data)
    imag_parts = np.imag(complex_data)

    n_complex = len(complex_data)
    n_real = num_real_samples

    if n_complex == 0:
        return np.zeros(n_real)

    real_data = np.zeros(n_real)
    real_data[0] = real_parts[0]

    for i in range(1, min(n_real - 1, n_complex)):
        estimate_from_imag = imag_parts[i - 1]
        estimate_from_real = real_parts[i] if i < n_complex else imag_parts[i - 1]
        real_data[i] = (estimate_from_imag + estimate_from_real) / 2

    if n_real > 1 and n_complex > 0:
        real_data[n_real - 1] = imag_parts[min(n_complex - 1, n_real - 2)]

    return real_data


def process_video_softcast(
    frames: np.ndarray,
    tx: SoftCastTransmitter,
    channels: ChannelCollection,
    channel_name: str,
    snr_db: float,
    flat_channel: bool = False,
    equalizer: str = 'zf',
    skip_cfo: bool = False,
) -> Dict[str, Any]:
    """Process video through SoftCast pipeline with channel simulation.

    Args:
        frames: Video frames of shape (H, W, T) in [0, 1]
        tx: SoftCastTransmitter instance
        channels: ChannelCollection instance
        channel_name: Name of channel to use
        snr_db: SNR in dB
        flat_channel: If True, use flat channel (AWGN only)
        equalizer: Equalizer type ('zf' or 'mmse')
        skip_cfo: If True, skip CFO estimation/correction

    Returns:
        Dict with metrics (psnr, ssim, etc.)
    """
    H, W, T = frames.shape

    # Step 1: Encode with SoftCast + OFDM wrapping
    tx_result = tx.encode_gop(frames)
    indices, means, vars_ = tx_result.metadata_raw

    # Step 2: Pass through channel with OFDM demodulation
    dummy_metadata = torch.zeros(10, dtype=torch.complex128)

    result = simulate_softcast_channel(
        pipeline=channels.pipelines[channel_name],
        metadata_signal=dummy_metadata,
        softcast_signal=tx_result.softcast_waveforms,
        snr_db=snr_db,
        softcast_pilot_cols=tx_result.softcast_pilot_cols,
        softcast_ofdm_config=tx.ofdm_config,
        add_awgn=True,
        flat_channel=flat_channel,
        equalizer=equalizer,
        skip_cfo=skip_cfo,
    )

    # rx_softcast is now OFDM-demodulated complex symbols
    rx_complex = result.rx_softcast
    noise_var_per_symbol = result.noise_var_per_symbol.cpu().numpy()

    # Step 3: Convert complex symbols back to real (I/Q demod)
    rx_complex_np = rx_complex.cpu().numpy()

    # Undo OFDM power normalization
    rx_complex_np = rx_complex_np * tx_result.softcast_ofdm_power_scale
    noise_var_scaled = noise_var_per_symbol * (tx_result.softcast_ofdm_power_scale ** 2)

    # Reshape to tx_mat format
    chunks_per_gop = len(indices)
    chunk_size = tx_result.tx_mat_real.shape[1]
    expected_size = chunks_per_gop * chunk_size

    # Convert to real via I/Q demod
    if tx_result.overlap_iq:
        rx_real = complex_to_real_iq_overlap(
            rx_complex_np, tx_result.tx_power_scale, expected_size
        )
        mean_noise_var = np.mean(noise_var_scaled) * (tx_result.tx_power_scale ** 2)
        noise_var_per_real = np.full(expected_size, mean_noise_var * 0.5)
        if expected_size > 0:
            noise_var_per_real[0] = mean_noise_var
        if expected_size > 1:
            noise_var_per_real[-1] = mean_noise_var
    else:
        rx_real = complex_to_real_iq(rx_complex_np, tx_result.tx_power_scale)
        noise_var_per_real = np.repeat(noise_var_scaled, 2) * (tx_result.tx_power_scale ** 2) / 2

    # Trim or pad to expected size
    if rx_real.size >= expected_size:
        rx_mat = rx_real[:expected_size].reshape(chunks_per_gop, chunk_size)
        noise_var_mat = noise_var_per_real[:expected_size].reshape(chunks_per_gop, chunk_size)
    else:
        rx_real_padded = np.concatenate([rx_real, np.zeros(expected_size - rx_real.size)])
        noise_var_padded = np.concatenate([
            noise_var_per_real,
            np.full(expected_size - noise_var_per_real.size, 1e6)
        ])
        rx_mat = rx_real_padded.reshape(chunks_per_gop, chunk_size)
        noise_var_mat = noise_var_padded.reshape(chunks_per_gop, chunk_size)

    rx_power = np.mean(rx_mat**2)
    tx_power = np.mean(tx_result.tx_mat_real**2)

    # Normalize rx_mat to have same power as tx_mat
    if rx_power > 1e-10:
        power_scale = np.sqrt(tx_power / rx_power)
        rx_mat_normalized = rx_mat * power_scale
        noise_var_mat_normalized = noise_var_mat * (power_scale ** 2)
    else:
        rx_mat_normalized = rx_mat
        noise_var_mat_normalized = noise_var_mat

    # Step 4: Compute per-chunk noise variance
    noise_per_chunk = np.mean(noise_var_mat_normalized, axis=1)
    coding_noises = np.diag(noise_per_chunk)

    # Step 5: Decode with ground-truth metadata
    softcast = SoftCast()
    reconstructed = softcast.decode(
        metadata=tx_result.metadata_raw,
        data=rx_mat_normalized,
        coding_noises=coding_noises,
        frames_per_gop=T,
        power_budget=tx.tx_config.power_budget,
        x_chunks=tx.tx_config.x_chunks,
        y_chunks=tx.tx_config.y_chunks,
        x_vid=H,
        y_vid=W,
    )
    reconstructed = np.clip(reconstructed, 0, 1)

    # Compute metrics
    psnr = compute_psnr(frames, reconstructed)
    ssim = compute_ssim(frames, reconstructed)

    return {
        'psnr_mean': psnr,
        'ssim_mean': ssim,
        'pre_eq_power': result.pre_eq_power,
        'rx_mat_power': float(rx_power),
        'tx_mat_power': float(tx_power),
        'num_chunks': chunks_per_gop,
        'chunk_size': chunk_size,
    }


def create_snr_sweep_plot(
    snr_sweep_results: List[Dict],
    categories: List[str],
    output_path: Path
):
    """Create SNR sweep plot showing metrics vs SNR."""
    snr_values = [r['snr_db'] for r in snr_sweep_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'clear': 'steelblue', 'turbid': 'seagreen', 'overall': 'purple'}

    # PSNR vs SNR
    ax = axes[0]
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
    ax = axes[1]
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='SoftCast Evaluation on UVE38K Dataset with Channel Simulation'
    )

    # Dataset arguments
    parser.add_argument('--uve-path', type=str, default=DEFAULT_UVE_PATH,
                        help=f'Path to UVE38K dataset root (default: {DEFAULT_UVE_PATH})')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum videos per category (default: all)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames per video (default: all)')

    # Video settings
    parser.add_argument('--resolution', type=int, default=128,
                        help='Video resolution (square, default: 128)')

    # Channel arguments
    parser.add_argument('--channel', type=str, default='NOF1',
                        help='Channel name for UWA (e.g., NOF1, KAU1, BCH1)')
    parser.add_argument('--channel-base-dir', type=str, default='input/channels',
                        help='Base directory containing channel .mat files')
    parser.add_argument('--flat-channel', action='store_true',
                        help='Use flat channel (AWGN only, no multipath)')

    # SNR arguments
    parser.add_argument('--snr-min', type=float, default=0,
                        help='Minimum SNR for sweep (default: 0)')
    parser.add_argument('--snr-max', type=float, default=30,
                        help='Maximum SNR for sweep (default: 30)')
    parser.add_argument('--snr-step', type=float, default=5,
                        help='SNR step size for sweep (default: 5)')

    # SoftCast arguments
    parser.add_argument('--equalizer', type=str, default='zf', choices=['zf', 'mmse'],
                        help='Equalizer type (default: zf)')
    parser.add_argument('--overlap-iq', action='store_true',
                        help='Use overlapping I/Q modulation')
    parser.add_argument('--skip-cfo', action='store_true',
                        help='Skip CFO estimation/correction')
    parser.add_argument('--x-chunks', type=int, default=8,
                        help='Number of horizontal chunks (default: 8)')
    parser.add_argument('--y-chunks', type=int, default=8,
                        help='Number of vertical chunks (default: 8)')
    parser.add_argument('--power-budget', type=float, default=1.0,
                        help='Power budget for SoftCast (default: 1.0)')

    # OFDM arguments
    parser.add_argument('--num-carriers', type=int, default=64,
                        help='Number of OFDM subcarriers (default: 64)')
    parser.add_argument('--cp-length', type=int, default=63,
                        help='Cyclic prefix length (default: 63)')
    parser.add_argument('--pilot-period', type=int, default=4,
                        help='Pilot period (default: 4)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results/softcast_uve',
                        help='Base output directory (default: results/softcast_uve)')

    args = parser.parse_args()

    # Verify dataset path
    uve_path = Path(args.uve_path)
    if not uve_path.exists():
        print(f"Error: UVE38K dataset path '{uve_path}' does not exist")
        return 1

    # SNR values
    snr_values = list(np.arange(args.snr_min, args.snr_max + args.snr_step/2, args.snr_step))
    print(f"SNR sweep: {snr_values} dB")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    channel_str = 'flat' if args.flat_channel else args.channel
    run_name = f"uve_softcast_{channel_str}_{args.equalizer}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize frame config and OFDM config
    frame_config = FramePrepConfig(
        num_carriers=args.num_carriers,
        cp_length=args.cp_length,
        oversample_q=8,
        bandwidth_hz=8e3,
        fc_hz=14e3,
    )
    ofdm_config = OFDMConfig(
        num_carriers=args.num_carriers,
        cp_length=args.cp_length,
        pilot_period=args.pilot_period,
    )

    # Initialize SoftCast transmitter
    tx_config = SoftCastTxConfig(
        frames_per_sec=30.0,
        data_symbols_per_sec=8000.0,
        power_budget=args.power_budget,
        x_chunks=args.x_chunks,
        y_chunks=args.y_chunks,
        overlap_iq=args.overlap_iq,
    )
    tx = SoftCastTransmitter(
        tx_config=tx_config,
        frame_config=frame_config,
    )

    # Print configuration
    print(f"\n{'='*70}")
    print("SOFTCAST UVE38K EVALUATION")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Resolution:        {args.resolution}x{args.resolution}")
    print(f"  Channel:           {'FLAT (AWGN only)' if args.flat_channel else args.channel}")
    print(f"  Equalizer:         {args.equalizer.upper()}")
    print(f"  Overlap I/Q:       {args.overlap_iq}")
    print(f"  Skip CFO:          {args.skip_cfo}")
    print(f"  Chunks:            {args.x_chunks}x{args.y_chunks}")
    print(f"  Power budget:      {args.power_budget}")
    print(f"  OFDM:              {args.num_carriers} carriers, CP={args.cp_length}")
    print(f"  SNR sweep:         {args.snr_min} to {args.snr_max} dB (step {args.snr_step})")

    # Initialize channel collection
    print(f"\nInitializing channel: {args.channel} (mode={EVAL_RECORDING_MODE}, seed={EVAL_RECORDING_SEED})")
    channels = ChannelCollection(
        channel_names=[args.channel],
        base_dir=Path(args.channel_base_dir),
        frame_config=frame_config,
        ofdm_config=ofdm_config,
        device=torch.device('cpu'),
        recording_mode=EVAL_RECORDING_MODE,
        recording_seed=EVAL_RECORDING_SEED,
    )

    # Auto-discover categories
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

                    # Load video frames
                    frames, total_frames, fps = load_video_frames(
                        video_path, args.resolution, args.max_frames
                    )

                    # Process through SoftCast
                    metrics = process_video_softcast(
                        frames=frames,
                        tx=tx,
                        channels=channels,
                        channel_name=args.channel,
                        snr_db=snr_db,
                        flat_channel=args.flat_channel,
                        equalizer=args.equalizer,
                        skip_cfo=args.skip_cfo,
                    )

                    video_end_time = time.time()
                    video_runtime = video_end_time - video_start_time

                    result = {
                        'video_name': video_name,
                        'video_path': str(video_path),
                        'category': category,
                        'snr_db': snr_db,
                        'frames_processed': frames.shape[2],
                        'total_frames': total_frames,
                        'fps': fps,
                        'psnr_mean': metrics['psnr_mean'],
                        'ssim_mean': metrics['ssim_mean'],
                        'num_chunks': metrics['num_chunks'],
                        'runtime_seconds': video_runtime,
                        'status': 'success',
                    }

                    all_results.append(result)
                    category_results[category].append(result)

                except Exception as e:
                    print(f"  ERROR processing {video_name}: {e}")
                    import traceback
                    traceback.print_exc()
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
                stats = {
                    'num_videos': len(successful),
                    'total_frames': sum(r['frames_processed'] for r in successful),
                    'psnr_mean': float(np.mean([r['psnr_mean'] for r in successful])),
                    'psnr_std': float(np.std([r['psnr_mean'] for r in successful])),
                    'ssim_mean': float(np.mean([r['ssim_mean'] for r in successful])),
                    'ssim_std': float(np.std([r['ssim_mean'] for r in successful])),
                }
                category_stats[cat] = stats

        # Compute overall stats for this SNR
        successful_results = [r for r in all_results if r.get('status') == 'success']
        overall_stats = {
            'snr_db': snr_db,
            'videos_successful': len(successful_results),
            'psnr_mean': float(np.mean([r['psnr_mean'] for r in successful_results])) if successful_results else 0,
            'ssim_mean': float(np.mean([r['ssim_mean'] for r in successful_results])) if successful_results else 0,
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
            f.write('category,video_name,psnr_mean,ssim_mean,frames,status\n')
            for r in all_results:
                if r.get('status') == 'success':
                    f.write(f"{r['category']},{r['video_name']},{r['psnr_mean']:.4f},"
                            f"{r['ssim_mean']:.6f},{r['frames_processed']},success\n")
                else:
                    f.write(f"{r.get('category', 'unknown')},{r['video_name']},0,0,0,failed\n")

    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time

    # Create sweep summary
    sweep_summary = {
        'run_info': {
            'timestamp': timestamp,
            'uve_path': str(uve_path),
            'output_dir': str(output_dir),
            'resolution': args.resolution,
            'categories': categories,
            'max_videos': args.max_videos,
            'max_frames': args.max_frames,
        },
        'channel_config': {
            'channel_name': args.channel,
            'flat_channel': args.flat_channel,
            'equalizer': args.equalizer,
            'snr_sweep': {'min': args.snr_min, 'max': args.snr_max, 'step': args.snr_step},
            'snr_values': snr_values,
        },
        'softcast_config': {
            'x_chunks': args.x_chunks,
            'y_chunks': args.y_chunks,
            'power_budget': args.power_budget,
            'overlap_iq': args.overlap_iq,
            'skip_cfo': args.skip_cfo,
        },
        'ofdm_config': {
            'num_carriers': args.num_carriers,
            'cp_length': args.cp_length,
            'pilot_period': args.pilot_period,
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
        f.write('snr_db,psnr_mean,ssim_mean')
        for cat in categories:
            f.write(f',{cat}_psnr,{cat}_ssim')
        f.write('\n')
        for r in snr_sweep_results:
            stats = r['overall_stats']
            f.write(f"{stats['snr_db']:.1f},{stats['psnr_mean']:.4f},{stats['ssim_mean']:.6f}")
            for cat in categories:
                cat_stats = r['category_stats'].get(cat, {})
                f.write(f",{cat_stats.get('psnr_mean', 0):.4f},{cat_stats.get('ssim_mean', 0):.6f}")
            f.write('\n')

    # Create plots
    create_snr_sweep_plot(snr_sweep_results, categories, output_dir / 'snr_sweep_plot.png')

    # Create category bar chart for highest SNR
    if snr_sweep_results:
        create_category_bar_chart(
            snr_sweep_results[-1]['category_stats'],
            output_dir / f'category_comparison_snr{snr_sweep_results[-1]["snr_db"]:.0f}dB.png'
        )

    # Print summary
    print(f"\n{'='*90}")
    print("SNR SWEEP COMPLETE")
    print(f"{'='*90}")
    print(f"SNR range: {args.snr_min} to {args.snr_max} dB (step {args.snr_step})")
    print(f"Total runtime: {format_duration(total_runtime)}")

    print(f"\n{'SNR':<8} {'PSNR':<10} {'SSIM':<10}")
    print(f"{'-'*30}")
    for r in snr_sweep_results:
        stats = r['overall_stats']
        print(f"{stats['snr_db']:<8.1f} {stats['psnr_mean']:<10.2f} {stats['ssim_mean']:<10.4f}")

    # Print per-category summary at best SNR
    best_snr_result = snr_sweep_results[-1]
    print(f"\nCategory Statistics at SNR={best_snr_result['snr_db']}dB:")
    for cat, stats in best_snr_result['category_stats'].items():
        print(f"\n  {cat.upper()} ({stats['num_videos']} videos, {stats['total_frames']} frames):")
        print(f"    PSNR: {stats['psnr_mean']:.2f} +/- {stats['psnr_std']:.2f} dB")
        print(f"    SSIM: {stats['ssim_mean']:.4f} +/- {stats['ssim_std']:.4f}")

    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - sweep_summary.json      (full sweep results)")
    print(f"  - sweep_summary.csv       (tabular sweep data)")
    print(f"  - snr_sweep_plot.png      (SNR vs metrics plot)")
    print(f"  - snr_XdB/                (per-SNR detailed results)")
    print(f"{'='*90}")

    return 0


if __name__ == '__main__':
    exit(main())
