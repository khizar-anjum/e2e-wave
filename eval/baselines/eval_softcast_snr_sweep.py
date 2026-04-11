#!/usr/bin/env python3
"""
SoftCast SNR Sweep Evaluation Script

Evaluates SoftCast video transmission over underwater acoustic channel at various SNR levels.
Measures PSNR and SSIM for each SNR level and outputs results to CSV.

Usage:
    python eval_softcast_snr_sweep.py \
        --video-root /path/to/ProcessedDataset \
        --channel-base /path/to/input/channels \
        --channel NOF1 \
        --output results/softcast_snr_sweep.csv
"""

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import video dataset from training script
from train_wave_bank_watermark_replicate import VideoDataset

# Import SoftCast integration
from python_replicate.softcast_integration import (
    SoftCastTransmitter,
    SoftCastTxConfig,
    SoftCastTxResult,
)
from python_replicate.channel_dataset import ChannelCollection
from python_replicate.frame_preparation import FramePrepConfig
from python_replicate.ofdm_mapper import OFDMConfig

# Import SoftCast decoder
from softcast import SoftCast


# Default configuration
DEFAULT_SNR_LEVELS = [0, 5, 10, 15, 20, 25, 30]
DEFAULT_RESOLUTION = 64
DEFAULT_FRAMES_PER_GOP = 8
DEFAULT_MAX_VIDEOS = 50
DEFAULT_CHANNEL = "NOF1"


@dataclass
class EvalConfig:
    """Configuration for SNR sweep evaluation."""
    video_root: str
    channel_base: str
    channel_name: str
    output_csv: str
    snr_levels: List[float]
    resolution: int
    frames_per_gop: int
    max_videos: int
    bandwidth_hz: float = 8e3
    fc_hz: float = 14e3
    use_flat_channel: bool = False
    device: str = "cpu"


@dataclass
class SNRResult:
    """Results for a single SNR level."""
    snr_db: float
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    num_samples: int


def parse_args() -> EvalConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SoftCast SNR Sweep Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-root",
        type=str,
        required=True,
        help="Root directory containing video dataset (with train/test subdirs)",
    )
    parser.add_argument(
        "--channel-base",
        type=str,
        default="/home/khizar/Temp/python_watermark/input/channels",
        help="Base directory containing channel .mat files",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=DEFAULT_CHANNEL,
        help="Channel name to use (e.g., NOF1, NCS1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="softcast_snr_sweep.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=0,
        help="Minimum SNR in dB",
    )
    parser.add_argument(
        "--snr-max",
        type=float,
        default=30,
        help="Maximum SNR in dB",
    )
    parser.add_argument(
        "--snr-step",
        type=float,
        default=5,
        help="SNR step size in dB",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Video resolution (square)",
    )
    parser.add_argument(
        "--frames-per-gop",
        type=int,
        default=DEFAULT_FRAMES_PER_GOP,
        help="Number of frames per GOP",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=DEFAULT_MAX_VIDEOS,
        help="Maximum number of videos to evaluate",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=8e3,
        help="Signal bandwidth in Hz",
    )
    parser.add_argument(
        "--fc",
        type=float,
        default=14e3,
        help="Carrier frequency in Hz",
    )
    parser.add_argument(
        "--flat-channel",
        action="store_true",
        help="Use flat (identity) channel instead of multipath",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Generate SNR levels
    snr_levels = list(np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step))

    return EvalConfig(
        video_root=args.video_root,
        channel_base=args.channel_base,
        channel_name=args.channel,
        output_csv=args.output,
        snr_levels=snr_levels,
        resolution=args.resolution,
        frames_per_gop=args.frames_per_gop,
        max_videos=args.max_videos,
        bandwidth_hz=args.bandwidth,
        fc_hz=args.fc,
        use_flat_channel=args.flat_channel,
        device=args.device,
    )


def video_tensor_to_softcast_format(video: torch.Tensor) -> np.ndarray:
    """Convert video tensor from VideoDataset format to SoftCast format.

    Args:
        video: Tensor of shape (C, T, H, W) in range [-1, 1]

    Returns:
        numpy array of shape (H, W, T) in range [0, 1], grayscale
    """
    # video shape: (C, T, H, W) in [-1, 1]
    # Convert to [0, 1]
    video_01 = (video + 1) / 2
    video_01 = video_01.clamp(0, 1)

    # Convert to grayscale if RGB (average channels)
    if video_01.shape[0] == 3:
        video_gray = video_01.mean(dim=0)  # (T, H, W)
    else:
        video_gray = video_01[0]  # (T, H, W)

    # Rearrange to (H, W, T)
    video_hwt = video_gray.permute(1, 2, 0)  # (H, W, T)

    return video_hwt.cpu().numpy()


def softcast_output_to_tensor(
    reconstructed: np.ndarray,
    target_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Convert SoftCast reconstructed output back to tensor format.

    Args:
        reconstructed: numpy array (H, W, T) in [0, 1]
        target_shape: Target shape (C, T, H, W)

    Returns:
        Tensor of shape (C, T, H, W) in [0, 1]
    """
    # reconstructed is (H, W, T), convert to (T, H, W)
    recon_thw = np.transpose(reconstructed, (2, 0, 1))

    # Convert to tensor
    recon_tensor = torch.from_numpy(recon_thw).float()

    # Add channel dimension and expand to match target
    C = target_shape[0]
    recon_tensor = recon_tensor.unsqueeze(0)  # (1, T, H, W)
    if C > 1:
        recon_tensor = recon_tensor.expand(C, -1, -1, -1)  # (C, T, H, W)

    return recon_tensor.clamp(0, 1)


def decode_softcast_with_gt_metadata(
    metadata: Tuple,
    rx_waveforms: torch.Tensor,
    noise_power: float,
    tx_config: SoftCastTxConfig,
    frames_per_gop: int,
    video_shape: Tuple[int, int],
    tx_power_scale: float = 1.0,
) -> np.ndarray:
    """Decode SoftCast using ground-truth metadata.

    Args:
        metadata: (indices, means, vars_) from encoder
        rx_waveforms: Received complex waveforms
        noise_power: Estimated noise power
        tx_config: Transmitter configuration
        frames_per_gop: Number of frames
        video_shape: (height, width) of video
        tx_power_scale: Power normalization factor from TX (to restore scale)

    Returns:
        Reconstructed frames as numpy array (H, W, T)
    """
    indices, means, vars_ = metadata

    # Convert received complex waveforms back to real via I/Q demod
    rx_complex = rx_waveforms.cpu().numpy()
    real_part = np.real(rx_complex)
    imag_part = np.imag(rx_complex)
    rx_real = np.column_stack([real_part, imag_part]).flatten()

    # Restore original scale (undo TX power normalization)
    rx_real = rx_real * tx_power_scale

    # Scale noise power accordingly (noise also gets scaled by tx_power_scale)
    noise_power = noise_power * (tx_power_scale ** 2)

    # Reshape to tx_mat format
    chunks_per_gop = len(indices)
    if chunks_per_gop == 0:
        return np.zeros((video_shape[0], video_shape[1], frames_per_gop))

    chunk_size = rx_real.shape[0] // chunks_per_gop if chunks_per_gop > 0 else 0
    if chunk_size > 0:
        rx_mat = rx_real[:chunks_per_gop * chunk_size].reshape(chunks_per_gop, chunk_size)
    else:
        return np.zeros((video_shape[0], video_shape[1], frames_per_gop))

    # Prepare noise covariance matrix
    coding_noises = np.eye(rx_mat.shape[0]) * noise_power

    # Decode with SoftCast
    softcast = SoftCast()
    try:
        reconstructed = softcast.decode(
            metadata=(indices, means, vars_),
            data=rx_mat,
            coding_noises=coding_noises,
            frames_per_gop=frames_per_gop,
            power_budget=tx_config.power_budget,
            x_chunks=tx_config.x_chunks,
            y_chunks=tx_config.y_chunks,
            x_vid=video_shape[0],
            y_vid=video_shape[1],
        )
    except Exception as e:
        warnings.warn(f"SoftCast decode failed: {e}")
        reconstructed = np.zeros((video_shape[0], video_shape[1], frames_per_gop))

    return reconstructed


def evaluate_at_snr(
    dataloader: DataLoader,
    channels: ChannelCollection,
    channel_name: str,
    tx: SoftCastTransmitter,
    tx_config: SoftCastTxConfig,
    snr_db: float,
    frames_per_gop: int,
    device: torch.device,
    use_flat_channel: bool = False,
) -> Tuple[List[float], List[float]]:
    """Evaluate all videos at a specific SNR level.

    Returns:
        Tuple of (psnr_list, ssim_list)
    """
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    psnr_list = []
    ssim_list = []

    for batch in dataloader:
        video = batch["video"]  # (B, C, T, H, W) or (C, T, H, W)

        # Handle batch dimension
        if video.dim() == 4:
            video = video.unsqueeze(0)

        for b in range(video.shape[0]):
            single_video = video[b]  # (C, T, H, W)
            C, T, H, W = single_video.shape

            # Convert to SoftCast format
            frames_np = video_tensor_to_softcast_format(single_video)

            # Limit to frames_per_gop
            if frames_np.shape[2] > frames_per_gop:
                frames_np = frames_np[:, :, :frames_per_gop]

            try:
                # Encode with SoftCast
                tx_result = tx.encode_gop(frames_np)

                # Transmit through channel
                result = channels.simulate_softcast(
                    channel_name=channel_name,
                    metadata_signal=tx_result.metadata_packet,
                    softcast_signal=tx_result.softcast_waveforms,
                    snr_db=snr_db,
                    add_awgn=True,
                    flat_channel=use_flat_channel,
                )

                # Decode using ground-truth metadata
                reconstructed = decode_softcast_with_gt_metadata(
                    metadata=tx_result.metadata_raw,
                    rx_waveforms=result.rx_softcast,
                    noise_power=result.estimated_noise_power,
                    tx_config=tx_config,
                    frames_per_gop=frames_np.shape[2],
                    video_shape=(H, W),
                    tx_power_scale=tx_result.tx_power_scale,
                )

                # Convert back to tensor format for metrics
                recon_tensor = softcast_output_to_tensor(
                    reconstructed, (C, frames_np.shape[2], H, W)
                ).to(device)

                # Original video in [0, 1]
                original = ((single_video[:, :frames_np.shape[2]] + 1) / 2).clamp(0, 1).to(device)

                # Reshape for per-frame metrics: (T, C, H, W)
                original_frames = original.permute(1, 0, 2, 3)
                recon_frames = recon_tensor.permute(1, 0, 2, 3)

                # Compute metrics
                psnr_val = psnr_metric(recon_frames, original_frames)
                ssim_val = ssim_metric(recon_frames, original_frames)

                psnr_list.append(psnr_val.item())
                ssim_list.append(ssim_val.item())

                psnr_metric.reset()
                ssim_metric.reset()

            except Exception as e:
                warnings.warn(f"Error processing video at SNR {snr_db}: {e}")
                continue

    return psnr_list, ssim_list


def run_snr_sweep(config: EvalConfig) -> List[SNRResult]:
    """Run full SNR sweep evaluation.

    Args:
        config: Evaluation configuration

    Returns:
        List of SNRResult for each SNR level
    """
    device = torch.device(config.device)

    print("=" * 60)
    print("SoftCast SNR Sweep Evaluation")
    print("=" * 60)
    print(f"Video root: {config.video_root}")
    print(f"Channel: {config.channel_name}")
    print(f"SNR levels: {config.snr_levels} dB")
    print(f"Resolution: {config.resolution}x{config.resolution}")
    print(f"Frames per GOP: {config.frames_per_gop}")
    print(f"Max videos: {config.max_videos}")
    print(f"Bandwidth: {config.bandwidth_hz/1000:.1f} kHz")
    print(f"Carrier freq: {config.fc_hz/1000:.1f} kHz")
    print(f"Flat channel: {config.use_flat_channel}")
    print("=" * 60)

    # Initialize video dataset
    print("\nLoading video dataset...")
    dataset = VideoDataset(
        data_folder=config.video_root,
        sequence_length=config.frames_per_gop,
        train=False,
        resolution=config.resolution,
        max_length=config.max_videos,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} video clips")

    # Initialize frame and OFDM configs
    frame_config = FramePrepConfig(
        num_carriers=64,
        cp_length=30,
        oversample_q=8,
        bandwidth_hz=config.bandwidth_hz,
        fc_hz=config.fc_hz,
        train_length=0,
    )
    ofdm_config = OFDMConfig(
        num_carriers=64,
        cp_length=30,
        pilot_period=4,
    )

    # Check channel exists
    channel_path = Path(config.channel_base) / config.channel_name / "mat" / f"{config.channel_name}_001.mat"
    if not channel_path.exists():
        raise FileNotFoundError(f"Channel file not found: {channel_path}")

    # Initialize channel collection
    print(f"Initializing channel: {config.channel_name}")
    channels = ChannelCollection(
        channel_names=[config.channel_name],
        base_dir=Path(config.channel_base),
        frame_config=frame_config,
        ofdm_config=ofdm_config,
        device=device,
    )

    # Initialize SoftCast transmitter
    tx_config = SoftCastTxConfig(
        frames_per_sec=30.0,
        data_symbols_per_sec=8000.0,
        power_budget=1.0,
        x_chunks=8,
        y_chunks=8,
        metadata_modulation='QPSK',
        use_fec=False,
    )
    tx = SoftCastTransmitter(tx_config=tx_config, frame_config=frame_config, device=device)

    # Run SNR sweep
    results = []
    print("\nRunning SNR sweep...")
    print("-" * 60)

    for snr_db in config.snr_levels:
        print(f"\nEvaluating at SNR = {snr_db:.1f} dB...")

        psnr_list, ssim_list = evaluate_at_snr(
            dataloader=dataloader,
            channels=channels,
            channel_name=config.channel_name,
            tx=tx,
            tx_config=tx_config,
            snr_db=snr_db,
            frames_per_gop=config.frames_per_gop,
            device=device,
            use_flat_channel=config.use_flat_channel,
        )

        if psnr_list:
            result = SNRResult(
                snr_db=snr_db,
                psnr_mean=np.mean(psnr_list),
                psnr_std=np.std(psnr_list),
                ssim_mean=np.mean(ssim_list),
                ssim_std=np.std(ssim_list),
                num_samples=len(psnr_list),
            )
            results.append(result)
            print(f"  PSNR: {result.psnr_mean:.2f} ± {result.psnr_std:.2f} dB")
            print(f"  SSIM: {result.ssim_mean:.4f} ± {result.ssim_std:.4f}")
            print(f"  Samples: {result.num_samples}")
        else:
            print(f"  No valid samples at SNR = {snr_db} dB")

    return results


def save_results_csv(results: List[SNRResult], output_path: str):
    """Save results to CSV file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'snr_db', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std', 'num_samples'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'snr_db': r.snr_db,
                'psnr_mean': r.psnr_mean,
                'psnr_std': r.psnr_std,
                'ssim_mean': r.ssim_mean,
                'ssim_std': r.ssim_std,
                'num_samples': r.num_samples,
            })

    print(f"\nResults saved to: {output_path}")


def print_results_table(results: List[SNRResult]):
    """Print results in a formatted table."""
    print("\n" + "=" * 60)
    print("SNR Sweep Results")
    print("=" * 60)
    print(f"{'SNR (dB)':>10} {'PSNR (dB)':>15} {'SSIM':>15} {'Samples':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r.snr_db:>10.1f} {r.psnr_mean:>7.2f} ± {r.psnr_std:<5.2f} "
              f"{r.ssim_mean:>7.4f} ± {r.ssim_std:<6.4f} {r.num_samples:>10}")
    print("=" * 60)


def main():
    """Main entry point."""
    config = parse_args()

    try:
        results = run_snr_sweep(config)

        if results:
            print_results_table(results)
            save_results_csv(results, config.output_csv)
        else:
            print("No results to save.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
