#!/usr/bin/env python3
"""
Test script for VQ-VAE model.
Processes an entire video by splitting into chunks, reconstructs each chunk,
and outputs:
- Full reconstructed video
- PSNR/SSIM plots per frame for the entire video

Usage:
    python test_vqvae.py --ckpt path/to/checkpoint.pth.tar --video path/to/video.mp4
    python test_vqvae.py --ckpt path/to/checkpoint.pth.tar --video path/to/video.mp4 --output_dir results
    python test_vqvae.py --ckpt path/to/checkpoint.pth.tar --video path/to/video.mp4 --max_frames 300
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from videogpt.train_utils import load_model
from videogpt.models.vqvae import compute_psnr, compute_ssim


def load_full_video(video_path, resolution, max_frames=None):
    """Load and preprocess entire video."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    frame_indices = list(range(total_frames))
    frames = vr.get_batch(frame_indices).asnumpy()  # [T, H, W, C]

    # Convert to tensor and preprocess
    video = torch.from_numpy(frames).float()  # [T, H, W, C]
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

    original_h, original_w = video.shape[2], video.shape[3]

    # Resize to target resolution
    video = F.interpolate(video, size=(resolution, resolution), mode='bilinear', align_corners=False)

    # Normalize to [-0.5, 0.5]
    video = video / 255.0 - 0.5

    # Reshape to [C, T, H, W]
    video = video.permute(1, 0, 2, 3)

    return video, total_frames, len(vr), (original_h, original_w)


def compute_per_frame_metrics(original, reconstructed):
    """Compute PSNR and SSIM for each frame."""
    # original and reconstructed: [C, T, H, W]
    c, t, h, w = original.shape

    psnr_per_frame = []
    ssim_per_frame = []

    for i in range(t):
        orig_frame = original[:, i:i+1, :, :]  # [C, 1, H, W]
        recon_frame = reconstructed[:, i:i+1, :, :]

        # Add batch dimension for compute functions: [1, C, 1, H, W]
        orig_frame = orig_frame.unsqueeze(0)
        recon_frame = recon_frame.unsqueeze(0)

        psnr = compute_psnr(recon_frame, orig_frame).item()
        ssim = compute_ssim(recon_frame, orig_frame).item()

        psnr_per_frame.append(psnr)
        ssim_per_frame.append(ssim)

    return psnr_per_frame, ssim_per_frame


def process_video_in_chunks(model, video, chunk_size, device):
    """Process entire video by splitting into chunks and reconstructing each.

    Args:
        model: VQ-VAE model
        video: full video tensor [C, T, H, W]
        chunk_size: number of frames per chunk (n_frames from model)
        device: torch device

    Returns:
        reconstructed: full reconstructed video [C, T, H, W]
        psnr_per_frame: list of PSNR values
        ssim_per_frame: list of SSIM values
    """
    c, total_frames, h, w = video.shape

    reconstructed_chunks = []
    psnr_per_frame = []
    ssim_per_frame = []

    # Calculate number of chunks
    n_chunks = (total_frames + chunk_size - 1) // chunk_size

    print(f"Processing {total_frames} frames in {n_chunks} chunks of {chunk_size} frames each...")

    for i in tqdm(range(n_chunks), desc="Reconstructing"):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_frames)

        # Get chunk
        chunk = video[:, start_idx:end_idx, :, :]  # [C, chunk_len, H, W]
        actual_chunk_len = end_idx - start_idx

        # Pad if last chunk is smaller than chunk_size
        if actual_chunk_len < chunk_size:
            padding = torch.zeros(c, chunk_size - actual_chunk_len, h, w)
            chunk = torch.cat([chunk, padding], dim=1)

        # Add batch dimension and move to device
        chunk = chunk.unsqueeze(0).to(device)  # [1, C, chunk_size, H, W]

        with torch.no_grad():
            recon_chunk = model.get_reconstruction(x=chunk)

        # Remove padding if needed
        recon_chunk = recon_chunk.squeeze(0)[:, :actual_chunk_len, :, :]  # [C, actual_len, H, W]
        chunk = chunk.squeeze(0)[:, :actual_chunk_len, :, :]

        # Compute per-frame metrics for this chunk
        chunk_psnr, chunk_ssim = compute_per_frame_metrics(chunk, recon_chunk)
        psnr_per_frame.extend(chunk_psnr)
        ssim_per_frame.extend(chunk_ssim)

        reconstructed_chunks.append(recon_chunk.cpu())

    # Concatenate all chunks
    reconstructed = torch.cat(reconstructed_chunks, dim=1)  # [C, T, H, W]

    return reconstructed, psnr_per_frame, ssim_per_frame


def save_video(frames, output_path, fps=8):
    """Save frames as MP4 video using imageio."""
    try:
        import imageio
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
        print(f"Saved video to {output_path}")
    except ImportError:
        print("imageio not installed. Saving as individual frames instead.")
        output_dir = output_path.replace('.mp4', '_frames')
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(output_dir, f'frame_{i:04d}.png'))
        print(f"Saved {len(frames)} frames to {output_dir}")


def save_outputs(original, reconstructed, psnr_per_frame, ssim_per_frame, output_dir, start_frame, video_name, fps=30):
    """Save reconstructed video, comparison video, and metric plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert from [C, T, H, W] in [-0.5, 0.5] to [T, H, W, C] in [0, 255]
    orig_frames = ((original + 0.5) * 255).clamp(0, 255).permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    recon_frames = ((reconstructed + 0.5) * 255).clamp(0, 255).permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)

    # Create side-by-side comparison frames
    comparison_frames = []
    for i in range(orig_frames.shape[0]):
        combined = np.concatenate([orig_frames[i], recon_frames[i]], axis=1)
        comparison_frames.append(combined)
    comparison_frames = np.array(comparison_frames)

    # Save videos
    save_video(orig_frames, os.path.join(output_dir, f'{video_name}_original.mp4'), fps=fps)
    save_video(recon_frames, os.path.join(output_dir, f'{video_name}_reconstructed.mp4'), fps=fps)
    save_video(comparison_frames, os.path.join(output_dir, f'{video_name}_comparison.mp4'), fps=fps)

    # Create metric plots
    n_frames = len(psnr_per_frame)
    frame_indices = list(range(start_frame, start_frame + n_frames))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # PSNR plot
    axes[0].plot(frame_indices, psnr_per_frame, 'b-o', linewidth=2, markersize=6)
    axes[0].axhline(y=np.mean(psnr_per_frame), color='r', linestyle='--', label=f'Mean: {np.mean(psnr_per_frame):.2f} dB')
    axes[0].set_xlabel('Frame Index', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR per Frame', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(frame_indices[0], frame_indices[-1])

    # SSIM plot
    axes[1].plot(frame_indices, ssim_per_frame, 'g-o', linewidth=2, markersize=6)
    axes[1].axhline(y=np.mean(ssim_per_frame), color='r', linestyle='--', label=f'Mean: {np.mean(ssim_per_frame):.4f}')
    axes[1].set_xlabel('Frame Index', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM per Frame', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(frame_indices[0], frame_indices[-1])
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{video_name}_metrics.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved metrics plot to {plot_path}")

    # Also save metrics as CSV
    csv_path = os.path.join(output_dir, f'{video_name}_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('frame,psnr,ssim\n')
        for i, (psnr, ssim) in enumerate(zip(psnr_per_frame, ssim_per_frame)):
            f.write(f'{start_frame + i},{psnr:.4f},{ssim:.6f}\n')
    print(f"Saved metrics CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Test VQ-VAE reconstruction on a video')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to VQ-VAE checkpoint')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum frames to process (default: all)')
    parser.add_argument('--resolution', type=int, default=None, help='Resolution to use (default: from checkpoint, e.g. 64, 128, 256, 512)')
    parser.add_argument('--output_dir', type=str, default='vqvae_test_output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--fps', type=int, default=None, help='Output video FPS (default: same as input)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    # Get config from checkpoint
    dset_configs = ckpt['dset_configs']
    model_resolution = dset_configs['resolution']
    n_frames = dset_configs['n_frames']

    # Use specified resolution or fall back to model's trained resolution
    resolution = args.resolution if args.resolution is not None else model_resolution

    if resolution != model_resolution:
        print(f"WARNING: Model was trained at {model_resolution}x{model_resolution}, using {resolution}x{resolution}")
        print(f"         Results may be suboptimal if resolutions differ significantly.")

    print(f"Model config: trained_resolution={model_resolution}, using_resolution={resolution}, chunk_size={n_frames} frames")

    # Load model with potentially different input shape
    # We need to recreate the model with the new resolution
    from videogpt.config_model import config_model
    hp = ckpt['hp'].copy()
    # Remove input_shape if present, we'll override it
    hp.pop('input_shape', None)
    model, _ = config_model(
        configs_str='',
        input_shape=(n_frames, resolution, resolution),
        cond_types=tuple(),
        **hp
    )

    # Handle state dict loading with potential resolution mismatch
    state_dict = ckpt['state_dict']
    model_state = model.state_dict()

    # Filter and potentially interpolate mismatched keys
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state_dict[k] = v
            elif 'pos_embd' in k:
                # Interpolate positional embeddings
                target_shape = model_state[k].shape
                if len(v.shape) == 2 and len(target_shape) == 2:
                    # Shape is [seq_len, emb_dim]
                    v_interp = F.interpolate(
                        v.unsqueeze(0).unsqueeze(0),  # [1, 1, seq_len, emb_dim]
                        size=target_shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    filtered_state_dict[k] = v_interp
                    print(f"Interpolated {k}: {v.shape} -> {target_shape}")
                else:
                    print(f"Skipping {k}: shape mismatch {v.shape} vs {target_shape}")
            else:
                print(f"Skipping {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
        else:
            print(f"Skipping {k}: not in model")

    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Load full video
    print(f"Loading video from {args.video}")
    video, frames_to_process, total_frames_in_video, original_size = load_full_video(
        args.video, resolution, max_frames=args.max_frames
    )
    print(f"Video: {total_frames_in_video} total frames, processing {frames_to_process} frames")
    print(f"Original size: {original_size[1]}x{original_size[0]}, resized to: {resolution}x{resolution}")

    # Get input video FPS for output
    vr = VideoReader(args.video, ctx=cpu(0))
    input_fps = vr.get_avg_fps()
    output_fps = args.fps if args.fps else int(input_fps)
    print(f"Input FPS: {input_fps:.1f}, Output FPS: {output_fps}")

    # Process video in chunks
    reconstructed, psnr_per_frame, ssim_per_frame = process_video_in_chunks(
        model, video, n_frames, device
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"Results Summary ({frames_to_process} frames):")
    print(f"  PSNR: {np.mean(psnr_per_frame):.2f} dB (mean), {np.std(psnr_per_frame):.2f} (std)")
    print(f"  SSIM: {np.mean(ssim_per_frame):.4f} (mean), {np.std(ssim_per_frame):.4f} (std)")
    print(f"  PSNR range: [{np.min(psnr_per_frame):.2f}, {np.max(psnr_per_frame):.2f}] dB")
    print(f"  SSIM range: [{np.min(ssim_per_frame):.4f}, {np.max(ssim_per_frame):.4f}]")
    print(f"{'='*50}\n")

    # Get video name for output files
    video_name = os.path.splitext(os.path.basename(args.video))[0]

    # Save outputs (videos + plots + CSV)
    save_outputs(
        video,
        reconstructed,
        psnr_per_frame,
        ssim_per_frame,
        args.output_dir,
        0,  # start_frame is 0 since we process from beginning
        video_name,
        fps=output_fps
    )

    print(f"\nDone! All outputs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
