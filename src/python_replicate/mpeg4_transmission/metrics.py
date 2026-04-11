"""
Quality metrics for MPEG4 transmission simulation.

Provides functions for computing:
- Bit Error Rate (BER)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
"""

import math
from typing import Tuple, List, Optional
from dataclasses import dataclass

import numpy as np

# Try to import cv2 for SSIM, fall back to simplified version if not available
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def compute_ber(
    original_bytes: bytes,
    received_bytes: bytes,
) -> Tuple[float, int, int]:
    """Compute Bit Error Rate between original and received bytes.

    Args:
        original_bytes: Original transmitted bytes
        received_bytes: Received bytes (possibly corrupted)

    Returns:
        Tuple of (ber, num_errors, total_bits)
    """
    # Convert to bit arrays
    original_bits = np.unpackbits(np.frombuffer(original_bytes, dtype=np.uint8))
    received_bits = np.unpackbits(np.frombuffer(received_bytes, dtype=np.uint8))

    # Handle length mismatch
    min_len = min(len(original_bits), len(received_bits))
    if min_len == 0:
        return 1.0, 0, 0

    # Count errors in overlapping portion
    errors = np.sum(original_bits[:min_len] != received_bits[:min_len])

    # Add errors for any missing/extra bits
    length_diff = abs(len(original_bits) - len(received_bits))
    errors += length_diff

    total_bits = max(len(original_bits), len(received_bits))

    ber = errors / total_bits if total_bits > 0 else 0.0

    return float(ber), int(errors), int(total_bits)


def compute_ber_from_bits(
    original_bits: np.ndarray,
    received_bits: np.ndarray,
) -> Tuple[float, int, int]:
    """Compute BER from bit arrays.

    Args:
        original_bits: Original bit array (0s and 1s)
        received_bits: Received bit array

    Returns:
        Tuple of (ber, num_errors, total_bits)
    """
    min_len = min(len(original_bits), len(received_bits))
    if min_len == 0:
        return 1.0, 0, 0

    errors = np.sum(original_bits[:min_len] != received_bits[:min_len])
    length_diff = abs(len(original_bits) - len(received_bits))
    errors += length_diff

    total_bits = max(len(original_bits), len(received_bits))
    ber = errors / total_bits if total_bits > 0 else 0.0

    return float(ber), int(errors), int(total_bits)


def img_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """Compute PSNR between two images.

    Args:
        img1: First image (any shape, values in [0, max_val])
        img2: Second image (same shape as img1)
        max_val: Maximum pixel value (1.0 for normalized, 255 for uint8)

    Returns:
        PSNR value in dB
    """
    # Normalize to [0, 1] if needed
    if max_val != 1.0:
        img1 = img1.astype(np.float64) / max_val
        img2 = img2.astype(np.float64) / max_val
    else:
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)

    if mse < 1e-10:
        return 100.0

    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr


def _ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM for single channel images using Gaussian weighting.

    Args:
        img1: First image (H, W)
        img2: Second image (H, W)

    Returns:
        SSIM value
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if HAS_CV2:
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean())
    else:
        # Simplified SSIM without windowing
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim)


def img_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """Compute SSIM between two images.

    Args:
        img1: First image (H, W) or (H, W, C) or (C, H, W)
        img2: Second image (same shape as img1)
        max_val: Maximum pixel value (1.0 for normalized, 255 for uint8)

    Returns:
        SSIM value
    """
    if not img1.shape == img2.shape:
        raise ValueError(f'Input images must have the same dimensions: {img1.shape} vs {img2.shape}')

    # Normalize to [0, 1]
    if max_val != 1.0:
        img1 = img1.astype(np.float64) / max_val
        img2 = img2.astype(np.float64) / max_val
    else:
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

    if img1.ndim == 2:
        return _ssim_single_channel(img1, img2)
    elif img1.ndim == 3:
        # Check if channels first or last
        if img1.shape[0] in (1, 3):  # Channels first (C, H, W)
            ssims = [_ssim_single_channel(img1[i], img2[i]) for i in range(img1.shape[0])]
        else:  # Channels last (H, W, C)
            ssims = [_ssim_single_channel(img1[:, :, i], img2[:, :, i]) for i in range(img1.shape[-1])]
        return float(np.mean(ssims))
    else:
        raise ValueError(f'Wrong input image dimensions: {img1.ndim}')


def compute_psnr_per_frame(
    original_frames: np.ndarray,
    received_frames: np.ndarray,
    max_val: float = 255.0,
) -> np.ndarray:
    """Compute PSNR for each frame.

    Args:
        original_frames: Original frames (N, H, W, C) or (N, H, W)
        received_frames: Received frames (same shape)
        max_val: Maximum pixel value

    Returns:
        Array of PSNR values for each frame
    """
    if original_frames.shape != received_frames.shape:
        raise ValueError(f'Frame shapes must match: {original_frames.shape} vs {received_frames.shape}')

    num_frames = min(len(original_frames), len(received_frames))
    psnr_values = np.zeros(num_frames)

    for i in range(num_frames):
        psnr_values[i] = img_psnr(original_frames[i], received_frames[i], max_val)

    return psnr_values


def compute_ssim_per_frame(
    original_frames: np.ndarray,
    received_frames: np.ndarray,
    max_val: float = 255.0,
) -> np.ndarray:
    """Compute SSIM for each frame.

    Args:
        original_frames: Original frames (N, H, W, C) or (N, H, W)
        received_frames: Received frames (same shape)
        max_val: Maximum pixel value

    Returns:
        Array of SSIM values for each frame
    """
    if original_frames.shape != received_frames.shape:
        raise ValueError(f'Frame shapes must match: {original_frames.shape} vs {received_frames.shape}')

    num_frames = min(len(original_frames), len(received_frames))
    ssim_values = np.zeros(num_frames)

    for i in range(num_frames):
        try:
            ssim_values[i] = img_ssim(original_frames[i], received_frames[i], max_val)
        except Exception:
            ssim_values[i] = 0.0  # If SSIM computation fails

    return ssim_values


@dataclass
class QualityMetrics:
    """Container for all quality metrics."""
    ber: float
    bit_errors: int
    total_bits: int

    psnr_per_frame: np.ndarray
    ssim_per_frame: np.ndarray

    @property
    def psnr_mean(self) -> float:
        return float(np.mean(self.psnr_per_frame)) if len(self.psnr_per_frame) > 0 else 0.0

    @property
    def ssim_mean(self) -> float:
        return float(np.mean(self.ssim_per_frame)) if len(self.ssim_per_frame) > 0 else 0.0

    @property
    def psnr_std(self) -> float:
        return float(np.std(self.psnr_per_frame)) if len(self.psnr_per_frame) > 0 else 0.0

    @property
    def ssim_std(self) -> float:
        return float(np.std(self.ssim_per_frame)) if len(self.ssim_per_frame) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'ber': self.ber,
            'bit_errors': self.bit_errors,
            'total_bits': self.total_bits,
            'psnr_mean': self.psnr_mean,
            'psnr_std': self.psnr_std,
            'ssim_mean': self.ssim_mean,
            'ssim_std': self.ssim_std,
        }


def compute_all_metrics(
    original_bytes: bytes,
    received_bytes: bytes,
    original_frames: np.ndarray,
    received_frames: np.ndarray,
    max_val: float = 255.0,
) -> QualityMetrics:
    """Compute all quality metrics.

    Args:
        original_bytes: Original video bytes
        received_bytes: Received video bytes
        original_frames: Original video frames (N, H, W, C)
        received_frames: Received video frames
        max_val: Maximum pixel value

    Returns:
        QualityMetrics with all computed values
    """
    ber, bit_errors, total_bits = compute_ber(original_bytes, received_bytes)

    # Compute frame metrics if frames are available
    if len(original_frames) > 0 and len(received_frames) > 0:
        # Match frame counts
        min_frames = min(len(original_frames), len(received_frames))
        orig = original_frames[:min_frames]
        recv = received_frames[:min_frames]

        psnr_per_frame = compute_psnr_per_frame(orig, recv, max_val)
        ssim_per_frame = compute_ssim_per_frame(orig, recv, max_val)
    else:
        psnr_per_frame = np.array([])
        ssim_per_frame = np.array([])

    return QualityMetrics(
        ber=ber,
        bit_errors=bit_errors,
        total_bits=total_bits,
        psnr_per_frame=psnr_per_frame,
        ssim_per_frame=ssim_per_frame,
    )
