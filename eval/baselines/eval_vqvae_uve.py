#!/usr/bin/env python3
"""
VQ-VAE Evaluation Script for UVE38K Dataset with Channel Simulation.

Evaluates VQ-VAE video reconstruction quality across the UVE38K dataset
(clear and turbid underwater video categories) with optional channel simulation.

Features:
- Processes videos by category (clear/turbid) and reports per-category statistics
- Records per-video and per-frame metrics (PSNR, SSIM)
- Computes semantic relevance based on codebook L2 distances
- Supports multiple channel types (AWGN, UWA channels)
- Supports configurable FEC and modulation (BPSK/QPSK)

Usage:
    # No channel (direct reconstruction)
    python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_*.tar

    # With AWGN channel
    python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_*.tar --channel-type awgn --snr 15

    # With UWA channel and FEC
    python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_*.tar --channel-type uwa --channel NOF1 \
        --snr 10 --fec dvbs2_ldpc --fec-rate 0.25 --modulation QPSK
"""

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm

from videogpt.config_model import config_model
from videogpt.models.vqvae import compute_psnr, compute_ssim

# Channel simulation imports
from python_replicate.fec_codec import get_fec_codec, PassthroughFEC
from python_replicate.frame_preparation import FramePrepConfig
from python_replicate.ofdm_mapper import OFDMMapper, OFDMConfig
from python_replicate.channel_dataset import FrameAssembler, ChannelCollection
from python_replicate.channel_replay import load_channel_sounding, replay_filter
from python_replicate.signal_utils import root_raised_cosine, upfirdn_torch


# Channel collection settings for evaluation
EVAL_RECORDING_MODE = "random"  # "first", "random", or "fixed"
EVAL_RECORDING_SEED = 123


# Default UVE38K dataset path
DEFAULT_UVE_PATH = os.environ.get("E2E_WAVE_UVE_DIR", "data/uve38k/10_sec_clips")

# Video extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


def compute_relevance_matrix(codebook: torch.Tensor) -> torch.Tensor:
    """
    Compute semantic relevance matrix from codebook geometry.
    Uses L2 distance (consistent with VQ-VAE quantization).

    Args:
        codebook: Tensor of shape [n_codebooks, codes_per_book, embedding_dim]
                  or [codes_per_book, embedding_dim] for single codebook

    Returns:
        relevance: Tensor of shape [n_codebooks, codes_per_book, codes_per_book]
                   or [codes_per_book, codes_per_book] for single codebook
                   where relevance[..., i, j] = 1.0 means i==j (perfect match)
                   and relevance[..., i, j] = 0.0 means i and j are maximally distant
    """
    if codebook.dim() == 2:
        # Single codebook: [codes_per_book, embedding_dim]
        norm = (codebook ** 2).sum(dim=1, keepdim=True)
        dist_sq = norm + norm.t() - 2 * torch.matmul(codebook, codebook.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)

        d_min = dist_sq.min()
        d_max = dist_sq.max()
        dist_norm = (dist_sq - d_min) / (d_max - d_min + 1e-9)
        relevance = 1.0 - dist_norm
        return relevance

    # Multiple codebooks: [n_codebooks, codes_per_book, embedding_dim]
    n_codebooks, codes_per_book, embedding_dim = codebook.shape

    relevance_matrices = []
    for cb_idx in range(n_codebooks):
        cb = codebook[cb_idx]  # [codes_per_book, embedding_dim]
        norm = (cb ** 2).sum(dim=1, keepdim=True)
        dist_sq = norm + norm.t() - 2 * torch.matmul(cb, cb.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)

        d_min = dist_sq.min()
        d_max = dist_sq.max()
        dist_norm = (dist_sq - d_min) / (d_max - d_min + 1e-9)
        relevance = 1.0 - dist_norm
        relevance_matrices.append(relevance)

    return torch.stack(relevance_matrices, dim=0)  # [n_codebooks, codes_per_book, codes_per_book]


def compute_token_relevance(
    tx_indices: torch.Tensor,
    rx_indices: torch.Tensor,
    relevance_matrix: torch.Tensor
) -> Tuple[float, int, int]:
    """
    Compute average relevance between transmitted and received tokens.

    Args:
        tx_indices: Original transmitted indices, shape (..., n_codebooks) or flat
        rx_indices: Received indices after channel, same shape as tx_indices
        relevance_matrix: Precomputed relevance matrix
                          [n_codebooks, codes_per_book, codes_per_book] or
                          [codes_per_book, codes_per_book] for single codebook

    Returns:
        avg_relevance: Average relevance score (0.0 to 1.0)
        correct_tokens: Number of exactly matched tokens
        total_tokens: Total number of tokens
    """
    tx_flat = tx_indices.flatten().long()
    rx_flat = rx_indices.flatten().long()

    if relevance_matrix.dim() == 2:
        # Single codebook: [codes_per_book, codes_per_book]
        relevances = relevance_matrix[tx_flat, rx_flat]
    else:
        # Multiple codebooks: [n_codebooks, codes_per_book, codes_per_book]
        # Indices have shape (..., n_codebooks) - the last dim is the codebook index
        n_codebooks = relevance_matrix.shape[0]

        # Reshape to (..., n_codebooks)
        original_shape = tx_indices.shape
        if original_shape[-1] == n_codebooks:
            # Indices are in shape (..., n_codebooks)
            tx_reshaped = tx_indices.view(-1, n_codebooks)  # [N, n_codebooks]
            rx_reshaped = rx_indices.view(-1, n_codebooks)  # [N, n_codebooks]

            relevances_list = []
            for cb_idx in range(n_codebooks):
                cb_relevance = relevance_matrix[cb_idx]  # [codes_per_book, codes_per_book]
                tx_cb = tx_reshaped[:, cb_idx].long()
                rx_cb = rx_reshaped[:, cb_idx].long()
                rel = cb_relevance[tx_cb, rx_cb]
                relevances_list.append(rel)

            # Average across all codebooks and positions
            relevances = torch.stack(relevances_list, dim=1).mean(dim=1)
        else:
            # Flat indices - assume single codebook or average all
            # Use first codebook's matrix
            relevances = relevance_matrix[0][tx_flat, rx_flat]

    avg_relevance = relevances.mean().item()

    # Count exact matches
    correct_tokens = int((tx_flat == rx_flat).sum().item())
    total_tokens = tx_flat.numel()

    return avg_relevance, correct_tokens, total_tokens


def create_fec_codec_from_rate(fec_type: str, target_rate: float, fec_repetitions: int = 3):
    """Create FEC codec with specified target rate."""
    config = {}

    if fec_type == 'none':
        return PassthroughFEC(), 1.0, config

    if fec_type == 'repetition':
        if target_rate is not None:
            reps = max(1, int(round(1.0 / target_rate)))
            if reps % 2 == 0:
                reps += 1
        else:
            reps = fec_repetitions
        codec = get_fec_codec('repetition', repetitions=reps)
        config['repetitions'] = reps
        return codec, codec.rate, config

    if fec_type == 'dvbs2_ldpc':
        try:
            from python_replicate.aff3ct_codecs import DVBS2LDPCCodec
            actual_rate, k, n = DVBS2LDPCCodec.find_code(target_rate or 0.5, frame='short')
            codec = DVBS2LDPCCodec(k=k, n=n)
            config['k'] = k
            config['n'] = n
            return codec, codec.rate, config
        except ImportError:
            print("Warning: DVB-S2 LDPC not available, falling back to repetition")
            return create_fec_codec_from_rate('repetition', target_rate, fec_repetitions)

    if fec_type == 'polar':
        try:
            if target_rate is not None:
                n = 1024
                k = max(64, int(n * target_rate))
                k = min(k, n - 64)
            else:
                k, n = 512, 1024
            codec = get_fec_codec('polar', k=k, n=n)
            config['k'] = k
            config['n'] = n
            return codec, codec.rate, config
        except ImportError:
            print("Warning: Polar codec not available, falling back to repetition")
            return create_fec_codec_from_rate('repetition', target_rate, fec_repetitions)

    if fec_type == 'turbo':
        try:
            from python_replicate.aff3ct_codecs import TurboCodec
            valid_k = TurboCodec.valid_k_values()
            k = valid_k[len(valid_k) // 2]
            codec = TurboCodec(k=k)
            config['k'] = k
            return codec, codec.rate, config
        except ImportError:
            print("Warning: Turbo codec not available, falling back to repetition")
            return create_fec_codec_from_rate('repetition', target_rate, fec_repetitions)

    if fec_type == 'ldpc':
        try:
            alist_path = 'py_aff3ct/lib/aff3ct/conf/dec/LDPC/CCSDS_64_128.alist'
            codec = get_fec_codec('ldpc', alist_path=alist_path)
            config['alist_path'] = alist_path
            return codec, codec.rate, config
        except ImportError:
            print("Warning: LDPC codec not available, falling back to repetition")
            return create_fec_codec_from_rate('repetition', target_rate, fec_repetitions)

    return PassthroughFEC(), 1.0, config


class ChannelSimulator:
    """Simulates transmission of VQ-VAE indices through an OFDM channel with FEC."""

    DEFAULT_BANDWIDTH_HZ = 8000.0
    DEFAULT_FC_HZ = 14000.0
    DEFAULT_CP_LENGTH = 63

    def __init__(
        self,
        codes_per_book: int = 1024,
        fec_type: str = 'none',
        fec_rate: Optional[float] = None,
        modulation: str = 'QPSK',
        channel_type: str = 'awgn',
        channel_name: str = 'NOF1',
        channel_base_dir: Path = None,
        num_carriers: int = 64,
        num_ofdm_symbols: int = 16,
        pilot_period: int = 4,
        device: torch.device = None,
        allow_reference_filter: bool = False,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.allow_reference_filter = allow_reference_filter
        self.codes_per_book = codes_per_book
        self.bits_per_index = int(math.ceil(math.log2(codes_per_book)))
        self.modulation = modulation
        self.channel_type = channel_type
        self.channel_name = channel_name
        self.num_carriers = num_carriers

        self.fec, self.actual_fec_rate, self.fec_config = create_fec_codec_from_rate(
            fec_type, fec_rate, fec_repetitions=3
        )
        self.fec_type = fec_type

        # Initialize channel using ChannelCollection for UWA
        self.channel = None
        self.channel_collection = None
        self.channel_pipeline = None

        # For UWA channels, load channel data first to derive correct cp_length
        # before creating expensive FrameAssembler/OFDMMapper objects
        selected_channel_path = None
        if channel_type == 'uwa' and channel_base_dir is not None:
            try:
                # Find channel recordings and select one based on mode/seed
                mat_dir = Path(channel_base_dir) / channel_name / "mat"
                paths = sorted(mat_dir.glob(f"{channel_name}_*.mat"))
                if not paths:
                    raise FileNotFoundError(f"No channel files found in {mat_dir}")

                # Select recording based on mode
                if EVAL_RECORDING_MODE == "first":
                    selected_channel_path = paths[0]
                elif EVAL_RECORDING_MODE == "fixed" or EVAL_RECORDING_MODE == "random":
                    rng = random.Random(EVAL_RECORDING_SEED) if EVAL_RECORDING_SEED is not None else random
                    selected_channel_path = paths[rng.randrange(len(paths))]
                else:
                    selected_channel_path = paths[0]

                # Load channel data (cheap - just loads .mat file)
                from python_replicate.channel_replay import load_channel_sounding
                self.channel = load_channel_sounding(selected_channel_path)
                self.channel.h = self.channel.h.to(self.device)
                print(f"Loaded channel: {channel_name} (mode={EVAL_RECORDING_MODE}, seed={EVAL_RECORDING_SEED})")
                print(f"  Recording: {selected_channel_path.name}")

            except (FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not load channel {channel_name}: {e}")
                print("Falling back to AWGN")

        # Now derive system params with channel loaded
        bandwidth_hz, fc_hz = self._derive_system_params()
        self.bandwidth_hz = bandwidth_hz
        self.fc_hz = fc_hz
        cp_length = self._derive_cp_length()
        self.cp_length = cp_length

        # Create configs with CORRECT cp_length (no more temp configs!)
        self.frame_config = FramePrepConfig(
            num_carriers=num_carriers,
            cp_length=cp_length,
            modulation_order=2 if modulation == 'BPSK' else 4,
            oversample_q=8,
            num_ofdm_symbols=num_ofdm_symbols,
            bandwidth_hz=bandwidth_hz,
            fc_hz=fc_hz,
            rolloff=0.25,
            sync_length=500,
            sc_length=128,
            train_length=0,
            span=8,
        )

        self.ofdm_config = OFDMConfig(
            num_carriers=num_carriers,
            cp_length=cp_length,
            pilot_period=pilot_period,
        )

        self.ofdm_mapper = OFDMMapper(self.ofdm_config)
        self.frame_assembler = FrameAssembler(self.frame_config, device=self.device)

        # Create ChannelCollection with correct configs (lazy-loaded, no pipeline created yet)
        if channel_type == 'uwa' and channel_base_dir is not None and self.channel is not None:
            self.channel_collection = ChannelCollection(
                channel_names=[channel_name],
                base_dir=Path(channel_base_dir),
                frame_config=self.frame_config,
                ofdm_config=self.ofdm_config,
                device=self.device,
                recording_mode=EVAL_RECORDING_MODE,
                recording_seed=EVAL_RECORDING_SEED,
            )
            # Pipeline will be created lazily on first get_pipeline() call if needed
        self.fs = self.frame_config.oversample_q * self.frame_config.bandwidth_hz

        # Statistics
        self.total_bits_tx = 0
        self.total_bit_errors = 0
        self.total_tokens = 0
        self.correct_tokens = 0
        self.total_relevance = 0.0

    def _derive_system_params(self) -> Tuple[float, float]:
        if self.channel is not None:
            bandwidth_hz = self.channel.fs_tau / 2.0
            fc_hz = self.channel.fc
        else:
            bandwidth_hz = self.DEFAULT_BANDWIDTH_HZ
            fc_hz = self.DEFAULT_FC_HZ
        return bandwidth_hz, fc_hz

    def _derive_cp_length(self, safety_margin: float = 1.1) -> int:
        if self.channel is None or self.channel.h.numel() == 0:
            return self.DEFAULT_CP_LENGTH

        taps = self.channel.h.shape[0]
        if taps <= 1 or self.channel.fs_tau <= 0:
            return self.DEFAULT_CP_LENGTH

        delay_sec = (taps - 1) / self.channel.fs_tau
        delay_sec *= safety_margin
        cp_samples = math.ceil(delay_sec * self.bandwidth_hz)
        cp_samples = max(cp_samples, 1)
        cp_samples = min(cp_samples, self.num_carriers - 1)
        return cp_samples

    def indices_to_bits(self, indices: torch.Tensor) -> np.ndarray:
        flat_indices = indices.cpu().numpy().flatten().astype(np.int64)
        bits_list = []
        for idx in flat_indices:
            binary = format(idx, f'0{self.bits_per_index}b')
            bits_list.extend([int(b) for b in binary])
        return np.array(bits_list, dtype=np.uint8)

    def bits_to_indices(self, bits: np.ndarray, num_indices: int) -> torch.Tensor:
        indices = []
        bits_per_idx = self.bits_per_index

        for i in range(num_indices):
            start = i * bits_per_idx
            end = start + bits_per_idx
            if end > len(bits):
                idx_bits = np.concatenate([bits[start:], np.zeros(end - len(bits), dtype=np.uint8)])
            else:
                idx_bits = bits[start:end]
            idx = int(''.join(str(b) for b in idx_bits), 2)
            idx = min(idx, self.codes_per_book - 1)
            indices.append(idx)

        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def modulate(self, bits: np.ndarray) -> torch.Tensor:
        bits_tensor = torch.from_numpy(bits.astype(np.float64)).to(self.device)

        if self.modulation == 'BPSK':
            mapped = 2 * bits_tensor - 1
            symbols = torch.complex(mapped, torch.zeros_like(mapped))
        else:
            if bits_tensor.numel() % 2 != 0:
                bits_tensor = torch.cat([bits_tensor, torch.zeros(1, dtype=bits_tensor.dtype, device=self.device)])
            bit_pairs = bits_tensor.view(-1, 2)
            symbols = torch.complex(
                2 * bit_pairs[:, 0] - 1,
                2 * bit_pairs[:, 1] - 1,
            ) / math.sqrt(2)

        return symbols

    def demodulate(self, symbols: torch.Tensor) -> np.ndarray:
        if self.modulation == 'BPSK':
            bits = (symbols.real >= 0).to(torch.int64)
            return bits.cpu().numpy().astype(np.uint8)
        else:
            real_bits = (symbols.real >= 0).to(torch.int64)
            imag_bits = (symbols.imag >= 0).to(torch.int64)
            bits = torch.stack([real_bits, imag_bits], dim=1).reshape(-1)
            return bits.cpu().numpy().astype(np.uint8)

    def add_awgn(self, signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        signal_power = torch.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.sqrt(noise_power) * torch.randn_like(signal)
        return signal + noise

    def transmit_indices(
        self,
        indices: torch.Tensor,
        snr_db: float,
        relevance_matrix: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, float, int, int, float]:
        """Transmit VQ-VAE indices through the channel.

        Args:
            indices: Tensor of codebook indices, shape (batch, t', h', w', n_codebooks)
            snr_db: Signal-to-noise ratio in dB
            relevance_matrix: Optional precomputed relevance matrix for semantic relevance

        Returns:
            Tuple of (received_indices, ber, num_errors, total_bits, relevance)
        """
        original_shape = indices.shape
        num_indices = indices.numel()

        original_bits = self.indices_to_bits(indices)
        num_original_bits = len(original_bits)

        encoded_bits = self.fec.encode(original_bits)
        symbols = self.modulate(encoded_bits)
        ofdm_signal, freq_grid, pilot_cols = self.ofdm_mapper.map(symbols, return_freq=True)
        wrap_result = self.frame_assembler.wrap_segments([ofdm_signal])
        passband = wrap_result.passband

        if self.channel is not None and self.channel_type == 'uwa':
            rx_passband = replay_filter(passband, self.fs, self.channel,
                                        allow_reference=self.allow_reference_filter)
        else:
            rx_passband = passband.clone()

        rms = torch.sqrt(torch.mean(rx_passband ** 2))
        if rms > 0:
            rx_passband = rx_passband / rms

        rx_passband = self.add_awgn(rx_passband, snr_db)
        rx_symbols = self._receive(rx_passband, freq_grid, pilot_cols)
        rx_bits = self.demodulate(rx_symbols)
        decoded_bits = self.fec.decode(rx_bits)
        decoded_bits = decoded_bits[:num_original_bits]

        min_len = min(len(original_bits), len(decoded_bits))
        bit_errors = int(np.sum(original_bits[:min_len] != decoded_bits[:min_len]))
        ber = bit_errors / min_len if min_len > 0 else 0.0

        self.total_bits_tx += min_len
        self.total_bit_errors += bit_errors

        rx_indices = self.bits_to_indices(decoded_bits, num_indices)

        # For relevance computation, reshape rx_indices back to original shape
        rx_indices_shaped = rx_indices.view(original_shape)

        indices_flat = indices.flatten()
        correct = int((indices_flat == rx_indices).sum().item())
        self.total_tokens += num_indices
        self.correct_tokens += correct

        # Compute relevance if matrix provided
        relevance = 1.0
        if relevance_matrix is not None:
            # Use shaped tensors for proper multi-codebook relevance computation
            relevance, _, _ = compute_token_relevance(
                indices, rx_indices_shaped, relevance_matrix
            )
            self.total_relevance += relevance * num_indices

        return rx_indices, ber, bit_errors, min_len, relevance

    def _receive(
        self,
        rx_passband: torch.Tensor,
        tx_freq_grid: torch.Tensor,
        pilot_cols: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.arange(rx_passband.numel(), dtype=torch.float64, device=rx_passband.device) / self.fs
        baseband = rx_passband.to(torch.float64) * torch.exp(-1j * 2 * math.pi * self.frame_config.fc_hz * t)

        rrc = root_raised_cosine(
            self.frame_config.rolloff,
            self.frame_config.span,
            self.frame_config.oversample_q
        ).to(self.device)
        filtered = upfirdn_torch(baseband, rrc, up=1, down=1)
        downsampled = filtered[::self.frame_config.oversample_q]

        sync_samples = self.frame_config.sync_length
        sc_samples = self.frame_config.sc_length
        train_samples = max(1, self.frame_config.train_length)
        preamble_samples = sync_samples + sc_samples + train_samples
        data_start = preamble_samples + self.frame_config.span

        if data_start >= len(downsampled):
            return torch.zeros(0, dtype=torch.cdouble, device=self.device)

        data_signal = downsampled[data_start:]

        sym_len = self.ofdm_config.num_carriers + self.ofdm_config.cp_length
        num_symbols = len(data_signal) // sym_len

        if num_symbols == 0:
            return torch.zeros(0, dtype=torch.cdouble, device=self.device)

        ofdm_symbols = data_signal[:num_symbols * sym_len].view(num_symbols, sym_len)
        without_cp = ofdm_symbols[:, self.ofdm_config.cp_length:]
        freq = torch.fft.fft(without_cp, dim=1) / self.ofdm_config.num_carriers
        freq = freq.t()

        freq_eq = self._equalize(freq, pilot_cols)
        data_symbols = self._extract_data(freq_eq, pilot_cols)

        return data_symbols

    def _equalize(self, freq: torch.Tensor, pilot_cols: torch.Tensor) -> torch.Tensor:
        if len(pilot_cols) == 0:
            return freq

        pilot_vals = freq[:, pilot_cols]
        expected = self.ofdm_config.pilot_value
        h_pilots = pilot_vals / expected

        num_symbols = freq.shape[1]
        h_est = torch.zeros_like(freq)

        for carrier in range(freq.shape[0]):
            pilot_indices = pilot_cols.cpu().numpy()
            pilot_values = h_pilots[carrier].cpu().numpy()

            for i in range(num_symbols):
                left_idx = np.searchsorted(pilot_indices, i, side='right') - 1
                right_idx = left_idx + 1

                if left_idx < 0:
                    h_est[carrier, i] = pilot_values[0]
                elif right_idx >= len(pilot_indices):
                    h_est[carrier, i] = pilot_values[-1]
                else:
                    left_pos = pilot_indices[left_idx]
                    right_pos = pilot_indices[right_idx]
                    alpha = (i - left_pos) / (right_pos - left_pos)
                    h_est[carrier, i] = (1 - alpha) * pilot_values[left_idx] + alpha * pilot_values[right_idx]

        freq_eq = freq / (h_est + 1e-9)
        return freq_eq

    def _extract_data(self, freq: torch.Tensor, pilot_cols: torch.Tensor) -> torch.Tensor:
        num_carriers, num_symbols = freq.shape
        pilot_set = set(pilot_cols.cpu().numpy().tolist())
        data_cols = [i for i in range(num_symbols) if i not in pilot_set]

        if len(data_cols) == 0:
            return torch.zeros(0, dtype=freq.dtype, device=freq.device)

        data = freq[:, data_cols]
        return data.t().reshape(-1)

    def reset_stats(self):
        self.total_bits_tx = 0
        self.total_bit_errors = 0
        self.total_tokens = 0
        self.correct_tokens = 0
        self.total_relevance = 0.0

    def get_overall_ber(self) -> float:
        if self.total_bits_tx == 0:
            return 0.0
        return self.total_bit_errors / self.total_bits_tx

    def get_token_accuracy(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.correct_tokens / self.total_tokens

    def get_average_relevance(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.total_relevance / self.total_tokens


def center_crop_to_square(video: torch.Tensor) -> torch.Tensor:
    """Center crop video frames to square aspect ratio."""
    t, c, h, w = video.shape
    crop_size = min(h, w)
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return video[:, :, top:top+crop_size, left:left+crop_size]


def load_full_video(video_path: str, resolution: int, max_frames: int = None):
    """Load and preprocess entire video."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()

    if max_frames is not None:
        frames_to_load = min(total_frames, max_frames)
    else:
        frames_to_load = total_frames

    frame_indices = list(range(frames_to_load))
    frames = vr.get_batch(frame_indices).asnumpy()

    video = torch.from_numpy(frames).float()
    video = video.permute(0, 3, 1, 2)

    original_h, original_w = video.shape[2], video.shape[3]

    video = center_crop_to_square(video)
    video = F.interpolate(video, size=(resolution, resolution), mode='bilinear', align_corners=False)
    video = video / 255.0 - 0.5
    video = video.permute(1, 0, 2, 3)

    return video, frames_to_load, total_frames, (original_h, original_w), fps


def compute_per_frame_metrics(original: torch.Tensor, reconstructed: torch.Tensor):
    """Compute PSNR and SSIM for each frame."""
    c, t, h, w = original.shape

    psnr_per_frame = []
    ssim_per_frame = []

    for i in range(t):
        orig_frame = original[:, i:i+1, :, :].unsqueeze(0)
        recon_frame = reconstructed[:, i:i+1, :, :].unsqueeze(0)

        psnr = compute_psnr(recon_frame, orig_frame).item()
        ssim = compute_ssim(recon_frame, orig_frame).item()

        psnr_per_frame.append(psnr)
        ssim_per_frame.append(ssim)

    return psnr_per_frame, ssim_per_frame


def process_video_in_chunks(
    model,
    video: torch.Tensor,
    chunk_size: int,
    device: torch.device,
    channel_sim: ChannelSimulator = None,
    snr_db: float = None,
    relevance_matrix: torch.Tensor = None,
):
    """Process video through VQ-VAE with optional channel simulation."""
    c, total_frames, h, w = video.shape

    reconstructed_chunks = []
    psnr_per_frame = []
    ssim_per_frame = []
    relevance_per_chunk = []

    if channel_sim is not None:
        channel_sim.reset_stats()

    n_chunks = (total_frames + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_frames)

        chunk = video[:, start_idx:end_idx, :, :]
        actual_chunk_len = end_idx - start_idx

        if actual_chunk_len < chunk_size:
            padding = torch.zeros(c, chunk_size - actual_chunk_len, h, w)
            chunk = torch.cat([chunk, padding], dim=1)

        chunk = chunk.unsqueeze(0).to(device)

        with torch.no_grad():
            if channel_sim is not None:
                _, encodings = model.encode(x=chunk)

                rx_encodings, ber, bit_errors, total_bits, relevance = channel_sim.transmit_indices(
                    encodings, snr_db, relevance_matrix
                )
                relevance_per_chunk.append(relevance)

                rx_encodings = rx_encodings.view(encodings.shape).to(device)
                recon_chunk = model.decode(rx_encodings)
            else:
                recon_chunk = model.get_reconstruction(x=chunk)

        recon_chunk = recon_chunk.squeeze(0)[:, :actual_chunk_len, :, :]
        chunk = chunk.squeeze(0)[:, :actual_chunk_len, :, :]

        chunk_psnr, chunk_ssim = compute_per_frame_metrics(chunk, recon_chunk)
        psnr_per_frame.extend(chunk_psnr)
        ssim_per_frame.extend(chunk_ssim)

        reconstructed_chunks.append(recon_chunk.cpu())

    reconstructed = torch.cat(reconstructed_chunks, dim=1)

    channel_stats = None
    if channel_sim is not None:
        channel_stats = {
            'overall_ber': channel_sim.get_overall_ber(),
            'total_bits': channel_sim.total_bits_tx,
            'total_errors': channel_sim.total_bit_errors,
            'token_accuracy': channel_sim.get_token_accuracy(),
            'total_tokens': channel_sim.total_tokens,
            'correct_tokens': channel_sim.correct_tokens,
            'average_relevance': channel_sim.get_average_relevance(),
            'snr_db': snr_db,
        }

    return reconstructed, psnr_per_frame, ssim_per_frame, channel_stats


def find_video_files(input_dir: Path) -> List[Path]:
    """Find all video files in directory."""
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_dir.glob(f'*{ext}'))
        video_files.extend(input_dir.glob(f'*{ext.upper()}'))
    return sorted(video_files)


def load_vqvae_model(ckpt_path: str, resolution: int, device: torch.device):
    """Load VQ-VAE model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)

    dset_configs = ckpt['dset_configs']
    model_resolution = dset_configs['resolution']
    n_frames = dset_configs['n_frames']

    hp = ckpt['hp'].copy()
    hp.pop('input_shape', None)

    codes_per_book = hp.get('codes_per_book', 1024)

    actual_resolution = resolution if resolution is not None else model_resolution

    model, _ = config_model(
        configs_str='',
        input_shape=(n_frames, actual_resolution, actual_resolution),
        cond_types=tuple(),
        **hp
    )

    state_dict = ckpt['state_dict']
    model_state = model.state_dict()

    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state_dict[k] = v
            elif 'pos_embd' in k:
                target_shape = model_state[k].shape
                if len(v.shape) == 2 and len(target_shape) == 2:
                    v_interp = F.interpolate(
                        v.unsqueeze(0).unsqueeze(0),
                        size=target_shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)
                    filtered_state_dict[k] = v_interp

    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, n_frames, model_resolution, actual_resolution, codes_per_book


def get_codebook_from_model(model) -> torch.Tensor:
    """Extract codebook embeddings from VQ-VAE model.

    Returns:
        embeddings: Tensor of shape [n_codebooks, codes_per_book, embedding_dim]
                    or [codes_per_book, embedding_dim] for single codebook
    """
    # The codebook is in model.codebook.embeddings (registered buffer)
    # Shape: (n_codebooks, codes_per_book, embedding_dim)
    if hasattr(model, 'codebook'):
        if hasattr(model.codebook, 'embeddings'):
            return model.codebook.embeddings.data.clone()

    # Fallback: try to find it in named buffers
    for name, buf in model.named_buffers():
        if 'embeddings' in name.lower() and buf.dim() >= 2:
            return buf.clone()

    raise ValueError("Could not find codebook embeddings in model")


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


def save_video(frames: np.ndarray, output_path: str, fps: int = 30):
    """Save frames as MP4 video."""
    try:
        import imageio
        imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
    except ImportError:
        pass  # Skip if imageio not available


def create_category_summary_plot(
    category_results: Dict[str, List[Dict]],
    output_path: Path,
    has_channel: bool
):
    """Create summary plots comparing categories."""
    fig_height = 12 if has_channel else 8
    n_rows = 3 if has_channel else 2

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, fig_height))

    categories = list(category_results.keys())
    colors = {'clear': 'steelblue', 'turbid': 'seagreen'}

    # PSNR comparison
    ax = axes[0]
    for i, cat in enumerate(categories):
        results = category_results[cat]
        if not results:
            continue
        psnr_means = [r['psnr_mean'] for r in results]
        x = np.arange(len(psnr_means))
        ax.bar(x + i * 0.4, psnr_means, width=0.35, label=f'{cat.capitalize()}',
               color=colors.get(cat, 'gray'), alpha=0.8)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR by Video and Category')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # SSIM comparison
    ax = axes[1]
    for i, cat in enumerate(categories):
        results = category_results[cat]
        if not results:
            continue
        ssim_means = [r['ssim_mean'] for r in results]
        x = np.arange(len(ssim_means))
        ax.bar(x + i * 0.4, ssim_means, width=0.35, label=f'{cat.capitalize()}',
               color=colors.get(cat, 'gray'), alpha=0.8)
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM by Video and Category')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    # Relevance comparison (if channel simulation)
    if has_channel:
        ax = axes[2]
        for i, cat in enumerate(categories):
            results = category_results[cat]
            if not results:
                continue
            relevance_means = [r.get('average_relevance', 1.0) for r in results]
            x = np.arange(len(relevance_means))
            ax.bar(x + i * 0.4, relevance_means, width=0.35, label=f'{cat.capitalize()}',
                   color=colors.get(cat, 'gray'), alpha=0.8)
        ax.set_ylabel('Average Relevance')
        ax.set_title('Semantic Relevance by Video and Category')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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

    # Relevance vs SNR
    ax = axes[1, 0]
    overall_rel = [r['overall_stats'].get('average_relevance', 1.0) for r in snr_sweep_results]
    ax.plot(snr_values, overall_rel, 'o-', color=colors['overall'], linewidth=2, markersize=8, label='Overall')
    for cat in categories:
        cat_rel = [r['category_stats'].get(cat, {}).get('average_relevance', 1.0) for r in snr_sweep_results]
        ax.plot(snr_values, cat_rel, 's--', color=colors.get(cat, 'gray'), linewidth=1.5, markersize=6, label=cat.capitalize())
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Average Relevance')
    ax.set_title('Semantic Relevance vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # BER vs SNR (log scale)
    ax = axes[1, 1]
    overall_ber = [r['overall_stats'].get('ber', 0) for r in snr_sweep_results]
    # Avoid log(0) by using small value
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
    has_channel: bool
):
    """Create bar chart comparing category averages."""
    categories = list(category_stats.keys())
    if not categories:
        return

    n_metrics = 4 if has_channel else 2
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

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

    if has_channel:
        # Token Accuracy
        acc_vals = [category_stats[c].get('token_accuracy', 1.0) for c in categories]
        bars = axes[2].bar(x, acc_vals,
                           color=[colors.get(c, 'gray') for c in categories], alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([c.capitalize() for c in categories])
        axes[2].set_ylabel('Token Accuracy')
        axes[2].set_title('Average Token Accuracy by Category')
        axes[2].set_ylim(0, 1)
        for bar, val in zip(bars, acc_vals):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f'{val:.4f}', ha='center', va='bottom')

        # Relevance
        rel_vals = [category_stats[c].get('average_relevance', 1.0) for c in categories]
        bars = axes[3].bar(x, rel_vals,
                           color=[colors.get(c, 'gray') for c in categories], alpha=0.8)
        axes[3].set_xticks(x)
        axes[3].set_xticklabels([c.capitalize() for c in categories])
        axes[3].set_ylabel('Average Relevance')
        axes[3].set_title('Average Semantic Relevance by Category')
        axes[3].set_ylim(0, 1)
        for bar, val in zip(bars, rel_vals):
            axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f'{val:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='VQ-VAE Evaluation on UVE38K Dataset with Channel Simulation'
    )

    # Model arguments
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to VQ-VAE checkpoint')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Resolution override (default: from checkpoint)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--channel-device', type=str, default=None,
                        help='Device for channel simulation (default: same as --device)')

    # Dataset arguments
    parser.add_argument('--uve-path', type=str, default=DEFAULT_UVE_PATH,
                        help=f'Path to UVE38K dataset (default: {DEFAULT_UVE_PATH})')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames per video (default: all)')
    parser.add_argument('--categories', type=str, nargs='+', default=['clear', 'turbid'],
                        help='Categories to evaluate (default: clear turbid)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results/vqvae',
                        help='Base output directory (default: results/vqvae)')
    parser.add_argument('--skip-videos', action='store_true',
                        help='Skip saving video files (only save metrics)')

    # Channel arguments
    parser.add_argument('--channel-type', type=str, default=None, choices=['awgn', 'uwa'],
                        help='Channel type for simulation (omit for direct reconstruction)')
    parser.add_argument('--channel', type=str, default='NOF1',
                        help='Channel name for UWA (e.g., NOF1, KAU1, BCH1)')
    parser.add_argument('--channel-base-dir', type=str, default='input/channels',
                        help='Base directory containing channel .mat files')
    parser.add_argument('--snr', type=float, default=15.0, help='SNR in dB (single value)')
    parser.add_argument('--snr-min', type=float, default=None,
                        help='Minimum SNR for sweep (enables SNR sweep mode)')
    parser.add_argument('--snr-max', type=float, default=None,
                        help='Maximum SNR for sweep')
    parser.add_argument('--snr-step', type=float, default=5.0,
                        help='SNR step size for sweep (default: 5.0 dB)')

    # FEC arguments
    parser.add_argument('--fec', type=str, default='none',
                        choices=['none', 'repetition', 'ldpc', 'polar', 'turbo', 'dvbs2_ldpc', 'rsc'],
                        help='FEC codec type')
    parser.add_argument('--fec-rate', type=float, default=None,
                        help='Target FEC code rate (0.2-1.0)')

    # Modulation arguments
    parser.add_argument('--modulation', type=str, default='QPSK', choices=['BPSK', 'QPSK'],
                        help='Modulation scheme')

    # OFDM arguments
    parser.add_argument('--num-carriers', type=int, default=64, help='Number of OFDM subcarriers')
    parser.add_argument('--num-ofdm-symbols', type=int, default=16, help='OFDM symbols per frame')
    parser.add_argument('--pilot-period', type=int, default=4, help='Symbols between pilots')

    args = parser.parse_args()

    # Setup devices
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    channel_device = torch.device(args.channel_device) if args.channel_device else device
    print(f"Model device: {device}")
    if args.channel_type:
        print(f"Channel device: {channel_device}")

    # Verify dataset path
    uve_path = Path(args.uve_path)
    if not uve_path.exists():
        print(f"Error: UVE38K dataset path '{uve_path}' does not exist")
        return

    # Determine SNR values (single or sweep)
    if args.snr_min is not None and args.snr_max is not None:
        snr_values = list(np.arange(args.snr_min, args.snr_max + args.snr_step/2, args.snr_step))
        snr_sweep_mode = True
        print(f"SNR sweep mode: {snr_values} dB")
    else:
        snr_values = [args.snr]
        snr_sweep_mode = False

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.channel_type:
        channel_str = args.channel if args.channel_type == 'uwa' else 'awgn'
        if snr_sweep_mode:
            run_name = f"uve_eval_{channel_str}_{args.fec}_snr_sweep_{timestamp}"
        else:
            run_name = f"uve_eval_{channel_str}_{args.fec}_snr{args.snr:.0f}dB_{timestamp}"
    else:
        run_name = f"uve_eval_direct_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"\nLoading VQ-VAE model from {args.ckpt}")
    model, n_frames, model_resolution, actual_resolution, codes_per_book = load_vqvae_model(
        args.ckpt, args.resolution, device
    )
    print(f"Model loaded: resolution={actual_resolution}x{actual_resolution}, "
          f"chunk_size={n_frames} frames, codebook_size={codes_per_book}")

    # Extract codebook and compute relevance matrix
    relevance_matrix = None
    if args.channel_type:
        try:
            codebook = get_codebook_from_model(model)
            relevance_matrix = compute_relevance_matrix(codebook).to(channel_device)
            print(f"Computed relevance matrix: {relevance_matrix.shape}")
        except ValueError as e:
            print(f"Warning: Could not extract codebook: {e}")
            print("Relevance will default to 1.0 (perfect)")

    # Setup channel simulator
    channel_sim = None
    if args.channel_type:
        channel_sim = ChannelSimulator(
            codes_per_book=codes_per_book,
            fec_type=args.fec,
            fec_rate=args.fec_rate,
            modulation=args.modulation,
            channel_type=args.channel_type,
            channel_name=args.channel,
            channel_base_dir=Path(args.channel_base_dir),
            num_carriers=args.num_carriers,
            num_ofdm_symbols=args.num_ofdm_symbols,
            pilot_period=args.pilot_period,
            device=channel_device,
        )
        print(f"\nChannel simulation enabled:")
        print(f"  Channel type: {args.channel_type.upper()}")
        if args.channel_type == 'uwa':
            print(f"  Channel name: {args.channel}")
        if snr_sweep_mode:
            print(f"  SNR sweep: {args.snr_min} to {args.snr_max} dB (step {args.snr_step})")
        else:
            print(f"  SNR: {args.snr} dB")
        print(f"  Modulation: {args.modulation}")
        print(f"  FEC: {args.fec} (rate={channel_sim.actual_fec_rate:.3f})")

    # Find videos by category
    category_videos = {}
    for cat in args.categories:
        cat_path = uve_path / cat
        if cat_path.exists():
            videos = find_video_files(cat_path)
            category_videos[cat] = videos
            print(f"Found {len(videos)} videos in '{cat}' category")
        else:
            print(f"Warning: Category '{cat}' not found at {cat_path}")

    if not any(category_videos.values()):
        print("Error: No videos found in any category")
        return

    # Pre-load all videos to avoid reloading for each SNR
    print(f"\n{'='*70}")
    print("Loading videos...")
    print(f"{'='*70}\n")

    loaded_videos = {}  # {(category, video_name): (video_tensor, metadata)}
    for category, videos in category_videos.items():
        for video_path in tqdm(videos, desc=f"Loading {category}"):
            video_name = video_path.stem
            try:
                video, frames_processed, total_frames, original_size, fps = load_full_video(
                    str(video_path), actual_resolution, args.max_frames
                )
                loaded_videos[(category, video_name)] = {
                    'video': video,
                    'video_path': str(video_path),
                    'frames_processed': frames_processed,
                    'total_frames': total_frames,
                    'original_size': original_size,
                    'fps': fps,
                }
            except Exception as e:
                print(f"  ERROR loading {video_name}: {e}")

    # Process each SNR value
    snr_sweep_results = []  # List of {snr, category_stats, overall_stats}
    total_start_time = time.time()

    for snr_idx, snr_db in enumerate(snr_values):
        if snr_sweep_mode:
            print(f"\n{'='*70}")
            print(f"SNR = {snr_db:.1f} dB ({snr_idx+1}/{len(snr_values)})")
            print(f"{'='*70}\n")

        # Results for this SNR
        all_results = []
        category_results = {cat: [] for cat in args.categories}

        # Create SNR-specific output directory for sweep mode
        if snr_sweep_mode:
            snr_output_dir = output_dir / f"snr_{snr_db:.0f}dB"
            snr_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            snr_output_dir = output_dir

        for category, videos in category_videos.items():
            if not videos:
                continue

            if not snr_sweep_mode:
                print(f"\n--- Category: {category.upper()} ({len(videos)} videos) ---\n")

            cat_output_dir = snr_output_dir / category
            cat_output_dir.mkdir(parents=True, exist_ok=True)

            for video_path in tqdm(videos, desc=f"{category}" if not snr_sweep_mode else f"{category}@{snr_db:.0f}dB"):
                video_name = video_path.stem
                video_output_dir = cat_output_dir / video_name

                # Get pre-loaded video
                video_data = loaded_videos.get((category, video_name))
                if video_data is None:
                    all_results.append({
                        'video_name': video_name,
                        'video_path': str(video_path),
                        'category': category,
                        'status': 'failed',
                        'error': 'Failed to load video'
                    })
                    continue

                try:
                    video_start_time = time.time()
                    video = video_data['video']

                    # Process through VQ-VAE
                    reconstructed, psnr_per_frame, ssim_per_frame, channel_stats = process_video_in_chunks(
                        model, video, n_frames, device,
                        channel_sim=channel_sim,
                        snr_db=snr_db if channel_sim else None,
                        relevance_matrix=relevance_matrix,
                    )

                    video_end_time = time.time()
                    video_runtime = video_end_time - video_start_time

                    # Calculate metrics
                    result = {
                        'video_name': video_name,
                        'video_path': video_data['video_path'],
                        'category': category,
                        'snr_db': snr_db,
                        'frames_processed': video_data['frames_processed'],
                        'total_frames': video_data['total_frames'],
                        'original_size': list(video_data['original_size']),
                        'fps': video_data['fps'],
                        'psnr_mean': float(np.mean(psnr_per_frame)),
                        'psnr_std': float(np.std(psnr_per_frame)),
                        'psnr_min': float(np.min(psnr_per_frame)),
                        'psnr_max': float(np.max(psnr_per_frame)),
                        'ssim_mean': float(np.mean(ssim_per_frame)),
                        'ssim_std': float(np.std(ssim_per_frame)),
                        'ssim_min': float(np.min(ssim_per_frame)),
                        'ssim_max': float(np.max(ssim_per_frame)),
                        'runtime_seconds': video_runtime,
                        'status': 'success'
                    }

                    if channel_stats:
                        result.update({
                            'ber': channel_stats['overall_ber'],
                            'bit_errors': channel_stats['total_errors'],
                            'total_bits': channel_stats['total_bits'],
                            'token_accuracy': channel_stats['token_accuracy'],
                            'correct_tokens': channel_stats['correct_tokens'],
                            'total_tokens': channel_stats['total_tokens'],
                            'average_relevance': channel_stats['average_relevance'],
                        })

                    all_results.append(result)
                    category_results[category].append(result)

                    # Save per-video metrics (only for single SNR or first SNR in sweep)
                    if not args.skip_videos and (not snr_sweep_mode or snr_idx == 0):
                        video_output_dir.mkdir(parents=True, exist_ok=True)

                        csv_path = video_output_dir / f'{video_name}_metrics.csv'
                        with open(csv_path, 'w') as f:
                            f.write('frame,psnr,ssim\n')
                            for i, (psnr, ssim) in enumerate(zip(psnr_per_frame, ssim_per_frame)):
                                f.write(f'{i},{psnr:.4f},{ssim:.6f}\n')

                        orig_frames = ((video + 0.5) * 255).clamp(0, 255).permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
                        recon_frames = ((reconstructed + 0.5) * 255).clamp(0, 255).permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
                        comparison_frames = np.concatenate([orig_frames, recon_frames], axis=2)

                        save_video(recon_frames, str(video_output_dir / f'{video_name}_reconstructed.mp4'), fps=int(video_data['fps']))
                        save_video(comparison_frames, str(video_output_dir / f'{video_name}_comparison.mp4'), fps=int(video_data['fps']))

                except Exception as e:
                    print(f"  ERROR processing {video_name}: {e}")
                    all_results.append({
                        'video_name': video_name,
                        'video_path': str(video_path),
                        'category': category,
                        'snr_db': snr_db,
                        'status': 'failed',
                        'error': str(e)
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
                if channel_sim:
                    stats['token_accuracy'] = float(np.mean([r.get('token_accuracy', 1.0) for r in successful]))
                    stats['average_relevance'] = float(np.mean([r.get('average_relevance', 1.0) for r in successful]))
                    stats['ber'] = float(np.mean([r.get('ber', 0.0) for r in successful]))
                category_stats[cat] = stats

        # Compute overall stats for this SNR
        successful_results = [r for r in all_results if r.get('status') == 'success']
        overall_stats = {
            'snr_db': snr_db,
            'videos_successful': len(successful_results),
            'psnr_mean': float(np.mean([r['psnr_mean'] for r in successful_results])) if successful_results else 0,
            'ssim_mean': float(np.mean([r['ssim_mean'] for r in successful_results])) if successful_results else 0,
        }
        if channel_sim and successful_results:
            overall_stats['token_accuracy'] = float(np.mean([r.get('token_accuracy', 1.0) for r in successful_results]))
            overall_stats['average_relevance'] = float(np.mean([r.get('average_relevance', 1.0) for r in successful_results]))
            overall_stats['ber'] = float(np.mean([r.get('ber', 0.0) for r in successful_results]))

        snr_sweep_results.append({
            'snr_db': snr_db,
            'category_stats': category_stats,
            'overall_stats': overall_stats,
            'per_video_results': all_results,
        })

        # Save per-SNR summary
        if snr_sweep_mode:
            with open(snr_output_dir / 'summary.json', 'w') as f:
                json.dump({
                    'snr_db': snr_db,
                    'category_stats': category_stats,
                    'overall_stats': overall_stats,
                    'per_video_results': all_results,
                }, f, indent=2)

    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time

    # Use results from sweep
    if snr_sweep_mode:
        # Create SNR sweep summary
        sweep_summary = {
            'run_info': {
                'timestamp': timestamp,
                'checkpoint': args.ckpt,
                'uve_path': str(uve_path),
                'output_dir': str(output_dir),
                'resolution': actual_resolution,
                'model_resolution': model_resolution,
                'chunk_size': n_frames,
                'codebook_size': codes_per_book,
                'max_frames': args.max_frames,
                'device': str(device),
                'categories': args.categories,
            },
            'channel_config': {
                'channel_type': args.channel_type,
                'channel_name': args.channel if args.channel_type == 'uwa' else None,
                'snr_sweep': {'min': args.snr_min, 'max': args.snr_max, 'step': args.snr_step},
                'snr_values': snr_values,
                'fec_type': args.fec,
                'fec_rate_target': args.fec_rate,
                'fec_rate_actual': channel_sim.actual_fec_rate if channel_sim else None,
                'modulation': args.modulation,
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
            f.write('snr_db,psnr_mean,ssim_mean,token_accuracy,relevance,ber')
            for cat in args.categories:
                f.write(f',{cat}_psnr,{cat}_ssim,{cat}_relevance')
            f.write('\n')
            for r in snr_sweep_results:
                stats = r['overall_stats']
                f.write(f"{stats['snr_db']:.1f},{stats['psnr_mean']:.4f},{stats['ssim_mean']:.6f},"
                        f"{stats.get('token_accuracy', 1):.4f},{stats.get('average_relevance', 1):.4f},"
                        f"{stats.get('ber', 0):.6f}")
                for cat in args.categories:
                    cat_stats = r['category_stats'].get(cat, {})
                    f.write(f",{cat_stats.get('psnr_mean', 0):.4f},{cat_stats.get('ssim_mean', 0):.6f},"
                            f"{cat_stats.get('average_relevance', 1):.4f}")
                f.write('\n')

        # Create SNR sweep plots
        create_snr_sweep_plot(snr_sweep_results, args.categories, output_dir / 'snr_sweep_plot.png')

        # Print sweep summary
        print(f"\n{'='*90}")
        print("SNR SWEEP COMPLETE")
        print(f"{'='*90}")
        print(f"SNR range: {args.snr_min} to {args.snr_max} dB (step {args.snr_step})")
        print(f"Total runtime: {format_duration(total_runtime)}")

        print(f"\n{'SNR':<8} {'PSNR':<10} {'SSIM':<10} {'Accuracy':<10} {'Relevance':<10} {'BER':<12}")
        print(f"{'-'*60}")
        for r in snr_sweep_results:
            stats = r['overall_stats']
            print(f"{stats['snr_db']:<8.1f} {stats['psnr_mean']:<10.2f} {stats['ssim_mean']:<10.4f} "
                  f"{stats.get('token_accuracy', 1):<10.4f} {stats.get('average_relevance', 1):<10.4f} "
                  f"{stats.get('ber', 0):<12.6f}")

        print(f"\nOutputs saved to: {output_dir}/")
        print(f"  - sweep_summary.json      (full sweep results)")
        print(f"  - sweep_summary.csv       (tabular sweep data)")
        print(f"  - snr_sweep_plot.png      (SNR vs metrics plot)")
        print(f"  - snr_XdB/                (per-SNR detailed results)")
        print(f"{'='*90}")

    else:
        # Single SNR mode
        category_stats = snr_sweep_results[0]['category_stats']
        all_results = snr_sweep_results[0]['per_video_results']
        successful_results = [r for r in all_results if r.get('status') == 'success']

        # Build summary
        summary = {
            'run_info': {
                'timestamp': timestamp,
                'checkpoint': args.ckpt,
                'uve_path': str(uve_path),
                'output_dir': str(output_dir),
                'resolution': actual_resolution,
                'model_resolution': model_resolution,
                'chunk_size': n_frames,
                'codebook_size': codes_per_book,
                'max_frames': args.max_frames,
                'device': str(device),
                'categories': args.categories,
            },
            'channel_config': None,
            'runtime': {
                'total_seconds': total_runtime,
                'total_formatted': format_duration(total_runtime),
            },
            'overall_metrics': {
                'videos_processed': len(all_results),
                'videos_successful': len(successful_results),
                'videos_failed': len(all_results) - len(successful_results),
                'total_frames_processed': sum(r['frames_processed'] for r in successful_results),
                'psnr_mean': float(np.mean([r['psnr_mean'] for r in successful_results])) if successful_results else 0,
                'ssim_mean': float(np.mean([r['ssim_mean'] for r in successful_results])) if successful_results else 0,
            },
            'category_stats': category_stats,
            'per_video_results': all_results,
        }

        if channel_sim:
            summary['channel_config'] = {
                'channel_type': args.channel_type,
                'channel_name': args.channel if args.channel_type == 'uwa' else None,
                'snr_db': args.snr,
                'fec_type': args.fec,
                'fec_rate_target': args.fec_rate,
                'fec_rate_actual': channel_sim.actual_fec_rate,
                'modulation': args.modulation,
                'num_carriers': args.num_carriers,
                'cp_length': channel_sim.cp_length,
            }
            summary['overall_metrics']['token_accuracy'] = float(np.mean([r.get('token_accuracy', 1.0) for r in successful_results])) if successful_results else 0
            summary['overall_metrics']['average_relevance'] = float(np.mean([r.get('average_relevance', 1.0) for r in successful_results])) if successful_results else 0
            summary['overall_metrics']['ber'] = float(np.mean([r.get('ber', 0.0) for r in successful_results])) if successful_results else 0

        # Save summary JSON
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save summary CSV
        csv_path = output_dir / 'summary.csv'
        with open(csv_path, 'w') as f:
            if channel_sim:
                f.write('category,video_name,frames,psnr_mean,psnr_std,ssim_mean,ssim_std,ber,token_accuracy,relevance,status\n')
            else:
                f.write('category,video_name,frames,psnr_mean,psnr_std,ssim_mean,ssim_std,status\n')
            for r in all_results:
                if r.get('status') == 'success':
                    if channel_sim:
                        f.write(f"{r['category']},{r['video_name']},{r['frames_processed']},"
                                f"{r['psnr_mean']:.4f},{r['psnr_std']:.4f},"
                                f"{r['ssim_mean']:.6f},{r['ssim_std']:.6f},"
                                f"{r.get('ber', 0):.6f},{r.get('token_accuracy', 1):.4f},"
                                f"{r.get('average_relevance', 1):.4f},success\n")
                    else:
                        f.write(f"{r['category']},{r['video_name']},{r['frames_processed']},"
                                f"{r['psnr_mean']:.4f},{r['psnr_std']:.4f},"
                                f"{r['ssim_mean']:.6f},{r['ssim_std']:.6f},success\n")
                else:
                    if channel_sim:
                        f.write(f"{r.get('category', 'unknown')},{r['video_name']},0,0,0,0,0,0,0,0,failed\n")
                    else:
                        f.write(f"{r.get('category', 'unknown')},{r['video_name']},0,0,0,0,0,failed\n")

        # Create plots
        if successful_results:
            # Need to rebuild category_results for plotting
            category_results_plot = {cat: [] for cat in args.categories}
            for r in successful_results:
                category_results_plot[r['category']].append(r)
            create_category_summary_plot(category_results_plot, output_dir / 'summary_by_video.png', has_channel=channel_sim is not None)
            create_category_bar_chart(category_stats, output_dir / 'summary_by_category.png', has_channel=channel_sim is not None)

        # Print final summary
        print(f"\n{'='*70}")
        print("EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"Videos processed: {len(successful_results)}/{len(all_results)}")
        print(f"Total runtime: {format_duration(total_runtime)}")

        print(f"\nCategory Statistics:")
        for cat, stats in category_stats.items():
            print(f"\n  {cat.upper()} ({stats['num_videos']} videos, {stats['total_frames']} frames):")
            print(f"    PSNR: {stats['psnr_mean']:.2f} +/- {stats['psnr_std']:.2f} dB")
            print(f"    SSIM: {stats['ssim_mean']:.4f} +/- {stats['ssim_std']:.4f}")
            if channel_sim:
                print(f"    Token Accuracy: {stats.get('token_accuracy', 1.0):.4f}")
                print(f"    Average Relevance: {stats.get('average_relevance', 1.0):.4f}")
                print(f"    BER: {stats.get('ber', 0.0):.6f}")

        print(f"\nOutputs saved to: {output_dir}/")
        print(f"  - summary.json        (full results)")
        print(f"  - summary.csv         (tabular metrics)")
        print(f"  - summary_by_video.png    (per-video comparison)")
        print(f"  - summary_by_category.png (category averages)")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
