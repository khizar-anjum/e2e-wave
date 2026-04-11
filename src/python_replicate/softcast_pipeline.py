"""
SoftCast Channel Simulation Pipeline

This module provides the full SoftCast channel simulation pipeline,
analogous to watermark_pipeline.py for watermark signals.

Key feature: RMS normalization after replay_filter to normalize channel output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .channel_dataset import (
    _flatten_data_matrix,
    _resample_complex,
    _select_data_columns,
    _pll_correct,
    _interpolate_pilots,
    _estimate_channel_from_pilots,
)
from .channel_replay import ChannelSounding, replay_filter
from .frame_preparation import FramePrepConfig
from .ofdm_mapper import OFDMConfig
from .signal_utils import upfirdn_torch


# -----------------------------------------------------------------------------
# OFDM Demodulation for SoftCast
# -----------------------------------------------------------------------------

def _ofdm_demodulate_softcast(
    rx_baseband: torch.Tensor,
    frame_config: FramePrepConfig,
    ofdm_config: OFDMConfig,
    pilot_cols: torch.Tensor,
    rrc_filter: torch.Tensor,
    equalize: bool = True,
    noise_power: float = 0.0,
    equalizer: str = 'zf',
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Demodulate OFDM signal back to complex baseband symbols.

    This function performs the inverse of OFDM modulation:
    1. RRC matched filter + downsample
    2. CP removal + FFT
    3. Pilot-based equalization (ZF or MMSE)
    4. Extract data symbols

    Args:
        rx_baseband: Received complex baseband signal (already downconverted)
        frame_config: Frame preparation configuration
        ofdm_config: OFDM configuration
        pilot_cols: Pilot column indices
        rrc_filter: RRC filter for matched filtering
        equalize: Whether to apply pilot-based equalization
        noise_power: Pre-equalization AWGN noise power (for computing post-EQ noise var)
        equalizer: Equalization method - 'zf' (Zero-Forcing) or 'mmse' (MMSE)
            - ZF: X_hat = Y / H, noise_var = σ² / |H|²
            - MMSE: X_hat = H* / (|H|² + σ²) * Y, noise_var = σ² / (|H|² + σ²)
            MMSE is better at low SNR as it doesn't amplify noise at deep fades.

    Returns:
        Tuple of (data_symbols, noise_var_per_symbol, pre_eq_power):
        - data_symbols: Complex baseband data symbols (flattened)
        - noise_var_per_symbol: Per-symbol noise variance after equalization
        - pre_eq_power: Average power of data symbols before equalization
    """
    device = rx_baseband.device

    if rx_baseband.numel() == 0:
        empty = torch.zeros(0, dtype=torch.complex128, device=device)
        empty_noise = torch.zeros(0, dtype=torch.float64, device=device)
        return empty, empty_noise, 0.0

    # Input is already complex baseband (downconverted with CFO correction upstream)
    baseband = rx_baseband.to(torch.complex128)

    # 1. RRC matched filter + downsample
    matched = upfirdn_torch(baseband, rrc_filter, up=1, down=1)
    downsampled = matched[::frame_config.oversample_q]
    span = frame_config.span
    if downsampled.numel() > 2 * span:
        downsampled = downsampled[span:-span]

    # 3. CP removal + FFT
    sym_len = ofdm_config.num_carriers + ofdm_config.cp_length
    if sym_len == 0 or downsampled.numel() < sym_len:
        empty = torch.zeros(0, dtype=torch.complex128, device=device)
        empty_noise = torch.zeros(0, dtype=torch.float64, device=device)
        return empty, empty_noise, 0.0

    usable = (downsampled.numel() // sym_len) * sym_len
    if usable == 0:
        empty = torch.zeros(0, dtype=torch.complex128, device=device)
        empty_noise = torch.zeros(0, dtype=torch.float64, device=device)
        return empty, empty_noise, 0.0

    symbols = downsampled[:usable].view(-1, sym_len)
    without_cp = symbols[:, ofdm_config.cp_length:]
    freq = torch.fft.fft(without_cp, dim=1) / ofdm_config.num_carriers
    freq = freq.t()  # (num_carriers, n_symbols)

    n_symbols = freq.shape[1]
    if n_symbols == 0:
        empty = torch.zeros(0, dtype=torch.complex128, device=device)
        empty_noise = torch.zeros(0, dtype=torch.float64, device=device)
        return empty, empty_noise, 0.0

    # 4. Pilot-based equalization and per-subcarrier noise variance
    # Measure power right after FFT (before any processing)
    freq_power_after_fft = float(torch.mean(torch.abs(freq) ** 2).item())

    if equalize and pilot_cols is not None and pilot_cols.numel() > 0:
        freq = _pll_correct(freq, pilot_cols)

        channel_est = _estimate_channel_from_pilots(
            freq, pilot_cols, ofdm_config.pilot_value
        )

        # Measure pre-equalization power on data symbols (excluding pilots)
        data_matrix_pre_eq, data_cols_pre = _select_data_columns(freq, pilot_cols, n_symbols)
        if data_matrix_pre_eq.numel() > 0:
            pre_eq_power = float(torch.mean(torch.abs(data_matrix_pre_eq) ** 2).item())
        else:
            pre_eq_power = 0.0

        # Compute |H|² for equalization
        H_mag_sq = torch.abs(channel_est) ** 2

        if equalizer.lower() == 'mmse':
            # MMSE Equalization: W = H* / (|H|² + σ²)
            # - Balances noise enhancement vs signal distortion
            # - At deep fades: attenuates instead of amplifying noise
            # - At high SNR: approaches ZF
            #
            # Output: X_hat = W * Y = H* / (|H|² + σ²) * Y
            # Noise variance (MSE): σ² / (|H|² + σ²)
            denominator = H_mag_sq + noise_power
            denominator_clamped = torch.clamp(denominator, min=1e-9)

            # MMSE equalizer coefficient
            W_mmse = torch.conj(channel_est) / denominator_clamped

            # Apply MMSE equalization
            freq = freq * W_mmse

            # Per-subcarrier noise variance after MMSE
            # MSE = σ² / (|H|² + σ²) - this is the total distortion (noise + bias)
            noise_var_per_carrier = noise_power / denominator_clamped

        else:
            # ZF Equalization (default): W = 1/H
            # - Simple inversion of channel
            # - Amplifies noise at deep fades (when |H| is small)
            #
            # Output: X_hat = Y / H
            # Noise variance: σ² / |H|²
            H_mag_sq_clamped = torch.clamp(H_mag_sq, min=1e-6)  # Prevent blow-up
            noise_var_per_carrier = noise_power / H_mag_sq_clamped

            # ZF equalization
            freq = freq / (channel_est + 1e-9)
    else:
        # No equalization: uniform noise
        noise_var_per_carrier = torch.full(
            freq.shape, noise_power, dtype=torch.float64, device=device
        )
        # Pre-EQ power is same as FFT power when no equalization
        pre_eq_power = freq_power_after_fft

    # 5. Extract data (remove pilot columns)
    data_matrix, data_cols = _select_data_columns(freq, pilot_cols, n_symbols)

    # Also extract noise variance for data columns only
    if data_cols.numel() > 0:
        noise_var_matrix = noise_var_per_carrier.index_select(1, data_cols)
    else:
        noise_var_matrix = torch.zeros(freq.shape[0], 0, dtype=torch.float64, device=device)

    data_flat = _flatten_data_matrix(data_matrix)
    noise_var_flat = noise_var_matrix.flatten()  # Same order as data_flat

    return data_flat, noise_var_flat, pre_eq_power


# -----------------------------------------------------------------------------
# SoftCast Simulation Result
# -----------------------------------------------------------------------------

@dataclass
class SoftCastSimulationResult:
    """Result of SoftCast channel simulation.

    Attributes:
        rx_signal: Full received signal (metadata + SoftCast)
        rx_metadata: Received metadata portion
        rx_softcast: Received SoftCast waveforms portion (OFDM-demodulated)
        clean_signal: Channel output before AWGN
        metadata_range: Sample indices (start, end) for metadata
        softcast_range: Sample indices (start, end) for SoftCast
        snr_db: Applied SNR
        channel_meta: Channel impulse response metadata
        estimated_noise_power: Estimated noise power from channel (pre-equalization)
        noise_var_per_symbol: Per-symbol noise variance after equalization
        pre_eq_power: Average power of data symbols before equalization
        channel_rms_normalization: RMS normalization factor applied after channel
            (multiply RX by this to restore original scale)
    """
    rx_signal: torch.Tensor
    rx_metadata: torch.Tensor
    rx_softcast: torch.Tensor
    clean_signal: torch.Tensor
    metadata_range: Tuple[int, int]
    softcast_range: Tuple[int, int]
    snr_db: float
    channel_meta: ChannelSounding
    estimated_noise_power: float
    noise_var_per_symbol: torch.Tensor
    pre_eq_power: float
    channel_rms_normalization: float = 1.0


# -----------------------------------------------------------------------------
# Main SoftCast Channel Simulation Pipeline
# -----------------------------------------------------------------------------

def simulate_softcast_channel(
    pipeline,  # ChannelSimulationPipeline instance
    metadata_signal: torch.Tensor,
    softcast_signal: torch.Tensor,
    snr_db: float,
    softcast_pilot_cols: Optional[torch.Tensor] = None,
    softcast_ofdm_config: Optional[OFDMConfig] = None,
    generator: Optional[torch.Generator] = None,
    add_awgn: bool = True,
    flat_channel: bool = False,
    equalizer: str = 'zf',
    skip_cfo: bool = False,
) -> SoftCastSimulationResult:
    """Simulate SoftCast TDM signal through channel with RMS normalization.

    This function mirrors watermark_pipeline.py's simulate_watermark:
    - Passes signal through channel via replay_filter
    - **Applies RMS normalization to channel output** (key difference from old code)
    - Adds AWGN based on normalized signal power
    - OFDM demodulates the received signal

    Args:
        pipeline: ChannelSimulationPipeline instance with channel, frame_config, etc.
        metadata_signal: Complex baseband metadata OFDM signal
        softcast_signal: Complex baseband SoftCast OFDM signal
        snr_db: Target SNR in dB
        softcast_pilot_cols: Pilot column indices for SoftCast OFDM demod
        softcast_ofdm_config: OFDM config used by TX (for correct CP length)
        generator: Random generator for noise
        add_awgn: Whether to add Gaussian noise
        flat_channel: Use identity channel (AWGN only, no multipath)
        equalizer: Equalization method - 'zf' (Zero-Forcing) or 'mmse' (MMSE)
            - ZF: Simple channel inversion, amplifies noise at deep fades
            - MMSE: Balances noise enhancement vs distortion, better at low SNR
        skip_cfo: Skip CFO estimation/correction (for testing purposes)

    Returns:
        SoftCastSimulationResult with OFDM-demodulated signals and normalization factor
    """
    device = pipeline.device

    # Move signals to device
    metadata_signal = metadata_signal.to(device, dtype=torch.complex128)
    softcast_signal = softcast_signal.to(device, dtype=torch.complex128)

    # Record lengths for later splitting
    metadata_len = metadata_signal.numel()
    softcast_len = softcast_signal.numel()

    # TDM: concatenate metadata first, then SoftCast
    combined_baseband = torch.cat([metadata_signal, softcast_signal])

    # Add sync and preamble structure
    full_baseband = torch.cat([
        pipeline.frame_assembler.sync_signal.to(device),
        pipeline.frame_assembler.sc_signal.to(device),
        pipeline.frame_assembler.train_signal.to(device),
        combined_baseband,
    ])

    # Calculate offset for data portion
    preamble_len = (
        pipeline.frame_assembler.sync_signal.numel() +
        pipeline.frame_assembler.sc_signal.numel() +
        pipeline.frame_assembler.train_signal.numel()
    )

    # Convert to passband
    fs = pipeline.frame_config.oversample_q * pipeline.frame_config.bandwidth_hz
    t = torch.arange(full_baseband.numel(), dtype=torch.float64, device=device) / fs
    passband = torch.real(
        full_baseband * torch.exp(1j * 2 * math.pi * pipeline.frame_config.fc_hz * t)
    )

    # Pass through channel
    channel_rms_normalization = 1.0
    if flat_channel:
        signal = passband.view(1, 1, -1)
        impulse = torch.ones(1, 1, 1, dtype=signal.dtype, device=device)
        channel_out = F.conv1d(signal, impulse).view(-1)
    else:
        channel_out = replay_filter(passband, fs, pipeline.channel)

        # =====================================================================
        # RMS NORMALIZATION (like watermark_pipeline.py)
        # This is the key fix - normalize channel output to RMS ≈ 1
        # =====================================================================
        rms = torch.sqrt(torch.mean(channel_out ** 2))
        channel_rms_normalization = rms.item() if rms.item() > 0 else 1.0
        channel_out = channel_out / channel_rms_normalization

    # Calculate noise power and add AWGN
    # After normalization, signal_power should be ~1
    signal_power = torch.mean(channel_out ** 2).clamp_min(1e-12)
    noise_power = signal_power / (10 ** (snr_db / 10))

    if add_awgn:
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn(
            channel_out.numel(),
            dtype=channel_out.dtype,
            device=device,
            generator=generator,
        ) * noise_std
        rx_signal = channel_out + noise
    else:
        rx_signal = channel_out

    # Calculate ranges in received signal (passband samples)
    metadata_start = preamble_len
    metadata_end = preamble_len + metadata_len
    softcast_start = metadata_end
    softcast_end = softcast_start + softcast_len

    # Clamp to actual signal length
    actual_len = rx_signal.numel()
    metadata_end = min(metadata_end, actual_len)
    softcast_start = min(softcast_start, actual_len)
    softcast_end = min(softcast_end, actual_len)

    # === Step 1: Downconvert FULL signal to baseband first ===
    fs = pipeline.frame_config.oversample_q * pipeline.frame_config.bandwidth_hz
    t = torch.arange(rx_signal.numel(), device=device, dtype=torch.float64) / fs
    baseband = torch.complex(
        rx_signal.to(torch.float64),
        torch.zeros_like(rx_signal, dtype=torch.float64),
    ) * torch.exp(-1j * 2 * math.pi * pipeline.frame_config.fc_hz * t)

    # === Step 2: Schmidl-Cox CFO estimation (reuse logic from recover_data_sequences) ===
    sc_len = pipeline.frame_assembler.sc_signal.numel()
    sync_len = pipeline.frame_assembler.sync_signal.numel()

    cfo_scale = 1.0
    # Only perform CFO estimation for non-flat channels (flat channel has no Doppler/CFO)
    # skip_cfo flag allows skipping CFO for testing purposes
    if not skip_cfo and not flat_channel and sc_len > 0 and baseband.numel() >= sync_len + sc_len:
        sc_segment = baseband[sync_len : sync_len + sc_len]
        half = sc_len // 2
        if half > 0:
            first = sc_segment[:half]
            second = sc_segment[half : 2 * half]
            corr = torch.sum(second * torch.conj(first))
            phase = torch.atan2(corr.imag, corr.real)
            per_sample = phase / max(half, 1)
            fd = per_sample * fs / (2 * math.pi)
            scale = 1.0 + (fd / max(pipeline.frame_config.fc_hz, 1e-9))
            # Convert scale to Python float to avoid tensor issues with round()
            if hasattr(scale, 'item'):
                scale = float(scale.item())
            else:
                scale = float(scale)
            # Clamp scale to [0.5, 1.5]
            scale = max(0.5, min(1.5, scale))
            cfo_scale = scale

            if abs(scale - 1.0) > 1e-4:
                baseband = _resample_complex(baseband, scale)
                # Adjust all offsets after resampling
                metadata_start = int(round(metadata_start * scale))
                metadata_end = int(round(metadata_end * scale))
                softcast_start = int(round(softcast_start * scale))
                softcast_end = int(round(softcast_end * scale))

    # === Step 3: Extract baseband portions (not passband!) ===
    # Clamp again after potential resampling
    actual_len = baseband.numel()
    metadata_end = min(metadata_end, actual_len)
    softcast_start = min(softcast_start, actual_len)
    softcast_end = min(softcast_end, actual_len)

    rx_metadata_baseband = baseband[metadata_start:metadata_end]
    rx_softcast_baseband = baseband[softcast_start:softcast_end]

    # === Step 4: OFDM demodulate SoftCast stream ===
    # Use TX's OFDM config if provided (for correct CP length), else fall back to pipeline's
    demod_ofdm_config = softcast_ofdm_config if softcast_ofdm_config is not None else pipeline.ofdm_config
    if softcast_pilot_cols is not None:
        rx_softcast, noise_var_per_symbol, pre_eq_power = _ofdm_demodulate_softcast(
            rx_softcast_baseband,  # Now baseband, not passband
            pipeline.frame_config,
            demod_ofdm_config,
            softcast_pilot_cols.to(device),
            pipeline.frame_assembler.rrc,
            equalize=not flat_channel,  # Skip equalization for flat channel
            noise_power=float(noise_power.item()),  # Pass pre-EQ noise power
            equalizer=equalizer,  # ZF or MMSE
        )
    else:
        # Fallback: return baseband if no pilot cols provided
        rx_softcast = rx_softcast_baseband
        # Uniform noise variance for fallback case
        noise_var_per_symbol = torch.full(
            (rx_softcast_baseband.numel(),),
            float(noise_power.item()),
            dtype=torch.float64,
            device=device,
        )
        # Pre-EQ power for fallback
        pre_eq_power = float(torch.mean(torch.abs(rx_softcast_baseband) ** 2).item())

    # For metadata, return baseband (can add full demod later if needed)
    rx_metadata = rx_metadata_baseband

    return SoftCastSimulationResult(
        rx_signal=rx_signal,
        rx_metadata=rx_metadata,
        rx_softcast=rx_softcast,
        clean_signal=channel_out,
        metadata_range=(metadata_start, metadata_end),
        softcast_range=(softcast_start, softcast_end),
        snr_db=snr_db,
        channel_meta=pipeline.channel,
        estimated_noise_power=float(noise_power.item()),
        noise_var_per_symbol=noise_var_per_symbol,
        pre_eq_power=pre_eq_power,
        channel_rms_normalization=channel_rms_normalization,
    )
