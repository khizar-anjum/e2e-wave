"""
SoftCast Integration with Underwater Acoustic Channel Simulation

This module provides the interface between SoftCast video encoding and
the channel simulation pipeline. It handles:
- I/Q modulation (real -> complex conversion)
- Metadata packet generation (BPSK/QPSK OFDM)
- Time Division Multiplexing (metadata first, then SoftCast waveforms)
- Receiver-side decoding
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

# Add parent directory to path for softcast import
sys.path.insert(0, str(Path(__file__).parent.parent))

from softcast import SoftCast

from .fec_codec import FECCodec, PassthroughFEC, get_fec_codec
from .frame_preparation import FramePrepConfig
from .metadata_frame import (
    MetadataFrameBuilder,
    MetadataFrameConfig,
    MetadataFrameReceiver,
    MetadataFrameResult,
)
from .ofdm_mapper import OFDMConfig, OFDMMapper
from .signal_utils import root_raised_cosine, upfirdn_torch


@dataclass
class SoftCastTxConfig:
    """Configuration for SoftCast transmitter.

    Attributes:
        frames_per_sec: Video frame rate
        data_symbols_per_sec: Target transmission rate
        power_budget: Total power budget for transmission
        x_chunks: Spatial chunk dimension X (default 8)
        y_chunks: Spatial chunk dimension Y (default 8)
        metadata_modulation: 'BPSK' or 'QPSK' for metadata
        use_fec: Whether to apply FEC to metadata
        fec_type: FEC codec type ('none', 'repetition')
        fec_params: Additional FEC parameters
    """
    frames_per_sec: float = 30.0
    data_symbols_per_sec: float = 8000.0
    power_budget: float = 1.0
    x_chunks: int = 8
    y_chunks: int = 8
    metadata_modulation: str = 'QPSK'
    use_fec: bool = False
    fec_type: str = 'none'
    fec_params: dict = None
    overlap_iq: bool = False  # Overlapping I/Q modulation for diversity

    def __post_init__(self):
        if self.fec_params is None:
            self.fec_params = {}


@dataclass
class SoftCastTxResult:
    """Result of SoftCast encoding for transmission.

    Attributes:
        metadata_packet: OFDM-modulated metadata (complex baseband)
        softcast_waveforms: OFDM-wrapped SoftCast data (complex baseband)
        combined_signal: TDM combined signal (metadata + waveforms)
        metadata_length: Number of samples in metadata packet
        waveform_length: Number of samples in SoftCast waveforms
        metadata_raw: Raw metadata tuple (indices, means, vars_)
        tx_mat_real: Original real-valued tx_mat from SoftCast
        tx_power_scale: Power normalization factor from I/Q modulation
        softcast_pilot_cols: Pilot column indices for OFDM demodulation
        softcast_ofdm_power_scale: Power normalization factor from OFDM shaping
        softcast_num_iq_samples: Number of complex I/Q samples (before OFDM)
    """
    metadata_packet: torch.Tensor
    softcast_waveforms: torch.Tensor
    combined_signal: torch.Tensor
    metadata_length: int
    waveform_length: int
    metadata_raw: Tuple
    tx_mat_real: np.ndarray
    tx_power_scale: float = 1.0
    softcast_pilot_cols: Optional[torch.Tensor] = None
    softcast_ofdm_power_scale: float = 1.0
    softcast_num_iq_samples: int = 0
    overlap_iq: bool = False  # Whether overlapping I/Q modulation was used


class SoftCastTransmitter:
    """Wraps SoftCast encoder for channel transmission.

    Handles encoding video frames, generating metadata packets,
    and converting to complex waveforms for channel simulation.
    """

    def __init__(
        self,
        tx_config: SoftCastTxConfig,
        frame_config: Optional[FramePrepConfig] = None,
        metadata_config: Optional[MetadataFrameConfig] = None,
        device: torch.device = None,
    ):
        """Initialize SoftCast transmitter.

        Args:
            tx_config: SoftCast transmission configuration
            frame_config: OFDM frame configuration (for sampling rate etc.)
            metadata_config: Metadata frame configuration
            device: Torch device
        """
        self.tx_config = tx_config
        self.frame_config = frame_config or FramePrepConfig()
        self.device = device or torch.device('cpu')

        # Initialize FEC codec
        if tx_config.use_fec:
            self.fec = get_fec_codec(tx_config.fec_type, **tx_config.fec_params)
        else:
            self.fec = PassthroughFEC()

        # Calculate chunks_per_gop based on config
        # This matches the SoftCast calculation
        self.chunks_per_gop = self._calculate_chunks_per_gop()

        # Setup metadata config if not provided
        if metadata_config is None:
            modulation_order = 4 if tx_config.metadata_modulation.upper() == 'QPSK' else 2
            metadata_config = MetadataFrameConfig(
                modulation_order=modulation_order,
                chunks_per_gop=self.chunks_per_gop,
                x_chunks=tx_config.x_chunks,
                y_chunks=tx_config.y_chunks,
            )
        self.metadata_config = metadata_config

        # Initialize metadata frame builder
        self.metadata_builder = MetadataFrameBuilder(
            config=metadata_config,
            fec=self.fec,
        )

        # Initialize SoftCast encoder
        self.softcast = SoftCast()

        # Initialize OFDM mapper for SoftCast waveforms (same config as frame)
        self.ofdm_config = OFDMConfig(
            num_carriers=self.frame_config.num_carriers,
            cp_length=self.frame_config.cp_length,
            pilot_period=4,
        )
        self.ofdm_mapper = OFDMMapper(self.ofdm_config)

        # Initialize RRC filter for pulse shaping
        self.rrc = root_raised_cosine(
            rolloff=0.25,
            span=self.frame_config.span,
            sps=self.frame_config.oversample_q,
        ).to(torch.complex128)

    def _calculate_chunks_per_gop(self, frames_per_gop: int = 8) -> int:
        """Calculate chunks_per_gop matching SoftCast logic."""
        cfg = self.tx_config
        chunks_per_gop = int(
            (cfg.data_symbols_per_sec * frames_per_gop) /
            (cfg.frames_per_sec * cfg.x_chunks * cfg.y_chunks)
        )
        # Round down to nearest power of 2 (SoftCast requirement)
        if chunks_per_gop > 0 and not (chunks_per_gop & (chunks_per_gop - 1) == 0):
            chunks_per_gop = 2 ** int(np.log2(chunks_per_gop))
        return max(1, chunks_per_gop)

    def encode_gop(
        self,
        frames: np.ndarray,
        frames_per_gop: Optional[int] = None,
    ) -> SoftCastTxResult:
        """Encode a group of pictures for transmission.

        Args:
            frames: Video frames as numpy array (H, W, T) or (H, W, C, T)
                   Values should be normalized to [0, 1]
            frames_per_gop: Number of frames (inferred from shape if None)

        Returns:
            SoftCastTxResult with all transmission data
        """
        # Handle input shape
        if frames.ndim == 4:
            # (H, W, C, T) -> convert to grayscale (H, W, T)
            frames = np.mean(frames, axis=2)
        if frames.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, T), got shape {frames.shape}")

        frames_per_gop = frames_per_gop or frames.shape[2]

        # Update chunks_per_gop for actual frame count
        self.chunks_per_gop = self._calculate_chunks_per_gop(frames_per_gop)

        # Encode with SoftCast
        metadata, tx_mat = self.softcast.encode_frames(
            frames=frames,
            frames_per_sec=self.tx_config.frames_per_sec,
            data_symbols_per_sec=self.tx_config.data_symbols_per_sec,
            power_budget=self.tx_config.power_budget,
            x_chunks=self.tx_config.x_chunks,
            y_chunks=self.tx_config.y_chunks,
        )

        indices, means, vars_ = metadata

        # Build metadata packet
        metadata_result = self.metadata_builder.build_packet(
            indices=indices,
            means=means,
            vars_=vars_,
        )

        # Convert SoftCast output to complex via I/Q modulation
        if self.tx_config.overlap_iq:
            softcast_complex, tx_power_scale = self._real_to_complex_iq_overlap(tx_mat)
        else:
            softcast_complex, tx_power_scale = self._real_to_complex_iq(tx_mat)
        num_iq_samples = softcast_complex.numel()

        # Wrap in OFDM (adds pilots, IFFT, cyclic prefix)
        softcast_ofdm, freq_grid, pilot_cols = self.ofdm_mapper.map(
            softcast_complex, return_freq=True
        )

        # RRC pulse shaping
        softcast_shaped = upfirdn_torch(
            softcast_ofdm, self.rrc,
            up=self.frame_config.oversample_q, down=1
        )

        # Power normalization
        ofdm_power = torch.mean(torch.abs(softcast_shaped) ** 2)
        if ofdm_power > 0:
            ofdm_power_scale = torch.sqrt(ofdm_power).item()
            softcast_shaped = softcast_shaped / ofdm_power_scale
        else:
            ofdm_power_scale = 1.0

        # Move to device
        metadata_packet = metadata_result.baseband.to(self.device)
        softcast_waveforms = softcast_shaped.to(self.device)

        # TDM: concatenate metadata + SoftCast waveforms
        combined = torch.cat([metadata_packet, softcast_waveforms])

        return SoftCastTxResult(
            metadata_packet=metadata_packet,
            softcast_waveforms=softcast_waveforms,
            combined_signal=combined,
            metadata_length=metadata_packet.numel(),
            waveform_length=softcast_waveforms.numel(),
            metadata_raw=(indices, means, vars_),
            tx_mat_real=tx_mat,
            tx_power_scale=tx_power_scale,
            softcast_pilot_cols=pilot_cols,
            softcast_ofdm_power_scale=ofdm_power_scale,
            softcast_num_iq_samples=num_iq_samples,
            overlap_iq=self.tx_config.overlap_iq,
        )

    def _real_to_complex_iq(self, real_data: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Convert real-valued data to complex via I/Q modulation.

        Pairs consecutive real samples as I (real) and Q (imaginary).

        Args:
            real_data: Real-valued numpy array

        Returns:
            Tuple of:
                - Complex torch tensor with half the length
                - Power scale factor (multiply RX by this to restore original scale)
        """
        # Flatten to 1D
        flat = real_data.flatten()

        # Pad to even length if necessary
        if len(flat) % 2 != 0:
            flat = np.concatenate([flat, [0.0]])

        # Reshape to pairs and create complex
        pairs = flat.reshape(-1, 2)
        complex_data = torch.complex(
            torch.from_numpy(pairs[:, 0].astype(np.float64)),
            torch.from_numpy(pairs[:, 1].astype(np.float64)),
        )

        # Normalize power and track scale factor
        power = torch.mean(torch.abs(complex_data) ** 2)
        if power > 0:
            scale = torch.sqrt(power).item()
            complex_data = complex_data / scale
        else:
            scale = 1.0

        return complex_data, scale

    def _real_to_complex_iq_overlap(self, real_data: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Convert real-valued data to complex via OVERLAPPING I/Q modulation.

        Creates overlapping pairs: x[i] + j*x[i+1], x[i+1] + j*x[i+2], ...
        Each real value (except endpoints) appears in TWO I/Q samples,
        providing diversity for averaging at the receiver.

        Args:
            real_data: Real-valued numpy array

        Returns:
            Tuple of:
                - Complex torch tensor with (N-1) samples for N real values
                - Power scale factor (multiply RX by this to restore original scale)
        """
        # Flatten to 1D
        flat = real_data.flatten().astype(np.float64)
        n = len(flat)

        if n < 2:
            # Edge case: single value, no overlap possible
            complex_data = torch.complex(
                torch.tensor([flat[0]] if n > 0 else [0.0], dtype=torch.float64),
                torch.tensor([0.0], dtype=torch.float64),
            )
            return complex_data, 1.0

        # Create overlapping pairs: (x[0], x[1]), (x[1], x[2]), ..., (x[n-2], x[n-1])
        # This gives n-1 complex samples
        real_parts = flat[:-1]  # x[0] to x[n-2]
        imag_parts = flat[1:]   # x[1] to x[n-1]

        complex_data = torch.complex(
            torch.from_numpy(real_parts),
            torch.from_numpy(imag_parts),
        )

        # Normalize power and track scale factor
        power = torch.mean(torch.abs(complex_data) ** 2)
        if power > 0:
            scale = torch.sqrt(power).item()
            complex_data = complex_data / scale
        else:
            scale = 1.0

        return complex_data, scale


@dataclass
class SoftCastRxResult:
    """Result of SoftCast decoding.

    Attributes:
        reconstructed_frames: Decoded video frames (H, W, T)
        metadata: Extracted metadata (indices, means, vars_)
        ber_metadata: Estimated bit error rate for metadata
    """
    reconstructed_frames: np.ndarray
    metadata: Tuple
    ber_metadata: Optional[float] = None


class SoftCastReceiver:
    """Decodes received SoftCast signals back to video frames."""

    def __init__(
        self,
        tx_config: SoftCastTxConfig,
        metadata_config: Optional[MetadataFrameConfig] = None,
        fec: Optional[FECCodec] = None,
    ):
        """Initialize SoftCast receiver.

        Args:
            tx_config: Matching transmitter configuration
            metadata_config: Metadata frame configuration
            fec: FEC codec (should match transmitter)
        """
        self.tx_config = tx_config
        self.fec = fec or PassthroughFEC()

        # Setup metadata config
        if metadata_config is None:
            modulation_order = 4 if tx_config.metadata_modulation.upper() == 'QPSK' else 2
            metadata_config = MetadataFrameConfig(
                modulation_order=modulation_order,
                x_chunks=tx_config.x_chunks,
                y_chunks=tx_config.y_chunks,
            )
        self.metadata_config = metadata_config

        # Initialize metadata receiver
        self.metadata_receiver = MetadataFrameReceiver(
            config=metadata_config,
            fec=self.fec,
        )

        # Initialize SoftCast decoder
        self.softcast = SoftCast()

    def decode(
        self,
        rx_metadata: torch.Tensor,
        rx_waveforms: torch.Tensor,
        noise_covariance: Union[float, np.ndarray],
        frames_per_gop: int,
        video_shape: Tuple[int, int] = (144, 176),
        channel_estimate: Optional[torch.Tensor] = None,
    ) -> SoftCastRxResult:
        """Decode received signals to video frames.

        Args:
            rx_metadata: Received metadata OFDM symbols (after demod)
            rx_waveforms: Received SoftCast complex waveforms
            noise_covariance: Noise covariance estimate (scalar or matrix)
            frames_per_gop: Number of frames in GOP
            video_shape: Original video (height, width)
            channel_estimate: Optional channel estimate for equalization

        Returns:
            SoftCastRxResult with reconstructed video
        """
        # Extract metadata
        indices, means, vars_ = self.metadata_receiver.extract_metadata(
            rx_symbols=rx_metadata,
            channel_estimate=channel_estimate,
        )

        # Convert received complex waveforms back to real via I/Q demod
        rx_real = self._complex_to_real_iq(rx_waveforms)

        # Reshape to tx_mat format
        chunks_per_gop = len(indices)
        chunk_size = rx_real.shape[0] // chunks_per_gop if chunks_per_gop > 0 else 0
        if chunk_size > 0:
            rx_mat = rx_real[:chunks_per_gop * chunk_size].reshape(chunks_per_gop, chunk_size)
        else:
            rx_mat = rx_real.reshape(1, -1)

        # Prepare noise covariance matrix
        if np.isscalar(noise_covariance):
            coding_noises = np.eye(rx_mat.shape[0]) * noise_covariance
        else:
            coding_noises = noise_covariance

        # Decode with SoftCast
        reconstructed = self.softcast.decode(
            metadata=(indices, means, vars_),
            data=rx_mat,
            coding_noises=coding_noises,
            frames_per_gop=frames_per_gop,
            power_budget=self.tx_config.power_budget,
            x_chunks=self.tx_config.x_chunks,
            y_chunks=self.tx_config.y_chunks,
            x_vid=video_shape[0],
            y_vid=video_shape[1],
        )

        return SoftCastRxResult(
            reconstructed_frames=reconstructed,
            metadata=(indices, means, vars_),
        )

    def _complex_to_real_iq(self, complex_data: torch.Tensor) -> np.ndarray:
        """Convert complex data back to real via I/Q demodulation.

        Interleaves real and imaginary parts.

        Args:
            complex_data: Complex torch tensor

        Returns:
            Real numpy array with twice the length
        """
        complex_np = complex_data.cpu().numpy()
        real_part = np.real(complex_np)
        imag_part = np.imag(complex_np)

        # Interleave: [r0, i0, r1, i1, ...]
        real_data = np.column_stack([real_part, imag_part]).flatten()
        return real_data


def split_tdm_signal(
    rx_signal: torch.Tensor,
    metadata_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split TDM signal into metadata and SoftCast components.

    Args:
        rx_signal: Received combined signal
        metadata_length: Length of metadata packet in samples

    Returns:
        Tuple of (metadata_signal, softcast_signal)
    """
    metadata = rx_signal[:metadata_length]
    softcast = rx_signal[metadata_length:]
    return metadata, softcast
