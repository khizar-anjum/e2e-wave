"""
Metadata Frame Builder for SoftCast Integration

Handles serialization, modulation (BPSK/QPSK), and OFDM mapping for
SoftCast metadata (indices, means, variances). The receiver knows
chunks_per_gop from config, so no length prefix is needed.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .fec_codec import FECCodec, PassthroughFEC
from .ofdm_mapper import OFDMConfig, OFDMMapper
from .signal_utils import root_raised_cosine, upfirdn_torch


@dataclass
class MetadataFrameConfig:
    """Configuration for metadata OFDM frame.

    Attributes:
        num_carriers: Number of OFDM subcarriers
        cp_length: Cyclic prefix length in samples
        modulation_order: 2 for BPSK, 4 for QPSK
        pilot_period: OFDM symbols between pilot columns
        pilot_value: Complex pilot symbol value
        oversample_q: Upsampling factor for RRC filtering
        bandwidth_hz: Signal bandwidth in Hz
        fc_hz: Carrier frequency in Hz
        rolloff: RRC filter rolloff factor
        span: RRC filter span in symbols
        chunks_per_gop: Number of DCT chunks (from SoftCast config)
        x_chunks: Spatial chunk dimension X (from SoftCast config)
        y_chunks: Spatial chunk dimension Y (from SoftCast config)
        frames_per_gop: Frames in GOP (from SoftCast config)
    """
    num_carriers: int = 16
    cp_length: int = 16
    modulation_order: int = 2  # 2=BPSK, 4=QPSK
    pilot_period: int = 4
    pilot_value: complex = 1 + 0j
    oversample_q: int = 8
    bandwidth_hz: float = 4e3
    fc_hz: float = 18e3  # Different from main signal to avoid overlap
    rolloff: float = 0.25
    span: int = 8
    # SoftCast parameters (needed for serialization)
    chunks_per_gop: int = 64
    x_chunks: int = 8
    y_chunks: int = 8
    frames_per_gop: int = 8

    def bits_per_index(self) -> int:
        """Bits needed for each index dimension."""
        max_val = max(self.x_chunks, self.y_chunks, self.frames_per_gop)
        return int(np.ceil(np.log2(max_val + 1)))

    def total_metadata_bits(self) -> int:
        """Total bits needed for all metadata."""
        bits_per_idx = self.bits_per_index()
        # Each index is (i, j, k) tuple
        index_bits = self.chunks_per_gop * 3 * bits_per_idx
        # means and vars_ are float32 (32 bits each)
        float_bits = self.chunks_per_gop * 32 * 2
        return index_bits + float_bits


@dataclass
class MetadataFrameResult:
    """Result of metadata frame building."""
    baseband: torch.Tensor
    passband: torch.Tensor
    ofdm_symbols: torch.Tensor
    freq_grid: torch.Tensor
    pilot_columns: torch.Tensor
    encoded_bits: np.ndarray
    fs: float


class MetadataSerializer:
    """Serializes SoftCast metadata to/from bit arrays."""

    def __init__(self, config: MetadataFrameConfig):
        self.config = config
        self.bits_per_idx = config.bits_per_index()

    def serialize(
        self,
        indices: List[Tuple[int, int, int]],
        means: np.ndarray,
        vars_: np.ndarray,
    ) -> np.ndarray:
        """Serialize metadata to bit array.

        Args:
            indices: List of (i, j, k) tuples
            means: Array of chunk means (float32)
            vars_: Array of chunk variances (float32)

        Returns:
            1D numpy array of bits (0/1)
        """
        bits = []

        # Serialize indices
        for i, j, k in indices:
            bits.extend(self._int_to_bits(i, self.bits_per_idx))
            bits.extend(self._int_to_bits(j, self.bits_per_idx))
            bits.extend(self._int_to_bits(k, self.bits_per_idx))

        # Pad if fewer indices than expected
        remaining = self.config.chunks_per_gop - len(indices)
        for _ in range(remaining):
            bits.extend([0] * (3 * self.bits_per_idx))

        # Serialize means (float32 -> 32 bits)
        for val in means:
            bits.extend(self._float32_to_bits(float(val)))
        for _ in range(self.config.chunks_per_gop - len(means)):
            bits.extend([0] * 32)

        # Serialize variances (float32 -> 32 bits)
        for val in vars_:
            bits.extend(self._float32_to_bits(float(val)))
        for _ in range(self.config.chunks_per_gop - len(vars_)):
            bits.extend([0] * 32)

        return np.array(bits, dtype=np.uint8)

    def deserialize(
        self, bits: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int]], np.ndarray, np.ndarray]:
        """Deserialize bit array back to metadata.

        Args:
            bits: 1D numpy array of bits

        Returns:
            Tuple of (indices, means, vars_)
        """
        cursor = 0
        indices = []
        means = []
        vars_ = []

        # Read indices
        for _ in range(self.config.chunks_per_gop):
            i = self._bits_to_int(bits[cursor:cursor + self.bits_per_idx])
            cursor += self.bits_per_idx
            j = self._bits_to_int(bits[cursor:cursor + self.bits_per_idx])
            cursor += self.bits_per_idx
            k = self._bits_to_int(bits[cursor:cursor + self.bits_per_idx])
            cursor += self.bits_per_idx
            indices.append((i, j, k))

        # Read means
        for _ in range(self.config.chunks_per_gop):
            val = self._bits_to_float32(bits[cursor:cursor + 32])
            cursor += 32
            means.append(val)

        # Read variances
        for _ in range(self.config.chunks_per_gop):
            val = self._bits_to_float32(bits[cursor:cursor + 32])
            cursor += 32
            vars_.append(val)

        return indices, np.array(means, dtype=np.float32), np.array(vars_, dtype=np.float32)

    def _int_to_bits(self, val: int, num_bits: int) -> List[int]:
        """Convert integer to fixed-width bit list (MSB first)."""
        return [(val >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]

    def _bits_to_int(self, bits: np.ndarray) -> int:
        """Convert bit array to integer (MSB first)."""
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val

    def _float32_to_bits(self, val: float) -> List[int]:
        """Convert float32 to 32 bits (IEEE 754)."""
        packed = struct.pack('>f', val)
        bits = []
        for byte in packed:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    def _bits_to_float32(self, bits: np.ndarray) -> float:
        """Convert 32 bits to float32 (IEEE 754)."""
        bytes_list = []
        for i in range(0, 32, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | int(bits[i + j])
            bytes_list.append(byte)
        packed = bytes(bytes_list)
        return struct.unpack('>f', packed)[0]


class MetadataFrameBuilder:
    """Builds OFDM-modulated metadata packets for SoftCast.

    Handles serialization, optional FEC, modulation (BPSK/QPSK),
    and OFDM mapping with pilot symbols.
    """

    def __init__(
        self,
        config: MetadataFrameConfig,
        fec: Optional[FECCodec] = None,
    ):
        self.config = config
        self.fec = fec or PassthroughFEC()
        self.serializer = MetadataSerializer(config)

        # Setup OFDM mapper
        ofdm_config = OFDMConfig(
            num_carriers=config.num_carriers,
            cp_length=config.cp_length,
            pilot_period=config.pilot_period,
            pilot_value=config.pilot_value,
        )
        self.ofdm_mapper = OFDMMapper(ofdm_config)

        # RRC filter
        self.rrc = root_raised_cosine(
            config.rolloff, config.span, config.oversample_q
        )

    def build_packet(
        self,
        indices: List[Tuple[int, int, int]],
        means: np.ndarray,
        vars_: np.ndarray,
    ) -> MetadataFrameResult:
        """Build complete metadata packet.

        Args:
            indices: List of (i, j, k) index tuples from SoftCast
            means: Array of chunk means
            vars_: Array of chunk variances

        Returns:
            MetadataFrameResult with baseband and passband signals
        """
        # Serialize metadata to bits
        raw_bits = self.serializer.serialize(indices, means, vars_)

        # Apply FEC encoding
        encoded_bits = self.fec.encode(raw_bits)

        # Modulate bits to complex symbols
        symbols = self._modulate(encoded_bits)

        # OFDM mapping
        ofdm_signal, freq_grid, pilot_cols = self.ofdm_mapper.map(
            symbols, return_freq=True
        )

        # RRC pulse shaping
        baseband = upfirdn_torch(ofdm_signal, self.rrc, up=self.config.oversample_q, down=1)

        # Normalize power
        baseband = baseband / torch.sqrt(torch.mean(torch.abs(baseband) ** 2) + 1e-12)

        # Passband modulation
        fs = self.config.oversample_q * self.config.bandwidth_hz
        t = torch.arange(baseband.numel(), dtype=torch.float64) / fs
        passband = torch.real(
            baseband * torch.exp(1j * 2 * math.pi * self.config.fc_hz * t)
        )

        return MetadataFrameResult(
            baseband=baseband,
            passband=passband,
            ofdm_symbols=symbols,
            freq_grid=freq_grid,
            pilot_columns=pilot_cols,
            encoded_bits=encoded_bits,
            fs=fs,
        )

    def _modulate(self, bits: np.ndarray) -> torch.Tensor:
        """Modulate bits to BPSK or QPSK symbols.

        Args:
            bits: Binary array (0/1)

        Returns:
            Complex symbol tensor
        """
        bits = torch.from_numpy(bits.astype(np.float64))
        bits_per_symbol = int(math.log2(self.config.modulation_order))

        if bits_per_symbol == 1:
            # BPSK: 0 -> -1, 1 -> +1
            mapped = 2 * bits - 1
            symbols = torch.complex(mapped, torch.zeros_like(mapped))
        elif bits_per_symbol == 2:
            # QPSK: pairs of bits to I/Q
            # Pad to even length if needed
            if bits.numel() % 2 != 0:
                bits = torch.cat([bits, torch.zeros(1, dtype=bits.dtype)])
            bit_pairs = bits.view(-1, 2)
            symbols = torch.complex(
                2 * bit_pairs[:, 0] - 1,
                2 * bit_pairs[:, 1] - 1,
            ) / math.sqrt(2)
        else:
            raise NotImplementedError(
                f"Modulation order {self.config.modulation_order} not supported"
            )

        return symbols


class MetadataFrameReceiver:
    """Demodulates and extracts metadata from received OFDM packet."""

    def __init__(
        self,
        config: MetadataFrameConfig,
        fec: Optional[FECCodec] = None,
    ):
        self.config = config
        self.fec = fec or PassthroughFEC()
        self.serializer = MetadataSerializer(config)

    def extract_metadata(
        self,
        rx_symbols: torch.Tensor,
        channel_estimate: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, int, int]], np.ndarray, np.ndarray]:
        """Extract metadata from received OFDM symbols.

        Args:
            rx_symbols: Received complex symbols (after OFDM demod)
            channel_estimate: Optional channel estimate for equalization

        Returns:
            Tuple of (indices, means, vars_)
        """
        # Equalize if channel estimate provided
        if channel_estimate is not None:
            rx_symbols = rx_symbols / (channel_estimate + 1e-12)

        # Demodulate symbols to bits
        bits = self._demodulate(rx_symbols)

        # Apply FEC decoding
        decoded_bits = self.fec.decode(bits)

        # Deserialize to metadata
        return self.serializer.deserialize(decoded_bits)

    def _demodulate(self, symbols: torch.Tensor) -> np.ndarray:
        """Demodulate BPSK or QPSK symbols to bits.

        Args:
            symbols: Complex symbol tensor

        Returns:
            Binary numpy array
        """
        symbols = symbols.cpu().numpy()
        bits_per_symbol = int(math.log2(self.config.modulation_order))

        if bits_per_symbol == 1:
            # BPSK: real part sign
            bits = (np.real(symbols) > 0).astype(np.uint8)
        elif bits_per_symbol == 2:
            # QPSK: sign of I and Q
            i_bits = (np.real(symbols) > 0).astype(np.uint8)
            q_bits = (np.imag(symbols) > 0).astype(np.uint8)
            bits = np.column_stack([i_bits, q_bits]).flatten()
        else:
            raise NotImplementedError(
                f"Modulation order {self.config.modulation_order} not supported"
            )

        return bits
