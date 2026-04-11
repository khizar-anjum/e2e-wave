"""
MPEG4 video transmission pipeline over underwater acoustic channels.

This module provides the main pipeline for simulating MPEG4 video transmission
through underwater acoustic channels with configurable FEC and modulation.
"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List, Union

import numpy as np
import torch

from ..fec_codec import get_fec_codec, FECCodec, PassthroughFEC
from ..channel_dataset import FrameAssembler
from ..channel_replay import load_channel_sounding, replay_filter, ChannelSounding
from ..frame_preparation import FramePrepConfig
from ..ofdm_mapper import OFDMMapper, OFDMConfig
from ..signal_utils import root_raised_cosine, upfirdn_torch

from .config import Mpeg4SimConfig, SimulationResult
from .video_utils import (
    reencode_video,
    extract_frames,
    extract_frames_from_bytes,
    get_video_info,
    bytes_to_video_file,
)
from .metrics import compute_ber_from_bits, compute_all_metrics, QualityMetrics


class Mpeg4Pipeline:
    """Pipeline for MPEG4 video transmission simulation.

    This pipeline:
    1. Encodes video to target bitrate using ffmpeg
    2. Converts bytes to bits and applies FEC
    3. Modulates with BPSK/QPSK
    4. Maps to OFDM symbols and converts to passband
    5. Applies channel (AWGN or real UWA channel)
    6. Demodulates and decodes
    7. Reconstructs video and computes quality metrics
    """

    # Default system parameters (used for AWGN or when channel doesn't specify)
    DEFAULT_BANDWIDTH_HZ = 8000.0
    DEFAULT_FC_HZ = 14000.0
    DEFAULT_CP_LENGTH = 30

    def __init__(self, config: Mpeg4SimConfig, device: Optional[torch.device] = None):
        """Initialize the pipeline.

        Args:
            config: Simulation configuration
            device: Torch device (defaults to CPU)
        """
        self.config = config
        self.device = device or torch.device('cpu')

        # Setup FEC codec
        self.fec = self._create_fec_codec(config)

        # Load channel first (needed to derive system parameters)
        self.channel: Optional[ChannelSounding] = None
        if config.channel_type == 'uwa':
            # Find channel recordings
            mat_dir = config.channel_base_dir / config.channel_name / 'mat'
            channel_paths = sorted(mat_dir.glob(f'{config.channel_name}_*.mat')) if mat_dir.exists() else []

            if channel_paths:
                # Select recording based on mode
                if config.channel_recording_mode == 'first':
                    channel_path = channel_paths[0]
                elif config.channel_recording_mode in ('random', 'fixed'):
                    rng = random.Random(config.channel_recording_seed) if config.channel_recording_seed is not None else random
                    channel_path = channel_paths[rng.randrange(len(channel_paths))]
                else:
                    channel_path = channel_paths[0]

                self.channel = load_channel_sounding(channel_path)
                self.channel.h = self.channel.h.to(self.device)
                print(f"Loaded channel: {config.channel_name} (mode={config.channel_recording_mode}, seed={config.channel_recording_seed})")
                print(f"  Recording: {channel_path.name}")
            else:
                print(f"Warning: No channel files found in {mat_dir}, falling back to AWGN")

        # Derive system parameters from channel
        bandwidth_hz, fc_hz = self._derive_system_params()
        self.bandwidth_hz = bandwidth_hz
        self.fc_hz = fc_hz

        # Auto-derive CP length based on channel delay spread
        cp_length = self._derive_cp_length(bandwidth_hz, config.num_carriers)
        self.cp_length = cp_length

        # Setup frame configuration with derived parameters
        self.frame_config = FramePrepConfig(
            num_carriers=config.num_carriers,
            cp_length=cp_length,
            modulation_order=2 if config.modulation == 'BPSK' else 4,
            oversample_q=8,
            num_ofdm_symbols=config.num_ofdm_symbols,
            bandwidth_hz=bandwidth_hz,
            fc_hz=fc_hz,
            rolloff=0.25,
            sync_length=500,
            sc_length=128,
            train_length=0,
            span=8,
        )

        # Setup OFDM configuration
        self.ofdm_config = OFDMConfig(
            num_carriers=config.num_carriers,
            cp_length=cp_length,
            pilot_period=config.pilot_period,
        )

        # Initialize components
        self.ofdm_mapper = OFDMMapper(self.ofdm_config)
        self.frame_assembler = FrameAssembler(self.frame_config, device=self.device)

        # Sampling rate
        self.fs = self.frame_config.oversample_q * self.frame_config.bandwidth_hz

    def _derive_system_params(self) -> Tuple[float, float]:
        """Derive bandwidth_hz and fc_hz from channel parameters.

        For UWA channels:
            - bandwidth_hz = channel.fs_tau / 2 (Nyquist rate)
            - fc_hz = channel.fc (carrier frequency from channel sounding)

        For AWGN:
            - Uses default values (8000 Hz, 14000 Hz)

        Returns:
            Tuple of (bandwidth_hz, fc_hz)
        """
        if self.channel is not None:
            # Derive from channel parameters
            bandwidth_hz = self.channel.fs_tau / 2.0  # Nyquist
            fc_hz = self.channel.fc
            print(f"Derived from channel: bandwidth={bandwidth_hz:.0f} Hz, fc={fc_hz:.0f} Hz")
        else:
            # Use defaults for AWGN
            bandwidth_hz = self.DEFAULT_BANDWIDTH_HZ
            fc_hz = self.DEFAULT_FC_HZ

        return bandwidth_hz, fc_hz

    def _create_fec_codec(self, config: Mpeg4SimConfig) -> FECCodec:
        """Create FEC codec based on configuration.

        Args:
            config: Simulation configuration

        Returns:
            Configured FEC codec instance
        """
        fec_type = config.fec_type

        if fec_type == 'none':
            return PassthroughFEC()
        elif fec_type == 'repetition':
            return get_fec_codec('repetition', repetitions=config.fec_repetitions)
        elif fec_type == 'ldpc':
            return get_fec_codec('ldpc',
                                 alist_path=config.ldpc_alist_path,
                                 max_iter=config.ldpc_max_iter)
        elif fec_type == 'dvbs2_ldpc':
            from ..aff3ct_codecs import DVBS2LDPCCodec
            return DVBS2LDPCCodec(k=config.dvbs2_ldpc_k,
                                  n=config.dvbs2_ldpc_n,
                                  max_iter=config.dvbs2_ldpc_max_iter)
        elif fec_type == 'polar':
            return get_fec_codec('polar',
                                 k=config.polar_k,
                                 n=config.polar_n,
                                 sigma=config.polar_sigma)
        elif fec_type == 'turbo':
            return get_fec_codec('turbo',
                                 k=config.turbo_k,
                                 max_iter=config.turbo_max_iter)
        elif fec_type == 'rsc':
            return get_fec_codec('rsc', k=config.rsc_k)
        else:
            raise ValueError(f"Unknown FEC type: {fec_type}")

    def _derive_cp_length(
        self,
        bandwidth_hz: float,
        num_carriers: int,
        safety_margin: float = 1.1,
    ) -> int:
        """Derive CP length from channel delay spread.

        Args:
            bandwidth_hz: Signal bandwidth in Hz
            num_carriers: Number of OFDM subcarriers
            safety_margin: Multiplier for delay spread (default 1.1)

        Returns:
            Derived CP length in samples
        """
        if self.channel is None or self.channel.h.numel() == 0:
            return self.DEFAULT_CP_LENGTH

        taps = self.channel.h.shape[0]
        if taps <= 1 or self.channel.fs_tau <= 0:
            return self.DEFAULT_CP_LENGTH

        # Compute delay spread in seconds
        delay_sec = (taps - 1) / self.channel.fs_tau
        delay_sec *= safety_margin

        # Convert to samples at baseband rate
        cp_samples = math.ceil(delay_sec * bandwidth_hz)
        cp_samples = max(cp_samples, 1)
        cp_samples = min(cp_samples, num_carriers - 1)

        print(f"Auto-derived CP length: {cp_samples} samples (from {taps} channel taps)")

        return cp_samples

    def bytes_to_bits(self, data: bytes) -> np.ndarray:
        """Convert bytes to bit array.

        Args:
            data: Input bytes

        Returns:
            Numpy array of bits (0s and 1s)
        """
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    def bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array back to bytes.

        Args:
            bits: Array of bits (0s and 1s)

        Returns:
            Bytes object
        """
        # Pad to multiple of 8 if needed
        pad_len = (8 - len(bits) % 8) % 8
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
        return np.packbits(bits.astype(np.uint8)).tobytes()

    def modulate(self, bits: np.ndarray) -> torch.Tensor:
        """Modulate bits to BPSK or QPSK symbols.

        Args:
            bits: Binary array (0/1)

        Returns:
            Complex symbol tensor
        """
        bits_tensor = torch.from_numpy(bits.astype(np.float64)).to(self.device)

        if self.config.modulation == 'BPSK':
            # BPSK: 0 -> -1, 1 -> +1
            mapped = 2 * bits_tensor - 1
            symbols = torch.complex(mapped, torch.zeros_like(mapped))
        else:
            # QPSK: pairs of bits to I/Q
            if bits_tensor.numel() % 2 != 0:
                bits_tensor = torch.cat([bits_tensor, torch.zeros(1, dtype=bits_tensor.dtype, device=self.device)])
            bit_pairs = bits_tensor.view(-1, 2)
            symbols = torch.complex(
                2 * bit_pairs[:, 0] - 1,
                2 * bit_pairs[:, 1] - 1,
            ) / math.sqrt(2)

        return symbols

    def demodulate(self, symbols: torch.Tensor) -> np.ndarray:
        """Demodulate symbols to bits using hard decision.

        Args:
            symbols: Complex symbol tensor

        Returns:
            Binary array (0/1)
        """
        if self.config.modulation == 'BPSK':
            # BPSK: sign of real part
            bits = (symbols.real >= 0).to(torch.int64)
            return bits.cpu().numpy().astype(np.uint8)
        else:
            # QPSK: signs of I and Q
            real_bits = (symbols.real >= 0).to(torch.int64)
            imag_bits = (symbols.imag >= 0).to(torch.int64)
            bits = torch.stack([real_bits, imag_bits], dim=1).reshape(-1)
            return bits.cpu().numpy().astype(np.uint8)

    def demodulate_soft(self, symbols: torch.Tensor, noise_var: float) -> np.ndarray:
        """Compute log-likelihood ratios (LLRs) for soft-decision decoding.

        This method computes LLRs from received symbols using the channel
        noise variance. LLRs provide reliability information that enables
        significantly better error correction with advanced FEC codes.

        Args:
            symbols: Complex received symbols
            noise_var: Noise variance (sigma^2). For SNR in dB:
                       noise_var = 1 / (10^(SNR_dB/10))

        Returns:
            1D array of LLRs (float32).
            Convention: positive LLR = more likely bit 0,
                       negative LLR = more likely bit 1.
        """
        # Avoid division by zero
        if noise_var <= 0:
            noise_var = 1e-10

        # Scale factor: 2/sigma^2 for AWGN channel
        # Note: Our modulation maps bit 0 -> -1, bit 1 -> +1
        # LLR convention: positive LLR = more likely bit 0
        # So we need: received -1 (from bit 0) -> positive LLR
        #             received +1 (from bit 1) -> negative LLR
        # Therefore: LLR = -symbol * scale
        scale = -2.0 / noise_var

        if self.config.modulation == 'BPSK':
            # BPSK: real part only
            llrs = symbols.real * scale
        else:
            # QPSK: I and Q channels
            # Scale by sqrt(2) to account for QPSK power normalization
            llr_i = symbols.real * scale * math.sqrt(2)
            llr_q = symbols.imag * scale * math.sqrt(2)
            # Interleave: [I0, Q0, I1, Q1, ...]
            llrs = torch.stack([llr_i, llr_q], dim=1).reshape(-1)

        return llrs.cpu().numpy().astype(np.float32)

    def add_awgn(self, signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add AWGN noise to signal.

        Args:
            signal: Input signal (real passband)
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Noisy signal
        """
        signal_power = torch.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.sqrt(noise_power) * torch.randn_like(signal)
        return signal + noise

    def transmit_symbols(
        self,
        symbols: torch.Tensor,
        snr_db: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transmit symbols through OFDM channel.

        Args:
            symbols: Complex data symbols
            snr_db: SNR in dB

        Returns:
            Tuple of (received_symbols, pilot_columns)
        """
        # OFDM modulation
        ofdm_signal, freq_grid, pilot_cols = self.ofdm_mapper.map(symbols, return_freq=True)

        # Frame assembly (add sync, SC preamble, convert to passband)
        wrap_result = self.frame_assembler.wrap_segments([ofdm_signal])
        passband = wrap_result.passband

        # Channel
        if self.channel is not None and self.config.channel_type == 'uwa':
            # Real UWA channel
            rx_passband = replay_filter(passband, self.fs, self.channel)
        else:
            # Flat channel (passthrough)
            rx_passband = passband.clone()

        # RMS normalization (ensures proper SNR calibration)
        # Normalize to unit RMS power before adding AWGN
        rms = torch.sqrt(torch.mean(rx_passband ** 2))
        if rms > 0:
            rx_passband = rx_passband / rms

        # Add AWGN (now calibrated to normalized signal)
        rx_passband = self.add_awgn(rx_passband, snr_db)

        # Receiver processing
        rx_symbols = self._receive(rx_passband, freq_grid, pilot_cols)

        return rx_symbols, pilot_cols

    def _receive(
        self,
        rx_passband: torch.Tensor,
        tx_freq_grid: torch.Tensor,
        pilot_cols: torch.Tensor,
    ) -> torch.Tensor:
        """Receive and demodulate OFDM signal.

        Args:
            rx_passband: Received passband signal
            tx_freq_grid: Transmitted frequency grid (for reference)
            pilot_cols: Pilot column indices

        Returns:
            Received symbols after equalization
        """
        # Downconvert to baseband
        t = torch.arange(rx_passband.numel(), dtype=torch.float64, device=rx_passband.device) / self.fs
        baseband = rx_passband.to(torch.float64) * torch.exp(-1j * 2 * math.pi * self.frame_config.fc_hz * t)

        # Matched filter (RRC)
        rrc = root_raised_cosine(
            self.frame_config.rolloff,
            self.frame_config.span,
            self.frame_config.oversample_q
        ).to(self.device)
        filtered = upfirdn_torch(baseband, rrc, up=1, down=1)

        # Downsample
        downsampled = filtered[::self.frame_config.oversample_q]

        # Skip preamble (sync + SC + train)
        sync_samples = self.frame_config.sync_length
        sc_samples = self.frame_config.sc_length
        train_samples = max(1, self.frame_config.train_length)
        preamble_samples = sync_samples + sc_samples + train_samples
        data_start = preamble_samples + self.frame_config.span

        if data_start >= len(downsampled):
            return torch.zeros(0, dtype=torch.cdouble, device=self.device)

        data_signal = downsampled[data_start:]

        # OFDM demodulation
        sym_len = self.ofdm_config.num_carriers + self.ofdm_config.cp_length
        num_symbols = len(data_signal) // sym_len

        if num_symbols == 0:
            return torch.zeros(0, dtype=torch.cdouble, device=self.device)

        # Reshape into OFDM symbols
        ofdm_symbols = data_signal[:num_symbols * sym_len].view(num_symbols, sym_len)

        # Remove CP
        without_cp = ofdm_symbols[:, self.ofdm_config.cp_length:]

        # FFT
        freq = torch.fft.fft(without_cp, dim=1) / self.ofdm_config.num_carriers
        freq = freq.t()  # (num_carriers, num_symbols)

        # Channel estimation and equalization using pilots
        freq_eq = self._equalize(freq, pilot_cols)

        # Extract data symbols (skip pilot columns)
        data_symbols = self._extract_data(freq_eq, pilot_cols)

        return data_symbols

    def _equalize(
        self,
        freq: torch.Tensor,
        pilot_cols: torch.Tensor,
    ) -> torch.Tensor:
        """Equalize using pilot-based channel estimation.

        Args:
            freq: Frequency domain symbols (num_carriers, num_symbols)
            pilot_cols: Pilot column indices

        Returns:
            Equalized frequency domain symbols
        """
        if len(pilot_cols) == 0:
            return freq

        # Get pilot columns
        pilot_vals = freq[:, pilot_cols]  # (num_carriers, num_pilots)

        # Expected pilot value
        expected = self.ofdm_config.pilot_value

        # Channel estimate at pilot positions
        h_pilots = pilot_vals / expected  # (num_carriers, num_pilots)

        # Interpolate channel estimate across all symbols
        num_symbols = freq.shape[1]
        h_est = torch.zeros_like(freq)

        for carrier in range(freq.shape[0]):
            # Simple linear interpolation
            pilot_indices = pilot_cols.cpu().numpy()
            pilot_values = h_pilots[carrier].cpu().numpy()

            for i in range(num_symbols):
                # Find surrounding pilots
                left_idx = np.searchsorted(pilot_indices, i, side='right') - 1
                right_idx = left_idx + 1

                if left_idx < 0:
                    h_est[carrier, i] = pilot_values[0]
                elif right_idx >= len(pilot_indices):
                    h_est[carrier, i] = pilot_values[-1]
                else:
                    # Linear interpolation
                    left_pos = pilot_indices[left_idx]
                    right_pos = pilot_indices[right_idx]
                    alpha = (i - left_pos) / (right_pos - left_pos)
                    h_est[carrier, i] = (1 - alpha) * pilot_values[left_idx] + alpha * pilot_values[right_idx]

        # Equalize
        freq_eq = freq / (h_est + 1e-9)

        return freq_eq

    def _extract_data(
        self,
        freq: torch.Tensor,
        pilot_cols: torch.Tensor,
    ) -> torch.Tensor:
        """Extract data symbols from frequency grid (skip pilots).

        Args:
            freq: Frequency domain symbols (num_carriers, num_symbols)
            pilot_cols: Pilot column indices

        Returns:
            Flattened data symbols
        """
        num_carriers, num_symbols = freq.shape
        pilot_set = set(pilot_cols.cpu().numpy().tolist())

        data_cols = [i for i in range(num_symbols) if i not in pilot_set]
        if len(data_cols) == 0:
            return torch.zeros(0, dtype=freq.dtype, device=freq.device)

        data = freq[:, data_cols]  # (num_carriers, num_data_symbols)
        return data.t().reshape(-1)  # Flatten

    def transmit_bytes(
        self,
        data: bytes,
        snr_db: float,
    ) -> Tuple[bytes, float, int, int]:
        """Transmit bytes through channel.

        Uses soft-decision decoding with LLRs when the FEC codec supports it,
        otherwise falls back to hard-decision decoding.

        Args:
            data: Input bytes
            snr_db: SNR in dB

        Returns:
            Tuple of (received_bytes, ber, num_errors, total_bits)
        """
        # Bytes to bits
        original_bits = self.bytes_to_bits(data)
        num_original_bits = len(original_bits)

        # FEC encode
        encoded_bits = self.fec.encode(original_bits)

        # Modulate
        symbols = self.modulate(encoded_bits)

        # Transmit through channel
        rx_symbols, _ = self.transmit_symbols(symbols, snr_db)

        # Demodulate and decode
        # Use soft decoding if FEC codec supports it AND channel is not UWA
        # Note: For UWA channels, the noise variance estimate is incorrect after
        # OFDM equalization (ZF enhances noise on weak subcarriers), causing
        # soft decoding to produce over-confident LLRs. Use hard decoding instead.
        use_soft_decoding = (
            self.fec.supports_soft_decoding() and
            self.config.channel_type != 'uwa'
        )

        if use_soft_decoding:
            # Compute noise variance from SNR
            # SNR = signal_power / noise_power
            # For unit signal power: noise_var = 1 / (10^(SNR_dB/10))
            noise_var = 1.0 / (10 ** (snr_db / 10))

            # Get LLRs from soft demodulation
            llrs = self.demodulate_soft(rx_symbols, noise_var)

            # Soft-decision FEC decode
            decoded_bits = self.fec.decode_soft(llrs)
        else:
            # Hard-decision demodulation and decoding
            rx_bits = self.demodulate(rx_symbols)
            decoded_bits = self.fec.decode(rx_bits)

        # Trim to original length
        decoded_bits = decoded_bits[:num_original_bits]

        # Compute BER
        ber, num_errors, total_bits = compute_ber_from_bits(original_bits, decoded_bits)

        # Bits to bytes
        rx_bytes = self.bits_to_bytes(decoded_bits)

        # Trim to original length
        rx_bytes = rx_bytes[:len(data)]

        return rx_bytes, ber, num_errors, total_bits

    def run_simulation(
        self,
        snr_db: float,
        output_dir: Optional[Path] = None,
    ) -> SimulationResult:
        """Run full video transmission simulation.

        Computes end-to-end quality by comparing received frames against a
        high-quality reference encoding (not the low-bitrate transmission
        encoding). This captures both compression and transmission degradation.

        Args:
            snr_db: SNR in dB
            output_dir: Optional output directory for results

        Returns:
            SimulationResult with all metrics
        """
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video info
        video_info = get_video_info(self.config.video_path)

        # Re-encode video at target bitrate
        with tempfile.TemporaryDirectory() as tmpdir:
            # High-quality reference at target resolution (for PSNR/SSIM)
            reference_path = Path(tmpdir) / 'reference.mp4'
            reencode_video(
                self.config.video_path,
                reference_path,
                resolution=self.config.resolution,
                bitrate_bps=1_000_000,  # High bitrate for quality reference
            )
            reference_frames = extract_frames(reference_path)

            # Low-bitrate encoding for transmission
            encoded_path = Path(tmpdir) / 'encoded.mp4'
            video_bytes = reencode_video(
                self.config.video_path,
                encoded_path,
                resolution=self.config.resolution,
                bitrate_bps=self.config.target_bitrate_bps,
            )

            # Extract encoded frames (for frame count reference)
            original_frames = extract_frames(encoded_path)

        # Calculate transmission time
        total_bits = len(video_bytes) * 8
        effective_bps = self.config.bits_per_frame / (
            self.frame_assembler.sync_signal.numel() +
            self.frame_assembler.sc_signal.numel() +
            self.frame_assembler.train_signal.numel() +
            (self.ofdm_config.num_carriers + self.ofdm_config.cp_length) *
            self.config.num_ofdm_symbols * self.frame_config.oversample_q
        ) * self.fs * self.config.fec_rate

        transmission_time = total_bits / effective_bps if effective_bps > 0 else 0

        # Transmit video bytes
        rx_bytes, ber, bit_errors, total_bits_tx = self.transmit_bytes(video_bytes, snr_db)

        # Attempt to reconstruct video
        rx_frames, playable, frames_decoded = extract_frames_from_bytes(
            rx_bytes,
            self.config.resolution[0],
            self.config.resolution[1],
        )

        # Compute video quality metrics against HIGH-QUALITY REFERENCE
        # This measures end-to-end quality including compression + transmission loss
        if len(reference_frames) > 0 and len(rx_frames) > 0:
            metrics = compute_all_metrics(
                video_bytes, rx_bytes,
                reference_frames, rx_frames,  # Compare against high-quality reference
                max_val=255.0,
            )
            psnr_mean = metrics.psnr_mean
            ssim_mean = metrics.ssim_mean
            psnr_per_frame = metrics.psnr_per_frame.tolist()
            ssim_per_frame = metrics.ssim_per_frame.tolist()
        else:
            psnr_mean = 0.0
            ssim_mean = 0.0
            psnr_per_frame = []
            ssim_per_frame = []

        return SimulationResult(
            snr_db=snr_db,
            ber=ber,
            bit_errors=bit_errors,
            total_bits=total_bits_tx,
            psnr_per_frame=psnr_per_frame,
            ssim_per_frame=ssim_per_frame,
            psnr_mean=psnr_mean,
            ssim_mean=ssim_mean,
            playable=playable,
            frames_decoded=frames_decoded,
            frames_total=len(original_frames),
            transmission_time_sec=transmission_time,
        )

    def run_snr_sweep(
        self,
        snr_range: List[float],
        output_dir: Optional[Path] = None,
    ) -> List[SimulationResult]:
        """Run simulation across multiple SNR values.

        Args:
            snr_range: List of SNR values in dB
            output_dir: Optional output directory

        Returns:
            List of SimulationResult for each SNR
        """
        results = []
        for snr_db in snr_range:
            print(f"Running simulation at SNR = {snr_db} dB...")
            result = self.run_simulation(snr_db, output_dir)
            results.append(result)
            print(f"  BER: {result.ber:.6f}, PSNR: {result.psnr_mean:.2f} dB, SSIM: {result.ssim_mean:.4f}")

        return results
