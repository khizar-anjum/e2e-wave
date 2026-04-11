"""
Unequal Error Protection (UEP) Pipeline for VP9 Temporal SVC.

Implements a 2-layer transmission scheme:
- Base layer (TID=0): Stronger FEC (e.g., rate 0.2)
- Enhancement layer (TID=1): Weaker FEC (e.g., rate 0.5)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch

from ..fec_codec import FECCodec
from .mpeg_pipeline import Mpeg4Pipeline
from .config import Mpeg4SimConfig
from .vp9_svc import (
    VP9SVCEncoder, VP9LayerData, VP9SuperframeParser,
    create_ivf_from_layers, create_ivf_base_only
)
from .video_utils import extract_frames_from_bytes
from .metrics import img_psnr, img_ssim


@dataclass
class UEPConfig:
    """Configuration for UEP transmission."""
    # FEC rates for each layer
    base_layer_fec_rate: float = 0.2    # Strong protection
    enhancement_layer_fec_rate: float = 0.5  # Weaker protection

    # Video encoding
    resolution: Tuple[int, int] = (64, 64)
    base_bitrate_kbps: int = 50
    total_bitrate_kbps: int = 100
    fps: Optional[float] = None

    # OFDM settings (inherited from Mpeg4SimConfig)
    num_carriers: int = 64
    num_ofdm_symbols: int = 16
    cp_length: int = 30
    pilot_period: int = 4
    modulation: str = 'QPSK'
    bandwidth_hz: float = 8e3
    fc_hz: float = 14e3

    # Channel settings
    channel_type: str = 'awgn'
    channel_name: str = 'NOF1'
    channel_base_dir: Path = field(default_factory=lambda: Path('input/channels'))

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path('results/vp9'))

    def __post_init__(self):
        if isinstance(self.channel_base_dir, str):
            self.channel_base_dir = Path(self.channel_base_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class UEPResult:
    """Results from UEP simulation."""
    snr_db: float

    # Base layer metrics
    base_ber: float
    base_bit_errors: int
    base_total_bits: int
    base_decoded_ok: bool

    # Enhancement layer metrics
    enhancement_ber: float
    enhancement_bit_errors: int
    enhancement_total_bits: int
    enhancement_decoded_ok: bool

    # Combined video metrics
    psnr_base_only: float      # PSNR with base layer only
    psnr_both_layers: float    # PSNR with both layers
    ssim_base_only: float
    ssim_both_layers: float

    # Overall metrics
    playable_base: bool
    playable_both: bool
    frames_decoded_base: int
    frames_decoded_both: int
    frames_total: int

    # Received video bytes for saving
    base_only_ivf: Optional[bytes] = None
    both_layers_ivf: Optional[bytes] = None

    # Comparison with uniform FEC
    uniform_ber: Optional[float] = None
    uniform_psnr: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            'snr_db': self.snr_db,
            'base_ber': self.base_ber,
            'base_bit_errors': self.base_bit_errors,
            'base_total_bits': self.base_total_bits,
            'base_decoded_ok': self.base_decoded_ok,
            'enhancement_ber': self.enhancement_ber,
            'enhancement_bit_errors': self.enhancement_bit_errors,
            'enhancement_total_bits': self.enhancement_total_bits,
            'enhancement_decoded_ok': self.enhancement_decoded_ok,
            'psnr_base_only': self.psnr_base_only,
            'psnr_both_layers': self.psnr_both_layers,
            'ssim_base_only': self.ssim_base_only,
            'ssim_both_layers': self.ssim_both_layers,
            'playable_base': self.playable_base,
            'playable_both': self.playable_both,
            'frames_decoded_base': self.frames_decoded_base,
            'frames_decoded_both': self.frames_decoded_both,
            'frames_total': self.frames_total,
            'uniform_ber': self.uniform_ber,
            'uniform_psnr': self.uniform_psnr,
        }


class UEPPipeline:
    """Pipeline for VP9 temporal SVC with Unequal Error Protection."""

    def __init__(
        self,
        config: UEPConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device('cpu')

        # Create FEC codecs for each layer
        self.base_fec = self._create_fec_codec(config.base_layer_fec_rate)
        self.enhancement_fec = self._create_fec_codec(config.enhancement_layer_fec_rate)

        # Create VP9 encoder
        self.vp9_encoder = VP9SVCEncoder(
            num_layers=2,
            base_bitrate_kbps=config.base_bitrate_kbps,
            total_bitrate_kbps=config.total_bitrate_kbps,
        )

        # Create underlying MPEG pipeline for modulation/channel
        # (with passthrough FEC since we handle FEC ourselves)
        self._mpeg_config = Mpeg4SimConfig(
            resolution=config.resolution,
            modulation=config.modulation,
            fec_type='none',  # We handle FEC separately
            num_carriers=config.num_carriers,
            num_ofdm_symbols=config.num_ofdm_symbols,
            cp_length=config.cp_length,
            pilot_period=config.pilot_period,
            bandwidth_hz=config.bandwidth_hz,
            fc_hz=config.fc_hz,
            channel_type=config.channel_type,
            channel_name=config.channel_name,
            channel_base_dir=config.channel_base_dir,
        )
        self._mpeg_pipeline = Mpeg4Pipeline(self._mpeg_config, device=device)

        # Parser for splitting layer bytes back to frames
        self._parser = VP9SuperframeParser()

    def _create_fec_codec(self, target_rate: float) -> FECCodec:
        """Create DVB-S2 LDPC codec for target rate."""
        from ..aff3ct_codecs import DVBS2LDPCCodec
        rate, k, n = DVBS2LDPCCodec.find_code(target_rate, frame='short')
        print(f"  FEC codec: rate={rate:.3f} (K={k}, N={n})")
        return DVBS2LDPCCodec(k=k, n=n)

    def transmit_layer(
        self,
        layer_bytes: bytes,
        fec: FECCodec,
        snr_db: float,
    ) -> Tuple[bytes, float, int, int]:
        """Transmit a single layer through the channel.

        Args:
            layer_bytes: Raw layer bytes to transmit
            fec: FEC codec to use
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Tuple of (received_bytes, ber, errors, total_bits)
        """
        if len(layer_bytes) == 0:
            return b'', 0.0, 0, 0

        # Convert to bits
        original_bits = self._mpeg_pipeline.bytes_to_bits(layer_bytes)
        num_original_bits = len(original_bits)

        # FEC encode
        encoded_bits = fec.encode(original_bits)

        # Modulate
        symbols = self._mpeg_pipeline.modulate(encoded_bits)

        # Transmit through channel
        rx_symbols, _ = self._mpeg_pipeline.transmit_symbols(symbols, snr_db)

        # Demodulate and decode
        if fec.supports_soft_decoding():
            noise_var = 1.0 / (10 ** (snr_db / 10))
            llrs = self._mpeg_pipeline.demodulate_soft(rx_symbols, noise_var)
            decoded_bits = fec.decode_soft(llrs)
        else:
            rx_bits = self._mpeg_pipeline.demodulate(rx_symbols)
            decoded_bits = fec.decode(rx_bits)

        # Trim to original length
        decoded_bits = decoded_bits[:num_original_bits]

        # Compute BER
        min_len = min(len(original_bits), len(decoded_bits))
        errors = int(np.sum(original_bits[:min_len] != decoded_bits[:min_len]))
        errors += abs(len(original_bits) - len(decoded_bits))
        total_bits = max(len(original_bits), len(decoded_bits))
        ber = errors / total_bits if total_bits > 0 else 0.0

        # Convert back to bytes
        rx_bytes = self._mpeg_pipeline.bits_to_bytes(decoded_bits)
        rx_bytes = rx_bytes[:len(layer_bytes)]

        return rx_bytes, float(ber), int(errors), int(total_bits)

    def run_simulation(
        self,
        video_path: Path,
        snr_db: float,
        compare_uniform: bool = True,
    ) -> UEPResult:
        """Run UEP simulation at a given SNR.

        Args:
            video_path: Path to input video
            snr_db: Signal-to-noise ratio in dB
            compare_uniform: If True, also run uniform FEC for comparison

        Returns:
            UEPResult with all metrics
        """
        # Encode video with VP9 temporal SVC
        ivf_bytes, temp_ivf = self.vp9_encoder.encode(
            video_path,
            resolution=self.config.resolution,
            fps=self.config.fps,
        )

        # Parse layers
        layer_data = self.vp9_encoder.parse_layers(ivf_bytes)

        print(f"  VP9 SVC encoded: {len(ivf_bytes)} bytes, {layer_data.total_frames} frames")
        print(f"    Base layer: {layer_data.base_layer_size} bytes, {len(layer_data.base_frame_indices)} frames")
        print(f"    Enhancement: {layer_data.enhancement_layer_size} bytes, {len(layer_data.enhancement_frame_indices)} frames")

        # Extract original frames for quality comparison
        original_frames, _, _ = self._extract_frames_from_ivf(ivf_bytes)

        # Transmit base layer with strong FEC
        rx_base, base_ber, base_errors, base_bits = self.transmit_layer(
            layer_data.base_layer_bytes,
            self.base_fec,
            snr_db,
        )

        # Transmit enhancement layer with weaker FEC
        rx_enhancement, enh_ber, enh_errors, enh_bits = self.transmit_layer(
            layer_data.enhancement_layer_bytes,
            self.enhancement_fec,
            snr_db,
        )

        # Split received bytes back into frames
        rx_base_frames = self._parser.split_layer_bytes(
            rx_base, layer_data.base_layer_frame_sizes
        )
        rx_enh_frames = self._parser.split_layer_bytes(
            rx_enhancement, layer_data.enhancement_layer_frame_sizes
        )

        # Reconstruct video with base layer only
        base_only_ivf = create_ivf_base_only(
            base_frames=rx_base_frames,
            base_indices=layer_data.base_frame_indices,
            width=self.config.resolution[0],
            height=self.config.resolution[1],
        )

        base_only_frames, base_playable, base_decoded = self._extract_frames_from_ivf(
            base_only_ivf
        )

        # Reconstruct video with both layers
        both_layers_ivf = create_ivf_from_layers(
            base_frames=rx_base_frames,
            enhancement_frames=rx_enh_frames,
            base_indices=layer_data.base_frame_indices,
            enhancement_indices=layer_data.enhancement_frame_indices,
            width=self.config.resolution[0],
            height=self.config.resolution[1],
        )

        both_frames, both_playable, both_decoded = self._extract_frames_from_ivf(
            both_layers_ivf
        )

        # Compute quality metrics
        psnr_base, ssim_base = self._compute_quality(original_frames, base_only_frames)
        psnr_both, ssim_both = self._compute_quality(original_frames, both_frames)

        # Optional: compare with uniform FEC
        uniform_ber = None
        uniform_psnr = None
        if compare_uniform:
            uniform_ber, uniform_psnr = self._run_uniform_comparison(
                ivf_bytes, snr_db, original_frames
            )

        # Clean up temp file
        try:
            temp_ivf.unlink()
        except Exception:
            pass

        return UEPResult(
            snr_db=snr_db,
            base_ber=base_ber,
            base_bit_errors=base_errors,
            base_total_bits=base_bits,
            base_decoded_ok=base_playable,
            enhancement_ber=enh_ber,
            enhancement_bit_errors=enh_errors,
            enhancement_total_bits=enh_bits,
            enhancement_decoded_ok=both_playable,
            psnr_base_only=psnr_base,
            psnr_both_layers=psnr_both,
            ssim_base_only=ssim_base,
            ssim_both_layers=ssim_both,
            playable_base=base_playable,
            playable_both=both_playable,
            frames_decoded_base=base_decoded,
            frames_decoded_both=both_decoded,
            frames_total=layer_data.total_frames,
            base_only_ivf=base_only_ivf if base_playable else None,
            both_layers_ivf=both_layers_ivf if both_playable else None,
            uniform_ber=uniform_ber,
            uniform_psnr=uniform_psnr,
        )

    def _extract_frames_from_ivf(
        self,
        ivf_bytes: bytes,
    ) -> Tuple[np.ndarray, bool, int]:
        """Extract frames from IVF bytes using FFmpeg."""
        import tempfile
        from pathlib import Path
        import subprocess

        if len(ivf_bytes) == 0:
            return np.array([]), False, 0

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.ivf', delete=False) as f:
            f.write(ivf_bytes)
            temp_path = Path(f.name)

        try:
            from .video_utils import get_ffmpeg_path
            cmd = [
                get_ffmpeg_path(),
                '-err_detect', 'ignore_err',
                '-i', str(temp_path),
                '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-'
            ]

            result = subprocess.run(cmd, capture_output=True)

            width, height = self.config.resolution
            frame_size = width * height * 3
            data = np.frombuffer(result.stdout, dtype=np.uint8)
            num_frames = len(data) // frame_size

            if num_frames > 0:
                frames = data[:num_frames * frame_size].reshape(
                    num_frames, height, width, 3
                )
                is_playable = result.returncode == 0
                return frames, is_playable, num_frames
            else:
                return np.array([]), False, 0
        finally:
            try:
                temp_path.unlink()
            except Exception:
                pass

    def _compute_quality(
        self,
        original: np.ndarray,
        received: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute PSNR and SSIM between original and received frames."""
        if len(original) == 0 or len(received) == 0:
            return 0.0, 0.0

        min_frames = min(len(original), len(received))
        psnr_values = [img_psnr(original[i], received[i], 255.0) for i in range(min_frames)]
        ssim_values = [img_ssim(original[i], received[i], 255.0) for i in range(min_frames)]

        return float(np.mean(psnr_values)), float(np.mean(ssim_values))

    def _run_uniform_comparison(
        self,
        ivf_bytes: bytes,
        snr_db: float,
        original_frames: np.ndarray,
    ) -> Tuple[float, float]:
        """Run transmission with uniform FEC for comparison.

        Uses an equivalent overall rate that approximately matches
        total coded bits from UEP.
        """
        # Calculate equivalent uniform rate
        # Average of base and enhancement rates
        equivalent_rate = (self.config.base_layer_fec_rate +
                          self.config.enhancement_layer_fec_rate) / 2

        # Create uniform FEC codec
        uniform_fec = self._create_fec_codec(equivalent_rate)

        # Transmit entire video with uniform FEC
        rx_bytes, ber, _, _ = self.transmit_layer(ivf_bytes, uniform_fec, snr_db)

        # Extract frames and compute PSNR
        rx_frames, _, _ = self._extract_frames_from_ivf(rx_bytes)

        psnr, _ = self._compute_quality(original_frames, rx_frames)

        return ber, psnr

    def run_snr_sweep(
        self,
        video_path: Path,
        snr_range: List[float],
        compare_uniform: bool = True,
    ) -> List[UEPResult]:
        """Run simulation across multiple SNR values.

        Args:
            video_path: Path to input video
            snr_range: List of SNR values in dB
            compare_uniform: If True, compare with uniform FEC

        Returns:
            List of UEPResult for each SNR
        """
        results = []
        for snr_db in snr_range:
            print(f"\n--- SNR = {snr_db} dB ---")
            result = self.run_simulation(video_path, snr_db, compare_uniform)
            results.append(result)

            print(f"  Base layer BER:        {result.base_ber:.6f}")
            print(f"  Enhancement layer BER: {result.enhancement_ber:.6f}")
            print(f"  PSNR (base only):      {result.psnr_base_only:.2f} dB")
            print(f"  PSNR (both layers):    {result.psnr_both_layers:.2f} dB")
            if result.uniform_psnr is not None:
                gain = result.psnr_both_layers - result.uniform_psnr
                print(f"  PSNR (uniform FEC):    {result.uniform_psnr:.2f} dB")
                print(f"  UEP gain:              {gain:+.2f} dB")

        return results
