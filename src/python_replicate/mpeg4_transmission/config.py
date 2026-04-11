"""
Configuration dataclasses for MPEG4 transmission simulation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass
class Mpeg4SimConfig:
    """Configuration for MPEG4 video transmission simulation.

    Attributes:
        video_path: Path to input video file
        resolution: Target video resolution (width, height)
        target_bitrate_bps: Target bitrate for MPEG4 encoding (0 = auto from channel)
        auto_bitrate: If True, calculate bitrate from channel capacity
        bitrate_efficiency: Fraction of channel capacity to use (0.0-1.0)

        codec: Video codec ('h264' or 'h265')
        fps: Video frame rate (None = keep original)

        channel_type: 'awgn' for flat channel or 'uwa' for real underwater acoustic
        channel_name: Channel name for UWA (e.g., 'NOF1', 'KAU1')
        channel_base_dir: Base directory containing channel .mat files
        snr_db: Signal-to-noise ratio in dB (can be single value or list for sweep)

        modulation: 'BPSK' (1 bit/symbol) or 'QPSK' (2 bits/symbol)

        fec_type: 'none' or 'repetition'
        fec_repetitions: Number of repetitions for repetition FEC

        output_dir: Directory for simulation results
    """
    # Video settings
    video_path: Path = field(default_factory=lambda: Path('input/video.mp4'))
    resolution: Tuple[int, int] = (64, 64)
    target_bitrate_bps: int = 0  # 0 = auto from channel capacity
    auto_bitrate: bool = True  # Calculate bitrate from channel capacity
    bitrate_efficiency: float = 1.0  # Use 100% of channel capacity
    codec: str = 'h264'  # 'h264' or 'h265'
    fps: Optional[float] = None  # None = keep original FPS

    # Channel settings
    channel_type: str = 'awgn'  # 'awgn' or 'uwa'
    channel_name: str = 'NOF1'
    channel_base_dir: Path = field(default_factory=lambda: Path('input/channels'))
    channel_recording_mode: str = 'first'  # 'first', 'random', or 'fixed'
    channel_recording_seed: Optional[int] = None  # Seed for random selection
    snr_db: float = 10.0

    # Modulation settings
    modulation: str = 'QPSK'  # 'BPSK' or 'QPSK'

    # FEC settings (uses existing fec_codec.py)
    # Basic: 'none', 'repetition'
    # Advanced (requires py_aff3ct): 'ldpc', 'dvbs2_ldpc', 'polar', 'turbo', 'rsc'
    fec_type: str = 'none'
    fec_repetitions: int = 3  # For repetition code only

    # LDPC settings (from alist file)
    ldpc_alist_path: Optional[str] = None  # Path to .alist or .qc file
    ldpc_max_iter: int = 50

    # DVB-S2 LDPC settings (for low rates 0.20-0.90)
    # Use DVBS2LDPCCodec.available_rates() to see all options
    dvbs2_ldpc_k: int = 3240   # Message length (K=3240 for rate 0.20)
    dvbs2_ldpc_n: int = 16200  # Codeword length (16200=short, 64800=normal)
    dvbs2_ldpc_max_iter: int = 50

    # Polar settings
    polar_k: int = 512  # Message length
    polar_n: int = 1024  # Codeword length (must be power of 2)
    polar_sigma: float = 0.5  # Design noise level

    # Turbo settings (K must be valid DVB-RCS2 size: 112, 128, 304, 408, 456, 680, 768, 800, 864, 880, etc.)
    turbo_k: int = 880
    turbo_max_iter: int = 8

    # RSC settings (N = 2*K + 4 automatically)
    rsc_k: int = 64

    # OFDM settings (defaults match existing pipeline)
    num_carriers: int = 64
    num_ofdm_symbols: int = 16
    cp_length: int = 30
    pilot_period: int = 4
    bandwidth_hz: float = 8e3
    fc_hz: float = 14e3

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path('results/mpeg4'))

    def __post_init__(self):
        # Convert string paths to Path objects
        if isinstance(self.video_path, str):
            self.video_path = Path(self.video_path)
        if isinstance(self.channel_base_dir, str):
            self.channel_base_dir = Path(self.channel_base_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Validate modulation
        if self.modulation.upper() not in ('BPSK', 'QPSK'):
            raise ValueError(f"modulation must be 'BPSK' or 'QPSK', got '{self.modulation}'")
        self.modulation = self.modulation.upper()

        # Validate channel type
        if self.channel_type.lower() not in ('awgn', 'uwa'):
            raise ValueError(f"channel_type must be 'awgn' or 'uwa', got '{self.channel_type}'")
        self.channel_type = self.channel_type.lower()

        # Validate FEC type
        valid_fec = ('none', 'repetition', 'ldpc', 'dvbs2_ldpc', 'polar', 'turbo', 'rsc')
        if self.fec_type.lower() not in valid_fec:
            raise ValueError(f"fec_type must be one of {valid_fec}, got '{self.fec_type}'")
        self.fec_type = self.fec_type.lower()

        # Validate LDPC requires alist_path
        if self.fec_type == 'ldpc' and not self.ldpc_alist_path:
            raise ValueError("ldpc_alist_path is required when fec_type='ldpc'")

        # Validate codec
        if self.codec.lower() not in ('h264', 'h265', 'libx264', 'libx265'):
            raise ValueError(f"codec must be 'h264' or 'h265', got '{self.codec}'")
        self.codec = self.codec.lower()

        # Validate bitrate_efficiency
        if not 0.0 < self.bitrate_efficiency <= 1.0:
            raise ValueError(f"bitrate_efficiency must be in (0, 1], got {self.bitrate_efficiency}")

    @property
    def bits_per_symbol(self) -> int:
        """Number of bits per modulation symbol."""
        return 1 if self.modulation == 'BPSK' else 2

    @property
    def data_symbols_per_frame(self) -> int:
        """Number of data symbols per OFDM frame (excluding pilots)."""
        data_ofdm_symbols = self.num_ofdm_symbols - (self.num_ofdm_symbols // self.pilot_period)
        return data_ofdm_symbols * self.num_carriers

    @property
    def bits_per_frame(self) -> int:
        """Number of data bits per OFDM frame."""
        return self.data_symbols_per_frame * self.bits_per_symbol

    @property
    def fec_rate(self) -> float:
        """FEC code rate (k/n). For advanced FEC, this is approximate."""
        if self.fec_type == 'none':
            return 1.0
        elif self.fec_type == 'repetition':
            return 1.0 / self.fec_repetitions
        elif self.fec_type == 'dvbs2_ldpc':
            return self.dvbs2_ldpc_k / self.dvbs2_ldpc_n
        elif self.fec_type == 'polar':
            return self.polar_k / self.polar_n
        elif self.fec_type == 'turbo':
            return 1.0 / 3.0  # Rate ~1/3
        elif self.fec_type == 'rsc':
            return self.rsc_k / (2 * self.rsc_k + 4)  # Rate ~0.48
        elif self.fec_type == 'ldpc':
            # LDPC rate depends on the matrix, return approximate 1/2
            return 0.5
        return 1.0

    @property
    def effective_bits_per_frame(self) -> int:
        """Effective data bits per frame after FEC overhead."""
        return int(self.bits_per_frame * self.fec_rate)

    @property
    def oversample_q(self) -> int:
        """Oversampling factor (fixed at 8)."""
        return 8

    @property
    def fs(self) -> float:
        """Sampling rate in Hz."""
        return self.oversample_q * self.bandwidth_hz

    @property
    def ofdm_symbol_samples(self) -> int:
        """Baseband samples per OFDM symbol (including CP)."""
        return self.num_carriers + self.cp_length

    @property
    def ofdm_frame_baseband_samples(self) -> int:
        """Total baseband samples for OFDM data portion."""
        return self.ofdm_symbol_samples * self.num_ofdm_symbols

    @property
    def preamble_baseband_samples(self) -> int:
        """Baseband samples for preamble (sync + SC + train).

        Note: These are approximate - actual length depends on RRC filter.
        Using sync=500, sc=128, train=0 as defaults.
        """
        sync_length = 500
        sc_length = 128
        train_length = 0
        return sync_length + sc_length + train_length

    @property
    def frame_baseband_samples(self) -> int:
        """Total baseband samples per transmission frame."""
        return self.preamble_baseband_samples + self.ofdm_frame_baseband_samples

    @property
    def frame_passband_samples(self) -> int:
        """Total passband samples per transmission frame (after upsampling)."""
        return self.frame_baseband_samples * self.oversample_q

    @property
    def frame_duration_sec(self) -> float:
        """Duration of one transmission frame in seconds."""
        return self.frame_passband_samples / self.fs

    @property
    def channel_bitrate_bps(self) -> float:
        """Raw channel bitrate (before FEC) in bits per second."""
        return self.bits_per_frame / self.frame_duration_sec

    @property
    def effective_bitrate_bps(self) -> float:
        """Effective data bitrate (after FEC overhead) in bits per second."""
        return self.effective_bits_per_frame / self.frame_duration_sec

    @property
    def video_bitrate_bps(self) -> int:
        """Target video bitrate based on channel capacity.

        If auto_bitrate is True and target_bitrate_bps is 0, calculates
        from channel capacity. Otherwise returns target_bitrate_bps.

        Note: FFmpeg encoders have minimum bitrate requirements (~1000 bps).
        If the calculated bitrate is below this, encoding will fail.
        """
        if self.auto_bitrate and self.target_bitrate_bps == 0:
            return int(self.effective_bitrate_bps * self.bitrate_efficiency)
        return self.target_bitrate_bps if self.target_bitrate_bps > 0 else int(self.effective_bitrate_bps * self.bitrate_efficiency)

    @property
    def video_bitrate_warning(self) -> Optional[str]:
        """Warning message if video bitrate is very low."""
        if self.video_bitrate_bps < 500:
            return (
                f"Video bitrate {self.video_bitrate_bps} bps is extremely low. "
                f"Consider: smaller resolution, lower FPS, or use H.265 codec."
            )
        return None

    @property
    def ffmpeg_codec(self) -> str:
        """FFmpeg codec name."""
        codec_map = {
            'h264': 'libx264',
            'libx264': 'libx264',
            'h265': 'libx265',
            'libx265': 'libx265',
        }
        return codec_map.get(self.codec, 'libx264')


@dataclass
class SimulationResult:
    """Results from a single simulation run.

    Attributes:
        snr_db: SNR used for this simulation
        ber: Bit error rate
        bit_errors: Total number of bit errors
        total_bits: Total number of bits transmitted
        psnr_per_frame: PSNR for each video frame
        ssim_per_frame: SSIM for each video frame
        psnr_mean: Mean PSNR across all frames
        ssim_mean: Mean SSIM across all frames
        playable: Whether the reconstructed video is playable
        frames_decoded: Number of frames successfully decoded
        frames_total: Total number of frames in original video
        transmission_time_sec: Simulated transmission time
    """
    snr_db: float
    ber: float
    bit_errors: int
    total_bits: int

    psnr_per_frame: Optional[List[float]] = None
    ssim_per_frame: Optional[List[float]] = None
    psnr_mean: Optional[float] = None
    ssim_mean: Optional[float] = None

    playable: bool = False
    frames_decoded: int = 0
    frames_total: int = 0

    transmission_time_sec: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'snr_db': self.snr_db,
            'ber': self.ber,
            'bit_errors': self.bit_errors,
            'total_bits': self.total_bits,
            'psnr_mean': self.psnr_mean,
            'ssim_mean': self.ssim_mean,
            'playable': self.playable,
            'frames_decoded': self.frames_decoded,
            'frames_total': self.frames_total,
            'transmission_time_sec': self.transmission_time_sec,
        }
