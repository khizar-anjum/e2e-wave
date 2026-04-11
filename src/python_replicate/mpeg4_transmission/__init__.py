"""
MPEG4 Video Transmission Simulation over Underwater Acoustic Channels.

This package provides tools for simulating MPEG4 video transmission through
underwater acoustic channels with configurable FEC and modulation schemes.

Also includes VP9 temporal SVC with Unequal Error Protection (UEP).

Usage:
    python -m python_replicate.mpeg4_transmission.run_simulation --help
"""

from .config import Mpeg4SimConfig, SimulationResult
from .mpeg_pipeline import Mpeg4Pipeline
from .metrics import compute_ber, compute_ber_from_bits, QualityMetrics
from .video_utils import get_video_info, reencode_video, extract_frames

# VP9 SVC with UEP
from .vp9_svc import VP9SVCEncoder, VP9SuperframeParser, VP9LayerData
from .uep_pipeline import UEPPipeline, UEPConfig, UEPResult

__all__ = [
    # MPEG4 pipeline
    'Mpeg4SimConfig',
    'SimulationResult',
    'Mpeg4Pipeline',
    'compute_ber',
    'compute_ber_from_bits',
    'QualityMetrics',
    'get_video_info',
    'reencode_video',
    'extract_frames',
    # VP9 SVC with UEP
    'VP9SVCEncoder',
    'VP9SuperframeParser',
    'VP9LayerData',
    'UEPPipeline',
    'UEPConfig',
    'UEPResult',
]
