"""
VP9 Temporal SVC encoding and parsing utilities.

Provides:
- VP9 encoding with temporal scalability (2 layers: base + enhancement)
- IVF container parsing to extract individual temporal layer frames
- Layer data structures for UEP transmission
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import subprocess
import tempfile
import struct

import numpy as np


def get_ffmpeg_path() -> str:
    """Get FFmpeg path, preferring system version."""
    import os
    if os.path.exists('/usr/bin/ffmpeg'):
        return '/usr/bin/ffmpeg'
    return 'ffmpeg'


@dataclass
class VP9LayerData:
    """Container for VP9 temporal layer data."""
    base_layer_bytes: bytes          # TID=0 frames (keyframes + base)
    enhancement_layer_bytes: bytes   # TID=1 frames
    base_layer_frame_sizes: List[int]
    enhancement_layer_frame_sizes: List[int]
    total_frames: int
    base_frame_indices: List[int]    # Original frame indices for base layer
    enhancement_frame_indices: List[int]

    @property
    def base_layer_size(self) -> int:
        return len(self.base_layer_bytes)

    @property
    def enhancement_layer_size(self) -> int:
        return len(self.enhancement_layer_bytes)


class VP9SVCEncoder:
    """Encode video with VP9 temporal SVC (2 layers).

    Uses FFmpeg with libvpx-vp9 and temporal scalability parameters.
    Layer 0 (base): keyframes + half framerate
    Layer 1 (enhancement): interpolation frames for full framerate
    """

    def __init__(
        self,
        num_layers: int = 2,
        base_bitrate_kbps: int = 50,
        total_bitrate_kbps: int = 100,
    ):
        """Initialize VP9 SVC encoder.

        Args:
            num_layers: Number of temporal layers (default 2)
            base_bitrate_kbps: Target bitrate for base layer
            total_bitrate_kbps: Total bitrate for all layers
        """
        self.num_layers = num_layers
        self.base_bitrate_kbps = base_bitrate_kbps
        self.total_bitrate_kbps = total_bitrate_kbps

    def encode(
        self,
        input_path: Path,
        resolution: Tuple[int, int] = (64, 64),
        fps: Optional[float] = None,
    ) -> Tuple[bytes, Path]:
        """Encode video with VP9 temporal SVC.

        Args:
            input_path: Path to input video
            resolution: Target (width, height)
            fps: Target framerate (None = keep original)

        Returns:
            Tuple of (ivf_bytes, temp_ivf_path)
        """
        with tempfile.NamedTemporaryFile(suffix='.ivf', delete=False) as f:
            output_path = Path(f.name)

        width, height = resolution

        # Build FFmpeg command for VP9 temporal SVC
        # ts_rate_decimator: base=2 (half fps), enhancement=1 (full fps)
        # ts_periodicity=2: pattern repeats every 2 frames
        # ts_layer_id=0,1: frame 0 -> layer 0, frame 1 -> layer 1
        cmd = [
            get_ffmpeg_path(), '-y', '-i', str(input_path),
            '-vf', f'crop=min(iw\\,ih):min(iw\\,ih),scale={width}:{height}',
            '-c:v', 'libvpx-vp9',
            '-b:v', f'{self.total_bitrate_kbps}k',
            '-ts-parameters',
            f'ts_number_layers={self.num_layers}:'
            f'ts_target_bitrate={self.base_bitrate_kbps},{self.total_bitrate_kbps}:'
            'ts_rate_decimator=2,1:'
            'ts_periodicity=2:'
            'ts_layer_id=0,1',
            '-an',  # No audio
            '-f', 'ivf',
            str(output_path)
        ]

        if fps is not None:
            # Insert fps filter before output
            cmd.insert(cmd.index('-c:v'), '-r')
            cmd.insert(cmd.index('-c:v'), str(fps))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"VP9 SVC encoding failed: {result.stderr}")

        with open(output_path, 'rb') as f:
            ivf_bytes = f.read()

        return ivf_bytes, output_path

    def parse_layers(self, ivf_bytes: bytes) -> VP9LayerData:
        """Parse IVF file and separate temporal layers.

        Args:
            ivf_bytes: Raw IVF file bytes

        Returns:
            VP9LayerData with separated base and enhancement layers
        """
        parser = VP9SuperframeParser()
        return parser.parse_ivf(ivf_bytes)


class VP9SuperframeParser:
    """Parse VP9/IVF container to extract temporal layer frames.

    IVF structure:
    - 32-byte header
    - For each frame:
      - 4-byte frame size (little-endian)
      - 8-byte timestamp (little-endian)
      - frame_size bytes of frame data

    With ts_layer_id=0,1 encoding pattern:
    - Even frames (0, 2, 4...) belong to layer 0 (base)
    - Odd frames (1, 3, 5...) belong to layer 1 (enhancement)
    """

    IVF_HEADER_SIZE = 32
    IVF_FRAME_HEADER_SIZE = 12

    def parse_ivf(self, ivf_bytes: bytes) -> VP9LayerData:
        """Parse IVF container and extract temporal layers.

        Args:
            ivf_bytes: Raw IVF file bytes

        Returns:
            VP9LayerData with separated layers
        """
        # Skip IVF header (32 bytes)
        offset = self.IVF_HEADER_SIZE

        base_frames = []
        enhancement_frames = []
        base_indices = []
        enhancement_indices = []
        frame_index = 0

        while offset < len(ivf_bytes):
            if offset + self.IVF_FRAME_HEADER_SIZE > len(ivf_bytes):
                break

            # Read IVF frame header
            frame_size = struct.unpack('<I', ivf_bytes[offset:offset+4])[0]
            # timestamp = struct.unpack('<Q', ivf_bytes[offset+4:offset+12])[0]
            offset += self.IVF_FRAME_HEADER_SIZE

            if offset + frame_size > len(ivf_bytes):
                break

            frame_data = ivf_bytes[offset:offset + frame_size]
            offset += frame_size

            # Determine layer by frame index (based on ts_layer_id=0,1 pattern)
            # Even frames -> layer 0 (base)
            # Odd frames -> layer 1 (enhancement)
            if frame_index % 2 == 0:
                base_frames.append(frame_data)
                base_indices.append(frame_index)
            else:
                enhancement_frames.append(frame_data)
                enhancement_indices.append(frame_index)

            frame_index += 1

        return VP9LayerData(
            base_layer_bytes=b''.join(base_frames),
            enhancement_layer_bytes=b''.join(enhancement_frames),
            base_layer_frame_sizes=[len(f) for f in base_frames],
            enhancement_layer_frame_sizes=[len(f) for f in enhancement_frames],
            total_frames=frame_index,
            base_frame_indices=base_indices,
            enhancement_frame_indices=enhancement_indices,
        )

    def split_layer_bytes(
        self,
        layer_bytes: bytes,
        frame_sizes: List[int],
    ) -> List[bytes]:
        """Split concatenated layer bytes back into individual frames.

        Args:
            layer_bytes: Concatenated frame bytes
            frame_sizes: List of individual frame sizes

        Returns:
            List of individual frame bytes
        """
        frames = []
        offset = 0
        for size in frame_sizes:
            if offset + size <= len(layer_bytes):
                frames.append(layer_bytes[offset:offset + size])
            else:
                # Handle truncated data
                frames.append(layer_bytes[offset:])
                break
            offset += size
        return frames


def create_ivf_from_layers(
    base_frames: List[bytes],
    enhancement_frames: List[bytes],
    base_indices: List[int],
    enhancement_indices: List[int],
    width: int = 64,
    height: int = 64,
    fps_num: int = 30,
    fps_den: int = 1,
) -> bytes:
    """Reconstruct IVF file from separated layer frames.

    Args:
        base_frames: List of base layer frame bytes
        enhancement_frames: List of enhancement layer frame bytes (can be empty)
        base_indices: Original frame indices for base layer
        enhancement_indices: Original frame indices for enhancement layer
        width, height: Video dimensions
        fps_num, fps_den: Frame rate as fraction

    Returns:
        IVF file bytes
    """
    # Create frame mapping (index -> frame_data)
    frame_map = {}
    for idx, frame in zip(base_indices, base_frames):
        frame_map[idx] = frame
    for idx, frame in zip(enhancement_indices, enhancement_frames):
        frame_map[idx] = frame

    # IVF header (32 bytes)
    header = b'DKIF'  # Signature
    header += struct.pack('<H', 0)  # Version
    header += struct.pack('<H', 32)  # Header length
    header += b'VP90'  # FourCC for VP9
    header += struct.pack('<HH', width, height)
    header += struct.pack('<II', fps_num, fps_den)
    header += struct.pack('<I', len(frame_map))  # Frame count
    header += struct.pack('<I', 0)  # Unused

    # Write frames in original order
    output = bytearray(header)
    for idx in sorted(frame_map.keys()):
        frame_data = frame_map[idx]
        # Frame header: size (4 bytes) + timestamp (8 bytes)
        # Timestamp in timebase units (fps_den / fps_num per frame)
        timestamp = idx * (fps_den * 1000000 // fps_num)  # microseconds
        frame_header = struct.pack('<I', len(frame_data))
        frame_header += struct.pack('<Q', timestamp)
        output.extend(frame_header)
        output.extend(frame_data)

    return bytes(output)


def create_ivf_base_only(
    base_frames: List[bytes],
    base_indices: List[int],
    width: int = 64,
    height: int = 64,
    fps_num: int = 30,
    fps_den: int = 1,
) -> bytes:
    """Create IVF with base layer frames only.

    When enhancement layer fails, we can still decode the base layer
    at half the framerate.

    Args:
        base_frames: List of base layer frame bytes
        base_indices: Original frame indices
        width, height: Video dimensions
        fps_num, fps_den: Frame rate as fraction

    Returns:
        IVF file bytes with base layer only
    """
    # For base-only, renumber frames sequentially
    sequential_indices = list(range(len(base_frames)))

    # Adjust fps for half framerate (base layer is every other frame)
    adjusted_fps_num = fps_num // 2 if fps_num > 1 else fps_num

    return create_ivf_from_layers(
        base_frames=base_frames,
        enhancement_frames=[],
        base_indices=sequential_indices,
        enhancement_indices=[],
        width=width,
        height=height,
        fps_num=adjusted_fps_num,
        fps_den=fps_den,
    )
