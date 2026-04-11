"""
Video utilities for MPEG4 transmission simulation.

Provides functions for:
- Re-encoding videos at target bitrates using ffmpeg
- Extracting frames as numpy arrays for quality metrics
- Reconstructing videos from (possibly corrupted) bytes
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass
import os

import numpy as np


def get_ffmpeg_path() -> str:
    """Get FFmpeg path, preferring system version with more codecs."""
    # Prefer system ffmpeg which typically has more codecs
    if os.path.exists('/usr/bin/ffmpeg'):
        return '/usr/bin/ffmpeg'
    return 'ffmpeg'


def get_ffprobe_path() -> str:
    """Get FFprobe path, preferring system version."""
    if os.path.exists('/usr/bin/ffprobe'):
        return '/usr/bin/ffprobe'
    return 'ffprobe'


@dataclass
class VideoInfo:
    """Information about a video file."""
    path: Path
    width: int
    height: int
    fps: float
    duration_sec: float
    num_frames: int
    bitrate_bps: int
    codec: str


def get_video_info(video_path: Path) -> VideoInfo:
    """Extract video metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo with video metadata
    """
    cmd = [
        get_ffprobe_path(), '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    import json
    data = json.loads(result.stdout)

    # Find video stream
    video_stream = None
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    # Parse frame rate (can be "30/1" format)
    fps_str = video_stream.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = map(float, fps_str.split('/'))
        fps = num / den if den != 0 else 30.0
    else:
        fps = float(fps_str)

    # Get duration and frame count
    duration = float(data.get('format', {}).get('duration', 0))
    num_frames = int(video_stream.get('nb_frames', int(fps * duration)))

    # Get bitrate
    bitrate = int(data.get('format', {}).get('bit_rate', 0))

    return VideoInfo(
        path=video_path,
        width=int(video_stream.get('width', 0)),
        height=int(video_stream.get('height', 0)),
        fps=fps,
        duration_sec=duration,
        num_frames=num_frames,
        bitrate_bps=bitrate,
        codec=video_stream.get('codec_name', 'unknown'),
    )


def reencode_video(
    input_path: Path,
    output_path: Path,
    resolution: Tuple[int, int] = (64, 64),
    bitrate_bps: int = 5000,
    fps: Optional[float] = None,
    codec: str = 'libx264',
) -> bytes:
    """Re-encode video to target resolution and bitrate.

    Args:
        input_path: Input video file
        output_path: Output video file
        resolution: Target (width, height)
        bitrate_bps: Target bitrate in bits per second
        fps: Target frame rate (None to keep original)
        codec: Video codec ('libx264', 'libx265', 'h264', 'h265')

    Returns:
        Video file contents as bytes

    Raises:
        RuntimeError: If FFmpeg fails
    """
    # Normalize codec name
    codec_map = {
        'h264': 'libx264',
        'h265': 'libx265',
        'libx264': 'libx264',
        'libx265': 'libx265',
        'vp9': 'libvpx-vp9',
        'libvpx-vp9': 'libvpx-vp9',
    }
    ffmpeg_codec = codec_map.get(codec.lower(), 'libx264')

    width, height = resolution

    # Center-crop to target aspect ratio, then scale to target resolution
    # This avoids distortion by cropping instead of stretching
    # crop='min(iw,ih*w/h)':'min(ih,iw*h/w)' calculates the largest centered crop
    # that matches the target aspect ratio
    crop_filter = f"crop='min(iw,ih*{width}/{height})':'min(ih,iw*{height}/{width})',scale={width}:{height}"

    cmd = [
        get_ffmpeg_path(), '-y', '-i', str(input_path),
        '-vf', crop_filter,
        '-b:v', f'{bitrate_bps}',
        '-maxrate', f'{bitrate_bps}',
        '-bufsize', f'{bitrate_bps * 2}',
        '-c:v', ffmpeg_codec,
    ]

    if fps is not None:
        cmd.extend(['-r', str(fps)])

    # Remove audio
    cmd.extend(['-an'])

    # Codec-specific options
    if ffmpeg_codec == 'libx264':
        cmd.extend(['-preset', 'medium', '-tune', 'zerolatency'])
    elif ffmpeg_codec == 'libx265':
        cmd.extend(['-preset', 'medium', '-tune', 'zerolatency', '-tag:v', 'hvc1'])

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Extract useful error from stderr
        stderr = result.stderr

        # Check for common errors
        if 'Unknown encoder' in stderr:
            raise RuntimeError(
                f"FFmpeg encoder '{ffmpeg_codec}' not installed. "
                f"For H.265, install libx265: sudo apt install libx265-dev, then rebuild FFmpeg. "
                f"Or use --codec h264 instead."
            )

        error_lines = [l for l in stderr.split('\n') if 'error' in l.lower() or 'bitrate' in l.lower() or 'Error' in l]
        error_msg = '\n'.join(error_lines) if error_lines else stderr[-500:]
        raise RuntimeError(
            f"FFmpeg encoding failed (bitrate={bitrate_bps} bps, codec={ffmpeg_codec}):\n{error_msg}"
        )

    # Read output file
    with open(output_path, 'rb') as f:
        return f.read()


def create_test_video_ffmpeg(
    output_path: Path,
    resolution: Tuple[int, int] = (64, 64),
    duration_sec: float = 0.5,
    fps: float = 30.0,
    bitrate_bps: int = 5000,
    codec: str = 'libx264',
    pattern: str = 'testsrc',
) -> bytes:
    """Create a test video using FFmpeg directly (no Python frame generation).

    Args:
        output_path: Output video file path
        resolution: Video resolution (width, height)
        duration_sec: Video duration in seconds
        fps: Frame rate
        bitrate_bps: Target bitrate
        codec: Video codec ('libx264', 'libx265')
        pattern: FFmpeg test pattern ('testsrc', 'testsrc2', 'smptebars', 'mandelbrot')

    Returns:
        Video file contents as bytes
    """
    # Normalize codec name
    codec_map = {
        'h264': 'libx264',
        'h265': 'libx265',
        'libx264': 'libx264',
        'libx265': 'libx265',
        'vp9': 'libvpx-vp9',
        'libvpx-vp9': 'libvpx-vp9',
    }
    ffmpeg_codec = codec_map.get(codec.lower(), 'libx264')

    width, height = resolution

    cmd = [
        get_ffmpeg_path(), '-y',
        '-f', 'lavfi',
        '-i', f'{pattern}=duration={duration_sec}:size={width}x{height}:rate={fps}',
        '-c:v', ffmpeg_codec,
        '-b:v', f'{bitrate_bps}',
        '-maxrate', f'{bitrate_bps}',
        '-bufsize', f'{bitrate_bps * 2}',
        '-pix_fmt', 'yuv420p',
    ]

    # Codec-specific options
    if ffmpeg_codec == 'libx264':
        cmd.extend(['-preset', 'fast', '-tune', 'zerolatency'])
    elif ffmpeg_codec == 'libx265':
        cmd.extend(['-preset', 'fast', '-tune', 'zerolatency', '-tag:v', 'hvc1'])

    cmd.append(str(output_path))

    subprocess.run(cmd, capture_output=True, check=True)

    with open(output_path, 'rb') as f:
        return f.read()


def extract_frames(video_path: Path, max_frames: Optional[int] = None) -> np.ndarray:
    """Extract video frames as numpy array.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None for all)

    Returns:
        Array of shape (num_frames, height, width, 3) with uint8 RGB values
    """
    # Get video info
    info = get_video_info(video_path)

    # Build ffmpeg command to output raw frames
    cmd = [
        get_ffmpeg_path(), '-i', str(video_path),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    ]

    if max_frames is not None:
        cmd.extend(['-frames:v', str(max_frames)])

    cmd.append('-')

    result = subprocess.run(cmd, capture_output=True, check=True)

    # Parse raw frames
    frame_size = info.width * info.height * 3
    data = np.frombuffer(result.stdout, dtype=np.uint8)
    num_frames = len(data) // frame_size

    frames = data[:num_frames * frame_size].reshape(
        num_frames, info.height, info.width, 3
    )

    return frames


def extract_frames_from_bytes(
    video_bytes: bytes,
    width: int,
    height: int,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, bool, int]:
    """Extract frames from video bytes (possibly corrupted).

    Args:
        video_bytes: Raw video file bytes
        width: Expected video width
        height: Expected video height
        max_frames: Maximum frames to extract

    Returns:
        Tuple of (frames_array, is_playable, num_frames_decoded)
        frames_array may be empty if video is completely corrupted
    """
    # Write bytes to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        temp_path = Path(f.name)

    try:
        # Try to extract frames with error tolerance
        cmd = [
            get_ffmpeg_path(),
            '-err_detect', 'ignore_err',
            '-i', str(temp_path),
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        ]

        if max_frames is not None:
            cmd.extend(['-frames:v', str(max_frames)])

        cmd.append('-')

        result = subprocess.run(cmd, capture_output=True)

        # Parse frames even if there were errors
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
        temp_path.unlink(missing_ok=True)


def reconstruct_video(
    video_bytes: bytes,
    output_path: Path,
    width: int,
    height: int,
    fps: float = 30.0,
) -> Tuple[bool, int, List[str]]:
    """Attempt to reconstruct video from (possibly corrupted) bytes.

    Args:
        video_bytes: Raw video file bytes (possibly corrupted)
        output_path: Path to write reconstructed video
        width: Expected video width
        height: Expected video height
        fps: Expected frame rate

    Returns:
        Tuple of (is_playable, frames_decoded, error_messages)
    """
    # Write bytes to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        temp_path = Path(f.name)

    errors = []

    try:
        # Try to transcode with error tolerance
        cmd = [
            get_ffmpeg_path(), '-y',
            '-err_detect', 'ignore_err',
            '-i', str(temp_path),
            '-c:v', 'libx264',
            '-preset', 'fast',
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            errors.append(result.stderr)

        # Check if output is valid
        if output_path.exists() and output_path.stat().st_size > 0:
            try:
                info = get_video_info(output_path)
                return True, info.num_frames, errors
            except Exception as e:
                errors.append(str(e))
                return False, 0, errors
        else:
            return False, 0, errors

    finally:
        temp_path.unlink(missing_ok=True)


def bytes_to_video_file(video_bytes: bytes, output_path: Path) -> None:
    """Write raw bytes to a video file.

    Args:
        video_bytes: Video file bytes
        output_path: Path to write the file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(video_bytes)
