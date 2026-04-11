from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch

from .channel_replay import ChannelSounding, load_channel_sounding, replay_filter


@dataclass
class Bookkeeping:
    nPackets: int
    nPacketsPerSounding: int
    packet_indices: torch.Tensor  # shape (nPacketsPerSounding, 2), 0-based
    nSoundings: int
    nBits: int
    effectiveBitRate: float
    velocities: torch.Tensor
    pad_leading: int
    pad_trailing: int
    signal_length: int


@dataclass
class WatermarkOutput:
    soundings: List[torch.Tensor]
    fs: float
    bookkeeping: Bookkeeping
    normalization_factor: float


def _select_soundings(files: Sequence[Path], howmany: str) -> List[Path]:
    if howmany == "single":
        return [files[0]]
    if howmany == "all":
        return list(files)
    raise ValueError("howmany must be 'single' or 'all'.")


def simulate_watermark(
    x: torch.Tensor,
    fs_x: float,
    n_bits: int,
    effective_bit_rate: float,
    channel_root: Path,
    channel_name: str,
    howmany: str = "all",
) -> WatermarkOutput:
    """Python port of matlab/watermark.m that keeps results in memory."""
    channel_dir = channel_root / channel_name / "mat"
    files = sorted(channel_dir.glob(f"{channel_name}_*.mat"))
    if not files:
        raise FileNotFoundError(f"No channel files found in {channel_dir}")
    selected_files = _select_soundings(files, howmany)
    first_channel = load_channel_sounding(selected_files[0])

    s1, s2 = first_channel.h.shape
    max_samples = math.floor((s1 - 1) * (1 / first_channel.fs_t) * fs_x)
    Lx = x.numel()
    L1 = math.ceil(s2 * (1 / first_channel.fs_tau) * fs_x)
    L2 = math.ceil(2e-3 * fs_x)
    packet_len = Lx + L1 + 2 * L2
    if packet_len == 0:
        raise ValueError("Packet length is zero.")
    n_packets_per_sounding = max_samples // packet_len
    if n_packets_per_sounding == 0:
        raise ValueError("Input signal too long for the selected channel.")

    zeros_lead = torch.zeros(L2, dtype=torch.float64)
    zeros_tail = torch.zeros(L1 + L2, dtype=torch.float64)
    x_padded = torch.cat([zeros_lead, x.to(torch.float64), zeros_tail])
    signal_train = x_padded.repeat(n_packets_per_sounding)

    starts = torch.arange(n_packets_per_sounding, dtype=torch.int64) * packet_len
    ends = starts + packet_len
    packet_indices = torch.stack([starts, ends], dim=1)

    soundings: List[torch.Tensor] = []
    normalization_factor = None
    velocities: List[float] = []
    for path in selected_files:
        channel = load_channel_sounding(path)
        velocities.append(channel.V0)
        y = replay_filter(signal_train, fs_x, channel)
        if normalization_factor is None:
            rms = torch.sqrt(torch.mean(y**2))
            normalization_factor = rms.item() if rms.item() > 0 else 1.0
        y = y / normalization_factor
        soundings.append(y)

    bk = Bookkeeping(
        nPackets=n_packets_per_sounding * len(soundings),
        nPacketsPerSounding=n_packets_per_sounding,
        packet_indices=packet_indices,
        nSoundings=len(soundings),
        nBits=n_bits,
        effectiveBitRate=effective_bit_rate,
        velocities=torch.tensor(velocities, dtype=torch.float64),
        pad_leading=L2,
        pad_trailing=L1 + L2,
        signal_length=Lx,
    )

    return WatermarkOutput(
        soundings=soundings,
        fs=fs_x,
        bookkeeping=bk,
        normalization_factor=normalization_factor or 1.0,
    )

