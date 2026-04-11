from __future__ import annotations

import math
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Tuple

import torch
from scipy.io import loadmat

from .signal_utils import resample_poly_torch

DISABLE_REPLAY_FILTER_REFERENCE = True


@dataclass
class ChannelSounding:
    h: torch.Tensor
    fs_t: float
    fs_tau: float
    fc: float
    V0: float


def load_channel_sounding(path: Path) -> ChannelSounding:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    h = torch.from_numpy(mat["h"]).to(torch.cdouble)
    return ChannelSounding(
        h=h,
        fs_t=float(mat["fs_t"]),
        fs_tau=float(mat["fs_tau"]),
        fc=float(mat["fc"]),
        V0=float(mat["V0"]),
    )


def _rational_ratio(value: float, max_denominator: int = 1000) -> Tuple[int, int]:
    frac = Fraction(value).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator


def _replay_filter_reference(
    x: torch.Tensor,
    fs_x: float,
    channel: ChannelSounding,
) -> torch.Tensor:
    """Slow reference implementation for verification."""
    device = x.device
    dtype = torch.cdouble
    t_in = torch.arange(x.numel(), dtype=torch.float64, device=device) / fs_x
    baseband = x.to(torch.float64)
    baseband = torch.complex(baseband, torch.zeros_like(baseband))
    baseband = baseband * torch.exp(-1j * 2 * math.pi * channel.fc * t_in)

    up, down = _rational_ratio(channel.fs_tau / fs_x)
    bb_resampled = resample_poly_torch(baseband, up, down)

    h_table = torch.flip(channel.h.t(), dims=(0,))
    K = h_table.shape[0]
    ir_count = h_table.shape[1]
    padded = torch.cat(
        [torch.zeros(K - 1, dtype=dtype, device=device), bb_resampled]
    )
    y = torch.zeros(bb_resampled.numel(), dtype=dtype, device=device)

    for k in range(y.numel()):
        block = k // K
        if ir_count == 1:
            ir = h_table[:, 0]
        else:
            frac = (k / K) - block
            n = min(block, ir_count - 2)
            ir = (1 - frac) * h_table[:, n] + frac * h_table[:, n + 1]
        segment = padded[k : k + K]
        y[k] = torch.dot(segment, ir)

    up2, down2 = _rational_ratio(fs_x / channel.fs_tau)
    y_time = resample_poly_torch(y, up2, down2)
    t_out = torch.arange(y_time.numel(), dtype=torch.float64, device=device) / fs_x
    passband = torch.real(y_time * torch.exp(1j * 2 * math.pi * channel.fc * t_out))
    return passband


def replay_filter(
    x: torch.Tensor,
    fs_x: float,
    channel: ChannelSounding,
) -> torch.Tensor:
    """Vectorized replayfilter; optional reference check via VERIFY_REPLAY_FILTER=1."""
    device = x.device
    dtype = torch.cdouble
    t_in = torch.arange(x.numel(), dtype=torch.float64, device=device) / fs_x
    baseband = torch.complex(
        x.to(torch.float64),
        torch.zeros_like(x, dtype=torch.float64, device=device),
    )
    baseband = baseband * torch.exp(-1j * 2 * math.pi * channel.fc * t_in)

    up, down = _rational_ratio(channel.fs_tau / fs_x)
    bb_resampled = resample_poly_torch(baseband, up, down)
    if bb_resampled.numel() == 0:
        return bb_resampled

    # Cache flipped impulse response on the right device.
    h_table = getattr(channel, "_h_table", None)
    if h_table is None or h_table.device != bb_resampled.device:
        h_table = torch.flip(channel.h.t().to(bb_resampled.device), dims=(0,))
        channel._h_table = h_table
    K, ir_count = h_table.shape

    padded = torch.cat(
        [torch.zeros(K - 1, dtype=dtype, device=bb_resampled.device), bb_resampled]
    )

    # The vectorized unfolding path materializes an (L x K) matrix, which can OOM for long signals.
    # Fall back to the reference implementation in that case.
    L_est = bb_resampled.numel()
    if L_est * K > 5_000_000 and not DISABLE_REPLAY_FILTER_REFERENCE:
        return _replay_filter_reference(x, fs_x, channel)

    segments = padded.unfold(0, K, step=1)  # (L, K)
    L = segments.shape[0]
    idx = torch.arange(L, device=bb_resampled.device, dtype=torch.float64)
    block = torch.floor(idx / float(K)).to(torch.long)
    frac = idx / float(K) - block.to(torch.float64)

    if ir_count == 1:
        ir_interp = h_table[:, 0].unsqueeze(0).expand(L, -1)
    else:
        base_idx = torch.clamp(block, max=ir_count - 2)
        gather_idx = base_idx.unsqueeze(0).expand(K, L)
        ir0 = torch.gather(h_table, 1, gather_idx).transpose(0, 1)  # (L, K)
        ir1 = torch.gather(h_table, 1, gather_idx + 1).transpose(0, 1)  # (L, K)
        ir_interp = ir0 + (ir1 - ir0) * frac.unsqueeze(1)

    y = torch.einsum("lk,lk->l", segments, ir_interp)

    up2, down2 = _rational_ratio(fs_x / channel.fs_tau)
    y_time = resample_poly_torch(y, up2, down2)
    t_out = torch.arange(y_time.numel(), dtype=torch.float64, device=bb_resampled.device) / fs_x
    passband = torch.real(y_time * torch.exp(1j * 2 * math.pi * channel.fc * t_out))

    if os.environ.get("VERIFY_REPLAY_FILTER") == "1":
        with torch.no_grad():
            ref = _replay_filter_reference(x.detach(), fs_x, channel)
            if ref.device != passband.device:
                ref = ref.to(passband.device)
            max_err = torch.max(torch.abs(passband - ref)).item()
            if max_err > 1e-4:
                print(f"[replay_filter] verification max_err={max_err:.2e}")
    return passband
