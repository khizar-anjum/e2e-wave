from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from scipy.signal import firwin

def root_raised_cosine(
    rolloff: float,
    span: int,
    sps: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Torch implementation of MATLAB's rcosdesign(..., 'sqrt')."""
    if not (0 < rolloff <= 1):
        raise ValueError("rolloff must be in (0, 1].")
    taps = span * sps + 1
    t = torch.linspace(
        -span / 2, span / 2, steps=taps, dtype=dtype, device=device
    )
    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    rrc = torch.zeros_like(t)
    sqrt_T = torch.sqrt(torch.tensor(1.0, dtype=dtype, device=device))
    for idx, ti in enumerate(t):
        if torch.isclose(ti, torch.tensor(0.0, dtype=dtype, device=device)):
            rrc[idx] = (1 + rolloff * (4 / pi - 1)) / sqrt_T
            continue
        if torch.isclose(
            torch.abs(ti), torch.tensor(1.0 / (4 * rolloff), dtype=dtype, device=device)
        ):
            term = (rolloff / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device)))
            trig = (1 + 2 / pi) * torch.sin(pi / (4 * rolloff))
            trig += (1 - 2 / pi) * torch.cos(pi / (4 * rolloff))
            rrc[idx] = term * trig / sqrt_T
            continue
        numerator = torch.cos(pi * (1 + rolloff) * ti) + torch.sin(
            pi * (1 - rolloff) * ti
        ) / (4 * rolloff * ti)
        denominator = 1 - (4 * rolloff * ti) ** 2
        rrc[idx] = (4 * rolloff / (pi * sqrt_T)) * numerator / denominator
    return rrc / math.sqrt(sps)


def upfirdn_torch(
    x: torch.Tensor,
    h: torch.Tensor,
    up: int = 1,
    down: int = 1,
) -> torch.Tensor:
    """Pure PyTorch upfirdn equivalent (complex64/complex128)."""
    if x.is_complex():
        x_complex = x.to(torch.cdouble)
    else:
        x_complex = x.to(torch.float64).to(torch.cdouble)
    if up > 1:
        upsampled = torch.zeros(
            x_complex.numel() * up,
            dtype=torch.cdouble,
            device=x.device,
        )
        upsampled[::up] = x_complex.reshape(-1)
    else:
        upsampled = x_complex.reshape(-1)
    h = h.to(dtype=torch.float64, device=x.device)
    kernel = torch.flip(h, dims=(0,)).view(1, 1, -1)
    y_real = F.conv1d(
        upsampled.real.view(1, 1, -1), kernel, padding=h.numel() - 1
    )
    y_imag = F.conv1d(
        upsampled.imag.view(1, 1, -1), kernel, padding=h.numel() - 1
    )
    y = torch.complex(y_real.view(-1), y_imag.view(-1))
    if down > 1:
        y = y[::down]
    trim = (up - 1) // down
    if trim > 0 and y.numel() > trim:
        y = y[:-trim]
    return y


def _firwin_lowpass(
    num_taps: int,
    cutoff: float,
    beta: float,
    dtype: torch.dtype,
    device: Optional[torch.device],
) -> torch.Tensor:
    """Mirror scipy.signal.firwin for low-pass taps."""
    taps = firwin(num_taps, cutoff, window=("kaiser", beta))
    return torch.from_numpy(taps).to(dtype=dtype, device=device)


def _output_len_like(h_len: int, x_len: int, up: int, down: int) -> int:
    naive = (x_len * up + h_len - 1) // down
    trim = (up - 1) // down
    return naive - trim


def resample_poly_torch(
    x: torch.Tensor,
    up: int,
    down: int,
    beta: float = 5.0,
) -> torch.Tensor:
    """Resample using a polyphase FIR filter (port of scipy.signal.resample_poly)."""
    if up != int(up) or down != int(down):
        raise ValueError("up and down must be integers.")
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1.")
    g = math.gcd(up, down)
    up //= g
    down //= g
    if up == down == 1:
        return x.clone()

    work = x.clone()
    if not work.is_complex():
        work = work.to(torch.float64)
    real_dtype = torch.float64
    device = work.device

    max_rate = max(up, down)
    half_len = 10 * max_rate
    taps = 2 * half_len + 1
    h = _firwin_lowpass(taps, 1.0 / max_rate, beta, real_dtype, device)
    h = h * up

    n_pre_pad = (down - (half_len % down)) % down
    n_pre_remove = (half_len + n_pre_pad) // down
    n_out = (work.numel() * up + down - 1) // down
    n_post_pad = 0
    while (
        _output_len_like(h.numel() + n_pre_pad + n_post_pad, work.numel(), up, down)
        < n_out + n_pre_remove
    ):
        n_post_pad += 1
    if n_pre_pad or n_post_pad:
        pad_pre = torch.zeros(n_pre_pad, dtype=h.dtype, device=device)
        pad_post = torch.zeros(n_post_pad, dtype=h.dtype, device=device)
        h = torch.cat([pad_pre, h, pad_post])

    y = upfirdn_torch(work, h, up=up, down=down)
    start = n_pre_remove
    end = start + n_out
    y = y[start:end]
    if x.is_complex():
        return y.to(x.dtype)
    return y.real.to(x.dtype)
