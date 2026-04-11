from __future__ import annotations

import pathlib
from fractions import Fraction

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly

from .channel_replay import load_channel_sounding, replay_filter
from .frame_preparation import FramePrepConfig, prepare_frame


def replay_filter_reference(
    x: np.ndarray,
    fs_x: float,
    h: np.ndarray,
    fs_tau: float,
    fc: float,
) -> np.ndarray:
    t_in = np.arange(len(x)) / fs_x
    baseband = x * np.exp(-1j * 2 * np.pi * fc * t_in)
    frac = Fraction(fs_tau / fs_x).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    bb = resample_poly(baseband, up, down)

    h_table = np.flipud(h.T)
    K = h_table.shape[0]
    padded = np.concatenate([np.zeros(K - 1, dtype=np.complex128), bb])
    y = np.zeros_like(bb, dtype=np.complex128)
    col_count = h_table.shape[1]
    for k in range(len(bb)):
        block = k // K
        if col_count == 1:
            ir = h_table[:, 0]
        else:
            frac = (k / K) - block
            n = min(block, col_count - 2)
            ir = (1 - frac) * h_table[:, n] + frac * h_table[:, n + 1]
        segment = padded[k : k + K]
        y[k] = np.vdot(segment, ir)

    frac2 = Fraction(fs_x / fs_tau).limit_denominator(1000)
    up2, down2 = frac2.numerator, frac2.denominator
    y = resample_poly(y, up2, down2)
    t_out = np.arange(len(y)) / fs_x
    return np.real(y * np.exp(1j * 2 * np.pi * fc * t_out))


def main() -> None:
    mat_path = pathlib.Path("matlab") / "qpsk_signal_OFDM.mat"
    ref = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    params = ref["params"]
    data_bits = torch.tensor(params.data_bits.astype(int), dtype=torch.int64)
    frame = prepare_frame(FramePrepConfig(), data_bits=data_bits)

    channel_path = pathlib.Path("input/channels/NOF1/mat/NOF1_001.mat")
    channel = load_channel_sounding(channel_path)
    torch_output = replay_filter(frame.passband, frame.fs, channel)

    np_output = replay_filter_reference(
        frame.passband.numpy(),
        frame.fs,
        channel.h.cpu().numpy(),
        channel.fs_tau,
        channel.fc,
    )
    ref_tensor = torch.from_numpy(np_output).to(torch.float64)
    rmse = torch.sqrt(torch.mean((torch_output.to(torch.float64) - ref_tensor) ** 2))
    print("Stage 2 – Channel replay")
    print(f" Output length : {torch_output.numel()} samples")
    print(f" RMSE vs ref   : {rmse.item():.3e}")


if __name__ == "__main__":
    main()
