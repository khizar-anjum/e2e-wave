from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from .frame_preparation import FramePrepConfig, prepare_frame
from .packet_retrieval import PacketRetriever
from .watermark_pipeline import simulate_watermark


def reference_packet_no_noise(wm_output, packet_number):
    bk = wm_output.bookkeeping
    idx0 = packet_number - 1
    sounding_idx = idx0 // bk.nPacketsPerSounding
    local_idx = idx0 % bk.nPacketsPerSounding
    start, end = bk.packet_indices[local_idx]
    packet = (
        wm_output.soundings[sounding_idx][start:end]
        .clone()
        .to(torch.float64)
        .numpy()
    )
    V0 = bk.velocities[sounding_idx].item()
    c = 1500.0
    factor = 1.0 / (1.0 - V0 / c)
    frac = Fraction(factor).limit_denominator(1000)
    packet = resample_poly(packet, frac.numerator, frac.denominator)
    return packet


def main() -> None:
    frame = prepare_frame(FramePrepConfig())
    signal_duration = frame.passband.numel() / frame.fs
    wm_output = simulate_watermark(
        frame.passband,
        frame.fs,
        frame.data_bits.numel(),
        frame.data_bits.numel() / signal_duration,
        Path("input/channels"),
        "NOF1",
        howmany="single",
    )
    retriever = PacketRetriever(wm_output)

    packet_number = 1
    clean_packet, fs = retriever.fetch(packet_number)
    ref_packet = reference_packet_no_noise(wm_output, packet_number)
    rmse = torch.sqrt(
        torch.mean(
            (clean_packet.to(torch.float64) - torch.from_numpy(ref_packet)) ** 2
        )
    )
    print("Stage 3 – Packet retrieval")
    print(f" Clean packet RMSE : {rmse.item():.3e}")

    snr_db = 10.0
    seed = 123
    noisy_packet, _ = retriever.fetch(packet_number, snr_db=snr_db, rng_seed=seed)
    clean_packet = clean_packet.to(torch.float64)
    rng = torch.Generator().manual_seed(seed)
    rand_offset = torch.rand(1, generator=rng).item()
    start = int(round((4 + 2 * rand_offset) * fs))
    padded = torch.zeros_like(noisy_packet)
    padded[start : start + clean_packet.numel()] = clean_packet
    noise = noisy_packet - padded
    Eb_est = torch.sum(clean_packet**2) / wm_output.bookkeeping.nBits
    sigma2 = torch.mean(noise**2)
    snr_est = 10 * torch.log10(Eb_est / (2 * sigma2))
    print(f" SNR target      : {snr_db:.2f} dB")
    print(f" SNR estimated   : {snr_est.item():.2f} dB")


if __name__ == "__main__":
    main()
