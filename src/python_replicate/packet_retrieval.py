from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

import torch

from .signal_utils import resample_poly_torch
from .watermark_pipeline import Bookkeeping, WatermarkOutput


@dataclass
class PacketRetriever:
    wm_output: WatermarkOutput

    def fetch(
        self,
        packet_number: int,
        snr_db: Optional[float] = None,
        rng_seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        bk = self.wm_output.bookkeeping
        if packet_number <= 0 or packet_number > bk.nPackets:
            raise ValueError(
                f"packet_number must be in [1, {bk.nPackets}], got {packet_number}"
            )
        idx0 = packet_number - 1
        sounding_idx = idx0 // bk.nPacketsPerSounding
        local_idx = idx0 % bk.nPacketsPerSounding
        start, end = bk.packet_indices[local_idx]
        packet = self.wm_output.soundings[sounding_idx][start:end].clone()

        c = 1500.0
        V0 = bk.velocities[sounding_idx].item()
        resampling_factor = 1.0 / (1.0 - V0 / c)
        frac = Fraction(resampling_factor).limit_denominator(1000)
        packet = resample_poly_torch(packet, frac.numerator, frac.denominator)

        if snr_db is None:
            return packet, self.wm_output.fs

        Eb = torch.sum(packet**2) / max(bk.nBits, 1)
        Eb = torch.clamp(Eb, min=1e-12)
        rng = torch.Generator()
        seed = rng_seed if rng_seed is not None else packet_number
        rng.manual_seed(int(seed))
        rand_offset = torch.rand(1, generator=rng).item()
        i_start = int(round((4 + 2 * rand_offset) * self.wm_output.fs))
        padding = int(round(10 * self.wm_output.fs))
        y = torch.zeros(packet.numel() + padding, dtype=torch.float64)
        y[i_start : i_start + packet.numel()] = packet

        awgn = torch.randn(y.shape, generator=rng)
        N0 = 2.0
        scaling = 10 * math.log10(Eb.item() / N0) - snr_db
        awgn = awgn * (10 ** (scaling / 20))
        y = y + awgn
        return y, self.wm_output.fs
