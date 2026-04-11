from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass
class OFDMConfig:
    num_carriers: int = 64
    cp_length: int = 30
    pilot_period: int = 4  # number of OFDM symbols per pilot block
    pilot_value: complex = 1 + 0j


class OFDMMapper:
    """Maps arbitrary complex samples onto OFDM tones with pilots + CP."""

    def __init__(self, config: OFDMConfig):
        self.config = config
        self.num_carriers = config.num_carriers
        self.cp_length = config.cp_length
        if config.pilot_period < 2:
            raise ValueError("pilot_period must be >= 2 to allocate data symbols.")
        self.pilot_period = int(config.pilot_period)
        self.pilot_value = torch.as_tensor(config.pilot_value, dtype=torch.complex128)

    def map(self, samples: torch.Tensor, return_freq: bool = False):
        """Return time-domain OFDM waveform (IFFT + CP) for given data symbols."""
        if samples.numel() == 0:
            return torch.zeros(0, dtype=torch.cdouble, device=samples.device)

        samples = samples.reshape(-1).to(torch.complex128)
        data_per_block = self.num_carriers * (self.pilot_period - 1)
        n_blocks = math.ceil(samples.numel() / data_per_block)
        total_needed = n_blocks * data_per_block
        if samples.numel() < total_needed:
            pad = torch.zeros(
                total_needed - samples.numel(), dtype=samples.dtype, device=samples.device
            )
            samples = torch.cat([samples, pad], dim=0)

        total_symbols = n_blocks * self.pilot_period
        freq = torch.zeros(
            (self.num_carriers, total_symbols), dtype=torch.complex128, device=samples.device
        )
        pilot_cols = []
        cursor = 0
        pilot_val = self.pilot_value.to(freq.dtype).to(freq.device)
        for block in range(n_blocks):
            base = block * self.pilot_period
            pilot_cols.append(base)
            freq[:, base] = pilot_val
            for offset in range(1, self.pilot_period):
                col = base + offset
                chunk = samples[cursor : cursor + self.num_carriers]
                freq[:, col] = chunk
                cursor += self.num_carriers

        time_domain = torch.fft.ifft(freq, dim=0) * self.num_carriers
        cp = time_domain[-self.cp_length :, :]
        with_cp = torch.cat([cp, time_domain], dim=0)
        serial = with_cp.t().reshape(-1)
        if return_freq:
            pilot_tensor = torch.tensor(pilot_cols, dtype=torch.long, device=samples.device)
            return serial, freq, pilot_tensor
        return serial
