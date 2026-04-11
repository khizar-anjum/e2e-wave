from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .signal_utils import root_raised_cosine, upfirdn_torch


@dataclass
class FramePrepConfig:
    num_carriers: int = 64
    cp_length: int = 30
    modulation_order: int = 4  # QPSK
    oversample_q: int = 8
    num_ofdm_symbols: int = 16
    bandwidth_hz: float = 8e3
    rolloff: float = 0.25
    sync_length: int = 500
    sc_length: int = 128  # Schmidl-Cox preamble length (samples before shaping)
    train_length: int = 30
    fc_hz: float = 14e3
    span: int = 8
    seed: int = 0


@dataclass
class FramePrepResult:
    data_bits: torch.Tensor
    qpsk_symbols: torch.Tensor
    data_with_cp: torch.Tensor
    rrc: torch.Tensor
    sync_seq: torch.Tensor
    sync_signal: torch.Tensor
    train_seq: torch.Tensor
    train_signal: torch.Tensor
    data_signal: torch.Tensor
    packet_baseband: torch.Tensor
    passband: torch.Tensor
    fs: float
    params: dict


def prepare_frame(
    config: FramePrepConfig, data_bits: Optional[torch.Tensor] = None
) -> FramePrepResult:
    torch.manual_seed(config.seed)
    bits_per_symbol = int(math.log2(config.modulation_order))
    total_bits = config.num_carriers * config.num_ofdm_symbols * bits_per_symbol
    if data_bits is None:
        data_bits = torch.randint(0, 2, (total_bits,), dtype=torch.int64)
    if data_bits.numel() != total_bits:
        raise ValueError(f"Expected {total_bits} bits, received {data_bits.numel()}.")

    symbols = None
    if bits_per_symbol == 1:
        mapped = 2 * data_bits.to(torch.float64) - 1
        symbols = torch.complex(mapped, torch.zeros_like(mapped))
    elif bits_per_symbol == 2:
        bit_pairs = data_bits.view(-1, 2).to(torch.float64)
        symbols = torch.complex(
            2 * bit_pairs[:, 0] - 1, 2 * bit_pairs[:, 1] - 1
        ) / math.sqrt(2)
    else:
        raise NotImplementedError(
            f"Modulation order {config.modulation_order} not supported."
        )
    num_fft = 1
    while num_fft < config.num_carriers:
        num_fft <<= 1
    symbols_matrix = (
        symbols.view(config.num_ofdm_symbols, config.num_carriers)
        .t()
        .contiguous()
    )
    freq_domain = torch.zeros(
        (num_fft, symbols_matrix.shape[1]), dtype=torch.cdouble
    )
    freq_domain[: config.num_carriers, :] = symbols_matrix.to(torch.cdouble)
    data_ifft = torch.fft.ifft(freq_domain, dim=0) * num_fft
    cp = data_ifft[-config.cp_length :, :]
    data_with_cp = torch.cat([cp, data_ifft], dim=0)
    serial_stream = data_with_cp.t().reshape(-1)

    rrc = root_raised_cosine(config.rolloff, config.span, config.oversample_q)
    sync_idx = torch.arange(config.sync_length, dtype=torch.float64)
    sync_seq = torch.exp(-1j * math.pi * (sync_idx**2) / config.sync_length)
    sync_signal = upfirdn_torch(sync_seq, rrc, up=config.oversample_q, down=1)
    sync_signal = sync_signal / torch.sqrt(
        torch.mean(torch.abs(sync_signal) ** 2)
    )

    # MATLAB's `zeros(size(train_length))` returns a scalar zero.
    train_seq = torch.zeros(1, dtype=torch.cdouble)
    train_signal = upfirdn_torch(train_seq, rrc, up=config.oversample_q, down=1)

    data_signal = upfirdn_torch(serial_stream, rrc, up=config.oversample_q, down=1)
    packet = torch.cat([sync_signal, train_signal, data_signal])

    fs = config.oversample_q * config.bandwidth_hz
    t = torch.arange(packet.numel(), dtype=torch.float64) / fs
    passband = torch.real(packet * torch.exp(1j * 2 * math.pi * config.fc_hz * t))

    params = {
        "sync_seq": sync_seq,
        "train_seq": train_seq,
        "sps": config.oversample_q,
        "rrc": rrc,
        "fc": config.fc_hz,
        "fs": fs,
        "ofdm_len": serial_stream.numel(),
        "data_bits": data_bits.clone(),
        "span": config.span,
        "packet": packet,
        "data_symbols": freq_domain[: config.num_carriers, :],
        "data_signal": data_signal,
        "num_fft": num_fft,
        "cp_length": config.cp_length,
    }

    return FramePrepResult(
        data_bits=data_bits,
        qpsk_symbols=symbols_matrix,
        data_with_cp=data_with_cp,
        rrc=rrc,
        sync_seq=sync_seq,
        sync_signal=sync_signal,
        train_seq=train_seq,
        train_signal=train_signal,
        data_signal=data_signal,
        packet_baseband=packet,
        passband=passband,
        fs=fs,
        params=params,
    )
