import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from python_replicate.channel_dataset import (  # noqa: E402
    _estimate_channel_from_pilots,
    _flatten_data_matrix,
    _select_data_columns,
    _pll_correct,
)
from python_replicate.channel_replay import load_channel_sounding, replay_filter  # noqa: E402
from python_replicate.frame_preparation import FramePrepConfig  # noqa: E402
from python_replicate.ofdm_mapper import OFDMConfig, OFDMMapper  # noqa: E402
from python_replicate.signal_utils import root_raised_cosine, upfirdn_torch  # noqa: E402


def map_symbols(samples: torch.Tensor, ofdm: OFDMMapper):
    serial, freq, pilots = ofdm.map(samples, return_freq=True)
    return serial, freq, pilots


def prepare_signal(serial: torch.Tensor, rrc: torch.Tensor, oversample_q: int) -> torch.Tensor:
    shaped = upfirdn_torch(serial, rrc, up=oversample_q, down=1)
    return shaped


def to_passband(baseband: torch.Tensor, fc: float, fs: float) -> torch.Tensor:
    t = torch.arange(baseband.numel(), device=baseband.device, dtype=torch.float64) / fs
    return torch.real(baseband * torch.exp(1j * 2 * math.pi * fc * t))


def run_ber(
    modulation: str,
    channel_paths: list[Path],
    device: torch.device,
    ofdm_cfg: OFDMConfig,
    cfg: FramePrepConfig,
) -> float:
    ofdm = OFDMMapper(ofdm_cfg)
    rrc = root_raised_cosine(cfg.rolloff, cfg.span, cfg.oversample_q).to(torch.float64).to(device)
    fs = cfg.oversample_q * cfg.bandwidth_hz

    # Bits and symbols
    if modulation.lower() == "bpsk":
        n_bits = 8192
        bits = torch.randint(0, 2, (n_bits,), device=device, dtype=torch.int64)
        symbols = (2 * bits - 1).to(torch.float64)
        symbols = torch.complex(symbols, torch.zeros_like(symbols))
    elif modulation.lower() == "qpsk":
        n_bits = 8192
        bits = torch.randint(0, 2, (n_bits,), device=device, dtype=torch.int64)
        bits = bits[: (bits.numel() // 2) * 2].view(-1, 2).to(torch.float64)
        symbols = torch.complex(2 * bits[:, 0] - 1, 2 * bits[:, 1] - 1) / math.sqrt(2)
    else:
        raise ValueError("modulation must be bpsk or qpsk")

    serial, freq, pilot_cols = map_symbols(symbols, ofdm)
    baseband = prepare_signal(serial, rrc, cfg.oversample_q)
    passband = to_passband(baseband, cfg.fc_hz, fs)

    freq_list = []
    chan_est_list = []
    for channel_path in channel_paths:
        channel = load_channel_sounding(channel_path)
        channel.h = channel.h.to(device)
        rx = replay_filter(passband.to(device), fs, channel)
        t = torch.arange(rx.numel(), device=device, dtype=torch.float64) / fs
        bb = torch.complex(rx.to(torch.float64), torch.zeros_like(rx, dtype=torch.float64)) * torch.exp(
            -1j * 2 * math.pi * cfg.fc_hz * t
        )
        matched = upfirdn_torch(bb, rrc, up=1, down=1)
        downsampled = matched[:: cfg.oversample_q]
        span = cfg.span
        if downsampled.numel() <= 2 * span:
            continue
        downsampled = downsampled[span:-span]
        sym_len = ofdm_cfg.num_carriers + ofdm_cfg.cp_length
        usable = (downsampled.numel() // sym_len) * sym_len
        if usable == 0:
            continue
        symbols_rx = downsampled[:usable].view(-1, sym_len)
        without_cp = symbols_rx[:, ofdm_cfg.cp_length :]
        freq_rx = torch.fft.fft(without_cp, dim=1) / ofdm_cfg.num_carriers
        freq_rx = freq_rx.t()
        freq_rx = _pll_correct(freq_rx, pilot_cols)
        freq_list.append(freq_rx)
        chan_est_list.append(_estimate_channel_from_pilots(freq_rx, pilot_cols, ofdm_cfg.pilot_value))

    if not freq_list:
        return 1.0
    min_sym = min(f.shape[1] for f in freq_list)
    freq_list = [f[:, :min_sym] for f in freq_list]
    chan_est_list = [h[:, :min_sym] for h in chan_est_list]
    num = sum(torch.conj(h) * y for h, y in zip(chan_est_list, freq_list))
    den = sum((h.real ** 2 + h.imag ** 2) for h in chan_est_list) + 1e-9
    freq_comb = num / den

    data_matrix, _ = _select_data_columns(freq_comb, pilot_cols, min_sym)
    if data_matrix.numel() == 0:
        return 1.0
    data_rx = _flatten_data_matrix(data_matrix)

    # Trim to original symbol count
    data_rx = data_rx[: symbols.numel()].to(torch.complex128)

    if modulation.lower() == "bpsk":
        decoded_bits = (data_rx.real >= 0).to(torch.int64)
        tx_bits = bits.to(torch.int64)
        errors = (decoded_bits[: tx_bits.numel()] != tx_bits).sum().item()
        total = tx_bits.numel()
    else:
        # QPSK decision
        real_bit = (data_rx.real >= 0).to(torch.int64)
        imag_bit = (data_rx.imag >= 0).to(torch.int64)
        decoded_bits = torch.stack([real_bit, imag_bit], dim=1).reshape(-1)
        tx_bits = bits.view(-1).to(torch.int64)
        errors = (decoded_bits[: tx_bits.numel()] != tx_bits).sum().item()
        total = tx_bits.numel()

    ber = errors / total if total > 0 else 1.0
    return ber


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    cp_lengths = [30, 60]
    pilot_periods = [4, 2]
    cfg = FramePrepConfig(sync_length=1023, sc_length=256)  # longer sync + SC preamble
    channel_dir = Path("/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/input/channels")
    mats: List[Tuple[str, List[Path]]] = []
    for ch_dir in sorted(channel_dir.iterdir()):
        if not ch_dir.is_dir():
            continue
        mat_files = sorted((ch_dir / "mat").glob(f"{ch_dir.name}_*.mat"))
        if not mat_files:
            continue
        mats.append((ch_dir.name, mat_files))

    for cp_len in cp_lengths:
        for pilot_p in pilot_periods:
            if cp_len >= OFDMConfig().num_carriers:
                continue
            ofdm_cfg = OFDMConfig(cp_length=cp_len, pilot_period=pilot_p)
            results = []
            for ch_name, mat_list in mats:
                simo = ch_name in {"BCH1", "KAU1", "KAU2"}
                mat_use = mat_list if simo else [mat_list[0]]
                for mod in ["bpsk", "qpsk"]:
                    ber = run_ber(mod, mat_use, device, ofdm_cfg, cfg)
                    results.append((ch_name, mod.upper(), ber))
            print(f"\nBER (no AWGN) cp_len={cp_len}, pilot_period={pilot_p}:")
            for ch, mod, ber in results:
                print(f"{ch:>6} | {mod}: {ber:.4e}")


if __name__ == "__main__":
    main()
