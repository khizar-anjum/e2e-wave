"""
SIMO BER experiment with per-antenna Doppler resampling and ICI mitigation.

This script is isolated from the main pipeline. It builds a BPSK/QPSK OFDM
payload with an SC preamble, replays it through Watermark channel mats for
all hydrophones, applies per-antenna coarse Doppler (Schmidl–Cox) resampling,
per-symbol pilot PLL, optional single-iteration ICI cancellation, and MRC
combining. Use this to benchmark hard channels (KAU/BCH) without touching
the training code.
"""

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from python_replicate.channel_replay import load_channel_sounding, replay_filter  # noqa: E402
from python_replicate.frame_preparation import FramePrepConfig  # noqa: E402
from python_replicate.ofdm_mapper import OFDMConfig, OFDMMapper  # noqa: E402
from python_replicate.signal_utils import root_raised_cosine, upfirdn_torch  # noqa: E402
from python_replicate.channel_dataset import _estimate_channel_from_pilots  # noqa: E402


def map_symbols(samples: torch.Tensor, ofdm: OFDMMapper) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    serial, freq, pilots = ofdm.map(samples, return_freq=True)
    return serial, freq, pilots


def prepare_signal(serial: torch.Tensor, rrc: torch.Tensor, oversample_q: int) -> torch.Tensor:
    return upfirdn_torch(serial, rrc, up=oversample_q, down=1)


def to_passband(baseband: torch.Tensor, fc: float, fs: float) -> torch.Tensor:
    t = torch.arange(baseband.numel(), device=baseband.device, dtype=torch.float64) / fs
    return torch.real(baseband * torch.exp(1j * 2 * math.pi * fc * t))


def schmidl_cox_rotate(rx_bb: torch.Tensor, sc_len: int, fs: float, fc: float) -> torch.Tensor:
    """Estimate CFO from SC preamble and apply phase rotation (no resampling)."""
    if sc_len <= 0 or rx_bb.numel() < sc_len:
        return rx_bb
    half = sc_len // 2
    if half == 0 or rx_bb.numel() < 2 * half:
        return rx_bb
    first = rx_bb[:half]
    second = rx_bb[half : 2 * half]
    corr = torch.sum(second * torch.conj(first))
    phase = torch.atan2(corr.imag, corr.real)
    per_sample = phase / max(half, 1)
    idx = torch.arange(rx_bb.numel(), device=rx_bb.device, dtype=torch.float64)
    rot = torch.exp(-1j * per_sample * idx)
    return rx_bb * rot


def pll_correct(freq: torch.Tensor, pilot_cols: torch.Tensor, alpha: float = 0.1, beta: float = 0.01) -> torch.Tensor:
    """Second-order PLL over pilot phases across symbols."""
    if pilot_cols.numel() == 0:
        return freq
    n_symbols = freq.shape[1]
    pilot_cols = pilot_cols.clamp(0, n_symbols - 1).to(torch.long)
    phase = torch.zeros(1, device=freq.device, dtype=torch.float64)
    freq_err = torch.zeros(1, device=freq.device, dtype=torch.float64)
    corrected = []
    for t in range(n_symbols):
        mask = pilot_cols == t
        if mask.any():
            pilots = freq[:, mask.nonzero(as_tuple=False).flatten()]
            if pilots.numel() > 0:
                mean_pilot = torch.mean(pilots)
                err = torch.atan2(mean_pilot.imag, mean_pilot.real)
                err = (err + torch.pi) % (2 * torch.pi) - torch.pi
                freq_err = freq_err + beta * err
                phase = phase + alpha * err + freq_err
                phase = (phase + torch.pi) % (2 * torch.pi) - torch.pi
        rot = torch.exp(-1j * phase)
        corrected.append(freq[:, t : t + 1] * rot)
    return torch.cat(corrected, dim=1)


def ici_cancel(freq: torch.Tensor, h_est: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """Single-iteration banded ICI cancellation using neighbors k-1,k,k+1."""
    y = freq
    h = h_est
    x_hat = y / (h + 1e-9)
    for _ in range(iters):
        y_clean = y.clone()
        # subtract neighbor interference
        y_clean[1:] -= h[1:] * x_hat[:-1]
        y_clean[:-1] -= h[:-1] * x_hat[1:]
        x_hat = y_clean / (h + 1e-9)
    return x_hat


def run_ber(
    modulation: str,
    channel_paths: List[Path],
    device: torch.device,
    ofdm_cfg: OFDMConfig,
    cfg: FramePrepConfig,
    use_sc: bool = True,
    use_ici: bool = False,
    ici_iters: int = 1,
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

    serial, _, pilot_cols = map_symbols(symbols, ofdm)
    baseband = prepare_signal(serial, rrc, cfg.oversample_q)
    if use_sc and cfg.sc_length > 0:
        sc_half = torch.ones(cfg.sc_length // 2, dtype=torch.cdouble, device=device)
        sc_seq = torch.cat([sc_half, sc_half], dim=0)
        sc_shaped = upfirdn_torch(sc_seq, rrc, up=cfg.oversample_q, down=1)
        baseband = torch.cat([sc_shaped, baseband], dim=0)
    else:
        sc_shaped = None
    passband = to_passband(baseband, cfg.fc_hz, fs)

    freq_list = []
    h_list = []
    for mat_path in channel_paths:
        ch = load_channel_sounding(mat_path)
        ch.h = ch.h.to(device)
        rx = replay_filter(passband, fs, ch)
        t = torch.arange(rx.numel(), device=device, dtype=torch.float64) / fs
        bb = torch.complex(rx.to(torch.float64), torch.zeros_like(rx, dtype=torch.float64)) * torch.exp(
            -1j * 2 * math.pi * cfg.fc_hz * t
        )
        drop_samples = 0
        if use_sc and sc_shaped is not None:
            bb = schmidl_cox_rotate(bb, sc_shaped.numel(), fs, cfg.fc_hz)
            drop_samples = cfg.sc_length  # drop in symbol units after decimation
        matched = upfirdn_torch(bb, rrc, up=1, down=1)
        downsampled = matched[:: cfg.oversample_q]
        span = cfg.span
        if downsampled.numel() <= 2 * span:
            continue
        downsampled = downsampled[span:-span]
        if drop_samples > 0:
            if downsampled.numel() <= drop_samples:
                continue
            downsampled = downsampled[drop_samples:]
        sym_len = ofdm_cfg.num_carriers + ofdm_cfg.cp_length
        usable = (downsampled.numel() // sym_len) * sym_len
        if usable == 0:
            continue
        symbols_rx = downsampled[:usable].view(-1, sym_len)
        without_cp = symbols_rx[:, ofdm_cfg.cp_length :]
        freq_rx = torch.fft.fft(without_cp, dim=1) / ofdm_cfg.num_carriers
        freq_rx = freq_rx.t()
        freq_rx = pll_correct(freq_rx, pilot_cols)
        h_est = _estimate_channel_from_pilots(freq_rx, pilot_cols, ofdm_cfg.pilot_value)
        if use_ici:
            freq_rx = ici_cancel(freq_rx, h_est, iters=ici_iters)
        freq_list.append(freq_rx)
        h_list.append(h_est)

    if not freq_list:
        return 1.0

    min_sym = min(f.shape[1] for f in freq_list)
    freq_list = [f[:, :min_sym] for f in freq_list]
    h_list = [h[:, :min_sym] for h in h_list]
    num = sum(torch.conj(h) * y for h, y in zip(h_list, freq_list))
    den = sum((h.real ** 2 + h.imag ** 2) for h in h_list) + 1e-9
    freq_comb = num / den  # already equalized estimate

    data_mask = torch.ones(min_sym, dtype=torch.bool, device=device)
    if pilot_cols.numel() > 0:
        pilot_cols_clamped = pilot_cols.clamp(0, min_sym - 1).to(torch.long)
        data_mask[pilot_cols_clamped] = False
    data_cols = torch.nonzero(data_mask, as_tuple=False).reshape(-1)
    data_matrix = freq_comb.index_select(1, data_cols)
    data_rx = data_matrix.permute(1, 0).contiguous().view(-1)

    data_rx = data_rx.to(torch.complex128)
    if modulation.lower() == "bpsk":
        decoded_bits = (data_rx.real >= 0).to(torch.int64)
        tx_bits = bits.to(torch.int64)
        total = min(tx_bits.numel(), decoded_bits.numel())
        if total == 0:
            return 1.0
        errors = (decoded_bits[:total] != tx_bits[:total]).sum().item()
    else:
        real_bit = (data_rx.real >= 0).to(torch.int64)
        imag_bit = (data_rx.imag >= 0).to(torch.int64)
        decoded_bits = torch.stack([real_bit, imag_bit], dim=1).reshape(-1)
        tx_bits = bits.view(-1).to(torch.int64)
        total = min(tx_bits.numel(), decoded_bits.numel())
        if total == 0:
            return 1.0
        errors = (decoded_bits[:total] != tx_bits[:total]).sum().item()

    ber = errors / total if total > 0 else 1.0
    return ber


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    cp_lengths = [30, 60]
    pilot_periods = [4, 2]
    cfg = FramePrepConfig(sync_length=1023, sc_length=256)
    channel_dir = Path("input/channels")
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
                    ber = run_ber(mod, mat_use, device, ofdm_cfg, cfg, use_sc=False, use_ici=True, ici_iters=1)
                    results.append((ch_name, mod.upper(), ber))
            print(f"\nBER (no AWGN) cp_len={cp_len}, pilot_period={pilot_p}:")
            for ch, mod, ber in results:
                print(f"{ch:>6} | {mod}: {ber:.4e}")


if __name__ == "__main__":
    main()
