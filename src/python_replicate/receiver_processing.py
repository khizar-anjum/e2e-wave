from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .signal_utils import upfirdn_torch


def _xcorr_fft(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.numel()
    m = y.numel()
    size = n + m - 1
    nfft = 1 << math.ceil(math.log2(max(size, 1)))
    X = torch.fft.fft(x, nfft)
    Y = torch.fft.fft(y, nfft)
    corr = torch.fft.ifft(X * torch.conj(Y))
    corr = torch.roll(corr, shifts=m - 1)[:size]
    lags = torch.arange(-(m - 1), n, device=x.device)
    return corr, lags


@dataclass
class ReceiverParams:
    fs: float
    fc: float
    rrc: torch.Tensor
    sps: int
    sync_seq: torch.Tensor
    train_seq: torch.Tensor
    span: int
    ofdm_len: int
    num_fft: int
    cp_length: int
    data_symbols: torch.Tensor
    pilot_columns: Optional[torch.Tensor] = None
    pilot_value: complex = 1 + 0j
    bits_per_symbol: int = 2


def extract_ofdm_symbols_with_ce(
    y: torch.Tensor,
    params: ReceiverParams,
    return_details: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[dict]]:
    device = y.device
    t = torch.arange(y.numel(), dtype=torch.float64, device=device) / params.fs
    y_bb = y.to(torch.float64)
    y_bb = torch.complex(y_bb, torch.zeros_like(y_bb))
    y_bb = y_bb * torch.exp(-1j * 2 * math.pi * params.fc * t)

    y_filtered = upfirdn_torch(y_bb, params.rrc.to(device), up=1, down=1)

    ref_seq = upfirdn_torch(
        params.sync_seq.to(device), params.rrc.to(device), up=params.sps, down=1
    )
    ref_seq = ref_seq / torch.sqrt(torch.mean(torch.abs(ref_seq) ** 2))
    ref_seq2 = upfirdn_torch(ref_seq, params.rrc.to(device), up=1, down=1)
    corr, lags = _xcorr_fft(y_filtered, ref_seq2)
    idx = torch.argmax(torch.abs(corr))
    frame_start = int(lags[idx].item())

    if frame_start < 0 or frame_start + ref_seq2.numel() >= y_filtered.numel():
        return (None, None, None) if return_details else (None, None)

    train_ref = upfirdn_torch(
        params.train_seq.to(device), params.rrc.to(device), up=params.sps, down=1
    )
    train_start = frame_start + ref_seq.numel()
    train_end = train_start + train_ref.numel()
    if train_end >= y_bb.numel():
        return (None, None, None) if return_details else (None, None)

    data_start = train_end
    p = params.sps
    lh = params.rrc.numel()
    offset = math.ceil((params.ofdm_len - 1) * p + lh)
    data_end = min(data_start + offset, y_bb.numel())
    if data_end <= data_start:
        return (None, None, None) if return_details else (None, None)

    y_data = y_bb[data_start:data_end]
    y_data = upfirdn_torch(y_data, params.rrc.to(device), up=1, down=1)
    resampled = y_data[:: params.sps]
    if resampled.numel() <= 2 * params.span:
        return (None, None, None) if return_details else (None, None)
    symbols_with_cp = resampled[params.span : -params.span]
    frame_len = params.num_fft + params.cp_length
    usable = (symbols_with_cp.numel() // frame_len) * frame_len
    if usable == 0:
        return (None, None, None) if return_details else (None, None)
    symbols_with_cp = symbols_with_cp[:usable]
    symbols = symbols_with_cp.view(-1, frame_len).transpose(0, 1)
    symbols = symbols[params.cp_length :, :]
    freq_symbols = torch.fft.fft(symbols, dim=0)
    freq_symbols = (freq_symbols / params.num_fft) * 2

    tx_symbols = params.data_symbols.to(device)
    tx_symbols = tx_symbols[:, : freq_symbols.shape[1]]
    H_est = freq_symbols / tx_symbols

    num_cols = freq_symbols.shape[1]
    mask = torch.ones_like(freq_symbols.real, dtype=torch.bool)

    if params.pilot_columns is not None and len(params.pilot_columns) > 0:
        pilot_cols = torch.as_tensor(
            params.pilot_columns, dtype=torch.long, device=device
        )
        pilot_cols = torch.unique(pilot_cols.clamp(0, num_cols - 1))
        if pilot_cols.numel() == 0:
            channel_est = torch.ones_like(freq_symbols)
        else:
            pilot_cols, _ = torch.sort(pilot_cols)
            pilot_val = torch.as_tensor(
                params.pilot_value, dtype=torch.complex128, device=device
            )
            known = freq_symbols[:, pilot_cols] / pilot_val
            channel_est = torch.zeros_like(freq_symbols)
            all_times = torch.arange(num_cols, dtype=torch.float64, device=device)

            def interp_time(pilot_pos: torch.Tensor, pilot_vals: torch.Tensor) -> torch.Tensor:
                if pilot_pos.numel() == 1:
                    return pilot_vals.repeat(all_times.numel())
                output = torch.empty(all_times.numel(), dtype=torch.complex128, device=device)
                for idx, t in enumerate(all_times):
                    if t <= pilot_pos[0]:
                        output[idx] = pilot_vals[0]
                        continue
                    if t >= pilot_pos[-1]:
                        output[idx] = pilot_vals[-1]
                        continue
                    hi = torch.searchsorted(pilot_pos, t)
                    lo = hi - 1
                    denom = pilot_pos[hi] - pilot_pos[lo]
                    weight = (t - pilot_pos[lo]) / denom
                    output[idx] = pilot_vals[lo] + weight * (pilot_vals[hi] - pilot_vals[lo])
                return output

            for row in range(freq_symbols.shape[0]):
                pilots = known[row]
                interp_vals = interp_time(
                    pilot_cols.to(torch.float64), pilots.to(torch.complex128)
                )
                channel_est[row, :] = interp_vals
                channel_est[row, pilot_cols] = pilots
        mask[:, pilot_cols] = False
    else:
        pilot_indices = torch.tensor([1, 5, 9, 13], dtype=torch.float64, device=device)
        pilot_cols = (pilot_indices - 1).long()
        valid = pilot_cols < freq_symbols.shape[1]
        pilot_cols = pilot_cols[valid]
        pilot_indices = pilot_indices[valid]

        if pilot_cols.numel() == 0:
            channel_est = torch.ones_like(freq_symbols)
        else:
            x_targets = torch.arange(
                1,
                freq_symbols.shape[1] + 1,
                dtype=torch.float64,
                device=device,
            )
            H_interp = torch.zeros_like(freq_symbols)
            for row in range(freq_symbols.shape[0]):
                pilots = H_est[row, pilot_cols]
                interp_vals = torch.empty_like(x_targets, dtype=torch.complex128)

                if pilot_indices.numel() == 1:
                    interp_vals[:] = pilots[0]
                else:
                    for j, xt in enumerate(x_targets):
                        if xt <= pilot_indices[0]:
                            slope = (pilots[1] - pilots[0]) / (
                                pilot_indices[1] - pilot_indices[0]
                            )
                            interp_vals[j] = pilots[0] + slope * (xt - pilot_indices[0])
                        elif xt >= pilot_indices[-1]:
                            slope = (pilots[-1] - pilots[-2]) / (
                                pilot_indices[-1] - pilot_indices[-2]
                            )
                            interp_vals[j] = pilots[-1] + slope * (xt - pilot_indices[-1])
                        else:
                            hi = torch.searchsorted(pilot_indices, xt)
                            lo = hi - 1
                            slope = (pilots[hi] - pilots[lo]) / (
                                pilot_indices[hi] - pilot_indices[lo]
                            )
                            interp_vals[j] = pilots[lo] + slope * (xt - pilot_indices[lo])

                H_interp[row, :] = interp_vals
                H_interp[row, pilot_cols] = pilots

            mask[:, pilot_cols] = False
            channel_est = H_interp

    eq = freq_symbols / channel_est
    real_sign = torch.sign(eq.real)
    imag_sign = torch.sign(eq.imag)
    real_tx = torch.sign(tx_symbols.real)
    imag_tx = torch.sign(tx_symbols.imag)
    bits_per_symbol = max(1, params.bits_per_symbol)
    if bits_per_symbol == 1:
        wrong_real = real_sign[mask] != real_tx[mask]
        total_errors = wrong_real.sum()
        total_bits = wrong_real.numel()
    else:
        wrong_real = real_sign[mask] != real_tx[mask]
        wrong_imag = imag_sign[mask] != imag_tx[mask]
        total_errors = wrong_real.sum() + wrong_imag.sum()
        total_bits = wrong_real.numel() + wrong_imag.numel()
    ber = total_errors.item() / total_bits if total_bits > 0 else None
    if return_details:
        details = {
            "freq_symbols": freq_symbols,
            "eq_symbols": eq,
            "tx_symbols": tx_symbols,
            "data_mask": mask,
            "channel_est": channel_est,
        }
        return channel_est, ber, details
    return channel_est, ber
