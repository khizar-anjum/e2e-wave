from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import correlate, upfirdn

from .frame_preparation import FramePrepConfig, prepare_frame
from .packet_retrieval import PacketRetriever
from .receiver_processing import ReceiverParams, extract_ofdm_symbols_with_ce
from .watermark_pipeline import simulate_watermark


def extract_reference(y, params_dict):
    fs = params_dict["fs"]
    fc = params_dict["fc"]
    rrc = params_dict["rrc"]
    sps = params_dict["sps"]
    sync_seq = params_dict["sync_seq"]
    train_seq = params_dict["train_seq"]
    span = params_dict["span"]
    num_fft = params_dict["num_fft"]
    cp_length = params_dict["cp_length"]
    data_symbols = params_dict["data_symbols"]
    ofdm_len = params_dict["ofdm_len"]

    t = np.arange(len(y)) / fs
    y_bb = y * np.exp(-1j * 2 * np.pi * fc * t)
    y_filtered = upfirdn(rrc, y_bb, up=1, down=1)

    ref_seq = upfirdn(rrc, sync_seq, up=sps, down=1)
    ref_seq = ref_seq / np.sqrt(np.mean(np.abs(ref_seq) ** 2))
    ref_seq2 = upfirdn(rrc, ref_seq, up=1, down=1)
    corr = correlate(y_filtered, ref_seq2, mode="full")
    lags = np.arange(-(len(ref_seq2) - 1), len(y_filtered))
    frame_start = lags[np.argmax(np.abs(corr))]

    if frame_start < 0 or frame_start + len(ref_seq2) >= len(y_filtered):
        return None, None

    train_ref = upfirdn(rrc, train_seq, up=sps, down=1)
    train_start = frame_start + len(ref_seq2)
    train_end = train_start + len(train_ref)
    if train_end >= len(y_bb):
        return None, None

    p = sps
    lh = len(rrc)
    offset = math.ceil((ofdm_len - 1) * p + lh)
    data_start = train_end
    data_end = data_start + offset
    if data_end > len(y_bb):
        return None, None

    y_data = y_bb[data_start:data_end]
    y_data = upfirdn(rrc, y_data, up=1, down=1)
    resampled = y_data[::sps]
    symbols_with_cp = resampled[span:-span]
    frame_len = num_fft + cp_length
    usable = (len(symbols_with_cp) // frame_len) * frame_len
    symbols_with_cp = symbols_with_cp[:usable]
    symbols = symbols_with_cp.reshape(-1, frame_len).T
    symbols = symbols[cp_length:, :]
    freq_symbols = np.fft.fft(symbols, axis=0)
    freq_symbols = (freq_symbols / num_fft) * 2

    tx_symbols = data_symbols[:, : freq_symbols.shape[1]]
    H_est = freq_symbols / tx_symbols

    pilot_idx = np.array([0, 4, 8, 12])
    H_interp = np.zeros_like(freq_symbols, dtype=complex)
    for row in range(freq_symbols.shape[0]):
        pilots = H_est[row, pilot_idx]
        x_points = pilot_idx
        x_new = np.arange(freq_symbols.shape[1])
        interp_real = np.interp(x_new, x_points, pilots.real)
        interp_imag = np.interp(x_new, x_points, pilots.imag)
        interp = interp_real + 1j * interp_imag
        H_interp[row, :] = interp
        H_interp[row, pilot_idx] = pilots

    eq = freq_symbols / H_interp
    real_sign = np.sign(eq.real)
    imag_sign = np.sign(eq.imag)
    real_tx = np.sign(tx_symbols.real)
    imag_tx = np.sign(tx_symbols.imag)
    mask = np.ones_like(real_sign, dtype=bool)
    mask[:, pilot_idx] = False
    wrong_real = real_sign[mask] != real_tx[mask]
    wrong_imag = imag_sign[mask] != imag_tx[mask]
    total_bits = wrong_real.size + wrong_imag.size
    ber = (wrong_real.sum() + wrong_imag.sum()) / total_bits
    return H_est, ber


def main() -> None:
    mat = loadmat("matlab/qpsk_signal_OFDM.mat", squeeze_me=True, struct_as_record=False)
    params_mat = mat["params"]
    data_bits = torch.tensor(params_mat.data_bits.astype(int), dtype=torch.int64)
    frame = prepare_frame(FramePrepConfig(), data_bits=data_bits)
    duration = frame.passband.numel() / frame.fs
    wm_output = simulate_watermark(
        frame.passband,
        frame.fs,
        frame.data_bits.numel(),
        frame.data_bits.numel() / duration,
        Path("input/channels"),
        "NOF1",
        howmany="single",
    )
    retriever = PacketRetriever(wm_output)
    packet, _ = retriever.fetch(packet_number=1, snr_db=15.0, rng_seed=21)

    rx_params = ReceiverParams(
        fs=frame.fs,
        fc=frame.params["fc"],
        rrc=frame.rrc,
        sps=frame.params["sps"],
        sync_seq=frame.params["sync_seq"],
        train_seq=frame.params["train_seq"],
        span=frame.params["span"],
        ofdm_len=frame.params["ofdm_len"],
        num_fft=frame.params["num_fft"],
        cp_length=frame.params["cp_length"],
        data_symbols=frame.params["data_symbols"],
        bits_per_symbol=2,
    )
    H_est, ber = extract_ofdm_symbols_with_ce(packet, rx_params)
    ref_params = {
        "fs": frame.fs,
        "fc": frame.params["fc"],
        "rrc": frame.rrc.cpu().numpy(),
        "sps": int(frame.params["sps"]),
        "sync_seq": frame.params["sync_seq"].cpu().numpy(),
        "train_seq": np.atleast_1d(frame.params["train_seq"].cpu().numpy()),
        "span": int(frame.params["span"]),
        "num_fft": int(frame.params["num_fft"]),
        "cp_length": int(frame.params["cp_length"]),
        "data_symbols": frame.params["data_symbols"].cpu().numpy(),
        "ofdm_len": int(frame.params["ofdm_len"]),
    }
    H_ref, ber_ref = extract_reference(packet.numpy(), ref_params)
    rmse = torch.sqrt(
        torch.mean(
            torch.abs(H_est - torch.from_numpy(H_ref)).to(torch.float64) ** 2
        )
    )
    print("Stage 4 – Receiver processing")
    print(f" H_est RMSE : {rmse.item():.3e}")
    if ber is not None:
        print(f" BER (torch) : {ber:.3e}")
    if ber_ref is not None:
        print(f" BER (ref)   : {ber_ref:.3e}")


if __name__ == "__main__":
    main()
