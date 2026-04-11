from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import sys
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from python_replicate.channel_dataset import ChannelSimulationPipeline
from python_replicate.frame_preparation import FramePrepConfig, prepare_frame
from python_replicate.ofdm_mapper import OFDMConfig
from python_replicate.receiver_processing import (
    ReceiverParams,
    extract_ofdm_symbols_with_ce,
)
from python_replicate.waveform_bank import ComplexWaveformSystem
OUTPUT_DIR = ROOT / "python_replicate" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = OUTPUT_DIR / "zcz_correlation.pt"
FIG_PATH = OUTPUT_DIR / "zcz_correlation.png"
CHANNEL_PATH = (
    ROOT
    / "input"
    / "channels"
    / "NOF1"
    / "mat"
    / "NOF1_001.mat"
)

FRAME_CONFIG = FramePrepConfig()
OFDM_CFG = OFDMConfig()
ZCZ_LENGTH = 64
ZCZ_ROOT = 1


def zadoff_chu_sequence(length: int, root: int = 1) -> torch.Tensor:
    n = torch.arange(length, dtype=torch.float64)
    seq = torch.exp(
        -1j * math.pi * root * n * (n + 1) / length
    )
    seq = seq / torch.sqrt(torch.mean(torch.abs(seq) ** 2))
    return seq


def build_zcz_bank(length: int, root: int) -> ComplexWaveformSystem:
    seq = zadoff_chu_sequence(length, root)
    freq = torch.fft.fft(seq, norm="ortho")
    bank = ComplexWaveformSystem(
        num_tokens=1,
        output_seq_len=length,
        use_temperature=False,
    )
    with torch.no_grad():
        bank.freq_real.zero_()
        bank.freq_imag.zero_()
        bank.freq_real[0, : freq.numel()] = freq.real.to(bank.freq_real.dtype)
        bank.freq_imag[0, : freq.numel()] = freq.imag.to(bank.freq_imag.dtype)
    return bank


def build_receiver_params(
    frame_template,
    data_symbols: torch.Tensor,
    pilot_columns: torch.Tensor,
) -> ReceiverParams:
    return ReceiverParams(
        fs=frame_template.fs,
        fc=frame_template.params["fc"],
        rrc=frame_template.rrc,
        sps=frame_template.params["sps"],
        sync_seq=frame_template.params["sync_seq"],
        train_seq=frame_template.params["train_seq"],
        span=frame_template.params["span"],
        ofdm_len=frame_template.params["ofdm_len"],
        num_fft=frame_template.params["num_fft"],
        cp_length=frame_template.params["cp_length"],
        data_symbols=data_symbols,
        pilot_columns=pilot_columns,
        pilot_value=OFDM_CFG.pilot_value,
        bits_per_symbol=2,
    )


def normalized_cross_correlation(
    tx: torch.Tensor,
    rx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tx = tx.reshape(-1)
    rx = rx.reshape(-1)
    n = tx.numel()
    lags = torch.arange(-(n - 1), n, dtype=torch.int64)
    corr = torch.empty(2 * n - 1, dtype=torch.complex128)
    for i, lag in enumerate(range(-(n - 1), n)):
        if lag >= 0:
            tx_seg = tx[lag:]
            rx_seg = rx[: n - lag]
        else:
            tx_seg = tx[: n + lag]
            rx_seg = rx[-lag:]
        if tx_seg.numel() == 0:
            corr[i] = 0
            continue
        corr[i] = torch.sum(tx_seg * torch.conj(rx_seg))
    denom = torch.linalg.norm(tx) * torch.linalg.norm(rx) + 1e-8
    corr = corr / denom
    return corr, lags


def tokens_needed_per_frame(
    pipeline: ChannelSimulationPipeline,
    bank: ComplexWaveformSystem,
) -> int:
    data_cols_per_block = pipeline.ofdm_mapper.pilot_period - 1
    num_blocks = max(
        1,
        math.ceil(FRAME_CONFIG.num_ofdm_symbols / pipeline.ofdm_mapper.pilot_period),
    )
    total_data_columns = num_blocks * data_cols_per_block
    total_samples = total_data_columns * pipeline.ofdm_mapper.num_carriers
    tokens_needed = math.ceil(total_samples / bank.output_seq_len)
    return tokens_needed


def main() -> None:
    pipeline = ChannelSimulationPipeline(
        channel_path=CHANNEL_PATH,
        frame_config=FRAME_CONFIG,
        ofdm_config=OFDM_CFG,
    )
    frame_template = prepare_frame(FRAME_CONFIG)
    bank = build_zcz_bank(ZCZ_LENGTH, ZCZ_ROOT)
    num_tokens = tokens_needed_per_frame(pipeline, bank)
    tokens = torch.zeros(num_tokens, dtype=torch.long)

    sim = pipeline.simulate_video(
        bank,
        [tokens],
        snr_schedule=torch.tensor([0.0]),
        add_awgn=False,
    )
    freq_grid = sim.tx_freq_grids[0]
    pilot_columns = sim.pilot_columns[0]
    params = build_receiver_params(frame_template, freq_grid, pilot_columns)
    _, _, details = extract_ofdm_symbols_with_ce(
        sim.rx_waveform,
        params,
        return_details=True,
    )
    if details is None:
        raise RuntimeError("Receiver failed to lock onto the frame.")

    mask = details["data_mask"]
    tx = details["tx_symbols"][mask]
    rx = details["freq_symbols"][mask]
    eq = details["eq_symbols"][mask]

    raw_corr, lags = normalized_cross_correlation(tx, rx)
    eq_corr, _ = normalized_cross_correlation(tx, eq)

    torch.save(
        {
            "lags": lags,
            "raw_corr": raw_corr,
            "equalized_corr": eq_corr,
        },
        DATA_PATH,
    )
    print(f"Saved correlation tensors to {DATA_PATH}")

    plt.figure(figsize=(10, 5))
    plt.plot(
        lags.cpu().numpy(),
        torch.abs(raw_corr).detach().cpu().numpy(),
        label="Before CE",
    )
    plt.plot(
        lags.cpu().numpy(),
        torch.abs(eq_corr).detach().cpu().numpy(),
        label="After CE",
    )
    plt.title("ZCZ Tx/Rx Correlation Across Lags")
    plt.xlabel("Lag")
    plt.ylabel("|Correlation|")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200)
    plt.close()
    print(f"Correlation figure saved to {FIG_PATH}")


if __name__ == "__main__":
    main()
