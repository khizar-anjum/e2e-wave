from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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


CHANNEL_PATH = ROOT / "input" / "channels" / "NOF1" / "mat" / "NOF1_001.mat"
FRAME_CONFIG = FramePrepConfig()
OFDM_CFG = OFDMConfig()
NUM_TOKENS = 8
WAVEFORM_LEN = 40
OUTPUT_DIR = ROOT / "python_replicate" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH = OUTPUT_DIR / "waveform_alignment.png"
DATA_PATH = OUTPUT_DIR / "waveform_alignment.pt"


def build_receiver_params(
    template,
    data_symbols: torch.Tensor,
    pilot_columns: torch.Tensor,
) -> ReceiverParams:
    return ReceiverParams(
        fs=template.fs,
        fc=template.params["fc"],
        rrc=template.rrc,
        sps=template.params["sps"],
        sync_seq=template.params["sync_seq"],
        train_seq=template.params["train_seq"],
        span=template.params["span"],
        ofdm_len=template.params["ofdm_len"],
        num_fft=template.params["num_fft"],
        cp_length=template.params["cp_length"],
        data_symbols=data_symbols,
        pilot_columns=pilot_columns,
        pilot_value=OFDM_CFG.pilot_value,
        bits_per_symbol=2,
    )


def flatten_data(symbols: torch.Tensor, pilot_columns: torch.Tensor) -> torch.Tensor:
    num_cols = symbols.shape[1]
    mask = torch.ones(num_cols, dtype=torch.bool, device=symbols.device)
    if pilot_columns.numel() > 0:
        mask[pilot_columns.clamp(0, num_cols - 1)] = False
    data_cols = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if data_cols.numel() == 0:
        return torch.zeros(0, dtype=symbols.dtype, device=symbols.device)
    data_matrix = symbols.index_select(1, data_cols)
    return data_matrix.permute(1, 0).contiguous().reshape(-1)


def equalized_sequence(
    pipeline: ChannelSimulationPipeline,
    sim,
    frame_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    template = prepare_frame(pipeline.frame_config)
    pilot_cols = sim.pilot_columns[frame_idx]
    params = build_receiver_params(template, sim.tx_freq_grids[frame_idx], pilot_cols)
    _, _, details = extract_ofdm_symbols_with_ce(
        sim.rx_waveform,
        params,
        return_details=True,
    )
    if details is None:
        raise RuntimeError("Failed to recover frame for equalization.")
    freq_symbols = details["freq_symbols"]
    eq_symbols = details["eq_symbols"]
    pilot_cols = pilot_cols.to(freq_symbols.device)
    raw_flat = flatten_data(freq_symbols, pilot_cols)
    eq_flat = flatten_data(eq_symbols, pilot_cols)
    return raw_flat, eq_flat


def main() -> None:
    torch.manual_seed(0)
    pipeline = ChannelSimulationPipeline(
        channel_path=CHANNEL_PATH,
        frame_config=FRAME_CONFIG,
        ofdm_config=OFDM_CFG,
    )
    bank = ComplexWaveformSystem(
        num_tokens=NUM_TOKENS,
        output_seq_len=WAVEFORM_LEN,
        use_temperature=False,
    )
    tokens = torch.tensor([0], dtype=torch.long)
    sim = pipeline.simulate_video(
        bank,
        [tokens],
        snr_schedule=torch.tensor([0.0]),
        add_awgn=False,
    )
    tx_waveform = sim.tx_waveforms[0][0].detach().cpu()
    recovered = pipeline.recover_data_sequences(sim)[0][0].detach().cpu()
    raw_flat, eq_flat = equalized_sequence(pipeline, sim, frame_idx=0)
    raw_waveform = raw_flat[: WAVEFORM_LEN].detach().cpu()
    eq_waveform = eq_flat[: WAVEFORM_LEN].detach().cpu()

    cos_raw = F.cosine_similarity(
        torch.view_as_real(tx_waveform).reshape(-1),
        torch.view_as_real(raw_waveform).reshape(-1),
        dim=0,
    ).item()
    cos_eq = F.cosine_similarity(
        torch.view_as_real(tx_waveform).reshape(-1),
        torch.view_as_real(eq_waveform).reshape(-1),
        dim=0,
    ).item()
    print(f"Cosine similarity Tx vs raw: {cos_raw:.4f}")
    print(f"Cosine similarity Tx vs equalized: {cos_eq:.4f}")

    torch.save(
        {
            "tx": tx_waveform,
            "rx_raw": recovered,
            "rx_pre_eq": raw_waveform,
            "rx_post_eq": eq_waveform,
            "cos_raw": cos_raw,
            "cos_eq": cos_eq,
        },
        DATA_PATH,
    )

    x = torch.arange(WAVEFORM_LEN)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, tx_waveform.real, label="Tx Real", linewidth=2)
    plt.plot(x, raw_waveform.real, label="Rx Raw Real", linestyle="--")
    plt.plot(x, eq_waveform.real, label="Rx EQ Real", linestyle=":")
    plt.ylabel("Real")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(x, tx_waveform.imag, label="Tx Imag", linewidth=2)
    plt.plot(x, raw_waveform.imag, label="Rx Raw Imag", linestyle="--")
    plt.plot(x, eq_waveform.imag, label="Rx EQ Imag", linestyle=":")
    plt.xlabel("Sample")
    plt.ylabel("Imag")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200)
    plt.close()
    print(f"Saved waveform comparison plot to {FIG_PATH}")


if __name__ == "__main__":
    main()
