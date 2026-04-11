from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from python_replicate.channel_dataset import ChannelSimulationPipeline
from python_replicate.ofdm_mapper import OFDMConfig
from python_replicate.waveform_bank import ComplexWaveformSystem

CHANNEL_FILE = Path("input/channels/NOF1/mat/NOF1_001.mat")
FRAME_CONFIG = None  # Lazy import to avoid circular dependencies
OFDM_CONFIG = OFDMConfig()


def _load_frame_config():
    global FRAME_CONFIG
    if FRAME_CONFIG is None:
        from python_replicate.frame_preparation import FramePrepConfig

        FRAME_CONFIG = FramePrepConfig()
    return FRAME_CONFIG


def build_constellation_bank(name: str) -> ComplexWaveformSystem:
    name = name.upper()
    if name == "BPSK":
        symbols = torch.tensor([1 + 0j, -1 + 0j], dtype=torch.complex128)
    elif name == "QPSK":
        symbols = torch.tensor(
            [
                (1 + 1j) / math.sqrt(2),
                (1 - 1j) / math.sqrt(2),
                (-1 + 1j) / math.sqrt(2),
                (-1 - 1j) / math.sqrt(2),
            ],
            dtype=torch.complex128,
        )
    else:
        raise ValueError("Supported constellations: BPSK, QPSK")

    bank = ComplexWaveformSystem(num_tokens=symbols.numel(), output_seq_len=1, use_temperature=False)
    with torch.no_grad():
        bank.freq_real.zero_()
        bank.freq_imag.zero_()
        bank.freq_real[:, 0] = symbols.real.to(bank.freq_real.dtype)
        bank.freq_imag[:, 0] = symbols.imag.to(bank.freq_imag.dtype)
    return bank


def build_training_bank() -> ComplexWaveformSystem:
    torch.manual_seed(0)
    return ComplexWaveformSystem(num_tokens=8192, output_seq_len=40, use_temperature=True)


def estimate_channel_response(
    bank: ComplexWaveformSystem,
    num_data_slots: int,
    seed: int,
    deterministic: bool = False,
) -> torch.Tensor:
    frame_config = _load_frame_config()
    pipeline = ChannelSimulationPipeline(
        channel_path=CHANNEL_FILE, frame_config=frame_config, ofdm_config=OFDM_CONFIG
    )
    waveform_len = bank.output_seq_len
    total_tokens = math.ceil(num_data_slots / waveform_len)
    total_samples = total_tokens * waveform_len
    generator = torch.Generator().manual_seed(seed)
    if deterministic:
        base = torch.arange(bank.num_tokens, dtype=torch.long)
        repeats = math.ceil(total_tokens / base.numel())
        tokens = base.repeat(repeats)[:total_tokens]
    else:
        tokens = torch.randint(0, bank.num_tokens, (total_tokens,), generator=generator)
    sim = pipeline.simulate_video(
        bank,
        [tokens],
        snr_schedule=torch.tensor([0.0]),
        generator=generator,
        add_awgn=False,
    )
    rx_sequences = pipeline.recover_data_sequences(sim)[0].reshape(-1)
    tx_sequences = sim.tx_waveforms[0].reshape(-1)
    usable = min(rx_sequences.numel(), tx_sequences.numel(), num_data_slots)
    if usable < num_data_slots:
        raise RuntimeError("Recovered fewer data symbols than expected.")
    per_column = pipeline.ofdm_mapper.num_carriers
    n_columns = num_data_slots // per_column
    ratios = (rx_sequences[:num_data_slots] / tx_sequences[:num_data_slots]).reshape(
        n_columns, per_column
    )
    return ratios


def main() -> None:
    frame_config = _load_frame_config()
    data_cols_per_block = OFDM_CONFIG.pilot_period - 1
    num_blocks = max(
        1, math.ceil(frame_config.num_ofdm_symbols / OFDM_CONFIG.pilot_period)
    )
    num_data_slots = num_blocks * data_cols_per_block * OFDM_CONFIG.num_carriers
    estimates: Dict[str, torch.Tensor] = {}
    configs = [
        ("DL", build_training_bank, 0, False),
        ("BPSK", lambda: build_constellation_bank("BPSK"), 0, True),
        ("QPSK", lambda: build_constellation_bank("QPSK"), 0, True),
    ]
    for label, builder, seed, deterministic in configs:
        bank = builder()
        estimates[label] = estimate_channel_response(
            bank, num_data_slots, seed, deterministic=deterministic
        )

    def compare(a: str, b: str) -> None:
        min_symbols = min(estimates[a].shape[0], estimates[b].shape[0])
        diff = estimates[a][:min_symbols] - estimates[b][:min_symbols]
        rel = torch.linalg.norm(diff) / torch.linalg.norm(estimates[b][:min_symbols])
        max_abs = torch.max(torch.abs(diff)).item()
        print(f"{a} vs {b}: relative error={rel.item():.6f}, max |diff|={max_abs:.6f}")

    compare("DL", "BPSK")
    compare("DL", "QPSK")
    compare("BPSK", "QPSK")


if __name__ == "__main__":
    main()
