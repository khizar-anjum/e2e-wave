from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from python_replicate.channel_dataset import ChannelSimulationPipeline
from python_replicate.frame_preparation import FramePrepConfig, prepare_frame
from python_replicate.ofdm_mapper import OFDMConfig
from python_replicate.receiver_processing import ReceiverParams, extract_ofdm_symbols_with_ce
from python_replicate.waveform_bank import ComplexWaveformSystem

CHANNEL_PATH = Path(
    os.environ.get("E2E_WAVE_CHANNELS_DIR", "data/channels")
) / "NOF1" / "mat" / "NOF1_001.mat"
FRAME_CONFIG = FramePrepConfig()
OFDM_CFG = OFDMConfig()
RESULT_PATH = Path(
    os.environ.get("E2E_WAVE_OUT_DIR", "output")
) / "channel_estimator_validation.pt"

def make_waveform_bank(kind: str) -> ComplexWaveformSystem:
    kind = kind.upper()
    if kind == "BPSK":
        const = torch.tensor([1 + 0j, -1 + 0j], dtype=torch.complex128)
        bank = ComplexWaveformSystem(num_tokens=len(const), output_seq_len=1, use_temperature=False)
        with torch.no_grad():
            bank.freq_real.zero_()
            bank.freq_imag.zero_()
            bank.freq_real[:, 0] = const.real.to(bank.freq_real.dtype)
            bank.freq_imag[:, 0] = const.imag.to(bank.freq_imag.dtype)
        return bank
    if kind == "QPSK":
        const = torch.tensor(
            [
                (1 + 1j) / math.sqrt(2),
                (1 - 1j) / math.sqrt(2),
                (-1 + 1j) / math.sqrt(2),
                (-1 - 1j) / math.sqrt(2),
            ],
            dtype=torch.complex128,
        )
        bank = ComplexWaveformSystem(num_tokens=len(const), output_seq_len=1, use_temperature=False)
        with torch.no_grad():
            bank.freq_real.zero_()
            bank.freq_imag.zero_()
            bank.freq_real[:, 0] = const.real.to(bank.freq_real.dtype)
            bank.freq_imag[:, 0] = const.imag.to(bank.freq_imag.dtype)
        return bank
    if kind == "DL":
        return ComplexWaveformSystem(num_tokens=8192, output_seq_len=40, use_temperature=True)
    raise ValueError(f"Unknown payload type {kind}")


def build_receiver_params(
    template_frame,
    data_symbols: torch.Tensor,
    pilot_columns: torch.Tensor,
    bits_per_symbol: int,
) -> ReceiverParams:
    return ReceiverParams(
        fs=template_frame.fs,
        fc=template_frame.params["fc"],
        rrc=template_frame.rrc,
        sps=template_frame.params["sps"],
        sync_seq=template_frame.params["sync_seq"],
        train_seq=template_frame.params["train_seq"],
        span=template_frame.params["span"],
        ofdm_len=template_frame.params["ofdm_len"],
        num_fft=template_frame.params["num_fft"],
        cp_length=template_frame.params["cp_length"],
        data_symbols=data_symbols,
        pilot_columns=pilot_columns,
        pilot_value=OFDM_CFG.pilot_value,
        bits_per_symbol=bits_per_symbol,
    )


def run_case(
    pipeline: ChannelSimulationPipeline,
    frame_template,
    kind: str,
    seed: int = 0,
) -> Tuple[torch.Tensor, float, torch.Tensor, Dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    bank = make_waveform_bank(kind)
    data_cols_per_block = pipeline.ofdm_mapper.pilot_period - 1
    num_blocks = max(
        1, math.ceil(FRAME_CONFIG.num_ofdm_symbols / pipeline.ofdm_mapper.pilot_period)
    )
    total_data_columns = num_blocks * data_cols_per_block
    total_samples = total_data_columns * pipeline.ofdm_mapper.num_carriers
    tokens_needed = math.ceil(total_samples / bank.output_seq_len)
    tokens = torch.randint(0, bank.num_tokens, (tokens_needed,), generator=generator)
    sim = pipeline.simulate_video(
        bank,
        [tokens],
        snr_schedule=torch.tensor([0.0]),
        generator=generator,
        add_awgn=False,
    )
    freq_grid = sim.tx_freq_grids[0]
    pilot_columns = sim.pilot_columns[0]
    bits = 1 if kind == "BPSK" else 2
    params = build_receiver_params(frame_template, freq_grid, pilot_columns, bits)
    H_est, ber, details = extract_ofdm_symbols_with_ce(
        sim.rx_waveform, params, return_details=True
    )
    if H_est is None:
        raise RuntimeError(f"Synchronization failed for payload {kind}")
    eq = details["eq_symbols"]
    tx = details["tx_symbols"]
    rx = details["freq_symbols"]
    mask = details["data_mask"]
    errors = torch.abs(eq - tx) ** 2
    mse_samples = errors[mask].reshape(-1).to(torch.float32)
    tx_data = tx[mask].reshape(-1)
    rx_data = rx[mask].reshape(-1)
    eq_data = eq[mask].reshape(-1)
    max_trace = min(2000, tx_data.numel())
    traces = {
        "tx": tx_data[:max_trace].detach().cpu(),
        "rx": rx_data[:max_trace].detach().cpu(),
        "eq": eq_data[:max_trace].detach().cpu(),
    }
    return (
        H_est.detach().cpu(),
        float("nan") if ber is None else float(ber),
        mse_samples.detach().cpu().reshape(-1),
        traces,
    )


def compare_channels(
    a: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[float, float]:
    min_symbols = min(a.shape[1], b.shape[1])
    a_trim = a[:, :min_symbols]
    b_trim = b[:, :min_symbols]
    diff = a_trim - b_trim
    rel = torch.linalg.norm(diff) / torch.linalg.norm(b_trim)
    max_abs = torch.max(torch.abs(diff)).item()
    return rel.item(), max_abs


def main() -> None:
    pipeline = ChannelSimulationPipeline(
        channel_path=CHANNEL_PATH,
        frame_config=FRAME_CONFIG,
        ofdm_config=OFDM_CFG,
    )
    template_frame = prepare_frame(FRAME_CONFIG)

    results: Dict[str, Dict[str, torch.Tensor]] = {}
    estimates: Dict[str, torch.Tensor] = {}
    for kind in ("DL", "BPSK", "QPSK"):
        H_est, ber, mse_samples, traces = run_case(pipeline, template_frame, kind)
        estimates[kind] = H_est
        metrics = {
            "ber": ber if not (kind == "DL") else float("nan"),
            "mse_samples": mse_samples,
            "traces": traces,
        }
        if kind == "DL":
            print(
                f"{kind}: H_est shape={tuple(H_est.shape)}, MSE samples mean={mse_samples.mean().item():.3e}"
            )
        else:
            print(
                f"{kind}: H_est shape={tuple(H_est.shape)}, BER={ber:.3e}"
                if not math.isnan(ber)
                else f"{kind}: H_est shape={tuple(H_est.shape)}, BER=N/A"
            )
        results[kind] = metrics

    comparisons = {
        ("DL", "BPSK"): compare_channels(estimates["DL"], estimates["BPSK"]),
        ("DL", "QPSK"): compare_channels(estimates["DL"], estimates["QPSK"]),
        ("BPSK", "QPSK"): compare_channels(estimates["BPSK"], estimates["QPSK"]),
    }
    for (a, b), (rel, max_abs) in comparisons.items():
        print(f"{a} vs {b}: relative error={rel:.6f}, max |diff|={max_abs:.6f}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"estimates": estimates, "metrics": results, "comparisons": comparisons}, RESULT_PATH)
    print(f"Saved channel estimates for plotting to {RESULT_PATH}")


if __name__ == "__main__":
    main()
