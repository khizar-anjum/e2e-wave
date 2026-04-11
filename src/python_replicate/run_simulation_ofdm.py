from __future__ import annotations

import argparse
from pathlib import Path

import torch
from scipy.io import loadmat

from .frame_preparation import FramePrepConfig, prepare_frame
from .packet_retrieval import PacketRetriever
from .receiver_processing import ReceiverParams, extract_ofdm_symbols_with_ce
from .watermark_pipeline import simulate_watermark


def build_receiver_params(frame) -> ReceiverParams:
    return ReceiverParams(
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Python port of run_simulation_OFDM.m")
    p.add_argument("--channel", default="NOF1", help="Channel archive to use")
    p.add_argument("--howmany", default="single", choices=["single", "all"])
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=50.0)
    p.add_argument("--snr-steps", type=int, default=6)
    p.add_argument("--max-packets", type=int, default=1)
    p.add_argument(
        "--matlab-signal",
        default="matlab/qpsk_signal_OFDM.mat",
        help="Use MATLAB-generated OFDM bits for reproducibility.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if Path(args.matlab_signal).is_file():
        mat = loadmat(args.matlab_signal, squeeze_me=True, struct_as_record=False)
        data_bits = torch.tensor(mat["params"].data_bits.astype(int), dtype=torch.int64)
        frame = prepare_frame(FramePrepConfig(), data_bits=data_bits)
    else:
        frame = prepare_frame(FramePrepConfig())

    duration = frame.passband.numel() / frame.fs
    wm_output = simulate_watermark(
        frame.passband,
        frame.fs,
        frame.data_bits.numel(),
        frame.data_bits.numel() / duration,
        Path("input/channels"),
        args.channel,
        howmany=args.howmany,
    )
    retriever = PacketRetriever(wm_output)
    rx_params = build_receiver_params(frame)

    snr_values = torch.linspace(args.snr_min, args.snr_max, args.snr_steps)
    max_packets = min(args.max_packets, wm_output.bookkeeping.nPackets)
    results = []

    for packet_idx in range(1, max_packets + 1):
        for snr in snr_values:
            packet, _ = retriever.fetch(packet_idx, snr_db=float(snr))
            H_est, ber = extract_ofdm_symbols_with_ce(packet, rx_params)
            results.append(
                {
                    "packet": packet_idx,
                    "snr": float(snr),
                    "ber": ber,
                    "H_shape": None if H_est is None else tuple(H_est.shape),
                }
            )
            print(
                f"Packet {packet_idx}/{max_packets} SNR={snr:.1f} dB -> "
                f"{'BER=%.3e' % ber if ber is not None else 'sync failure'}"
            )

    out_dir = Path("python_replicate/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_dir / "ofdm_results.pt")
    print(f"Saved summary to {out_dir/'ofdm_results.pt'}")


if __name__ == "__main__":
    main()
