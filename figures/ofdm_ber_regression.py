"""OFDM BER over Watermark channels using the sync-based receiver.

This harness uses the validated receiver chain:

    prepare_frame      -> builds a frame with a Schmidl-Cox/chirp sync preamble
    replay_filter      -> convolves with the Watermark channel sounding (+RMS norm)
    PacketRetriever    -> extracts packets, applies per-sounding Doppler resampling
    extract_ofdm_symbols_with_ce
                       -> sync-correlates to find the frame start, then does
                          per-symbol pilot-interpolated channel estimation + BER

This is the receiver that produces the paper's BER numbers. Bits are generated
deterministically (seeded) so runs are reproducible without the MATLAB .mat file
that the original research script consumed.

Usage:
    source env.sh
    python figures/ofdm_ber_regression.py                  # all channels, BPSK+QPSK, no AWGN
    python figures/ofdm_ber_regression.py --frames 100 --mod qpsk
    python figures/ofdm_ber_regression.py --snr 0,5,10,15,20,25,30 --csv ber.csv  # BER-vs-SNR sweep
    python figures/ofdm_ber_regression.py --snr 0,5,10,15,20,25,30 \
        --fec none,ldpc_r73,ldpc_r33 --csv ber.csv                # add LDPC FEC curves

FEC: LDPC entries use hard-decision DVB-S2 decoding. The receiver's uncoded bit
error rate at each operating point defines a binary symmetric channel (BSC), and
the LDPC decoder is run over that BSC -- soft/LLR decoding is intentionally not
used (the equalized symbols are not reliably calibrated for soft metrics here).
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from python_replicate.channel_replay import load_channel_sounding, replay_filter  # noqa: E402
from python_replicate.frame_preparation import FramePrepConfig, prepare_frame  # noqa: E402
from python_replicate.packet_retrieval import PacketRetriever  # noqa: E402
from python_replicate.receiver_processing import (  # noqa: E402
    ReceiverParams,
    extract_ofdm_symbols_with_ce,
)
from python_replicate.watermark_pipeline import Bookkeeping, WatermarkOutput  # noqa: E402


# Channels whose multiple .mat files are simultaneous hydrophones of one
# recording (vertical array), not independent time soundings. For these we MRC-
# combine the hydrophones; everything else is single-hydrophone SISO. Confirmed
# from the .mat metadata: KAU1/KAU2 inputFiles are AAS*.ch01..ch16 sharing one
# soundingDate; BCH1 is a "SIMO channel sounder" recording with 4 hydrophones.
SIMO_CHANNELS = {"BCH1", "KAU1", "KAU2"}


@dataclass
class SweepStats:
    n_target: int
    n_ok: int
    n_sync_fail: int
    ber_mean: float
    ber_std: float
    ber_min: float
    ber_max: float
    mode: str = "SISO"


def _channel_root() -> Path:
    return Path(os.environ.get("E2E_WAVE_CHANNELS_DIR", "data/channels"))


def _build_fec(fec_arg: str):
    """Parse --fec into (include_none, [(label, codec), ...]).

    Labels match plot_ber_comparison.py's `fec` column: none, ldpc_r73, ldpc_r33.
    """
    include_none = False
    codecs = []
    for tok in fec_arg.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok == "none":
            include_none = True
        elif tok.startswith("ldpc_r"):
            # py_aff3ct is imported lazily inside the codec constructor, so catch the
            # ImportError around construction (not just the module import).
            try:
                from python_replicate.aff3ct_codecs import DVBS2LDPCCodec
                rate = int(tok[len("ldpc_r"):]) / 100.0
                _, k, n = DVBS2LDPCCodec.find_code(rate, frame="short")
                codecs.append((tok, DVBS2LDPCCodec(k=k, n=n)))
            except ImportError as e:
                raise SystemExit(
                    f"--fec {tok} requires py_aff3ct, which is not installed. Build it from "
                    "source (see 'Optional: LDPC FEC via py_aff3ct' in README.md), or drop "
                    "--fec / use --fec none for uncoded BER."
                ) from e
        else:
            raise ValueError(f"unknown --fec entry '{tok}' (use none, ldpc_r73, ldpc_r33)")
    return include_none, codecs


def _coded_ber_bsc(codec, p: float, n_codewords: int, rng) -> float:
    """Hard-decision coded BER. Model the receiver's hard bits as a BSC with
    crossover = uncoded BER p, then run the LDPC hard-decision decoder over random
    codewords. (Soft/LLR decoding is not used: the equalized symbols here are not
    reliably calibrated, so hard-decision is what holds up.)"""
    if not math.isfinite(p):
        return float("nan")
    p = min(max(p, 0.0), 1.0)
    K, N = codec._k, codec._n
    errors = total = 0
    for _ in range(n_codewords):
        info = rng.integers(0, 2, K).astype(np.int32)
        coded = codec.encode(info).astype(np.uint8)
        flips = (rng.random(coded.size) < p).astype(np.uint8)
        decoded = codec.decode(np.bitwise_xor(coded, flips))[:K]
        errors += int(np.count_nonzero(decoded != info))
        total += K
    return errors / total if total else float("nan")


def _summarize(bers: List[float], n_target: int, sync_fail: int, mode: str = "SISO") -> SweepStats:
    if bers:
        arr = np.asarray(bers, dtype=np.float64)
        return SweepStats(n_target, len(bers), sync_fail, float(arr.mean()),
                          float(arr.std()), float(arr.min()), float(arr.max()), mode)
    return SweepStats(n_target, 0, sync_fail, float("nan"), float("nan"),
                      float("nan"), float("nan"), mode)


def _mrc_combine_and_ber(details: List[dict], bits_per_symbol: int) -> Optional[float]:
    """Maximal-ratio combine per-hydrophone synced/equalized symbols, then BER.

    Each ``details`` entry comes from extract_ofdm_symbols_with_ce(return_details)
    for one hydrophone receiving the same transmitted frame: it carries the raw
    frequency-domain symbols Y_n and the channel estimate H_n. MRC forms
    sum(conj(H_n) * Y_n) / sum(|H_n|^2), which is the diversity-optimal combiner.
    """
    ncols = min(d["freq_symbols"].shape[1] for d in details)
    ys = [d["freq_symbols"][:, :ncols] for d in details]
    hs = [d["channel_est"][:, :ncols] for d in details]
    tx = details[0]["tx_symbols"][:, :ncols]
    mask = details[0]["data_mask"][:, :ncols]

    num = sum(torch.conj(h) * y for h, y in zip(hs, ys))
    den = sum((h.real ** 2 + h.imag ** 2) for h in hs) + 1e-12
    eq = num / den

    real_sign = torch.sign(eq.real)
    real_tx = torch.sign(tx.real)
    if bits_per_symbol == 1:
        wrong = real_sign[mask] != real_tx[mask]
        errors = int(wrong.sum().item())
        total = int(wrong.numel())
    else:
        wrong_r = real_sign[mask] != real_tx[mask]
        wrong_i = torch.sign(eq.imag)[mask] != torch.sign(tx.imag)[mask]
        errors = int((wrong_r.sum() + wrong_i.sum()).item())
        total = int(wrong_r.numel() + wrong_i.numel())
    return errors / total if total > 0 else None


def _compute_packets_per_sounding(channel_path: Path, fs_x: float, x_len: int) -> Tuple[int, int, int]:
    ch = load_channel_sounding(channel_path)
    s1, s2 = ch.h.shape
    max_samples = math.floor((s1 - 1) * (1 / ch.fs_t) * fs_x)
    L1 = math.ceil(s2 * (1 / ch.fs_tau) * fs_x)
    L2 = math.ceil(2e-3 * fs_x)
    packet_len = x_len + L1 + 2 * L2
    n_packets_per_sounding = max_samples // packet_len
    return int(n_packets_per_sounding), int(L1), int(L2)


def _simulate_watermark_subset(
    x: torch.Tensor,
    fs_x: float,
    n_bits: int,
    effective_bit_rate: float,
    channel_paths: List[Path],
    n_packets_per_sounding: int,
) -> WatermarkOutput:
    if not channel_paths:
        raise ValueError("channel_paths is empty.")

    first_channel = load_channel_sounding(channel_paths[0])
    s1, s2 = first_channel.h.shape
    max_samples = math.floor((s1 - 1) * (1 / first_channel.fs_t) * fs_x)
    Lx = x.numel()
    L1 = math.ceil(s2 * (1 / first_channel.fs_tau) * fs_x)
    L2 = math.ceil(2e-3 * fs_x)
    packet_len = Lx + L1 + 2 * L2
    if packet_len == 0:
        raise ValueError("Packet length is zero.")
    max_npps = max_samples // packet_len
    if max_npps == 0:
        raise ValueError("Input signal too long for the selected channel.")
    n_packets_per_sounding = int(min(max_npps, max(1, n_packets_per_sounding)))

    zeros_lead = torch.zeros(L2, dtype=torch.float64, device=x.device)
    zeros_tail = torch.zeros(L1 + L2, dtype=torch.float64, device=x.device)
    x_padded = torch.cat([zeros_lead, x.to(torch.float64), zeros_tail])
    signal_train = x_padded.repeat(n_packets_per_sounding)

    starts = torch.arange(n_packets_per_sounding, dtype=torch.int64) * packet_len
    ends = starts + packet_len
    packet_indices = torch.stack([starts, ends], dim=1)

    soundings: List[torch.Tensor] = []
    normalization_factor: Optional[float] = None
    velocities: List[float] = []
    for p in channel_paths:
        channel = load_channel_sounding(p)
        channel.h = channel.h.to(x.device)
        velocities.append(channel.V0)
        y = replay_filter(signal_train, fs_x, channel)
        if normalization_factor is None:
            rms = torch.sqrt(torch.mean(y**2))
            normalization_factor = rms.item() if rms.item() > 0 else 1.0
        y = y / float(normalization_factor)
        soundings.append(y)

    bk = Bookkeeping(
        nPackets=n_packets_per_sounding * len(soundings),
        nPacketsPerSounding=n_packets_per_sounding,
        packet_indices=packet_indices,
        nSoundings=len(soundings),
        nBits=n_bits,
        effectiveBitRate=effective_bit_rate,
        velocities=torch.tensor(velocities, dtype=torch.float64),
        pad_leading=L2,
        pad_trailing=L1 + L2,
        signal_length=Lx,
    )
    return WatermarkOutput(
        soundings=soundings,
        fs=fs_x,
        bookkeeping=bk,
        normalization_factor=float(normalization_factor or 1.0),
    )


def _build_rx_params(frame, bits_per_symbol: int) -> ReceiverParams:
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
        bits_per_symbol=bits_per_symbol,
    )


def sweep_channel(
    channel_name: str,
    n_frames: int,
    modulation: str,
    device: torch.device,
    snr_db: Optional[float] = None,
    seed: int = 0,
) -> SweepStats:
    if modulation.lower() == "qpsk":
        bits_per_symbol, mod_order = 2, 4
    elif modulation.lower() == "bpsk":
        bits_per_symbol, mod_order = 1, 2
    else:
        raise ValueError("modulation must be bpsk or qpsk")

    channel_dir = _channel_root() / channel_name / "mat"
    files = sorted(channel_dir.glob(f"{channel_name}_*.mat"))
    if not files:
        raise FileNotFoundError(f"No channel files found in {channel_dir}")

    # The Watermark IR is a baseband-equivalent response relative to channel.fc,
    # so the transmitted frame must be centered at channel.fc (otherwise the
    # downconversion inside replay_filter shifts the signal off-band and it aliases
    # through the internal fs_tau resample -- this is what broke BCH1 at fc=35 kHz).
    # Oversample enough that the passband carrier stays below Nyquist:
    # fs_x = oversample_q * bandwidth > 2*fc + bandwidth.
    base = FramePrepConfig(modulation_order=mod_order)
    ch_fc = float(load_channel_sounding(files[0]).fc)
    oversample_q = max(base.oversample_q,
                       math.ceil((2 * ch_fc + base.bandwidth_hz) / base.bandwidth_hz))
    prep_cfg = FramePrepConfig(modulation_order=mod_order, fc_hz=ch_fc, oversample_q=oversample_q)

    # Deterministic bits (replaces the research script's MATLAB qpsk_signal_OFDM.mat).
    bits_needed = prep_cfg.num_carriers * prep_cfg.num_ofdm_symbols * bits_per_symbol
    gen = torch.Generator().manual_seed(seed)
    data_bits = torch.randint(0, 2, (bits_needed,), generator=gen, dtype=torch.int64)

    frame = prepare_frame(prep_cfg, data_bits=data_bits)
    rx_params = _build_rx_params(frame, bits_per_symbol=bits_per_symbol)

    max_npps, _, _ = _compute_packets_per_sounding(files[0], frame.fs, frame.passband.numel())
    if max_npps <= 0:
        raise RuntimeError(f"{channel_name}: max nPacketsPerSounding computed as {max_npps}.")

    duration = frame.passband.numel() / frame.fs
    eff_rate = frame.data_bits.numel() / duration

    if channel_name in SIMO_CHANNELS and len(files) > 1:
        # SIMO: files[*] are simultaneous hydrophones. Replay the SAME frames
        # through every hydrophone (one sounding per hydrophone), then for each
        # transmitted frame MRC-combine all hydrophones' synced/equalized symbols.
        n_hydro = len(files)
        npps = min(max_npps, max(1, n_frames))
        wm = _simulate_watermark_subset(
            frame.passband.to(torch.float64).to(device), frame.fs,
            frame.data_bits.numel(), eff_rate, files, n_packets_per_sounding=npps,
        )
        retriever = PacketRetriever(wm)
        bers: List[float] = []
        sync_fail = 0
        for j in range(npps):
            details = []
            for n in range(n_hydro):
                pkt = n * npps + j + 1  # sounding index n == hydrophone, local frame j
                y, _ = retriever.fetch(pkt, snr_db=snr_db)
                y = y.to(device=device, dtype=torch.float64)
                _, _, det = extract_ofdm_symbols_with_ce(y, rx_params, return_details=True)
                if det is not None:
                    details.append(det)
            if not details:
                sync_fail += 1
                continue
            ber = _mrc_combine_and_ber(details, bits_per_symbol)
            if ber is None:
                sync_fail += 1
                continue
            bers.append(ber)
        return _summarize(bers, npps, sync_fail, mode=f"SIMO x{n_hydro}")

    # SISO: files are independent time soundings; run the single-hydrophone
    # receiver on each and average the resulting BER.
    n_soundings_needed = int(math.ceil(n_frames / float(max_npps)))
    n_soundings_needed = max(1, min(n_soundings_needed, len(files)))
    npps = int(math.ceil(n_frames / float(n_soundings_needed)))
    npps = min(npps, max_npps)
    selected = files[:n_soundings_needed]
    wm = _simulate_watermark_subset(
        frame.passband.to(torch.float64).to(device), frame.fs,
        frame.data_bits.numel(), eff_rate, selected, n_packets_per_sounding=npps,
    )
    retriever = PacketRetriever(wm)

    bers = []
    sync_fail = 0
    for pkt in range(1, min(n_frames, wm.bookkeeping.nPackets) + 1):
        y, _ = retriever.fetch(pkt, snr_db=snr_db)
        y = y.to(device=device, dtype=torch.float64)
        _, ber = extract_ofdm_symbols_with_ce(y, rx_params)
        if ber is None:
            sync_fail += 1
            continue
        bers.append(float(ber))
    return _summarize(bers, n_frames, sync_fail, mode="SISO")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=50, help="packets per (channel, modulation)")
    p.add_argument("--mod", nargs="+", default=["bpsk", "qpsk"], choices=["bpsk", "qpsk"])
    p.add_argument("--channels", nargs="+", default=["NOF1", "NCS1", "BCH1", "KAU1", "KAU2"])
    p.add_argument("--snr", type=str, default=None,
                   help="AWGN Eb/N0 in dB: a single value or a comma-separated sweep, "
                        "e.g. --snr 0,5,10,15,20,25,30 (default: no noise)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fec", type=str, default="none",
                   help="comma-separated FEC list: none, ldpc_r73, ldpc_r33. LDPC entries "
                        "use hard-decision DVB-S2 decoding over the measured BSC.")
    p.add_argument("--fec-codewords", type=int, default=20,
                   help="LDPC codewords per point for coded-BER estimation")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional CSV: channel,modulation,fec,snr_db,ber")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    include_none, fec_codecs = _build_fec(args.fec)
    fec_rng = np.random.default_rng(args.seed)

    # No --snr -> a single no-AWGN point. Otherwise sweep the comma-separated list.
    if args.snr is None:
        snr_points: List[Optional[float]] = [None]
    else:
        snr_points = [float(s) for s in args.snr.split(",") if s.strip() != ""]

    rows: List[Tuple[str, str, str, float, float]] = []  # (channel, mod, fec, snr_db, ber)
    for snr in snr_points:
        snr_label = "no AWGN" if snr is None else f"Eb/N0={snr:g} dB"
        print(f"\nBER via proper receiver ({snr_label}, {args.frames} frames/cell):")
        for ch in args.channels:
            for mod in args.mod:
                stats = sweep_channel(ch, args.frames, mod, device, snr_db=snr, seed=args.seed)
                snr_val = 0.0 if snr is None else snr
                p_unc = stats.ber_mean
                if include_none:
                    rows.append((ch, mod.upper(), "none", snr_val, p_unc))
                print(
                    f"{ch:>4} | {mod.upper():4} | {stats.mode:8} | "
                    f"n_ok={stats.n_ok:3d}/{stats.n_target:3d} sync_fail={stats.n_sync_fail:3d} | "
                    f"uncoded BER={p_unc:.4f} (std={stats.ber_std:.4f})"
                )
                for label, codec in fec_codecs:
                    cber = _coded_ber_bsc(codec, p_unc, args.fec_codewords, fec_rng)
                    rows.append((ch, mod.upper(), label, snr_val, cber))
                    print(f"{ch:>4} | {mod.upper():4} | {label:9} | coded BER={cber:.4e}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w") as fh:
            fh.write("channel,modulation,fec,snr_db,ber\n")
            for ch, mod, fec, snr_db, ber in rows:
                fh.write(f"{ch},{mod},{fec},{snr_db:g},{ber:.6f}\n")
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
