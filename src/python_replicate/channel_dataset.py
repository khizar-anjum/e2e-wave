from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .channel_replay import ChannelSounding, load_channel_sounding, replay_filter
from .frame_preparation import FramePrepConfig
from .ofdm_mapper import OFDMConfig, OFDMMapper
from .signal_utils import root_raised_cosine, upfirdn_torch
from .waveform_bank import ComplexWaveformSystem


@dataclass
class FrameWrapResult:
    passband: torch.Tensor
    baseband: torch.Tensor
    frame_sample_ranges: List[Tuple[int, int]]


def _sync_sequence(length: int) -> torch.Tensor:
    idx = torch.arange(length, dtype=torch.float64)
    return torch.exp(-1j * math.pi * (idx**2) / length)


def _train_sequence(length: int) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(0, dtype=torch.cdouble)
    return torch.zeros(length, dtype=torch.cdouble)


class FrameAssembler:
    """Wraps baseband data into sync + training + data segments."""

    def __init__(self, config: FramePrepConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device if device is not None else torch.device("cpu")
        self.fs = config.oversample_q * config.bandwidth_hz
        self.rrc = root_raised_cosine(
            config.rolloff, config.span, config.oversample_q
        ).to(torch.float64).to(self.device)

        sync_seq = _sync_sequence(config.sync_length).to(self.device)
        self.sync_signal = upfirdn_torch(
            sync_seq, self.rrc, up=config.oversample_q, down=1
        )
        self.sync_signal = self.sync_signal / torch.sqrt(
            torch.mean(torch.abs(self.sync_signal) ** 2)
        )

        # Schmidl-Cox preamble: two identical halves
        sc_half = torch.ones(config.sc_length // 2, dtype=torch.cdouble, device=self.device)
        sc_seq = torch.cat([sc_half, sc_half], dim=0)
        self.sc_signal = upfirdn_torch(sc_seq, self.rrc, up=self.config.oversample_q, down=1)
        self.sc_signal = self.sc_signal / torch.sqrt(
            torch.mean(torch.abs(self.sc_signal) ** 2)
        )

        train_seq = _train_sequence(config.train_length).to(self.device)
        if train_seq.numel() == 0:
            self.train_signal = torch.zeros(0, dtype=torch.cdouble, device=self.device)
        else:
            self.train_signal = upfirdn_torch(
                train_seq, self.rrc, up=self.config.oversample_q, down=1
            )

    def _shape_segment(self, segment: torch.Tensor) -> torch.Tensor:
        target_device = self.rrc.device
        if segment.numel() == 0:
            return torch.zeros(0, dtype=torch.cdouble, device=target_device)
        return upfirdn_torch(
            segment.to(target_device),
            self.rrc,
            up=self.config.oversample_q,
            down=1,
        )

    def wrap_segments(self, segments: Sequence[torch.Tensor]) -> FrameWrapResult:
        target_device = self.sync_signal.device
        shaped_segments: List[torch.Tensor] = []
        ranges: List[Tuple[int, int]] = []
        cursor = 0
        for seg in segments:
            shaped = self._shape_segment(seg.to(target_device))
            shaped_segments.append(shaped)
            start = cursor
            cursor += shaped.numel()
            ranges.append((start, cursor))

        data_signal = (
            torch.cat(shaped_segments) if shaped_segments else torch.zeros(0, dtype=torch.cdouble)
        )
        baseband = torch.cat(
            [
                self.sync_signal.to(target_device),
                self.sc_signal.to(target_device),
                self.train_signal.to(target_device),
                data_signal.to(target_device),
            ]
        )

        t = torch.arange(baseband.numel(), dtype=torch.float64, device=baseband.device) / self.fs
        passband = torch.real(baseband * torch.exp(1j * 2 * math.pi * self.config.fc_hz * t))

        offset = self.sync_signal.numel() + self.sc_signal.numel() + self.train_signal.numel()
        frame_ranges = [(start + offset, end + offset) for (start, end) in ranges]

        return FrameWrapResult(passband=passband, baseband=data_signal, frame_sample_ranges=frame_ranges)


class FramePayloadBuilder:
    """Maps waveform-bank outputs onto OFDM payloads per frame."""

    def __init__(self, ofdm_mapper: OFDMMapper):
        self.ofdm_mapper = ofdm_mapper

    def build_segments(
        self,
        waveform_bank: ComplexWaveformSystem,
        frame_tokens: Sequence[torch.Tensor],
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        bank = waveform_bank.normalize_power(waveform_bank.get_waveforms())
        waveform_len = waveform_bank.output_seq_len
        segments: List[torch.Tensor] = []
        originals: List[torch.Tensor] = []
        freq_grids: List[torch.Tensor] = []
        pilot_columns: List[torch.Tensor] = []
        for tokens in frame_tokens:
            flat = tokens.reshape(-1).to(dtype=torch.long, device=bank.device)
            if flat.numel() == 0:
                originals.append(
                    torch.zeros(0, waveform_len, dtype=torch.complex128, device=bank.device)
                )
                segments.append(torch.zeros(0, dtype=torch.complex128, device=bank.device))
                freq_grids.append(torch.zeros(0, dtype=torch.complex128, device=bank.device))
                pilot_columns.append(torch.zeros(0, dtype=torch.long, device=bank.device))
                continue
            frame_waveforms = bank.index_select(0, flat)
            originals.append(frame_waveforms)
            mapped, freq, pilots = self.ofdm_mapper.map(
                frame_waveforms.reshape(-1), return_freq=True
            )
            segments.append(mapped)
            freq_grids.append(freq)
            pilot_columns.append(pilots)
        return segments, originals, freq_grids, pilot_columns


def _flatten_data_matrix(data_matrix: torch.Tensor) -> torch.Tensor:
    if data_matrix.numel() == 0:
        return torch.zeros(0, dtype=data_matrix.dtype, device=data_matrix.device)
    return data_matrix.permute(1, 0).contiguous().reshape(-1)


def _resample_complex(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Resample 1D complex tensor by scale factor using linear interpolation."""
    # Convert scale to Python float if it's a tensor
    if hasattr(scale, 'item'):
        scale = float(scale.item())
    else:
        scale = float(scale)
    if scale <= 0:
        return x
    target_len = max(1, int(round(x.numel() * scale)))
    if target_len == x.numel():
        return x
    real_imag = torch.stack([x.real, x.imag], dim=0).unsqueeze(0)  # (1,2,T)
    resampled = F.interpolate(real_imag, size=target_len, mode="linear", align_corners=False)
    resampled = resampled.squeeze(0)
    return torch.complex(resampled[0], resampled[1])


def _select_data_columns(
    freq: torch.Tensor, pilot_cols: torch.Tensor, num_symbols: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_symbols == 0:
        empty = freq.new_zeros(freq.shape[0], 0)
        return empty, torch.zeros(0, dtype=torch.long, device=freq.device)
    mask = torch.ones(num_symbols, dtype=torch.bool, device=freq.device)
    if pilot_cols.numel() > 0:
        mask[pilot_cols.clamp(0, num_symbols - 1)] = False
    data_cols = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if data_cols.numel() == 0:
        empty = freq.new_zeros(freq.shape[0], 0)
        return empty, data_cols
    data_matrix = freq.index_select(1, data_cols)
    return data_matrix, data_cols


def _pll_correct(
    freq: torch.Tensor,
    pilot_cols: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 0.01,
) -> torch.Tensor:
    """Second-order PLL on pilot phases across OFDM symbols.

    freq: (num_carriers, n_symbols)
    pilot_cols: indices of pilot symbols (shape (num_pilots,))
    alpha: proportional gain
    beta: integral gain
    Returns freq corrected per symbol.
    """
    if pilot_cols.numel() == 0:
        return freq
    n_symbols = freq.shape[1]
    # Clamp pilot indices to valid range and work in int
    pilot_cols = pilot_cols.clamp(0, n_symbols - 1).to(torch.long)
    phase = torch.zeros(1, device=freq.device, dtype=torch.float64)
    freq_err = torch.zeros(1, device=freq.device, dtype=torch.float64)
    corrected = []
    for t in range(n_symbols):
        # Select pilots belonging to this symbol only
        mask = pilot_cols == t
        if mask.any():
            pilots = freq[:, mask.nonzero(as_tuple=False).flatten()]
            if pilots.numel() > 0:
                mean_pilot = torch.mean(pilots)
                err = torch.atan2(mean_pilot.imag, mean_pilot.real)
                # Wrap error to [-pi, pi]
                err = (err + torch.pi) % (2 * torch.pi) - torch.pi
                freq_err = freq_err + beta * err
                phase = phase + alpha * err + freq_err
                # Wrap phase too
                phase = (phase + torch.pi) % (2 * torch.pi) - torch.pi
        rot = torch.exp(-1j * phase)
        corrected.append(freq[:, t : t + 1] * rot)
    return torch.cat(corrected, dim=1)


def _interpolate_pilots(
    pilot_pos: torch.Tensor,
    pilot_vals: torch.Tensor,
    num_cols: int,
) -> torch.Tensor:
    device = pilot_pos.device
    output = torch.empty(num_cols, dtype=torch.complex128, device=device)
    if pilot_pos.numel() == 0:
        output.fill_(1 + 0j)
        return output
    if pilot_pos.numel() == 1:
        output.fill_(pilot_vals[0])
        return output
    all_times = torch.arange(num_cols, dtype=pilot_pos.dtype, device=device)
    mask_low = all_times <= pilot_pos[0]
    mask_high = all_times >= pilot_pos[-1]
    mask_mid = (~mask_low) & (~mask_high)
    output[mask_low] = pilot_vals[0]
    output[mask_high] = pilot_vals[-1]
    if mask_mid.any():
        mid_times = all_times[mask_mid]
        idx = torch.searchsorted(pilot_pos, mid_times, right=False)
        idx = torch.clamp(idx, 1, pilot_pos.numel() - 1)
        lo = idx - 1
        hi = idx
        left = pilot_pos[lo]
        right = pilot_pos[hi]
        denom = (right - left).clamp(min=1e-9)
        weight = (mid_times - left) / denom
        interp = pilot_vals[lo] + (pilot_vals[hi] - pilot_vals[lo]) * weight
        output[mask_mid] = interp
    return output


def _estimate_channel_from_pilots(
    freq: torch.Tensor,
    pilot_cols: torch.Tensor,
    pilot_value: complex,
) -> torch.Tensor:
    num_cols = freq.shape[1]
    if num_cols == 0:
        return torch.ones_like(freq)
    pilot_cols = pilot_cols.reshape(-1)
    if pilot_cols.numel() == 0:
        return torch.ones_like(freq)
    pilot_cols = torch.unique(pilot_cols.clamp(0, num_cols - 1))
    pilot_pos = pilot_cols.to(torch.float64)
    pilot_val = torch.as_tensor(pilot_value, dtype=freq.dtype, device=freq.device)
    known = freq.index_select(1, pilot_cols) / pilot_val
    channel_est = torch.empty(freq.shape, dtype=torch.complex128, device=freq.device)
    for row in range(freq.shape[0]):
        pilot_vals = known[row].to(torch.complex128)
        interp = _interpolate_pilots(pilot_pos, pilot_vals, num_cols)
        channel_est[row] = interp
    channel_est[row, pilot_cols] = pilot_vals
    return channel_est.to(freq.dtype)


def _flatten_data_matrix(data_matrix: torch.Tensor) -> torch.Tensor:
    if data_matrix.numel() == 0:
        return torch.zeros(0, dtype=data_matrix.dtype, device=data_matrix.device)
    return data_matrix.permute(1, 0).contiguous().reshape(-1)


def _select_data_columns(
    freq: torch.Tensor, pilot_cols: torch.Tensor, num_symbols: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_symbols == 0:
        empty = freq.new_zeros(freq.shape[0], 0)
        return empty, torch.zeros(0, dtype=torch.long, device=freq.device)
    mask = torch.ones(num_symbols, dtype=torch.bool, device=freq.device)
    if pilot_cols.numel() > 0:
        mask[pilot_cols.clamp(0, num_symbols - 1)] = False
    data_cols = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if data_cols.numel() == 0:
        empty = freq.new_zeros(freq.shape[0], 0)
        return empty, data_cols
    data_matrix = freq.index_select(1, data_cols)
    return data_matrix, data_cols


def _interpolate_pilots(
    pilot_pos: torch.Tensor,
    pilot_vals: torch.Tensor,
    num_cols: int,
) -> torch.Tensor:
    device = pilot_pos.device
    output = torch.empty(num_cols, dtype=torch.complex128, device=device)
    if pilot_pos.numel() == 0:
        output.fill_(1 + 0j)
        return output
    if pilot_pos.numel() == 1:
        output.fill_(pilot_vals[0])
        return output
    all_times = torch.arange(num_cols, dtype=pilot_pos.dtype, device=device)
    mask_low = all_times <= pilot_pos[0]
    mask_high = all_times >= pilot_pos[-1]
    mask_mid = (~mask_low) & (~mask_high)
    output[mask_low] = pilot_vals[0]
    output[mask_high] = pilot_vals[-1]
    if mask_mid.any():
        mid_times = all_times[mask_mid]
        idx = torch.searchsorted(pilot_pos, mid_times, right=False)
        idx = torch.clamp(idx, 1, pilot_pos.numel() - 1)
        lo = idx - 1
        hi = idx
        left = pilot_pos[lo]
        right = pilot_pos[hi]
        denom = (right - left).clamp(min=1e-9)
        weight = (mid_times - left) / denom
        interp = pilot_vals[lo] + (pilot_vals[hi] - pilot_vals[lo]) * weight
        output[mask_mid] = interp
    return output


def _estimate_channel_from_pilots(
    freq: torch.Tensor,
    pilot_cols: torch.Tensor,
    pilot_value: complex,
) -> torch.Tensor:
    num_cols = freq.shape[1]
    if num_cols == 0:
        return torch.ones_like(freq)
    pilot_cols = pilot_cols.reshape(-1)
    if pilot_cols.numel() == 0:
        return torch.ones_like(freq)
    pilot_cols = torch.unique(pilot_cols.clamp(0, num_cols - 1))
    pilot_pos = pilot_cols.to(torch.float64)
    pilot_val = torch.as_tensor(pilot_value, dtype=freq.dtype, device=freq.device)
    known = freq.index_select(1, pilot_cols) / pilot_val
    channel_est = torch.empty(freq.shape, dtype=torch.complex128, device=freq.device)
    for row in range(freq.shape[0]):
        pilot_vals = known[row].to(torch.complex128)
        interp = _interpolate_pilots(pilot_pos, pilot_vals, num_cols)
        channel_est[row] = interp
        channel_est[row, pilot_cols] = pilot_vals
    return channel_est.to(freq.dtype)


def add_awgn_by_frame(
    signal: torch.Tensor,
    frame_ranges: Sequence[Tuple[int, int]],
    snr_schedule: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    noise = torch.zeros_like(signal)
    snr_schedule = snr_schedule.to(signal.dtype)
    for (start, end), snr in zip(frame_ranges, snr_schedule):
        start_i = max(0, int(start))
        end_i = min(len(signal), int(end))
        if end_i <= start_i:
            continue
        seg = signal[start_i:end_i]
        power = torch.mean(seg**2).clamp_min(1e-12)
        noise_power = power / (10 ** (snr / 10))
        noise_std = torch.sqrt(noise_power)
        segment_noise = torch.randn(
            end_i - start_i,
            dtype=signal.dtype,
            device=signal.device,
            generator=generator,
        ) * noise_std
        noise[start_i:end_i] = segment_noise
    return signal + noise


@dataclass
class SimulationResult:
    rx_waveform: torch.Tensor
    clean_waveform: torch.Tensor
    frame_ranges: List[Tuple[int, int]]
    snr_schedule: torch.Tensor
    channel_meta: ChannelSounding
    channel_path: Optional[Path]
    tx_waveforms: List[torch.Tensor]
    tx_freq_grids: List[torch.Tensor]
    pilot_columns: List[torch.Tensor]


class ChannelSimulationPipeline:
    """Runs waveform bank transmissions through recorded channels."""

    def __init__(
        self,
        channel_path: Path,
        frame_config: Optional[FramePrepConfig] = None,
        ofdm_config: Optional[OFDMConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.channel_path = channel_path
        self.frame_config = frame_config or FramePrepConfig()
        self.device = device if device is not None else torch.device("cpu")
        self.frame_assembler = FrameAssembler(self.frame_config, device=self.device)
        self.channel = load_channel_sounding(channel_path)
        self.channel.h = self.channel.h.to(self.device)
        base_config = ofdm_config or OFDMConfig()
        derived_cp = self._derive_cp_length(base_config, self.frame_config, self.channel)
        if derived_cp != base_config.cp_length:
            self.ofdm_config = replace(base_config, cp_length=derived_cp)
        else:
            self.ofdm_config = base_config
        self.ofdm_mapper = OFDMMapper(self.ofdm_config)
        self.payload_builder = FramePayloadBuilder(self.ofdm_mapper)

    @staticmethod
    def _derive_cp_length(
        ofdm_config: OFDMConfig,
        frame_config: FramePrepConfig,
        channel: ChannelSounding,
        safety_margin: float = 1.1,
    ) -> int:
        if channel is None or channel.h.numel() == 0:
            return ofdm_config.cp_length
        taps = channel.h.shape[0]
        if taps <= 1 or channel.fs_tau <= 0:
            return ofdm_config.cp_length
        delay_sec = (taps - 1) / channel.fs_tau
        delay_sec *= safety_margin
        cp_samples = math.ceil(delay_sec * frame_config.bandwidth_hz)
        cp_samples = max(cp_samples, 1)
        cp_samples = min(cp_samples, ofdm_config.num_carriers - 1)
        return cp_samples

    def simulate_video(
        self,
        waveform_bank: ComplexWaveformSystem,
        frame_tokens: Sequence[torch.Tensor],
        snr_schedule: Union[float, Sequence[float]],
        generator: Optional[torch.Generator] = None,
        add_awgn: bool = True,
        flat_channel: bool = False,
    ) -> SimulationResult:
        segments, originals, freq_grids, pilot_columns = self.payload_builder.build_segments(
            waveform_bank, frame_tokens
        )
        wrap = self.frame_assembler.wrap_segments(segments)

        if flat_channel:
            signal = wrap.passband.view(1, 1, -1)
            impulse = torch.ones(1, 1, 1, dtype=signal.dtype, device=signal.device)
            channel_out = F.conv1d(signal, impulse).view(-1)
        else:
            channel_out = replay_filter(
                wrap.passband,
                self.frame_config.oversample_q * self.frame_config.bandwidth_hz,
                self.channel,
            )

        snr_tensor = torch.as_tensor(
            snr_schedule, dtype=torch.float64, device=channel_out.device
        ).reshape(-1)
        n_frames = len(wrap.frame_sample_ranges)
        if snr_tensor.numel() == 1 and n_frames > 1:
            snr_tensor = snr_tensor.repeat(n_frames)
        if snr_tensor.numel() != n_frames:
            raise ValueError(
                f"SNR schedule length ({snr_tensor.numel()}) must equal number of frames ({n_frames}) "
                "or be a scalar for uniform SNR."
            )
        if add_awgn:
            noisy = add_awgn_by_frame(
                channel_out, wrap.frame_sample_ranges, snr_tensor, generator=generator
            )
        else:
            noisy = channel_out

        return SimulationResult(
            rx_waveform=noisy,
            clean_waveform=channel_out,
            frame_ranges=wrap.frame_sample_ranges,
            snr_schedule=snr_tensor,
            channel_meta=self.channel,
            channel_path=self.channel_path,
            tx_waveforms=originals,
            tx_freq_grids=freq_grids,
            pilot_columns=pilot_columns,
        )

    def recover_data_sequences(
        self, result: SimulationResult, equalize: bool = False
    ) -> List[torch.Tensor]:
        fs = self.frame_config.oversample_q * self.frame_config.bandwidth_hz
        t = (
            torch.arange(result.rx_waveform.numel(), device=result.rx_waveform.device, dtype=torch.float64)
            / fs
        )
        baseband = torch.complex(
            result.rx_waveform.to(torch.float64),
            torch.zeros_like(result.rx_waveform, dtype=torch.float64),
        ) * torch.exp(-1j * 2 * math.pi * self.frame_config.fc_hz * t)

        # Schmidl-Cox coarse Doppler/CFO using repeated preamble -> resample to undo scaling
        sc_len = self.frame_assembler.sc_signal.numel()
        frame_ranges_adj = result.frame_ranges
        if sc_len > 0 and baseband.numel() >= self.frame_assembler.sync_signal.numel() + sc_len:
            start = self.frame_assembler.sync_signal.numel()
            sc_segment = baseband[start : start + sc_len]
            half = sc_len // 2
            if half > 0:
                first = sc_segment[:half]
                second = sc_segment[half : 2 * half]
                corr = torch.sum(second * torch.conj(first))
                phase = torch.atan2(corr.imag, corr.real)
                per_sample = phase / max(half, 1)
                fd = per_sample * fs / (2 * math.pi)  # Hz
                scale = 1.0 + (fd / max(self.frame_config.fc_hz, 1e-9))
                scale = float(torch.clamp(torch.tensor(scale, device=baseband.device), 0.5, 1.5).item())
                if abs(scale - 1.0) > 1e-4:
                    baseband = _resample_complex(baseband, scale)
                    frame_ranges_adj = [
                        (int(round(s * scale)), int(round(e * scale))) for (s, e) in result.frame_ranges
                    ]
        recovered: List[torch.Tensor] = []
        for frame_idx, (start, end) in enumerate(frame_ranges_adj):
            expected = result.tx_waveforms[frame_idx]
            num_tokens = expected.shape[0]
            waveform_len = expected.shape[1] if num_tokens > 0 else 0

            segment = baseband[start:end]
            matched = upfirdn_torch(segment, self.frame_assembler.rrc, up=1, down=1)
            downsampled = matched[:: self.frame_config.oversample_q]
            span = self.frame_config.span
            if downsampled.numel() <= 2 * span or num_tokens == 0:
                recovered.append(torch.zeros_like(expected))
                continue
            downsampled = downsampled[span:-span]
            sym_len = self.ofdm_config.num_carriers + self.ofdm_config.cp_length
            usable = (downsampled.numel() // sym_len) * sym_len
            if usable == 0:
                recovered.append(torch.zeros_like(expected))
                continue
            symbols = downsampled[:usable].view(-1, sym_len)
            without_cp = symbols[:, self.ofdm_config.cp_length :]
            freq = torch.fft.fft(without_cp, dim=1) / self.ofdm_config.num_carriers
            n_symbols = freq.shape[0]
            freq = freq.t()  # shape (num_carriers, n_symbols)
            pilot_cols = result.pilot_columns[frame_idx].to(freq.device)
            freq_used = freq
            if equalize:
                freq = _pll_correct(freq, pilot_cols)
                channel_est = _estimate_channel_from_pilots(
                    freq, pilot_cols, self.ofdm_config.pilot_value
                )
                freq_used = freq / (channel_est + 1e-9)
            data_matrix, _ = _select_data_columns(freq_used, pilot_cols, n_symbols)
            if data_matrix.numel() == 0:
                recovered.append(torch.zeros_like(expected))
                continue
            data = _flatten_data_matrix(data_matrix)
            needed = num_tokens * waveform_len
            if data.numel() < needed:
                pad = torch.zeros(needed - data.numel(), dtype=data.dtype, device=data.device)
                data = torch.cat([data, pad], dim=0)
            recovered.append(data[:needed].view(num_tokens, waveform_len))
        return recovered


def estimate_required_samples(
    resolution: int,
    sequence_length: int,
    waveform_len: int,
    patch_size: int = 16,
    oversample: int = 8,
) -> int:
    tokens_per_frame = (resolution // patch_size) ** 2
    shaped_per_token = waveform_len * oversample
    per_frame = tokens_per_frame * shaped_per_token
    sync_train_margin = int(6000)
    return per_frame * sequence_length + sync_train_margin


class _LazyPipelineDict(dict):
    """Dict that lazily loads pipelines on first access."""

    def __init__(self, collection: "ChannelCollection"):
        super().__init__()
        self._collection = collection

    def __getitem__(self, channel_name: str):
        pipeline = super().get(channel_name)
        if pipeline is None and channel_name in self._collection._paths:
            pipeline = self._collection.get_pipeline(channel_name)
        return pipeline


class ChannelCollection:
    """Maintains separate channel pipelines for train/eval splits."""

    def __init__(
        self,
        channel_names: Sequence[str],
        base_dir: Path,
        frame_config: Optional[FramePrepConfig] = None,
        ofdm_config: Optional[OFDMConfig] = None,
        device: Optional[torch.device] = None,
        recording_mode: str = "first",
        recording_seed: Optional[int] = None,
        max_recordings: int = 0,
    ) -> None:
        self.frame_config = frame_config
        self.ofdm_config = ofdm_config
        self.device = device
        self.recording_mode = recording_mode
        self._rng = random.Random(recording_seed) if recording_seed is not None else random
        self._paths = {}
        self._fixed_paths = {}
        self._pipeline_cache = {}
        self.pipelines = _LazyPipelineDict(self)
        for name in channel_names:
            mat_dir = base_dir / name / "mat"
            paths = sorted(mat_dir.glob(f"{name}_*.mat"))
            if not paths:
                raise FileNotFoundError(f"No channel files found in {mat_dir}")
            if max_recordings and max_recordings > 0:
                paths = paths[:max_recordings]
            if recording_mode == "first":
                paths = paths[:1]
            self._paths[name] = paths
            if recording_mode == "fixed":
                idx = self._rng.randrange(len(paths))
                self._fixed_paths[name] = paths[idx]
        self.names = list(channel_names)

    def _select_path(self, channel_name: str) -> Path:
        paths = self._paths[channel_name]
        if self.recording_mode == "random":
            if hasattr(self._rng, "choice"):
                return self._rng.choice(paths)
            return paths[self._rng.randrange(len(paths))]
        if self.recording_mode == "fixed":
            return self._fixed_paths.get(channel_name, paths[0])
        return paths[0]

    def get_pipeline(
        self, channel_name: str, result: Optional[SimulationResult] = None
    ) -> ChannelSimulationPipeline:
        if channel_name not in self._paths:
            raise KeyError(f"Channel {channel_name} not in collection.")
        channel_path = None
        if result is not None and getattr(result, "channel_path", None) is not None:
            channel_path = Path(result.channel_path)
        if channel_path is None:
            channel_path = self._select_path(channel_name)
        key = (channel_name, channel_path)
        pipeline = self._pipeline_cache.get(key)
        if pipeline is None:
            pipeline = ChannelSimulationPipeline(
                channel_path,
                frame_config=self.frame_config,
                ofdm_config=self.ofdm_config,
                device=self.device,
            )
            self._pipeline_cache[key] = pipeline
            # Set default pipeline for backwards compatibility
            if self.pipelines.get(channel_name) is None:
                self.pipelines[channel_name] = pipeline
        return pipeline

    def simulate(
        self,
        channel_name: str,
        waveform_bank: ComplexWaveformSystem,
        frame_tokens: Sequence[torch.Tensor],
        snr_schedule: Union[float, Sequence[float]],
        generator: Optional[torch.Generator] = None,
        add_awgn: bool = True,
        flat_channel: bool = False,
    ) -> SimulationResult:
        pipeline = self.get_pipeline(channel_name)
        return pipeline.simulate_video(
            waveform_bank,
            frame_tokens,
            snr_schedule,
            generator=generator,
            add_awgn=add_awgn,
            flat_channel=flat_channel,
        )

    def sample(
        self,
        waveform_bank: ComplexWaveformSystem,
        frame_tokens: Sequence[torch.Tensor],
        snr_schedule: Union[float, Sequence[float]],
        generator: Optional[torch.Generator] = None,
        add_awgn: bool = True,
        flat_channel: bool = False,
    ) -> Tuple[str, SimulationResult]:
        name = random.choice(self.names)
        result = self.simulate(
            name,
            waveform_bank,
            frame_tokens,
            snr_schedule,
            generator,
            add_awgn=add_awgn,
            flat_channel=flat_channel,
        )
        return name, result
