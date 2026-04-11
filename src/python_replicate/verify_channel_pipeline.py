from __future__ import annotations

from pathlib import Path

import torch

from .channel_dataset import ChannelSimulationPipeline
from .frame_preparation import FramePrepConfig
from .waveform_bank import ComplexWaveformSystem


def main() -> None:
    bank = ComplexWaveformSystem(num_tokens=64, output_seq_len=40)
    pipeline = ChannelSimulationPipeline(
        channel_path=Path("input/channels/NOF1/mat/NOF1_001.mat"),
        frame_config=FramePrepConfig(),
    )

    rng = torch.Generator().manual_seed(0)
    frame_tokens = [
        torch.randint(0, 64, (8,), generator=rng),
        torch.randint(0, 64, (8,), generator=rng),
    ]
    snr_schedule = 20.0

    result = pipeline.simulate_video(bank, frame_tokens, snr_schedule, generator=rng)

    loss = result.rx_waveform.pow(2).mean()
    loss.backward()

    grad_norm = bank.freq_real.grad.norm().item()
    print(f"Gradient norm on waveform bank: {grad_norm:.6f}")


if __name__ == "__main__":
    main()
