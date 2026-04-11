from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn


class ComplexWaveformSystem(nn.Module):
    """Trainable complex waveform bank identical to train_wave_bank_awgn."""

    def __init__(
        self,
        num_tokens: int = 8192,
        output_seq_len: int = 128,
        use_temperature: bool = True,
        similarity_mode: str = "dot",
        mlp_hidden: int = 256,
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.output_seq_len = output_seq_len
        self.use_temperature = use_temperature
        self.similarity_mode = similarity_mode

        init_scale = 0.02
        self.freq_real = nn.Parameter(
            torch.randn(num_tokens, output_seq_len) * init_scale
        )
        self.freq_imag = nn.Parameter(
            torch.randn(num_tokens, output_seq_len) * init_scale
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

        if similarity_mode == "mlp":
            in_dim = output_seq_len * 4
            layers = []
            hidden = mlp_hidden
            prev_dim = in_dim
            depth = max(1, mlp_layers)
            for layer in range(depth - 1):
                layers.append(nn.Linear(prev_dim, hidden))
                layers.append(nn.GELU())
                prev_dim = hidden
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden, 1))
            self.sim_net = nn.Sequential(*layers)
        else:
            self.sim_net = None

    def get_waveforms(self) -> torch.Tensor:
        freq_domain = torch.complex(self.freq_real, self.freq_imag)
        time_domain = torch.fft.ifft(freq_domain, dim=-1, norm="ortho")
        return time_domain

    @staticmethod
    def normalize_power(signal: torch.Tensor) -> torch.Tensor:
        # Use squared magnitude to avoid torch.abs in similarity paths.
        mag_sq = signal.real**2 + signal.imag**2
        norm = torch.sqrt(torch.sum(mag_sq, dim=-1, keepdim=True) + 1e-8)
        return signal / norm

    @staticmethod
    def complex_awgn_channel(
        signal: torch.Tensor,
        snr_db: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        sig_power = torch.mean(torch.abs(signal) ** 2, dim=-1, keepdim=True)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise_std = torch.sqrt(noise_power / 2.0)
        noise_real = torch.randn_like(signal.real, generator=generator) * noise_std
        noise_imag = torch.randn_like(signal.imag, generator=generator) * noise_std
        noise = torch.complex(noise_real, noise_imag)
        return signal + noise

    def compute_similarity(
        self,
        rx_signal: torch.Tensor,
        bank: torch.Tensor,
        metric: Optional[str] = None,
    ) -> torch.Tensor:
        mode = metric or self.similarity_mode
        eps = 1e-9
        if mode == "dot":
            bank_conj = torch.conj(bank)
            complex_score = torch.matmul(rx_signal, bank_conj.t())
            sim = complex_score.real**2 + complex_score.imag**2
        elif mode == "cosine":
            rx_norm = rx_signal / (
                torch.linalg.vector_norm(rx_signal, dim=-1, keepdim=True) + 1e-8
            )
            bank_norm = bank / (
                torch.linalg.vector_norm(bank, dim=-1, keepdim=True) + 1e-8
            )
            bank_conj = torch.conj(bank_norm)
            complex_score = torch.matmul(rx_norm, bank_conj.t())
            sim = complex_score.real**2 + complex_score.imag**2
        elif mode == "l2":
            # Negative squared L2 distance (larger is better)
            rx_norm_sq = (rx_signal.real**2 + rx_signal.imag**2).sum(dim=-1, keepdim=True)
            bank_norm_sq = (bank.real**2 + bank.imag**2).sum(dim=-1)  # (N,)
            cross = torch.matmul(rx_signal, torch.conj(bank).t()).real  # (B, N)
            dist_sq = rx_norm_sq + bank_norm_sq.unsqueeze(0) - 2.0 * cross
            sim = -dist_sq
        elif mode == "si_l2":
            # Scale-invariant: residual energy after optimal complex gain projection
            rx_norm_sq = (rx_signal.real**2 + rx_signal.imag**2).sum(dim=-1, keepdim=True)  # (B,1)
            bank_norm_sq = (bank.real**2 + bank.imag**2).sum(dim=-1) + eps  # (N,)
            dot = torch.matmul(rx_signal, torch.conj(bank).t())
            dot_mag_sq = dot.real**2 + dot.imag**2  # (B, N)
            resid = rx_norm_sq + (-dot_mag_sq / bank_norm_sq.unsqueeze(0))
            sim = -resid
        elif mode == "si_dot":
            # Coherence: |<rx, bank>|^2 / (||rx||^2 ||bank||^2), in [0, 1]
            rx_norm_sq = (rx_signal.real**2 + rx_signal.imag**2).sum(dim=-1, keepdim=True) + eps
            bank_norm_sq = (bank.real**2 + bank.imag**2).sum(dim=-1) + eps
            dot = torch.matmul(rx_signal, torch.conj(bank).t())
            dot_mag_sq = dot.real**2 + dot.imag**2
            sim = dot_mag_sq / (rx_norm_sq * bank_norm_sq.unsqueeze(0))
        elif mode == "mlp":
            if self.sim_net is None:
                raise RuntimeError("MLP similarity requested but sim_net not initialized.")
            sim = self._mlp_similarity(rx_signal, bank)
        else:
            raise ValueError(f"Unsupported metric {mode}")
        if self.use_temperature:
            sim = sim / self.temperature
        return sim

    def _mlp_similarity(
        self, rx_signal: torch.Tensor, bank: torch.Tensor, chunk_size: int = 1024
    ) -> torch.Tensor:
        batch = rx_signal.shape[0]
        num_tokens = bank.shape[0]
        scores = []
        for start in range(0, num_tokens, chunk_size):
            end = min(num_tokens, start + chunk_size)
            bank_chunk = bank[start:end]  # (chunk, L)
            rx_expand = rx_signal.unsqueeze(1).expand(-1, end - start, -1)
            bank_expand = bank_chunk.unsqueeze(0).expand(batch, -1, -1)
            features = torch.cat(
                [
                    rx_expand.real,
                    rx_expand.imag,
                    bank_expand.real,
                    bank_expand.imag,
                ],
                dim=-1,
            )
            chunk_score = self.sim_net(features).squeeze(-1)
            scores.append(chunk_score)
        return torch.cat(scores, dim=1)
