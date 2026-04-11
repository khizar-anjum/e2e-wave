import argparse
import csv
import glob
import math
import os
import os.path as osp
import pickle
import random
import warnings
import numpy as np
from datetime import datetime
from typing import List, Tuple
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import io
import torchvision

# Ensure repo root and VideoGPT Implementation dir are on sys.path
repo_root = Path(__file__).resolve().parents[1]
vid_root = Path(__file__).resolve().parent
for p in (repo_root, vid_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pairwise_lambda import LambdaNDCGLoss2
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm

from python_replicate.channel_dataset import ChannelCollection
from python_replicate.frame_preparation import FramePrepConfig
from python_replicate.ofdm_mapper import OFDMConfig
from python_replicate.waveform_bank import ComplexWaveformSystem

# VideoGPT loader
from videogpt.train_utils import load_model as videogpt_load_model


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMM_DEVICE = device
ACCEL_DEVICE = device
NUM_TOKENS = 1024  # will be overwritten by loaded VideoGPT codebook size
FRAME_SEQ_LEN = 16  # will be overridden by checkpoint input_shape if present
WAVEFORM_LEN = 9
BATCH_SIZE_TRAIN = 8
VAL_BATCH_SIZE = 4
LR = 1e-3
USE_LAMBDA_LOSS = False  # if False, use cross-entropy with relevance as soft labels
CE_TARGET_TEMPERATURE = 0.01  # temperature applied to relevance before softmax for CE targets
TOP_K_TARGETS = 5  # number of most relevant tokens used in the target distribution
MIMIC_EVAL_TOKEN_GRID = True
PLOT_MAX_SAMPLES = 8
TRAIN_SNR_RANGE = (0.0, 30.0)
EVAL_SNR_MIN = 0.0
EVAL_SNR_MAX = 30.0
EVAL_SNR_STEP = 5.0
TRAIN_CHANNELS = ["NCS1"]
EVAL_CHANNELS = ["NOF1"]
CHANNEL_BASE = Path(os.environ.get("E2E_WAVE_CHANNELS_DIR", "data/channels"))
TRAIN_RECORDING_MODE = "random"  # "first", "random", or "fixed"
EVAL_RECORDING_MODE = "random"  # "first", "random", or "fixed"
EVAL_RECORDING_SEED = 123
MAX_RECORDINGS_PER_CHANNEL = 0  # 0 to use all recordings
# Preamble/pilot controls
TRAIN_LENGTH = 0  # set to 0 to drop the unused training segment
PILOT_PERIOD = 4  # number of OFDM symbols per pilot block (1 pilot + rest data)
VIDEO_RESOLUTION = 128
VAL_MAX_LENGTH = 200
LOG_DIR_BASE = Path(os.environ.get("E2E_WAVE_RUNS_DIR", "runs/watermark_videogpt"))
VIS_FPS = 4
SIMILARITY_METRIC = "l2" # "dot", "si_l2", or "mlp"
TRAIN_USE_AWGN = True
TRAIN_FLAT_CHANNEL = False
EVAL_USE_AWGN = True
EVAL_FLAT_CHANNEL = False
USE_SYNTHETIC_VAL = False  # if False, use real video dataset for validation


# ---------------------------------------------------------------------
# Dataset utilities (ported from AWGN trainer)
# ---------------------------------------------------------------------
def preprocess(
    video: torch.Tensor,
    resolution: int,
    sequence_length: int,
    in_channels: int = 3,
    sample_every_n_frames: int = 1,
) -> torch.Tensor:
    # Match VideoGPT preprocessing: resize/crop to resolution, center crop for eval, normalize to [-0.5, 0.5]
    if in_channels == 3:
        video = video.permute(0, 3, 1, 2).contiguous().float() / 255.0
    else:
        if video.shape[-1] == 3:
            video = video[:, :, :, 0]
        video = (
            F.one_hot(video.long(), num_classes=in_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )

    # Trim or pad (repeat) to sequence_length
    t, c, h, w = video.shape
    if sequence_length <= t:
        video = video[:sequence_length]
    else:
        reps = (sequence_length + t - 1) // t
        video = video.repeat(reps, 1, 1, 1)
        video = video[:sequence_length]

    if sample_every_n_frames > 1:
        video = video[::sample_every_n_frames]

    # Resize keeping aspect ratio, then center crop
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    t, c, h, w = video.shape
    w_start = max((w - resolution) // 2, 0)
    h_start = max((h - resolution) // 2, 0)
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
    video = video - 0.5  # normalize to [-0.5, 0.5] like VideoGPT dataset.py
    return video


def _parent_dir(path: str) -> str:
    return osp.basename(osp.dirname(path))


# ---------------------------------------------------------------------
# Video dataset (unchanged)
# ---------------------------------------------------------------------
class VideoDataset(Dataset):
    exts = ["avi", "mp4", "webm", "mov"]

    def __init__(
        self,
        data_folder: str,
        sequence_length: int,
        train: bool = False,
        resolution: int = 64,
        sample_every_n_frames: int = 1,
        max_length: int = None,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.sample_every_n_frames = sample_every_n_frames
        self.max_length = max_length
        folder = osp.join(data_folder, "train" if train else "test")
        if not osp.exists(folder):
            raise ValueError(f"Missing folder {folder}.")
        files = sum(
            [glob.glob(osp.join(folder, "**", f"*.{ext}"), recursive=True) for ext in self.exts],
            [],
        )
        if not files:
            raise ValueError(f"No video files found under {folder}.")
        self.classes = sorted(list({ _parent_dir(f) for f in files }))
        warnings.filterwarnings("ignore")
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=8)
            with open(cache_file, "wb") as f:
                pickle.dump(clips.metadata, f)
        else:
            with open(cache_file, "rb") as f:
                metadata = pickle.load(f)
            clips = VideoClips(files, sequence_length, _precomputed_metadata=metadata)
        self._clips = clips
        self.train = train

    def __len__(self) -> int:
        total = self._clips.num_clips()
        if self.max_length is None:
            return total
        return min(total, self.max_length)

    def __getitem__(self, idx: int):
        if self.max_length is not None:
            if self.train or idx >= self._clips.num_clips():
                idx = random.randint(0, self._clips.num_clips() - 1)
            elif not self.train:
                idx = idx % self._clips.num_clips()
        while True:
            try:
                video, _, _, video_idx = self._clips.get_clip(idx)
            except Exception:
                idx = (idx + 1) % self._clips.num_clips()
                continue
            break
        video_path = self._clips.video_paths[video_idx]
        processed = preprocess(
            video,
            self.resolution,
            sequence_length=self.sequence_length,
            sample_every_n_frames=self.sample_every_n_frames,
        )
        return {"video": processed, "path": video_path, "clip_idx": idx}


class EvalVideoDataset(Dataset):
    exts = ["avi", "mp4", "webm", "mov"]

    def __init__(
        self,
        data_folder: str,
        sequence_length: int,
        resolution: int = 64,
        sample_every_n_frames: int = 1,
        max_length: int = None,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.sample_every_n_frames = sample_every_n_frames
        self.max_length = max_length
        files = sum(
            [glob.glob(osp.join(data_folder, "**", f"*.{ext}"), recursive=True) for ext in self.exts],
            [],
        )
        if not files:
            raise ValueError(f"No video files found under {data_folder}.")
        warnings.filterwarnings("ignore")
        cache_file = osp.join(data_folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=8)
            with open(cache_file, "wb") as f:
                pickle.dump(clips.metadata, f)
        else:
            with open(cache_file, "rb") as f:
                metadata = pickle.load(f)
            clips = VideoClips(files, sequence_length, _precomputed_metadata=metadata)
        self._clips = clips

    def __len__(self) -> int:
        total = self._clips.num_clips()
        if self.max_length is None:
            return total
        return min(total, self.max_length)

    def __getitem__(self, idx: int):
        if self.max_length is not None and idx >= self._clips.num_clips():
            idx = idx % self._clips.num_clips()
        while True:
            try:
                video, _, _, video_idx = self._clips.get_clip(idx)
            except Exception:
                idx = (idx + 1) % self._clips.num_clips()
                continue
            break
        video_path = self._clips.video_paths[video_idx]
        processed = preprocess(
            video,
            self.resolution,
            sequence_length=self.sequence_length,
            sample_every_n_frames=self.sample_every_n_frames,
        )
        return {"video": processed, "path": video_path, "clip_idx": idx}


class CodebookIndexDataset(Dataset):
    def __init__(
        self, num_tokens: int = NUM_TOKENS, token_shape: Tuple[int, ...] = ()
    ) -> None:
        self.num_tokens = num_tokens
        self.token_shape = tuple(token_shape) if token_shape else ()

    def __len__(self) -> int:
        # Keep the synthetic dataset length modest so the number of batches matches expectations
        return self.num_tokens

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return random token ids, optionally as a spatial grid.
        shape = self.token_shape if self.token_shape else ()
        return torch.randint(0, self.num_tokens, shape, dtype=torch.long)


# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------
def compute_relevance_matrix(codebook: torch.Tensor) -> torch.Tensor:
    codebook = codebook.float()
    norm = (codebook**2).sum(dim=1, keepdim=True)
    dist_sq = norm + norm.t() - 2 * torch.matmul(codebook, codebook.t())
    dist_sq = torch.clamp(dist_sq, min=0.0)
    d_min = torch.min(dist_sq)
    d_max = torch.max(dist_sq)
    rel = 1.0 - (dist_sq - d_min) / (d_max - d_min + 1e-9)
    return rel


def make_grid_video(real: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    real = (real.detach().cpu() + 1) * 0.5
    recon = (recon.detach().cpu() + 1) * 0.5
    real = real.clamp(0, 1)
    recon = recon.clamp(0, 1)
    combined = torch.cat([real, recon], dim=4)
    combined = combined.permute(0, 2, 1, 3, 4)
    return combined


def make_grid_video_triplet(
    real: torch.Tensor, recon: torch.Tensor, perfect: torch.Tensor
) -> torch.Tensor:
    real = (real.detach().cpu() + 1) * 0.5
    recon = (recon.detach().cpu() + 1) * 0.5
    perfect = (perfect.detach().cpu() + 1) * 0.5
    real = real.clamp(0, 1)
    recon = recon.clamp(0, 1)
    perfect = perfect.clamp(0, 1)
    combined = torch.cat([real, recon, perfect], dim=4)
    combined = combined.permute(0, 2, 1, 3, 4)
    return combined


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class WatermarkBankTrainer(nn.Module):
    def __init__(self, fixed_relevance: torch.Tensor, num_tokens: int) -> None:
        super().__init__()
        self.comm = ComplexWaveformSystem(
            num_tokens=num_tokens,
            output_seq_len=WAVEFORM_LEN,
            use_temperature=True,
            similarity_mode=SIMILARITY_METRIC,
        ).to(COMM_DEVICE)
        self.lambda_loss = LambdaNDCGLoss2(sigma=1.0)
        self.register_buffer("fixed_relevance", fixed_relevance.to(COMM_DEVICE))
        self.frame_config = FramePrepConfig(train_length=TRAIN_LENGTH)
        self.ofdm_config = OFDMConfig(pilot_period=PILOT_PERIOD)
        self.train_channels = ChannelCollection(
            TRAIN_CHANNELS,
            CHANNEL_BASE,
            frame_config=self.frame_config,
            ofdm_config=self.ofdm_config,
            device=COMM_DEVICE,
            recording_mode=TRAIN_RECORDING_MODE,
            max_recordings=MAX_RECORDINGS_PER_CHANNEL,
        )
        self.eval_channels = ChannelCollection(
            EVAL_CHANNELS,
            CHANNEL_BASE,
            frame_config=self.frame_config,
            ofdm_config=self.ofdm_config,
            device=COMM_DEVICE,
            recording_mode=EVAL_RECORDING_MODE,
            recording_seed=EVAL_RECORDING_SEED,
            max_recordings=MAX_RECORDINGS_PER_CHANNEL,
        )

    def _pad_to_len(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() < self.comm.output_seq_len:
            pad = torch.zeros(
                self.comm.output_seq_len - waveform.numel(),
                dtype=waveform.dtype,
                device=waveform.device,
            )
            waveform = torch.cat([waveform, pad], dim=0)
        return waveform[: self.comm.output_seq_len]

    def _rx_feature(
        self,
        channel_collection: ChannelCollection,
        channel_name: str,
        sim_result,
        frame_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pipeline = channel_collection.get_pipeline(channel_name, sim_result)
        rx_sequences = pipeline.recover_data_sequences(sim_result, equalize=True)
        tx_frame = sim_result.tx_waveforms[frame_idx]
        if tx_frame.numel() == 0:
            empty = torch.zeros(0, dtype=torch.complex64, device=COMM_DEVICE)
            return empty, empty
        tx_waveform = tx_frame.reshape(-1, self.comm.output_seq_len)[0]
        rx_frame = rx_sequences[frame_idx]
        if rx_frame.numel() == 0:
            empty = torch.zeros(0, dtype=torch.complex64, device=COMM_DEVICE)
            return empty, tx_waveform.to(COMM_DEVICE)
        rx_waveform = rx_frame.reshape(-1, self.comm.output_seq_len)[0]
        rx_waveform = rx_waveform.to(COMM_DEVICE, dtype=self.comm.get_waveforms().dtype)
        tx_waveform = tx_waveform.to(COMM_DEVICE, dtype=self.comm.get_waveforms().dtype)
        rx_waveform = self.comm.normalize_power(rx_waveform.unsqueeze(0)).squeeze(0)
        tx_waveform = self.comm.normalize_power(tx_waveform.unsqueeze(0)).squeeze(0)
        rx_waveform = self._pad_to_len(rx_waveform)
        tx_waveform = self._pad_to_len(tx_waveform)
        return rx_waveform, tx_waveform

    def _simulate_single(
        self,
        idx: torch.Tensor,
        snr: float,
        channel_collection: ChannelCollection,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tokens = [idx.view(1)]
        channel_name, result = channel_collection.sample(
            self.comm,
            tokens,
            snr_schedule=snr,
            add_awgn=TRAIN_USE_AWGN,
            flat_channel=TRAIN_FLAT_CHANNEL,
        )
        rx_waveform, tx_waveform = self._rx_feature(channel_collection, channel_name, result, 0)
        return rx_waveform, tx_waveform, idx.item()

    def _complex_cos_similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_flat = torch.view_as_real(a).reshape(a.shape[0], -1)
        b_flat = torch.view_as_real(b).reshape(b.shape[0], -1)
        return F.cosine_similarity(a_flat, b_flat, dim=-1)

    def forward_train(
        self, indices: torch.Tensor, detach_reference: bool = False
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
        dict,
    ]:
        snrs = torch.empty(indices.shape[0], dtype=torch.float32, device=COMM_DEVICE).uniform_(
            *TRAIN_SNR_RANGE
        )
        frame_tokens = [idx.reshape(-1).to(COMM_DEVICE, dtype=torch.long) for idx in indices]
        channel_name, result = self.train_channels.sample(
            self.comm,
            frame_tokens,
            snr_schedule=snrs,
            add_awgn=TRAIN_USE_AWGN,
            flat_channel=TRAIN_FLAT_CHANNEL,
        )
        pipeline = self.train_channels.get_pipeline(channel_name, result)
        rx_sequences = pipeline.recover_data_sequences(result, equalize=True)
        channel_meta = {
            "V0": float(getattr(pipeline.channel, "V0", 0.0)),
            "fc_hz": float(getattr(pipeline.channel, "fc", 0.0)),
            "fs_tau": float(getattr(pipeline.channel, "fs_tau", 0.0)),
            "fs_t": float(getattr(pipeline.channel, "fs_t", 0.0)),
            "cp_length": int(pipeline.ofdm_config.cp_length),
            "pilot_period": int(pipeline.ofdm_config.pilot_period),
            "oversample_q": int(pipeline.frame_config.oversample_q),
            "num_carriers": int(pipeline.ofdm_config.num_carriers),
            "num_ofdm_symbols": int(pipeline.frame_config.num_ofdm_symbols),
            "frame_count": len(frame_tokens),
            "snr_mean": float(snrs.mean().item()),
            "snr_min": float(snrs.min().item()),
            "snr_max": float(snrs.max().item()),
        }

        rx_features: List[torch.Tensor] = []
        tx_refs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        mse_terms: List[torch.Tensor] = []
        snr_per_token: List[torch.Tensor] = []

        for frame_idx, (rx_frame, tx_frame) in enumerate(
            zip(rx_sequences, result.tx_waveforms)
        ):
            tx_waveform = tx_frame.reshape(-1, self.comm.output_seq_len)
            if tx_waveform.numel() == 0:
                continue
            rx_waveform = rx_frame.reshape(-1, self.comm.output_seq_len)
            if rx_waveform.numel() == 0:
                continue
            frame_targets = indices[frame_idx].reshape(-1).to(
                device=COMM_DEVICE, dtype=torch.long
            )
            num_tokens = frame_targets.numel()
            if num_tokens == 0:
                continue
            if tx_waveform.shape[0] < num_tokens:
                pad = torch.zeros(
                    num_tokens - tx_waveform.shape[0],
                    self.comm.output_seq_len,
                    dtype=tx_waveform.dtype,
                    device=tx_waveform.device,
                )
                tx_waveform = torch.cat([tx_waveform, pad], dim=0)
            if rx_waveform.shape[0] < num_tokens:
                pad = torch.zeros(
                    num_tokens - rx_waveform.shape[0],
                    self.comm.output_seq_len,
                    dtype=rx_waveform.dtype,
                    device=rx_waveform.device,
                )
                rx_waveform = torch.cat([rx_waveform, pad], dim=0)
            tx_vecs = tx_waveform[:num_tokens].to(
                COMM_DEVICE, dtype=self.comm.get_waveforms().dtype
            )
            rx_vecs = rx_waveform[:num_tokens].to(
                COMM_DEVICE, dtype=self.comm.get_waveforms().dtype
            )
            rx_vecs = self.comm.normalize_power(rx_vecs)
            tx_vecs = self.comm.normalize_power(tx_vecs)
            rx_features.append(rx_vecs)
            tx_refs.append(tx_vecs)
            targets.append(frame_targets)
            per_token_mse = F.mse_loss(
                torch.view_as_real(rx_vecs),
                torch.view_as_real(tx_vecs),
                reduction="none",
            ).mean(dim=-1).mean(dim=-1)
            mse_terms.append(per_token_mse)
            snr_per_token.append(snrs[frame_idx].repeat(num_tokens))

        if not rx_features:
            raise RuntimeError("All training samples failed through the channel.")

        rx_batch = torch.cat(rx_features, dim=0)
        tx_batch = torch.cat(tx_refs, dim=0)
        bank = self.comm.normalize_power(self.comm.get_waveforms())
        ref_bank = bank.detach() if detach_reference else bank
        scores = self.comm.compute_similarity(
            rx_batch, ref_bank, metric=SIMILARITY_METRIC
        )
        target_tensor = torch.cat(targets, dim=0)
        relevance = torch.index_select(self.fixed_relevance, 0, target_tensor)
        counts = torch.full(
            (scores.shape[0],), NUM_TOKENS, dtype=torch.long, device=COMM_DEVICE
        )
        k = min(TOP_K_TARGETS, relevance.shape[1])
        topk_vals, topk_idx = torch.topk(relevance, k=k, dim=1)
        topk_weights = torch.softmax(topk_vals / CE_TARGET_TEMPERATURE, dim=1)
        soft_targets = torch.zeros_like(relevance)
        soft_targets.scatter_(1, topk_idx, topk_weights)
        pred_probs = torch.softmax(scores, dim=1)
        if USE_LAMBDA_LOSS:
            loss = self.lambda_loss(scores, relevance, counts).mean()
        else:
            loss = -(soft_targets * F.log_softmax(scores, dim=1)).sum(dim=1).mean()
        mse_value = torch.cat(mse_terms, dim=0).mean()
        cos_sim = self._complex_cos_similarity(rx_batch, tx_batch).mean()
        pred = torch.argmax(scores, dim=-1)
        token_acc = (pred == target_tensor).float().mean()
        topk_k = min(TOP_K_TARGETS, scores.shape[1])
        topk_idx = torch.topk(scores, k=topk_k, dim=1).indices
        topk_acc = (topk_idx == target_tensor.unsqueeze(1)).any(dim=1).float().mean()
        pred_waves = bank.index_select(0, pred)
        target_waves = bank.index_select(0, target_tensor)
        token_l2 = (pred_waves - target_waves).pow(2).sum(dim=-1).mean().real
        return {
            "loss": loss,
            "mse": mse_value,
            "cos_sim": cos_sim,
            "token_acc": token_acc,
            "topk_acc": topk_acc,
            "token_l2": token_l2,
            "channel_name": channel_name,
            "channel_meta": channel_meta,
            "scores_mean": scores.mean().detach(),
            "scores_std": scores.std().detach(),
            "pred_hist": torch.bincount(pred, minlength=bank.shape[0]).float() / max(pred.numel(), 1),
            "targets": target_tensor.detach(),
            "preds": pred.detach(),
            "soft_targets": soft_targets.detach(),
            "pred_probs": pred_probs.detach(),
            "snrs": torch.cat(snr_per_token, dim=0).detach(),
        }

    def simulate_sequence_features(
        self,
        sequence_tokens: torch.Tensor,
        snr: float,
        channel_collection: ChannelCollection,
        add_awgn: bool = True,
        flat_channel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        frame_tokens = [
            frame.reshape(-1).to(device=COMM_DEVICE, dtype=torch.long)
            for frame in sequence_tokens
        ]
        channel_name, result = channel_collection.sample(
            self.comm,
            frame_tokens,
            snr_schedule=snr,
            add_awgn=add_awgn,
            flat_channel=flat_channel,
        )
        pipeline = channel_collection.get_pipeline(channel_name, result)
        rx_sequences = pipeline.recover_data_sequences(result, equalize=True)
        rx_list: List[torch.Tensor] = []
        counts: List[int] = []
        for frame_idx, tokens in enumerate(frame_tokens):
            needed = tokens.numel()
            counts.append(needed)
            rx_frame = rx_sequences[frame_idx]
            if rx_frame.shape[0] < needed:
                pad = torch.zeros(
                    needed - rx_frame.shape[0],
                    self.comm.output_seq_len,
                    dtype=rx_frame.dtype,
                    device=rx_frame.device,
                )
                rx_frame = torch.cat([rx_frame, pad], dim=0)
            rx_list.append(rx_frame[:needed])
        if rx_list:
            rx_flat = torch.cat(rx_list, dim=0)
        else:
            rx_flat = torch.zeros(
                0,
                self.comm.output_seq_len,
                dtype=torch.cdouble,
                device=COMM_DEVICE,
            )
        rx_flat = rx_flat.to(COMM_DEVICE, dtype=self.comm.get_waveforms().dtype)
        targets = torch.cat(frame_tokens, dim=0).to(torch.long)
        return rx_flat, targets, counts

    @torch.no_grad()
    def simulate_batch_sequences(
        self,
        sequences_tokens: torch.Tensor,
        snrs: torch.Tensor,
        channel_collection: ChannelCollection,
        add_awgn: bool = True,
        flat_channel: bool = False,
    ) -> Tuple[str, List[Tuple[torch.Tensor, torch.Tensor, List[int]]]]:
        """
        Simulate a batch of sequences in one channel call (faster than per-sequence).
        Returns a list per sequence: (rx_flat, targets, counts_per_frame).
        """
        B, T = sequences_tokens.shape[:2]
        frame_tokens: List[torch.Tensor] = []
        snr_per_frame: List[float] = []
        seq_frame_offsets: List[Tuple[int, int]] = []
        cursor = 0
        for b in range(B):
            start = cursor
            for t in range(T):
                ft = sequences_tokens[b, t].reshape(-1).to(device=COMM_DEVICE, dtype=torch.long)
                frame_tokens.append(ft)
                snr_per_frame.append(float(snrs[b].item()))
                cursor += 1
            seq_frame_offsets.append((start, cursor))

        channel_name, result = channel_collection.sample(
            self.comm,
            frame_tokens,
            snr_schedule=torch.tensor(snr_per_frame, device=COMM_DEVICE, dtype=torch.float64),
            add_awgn=add_awgn,
            flat_channel=flat_channel,
        )
        pipeline = channel_collection.get_pipeline(channel_name, result)
        rx_sequences = pipeline.recover_data_sequences(result, equalize=True)

        outputs: List[Tuple[torch.Tensor, torch.Tensor, List[int]]] = []
        frame_idx = 0
        for (start, end) in seq_frame_offsets:
            rx_list: List[torch.Tensor] = []
            counts: List[int] = []
            for idx_frame in range(start, end):
                needed = frame_tokens[idx_frame].numel()
                counts.append(needed)
                rx_frame = rx_sequences[idx_frame]
                if rx_frame.shape[0] < needed:
                    pad = torch.zeros(
                        needed - rx_frame.shape[0],
                        self.comm.output_seq_len,
                        dtype=rx_frame.dtype,
                        device=rx_frame.device,
                    )
                    rx_frame = torch.cat([rx_frame, pad], dim=0)
                rx_list.append(rx_frame[:needed])
                frame_idx += 1
            if rx_list:
                rx_flat = torch.cat(rx_list, dim=0)
            else:
                rx_flat = torch.zeros(
                    0,
                    self.comm.output_seq_len,
                    dtype=torch.cdouble,
                    device=COMM_DEVICE,
                )
            rx_flat = rx_flat.to(COMM_DEVICE, dtype=self.comm.get_waveforms().dtype)
            targets = torch.cat(frame_tokens[start:end], dim=0).to(torch.long)
            outputs.append((rx_flat, targets, counts))
        return channel_name, outputs


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def _snr_points(min_db: float, max_db: float, step_db: float) -> List[float]:
    if step_db <= 0:
        raise ValueError("step_db must be > 0.")
    points = []
    value = min_db
    while value <= max_db + 1e-6:
        points.append(round(float(value), 6))
        value += step_db
    return points


def _psnr_from_mse(mse: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return 10.0 * torch.log10((data_range**2) / (mse + 1e-12))


def _tensor_to_video_uint8(video_bcthw: torch.Tensor) -> torch.Tensor:
    video = (video_bcthw.detach().cpu() + 1) * 0.5
    video = video.clamp(0, 1)
    video = video.squeeze(0).permute(1, 2, 3, 0).contiguous()
    return (video * 255.0).round().to(torch.uint8)


def _write_video(path: Path, video_bcthw: torch.Tensor, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    video = _tensor_to_video_uint8(video_bcthw)
    torchvision.io.write_video(path.as_posix(), video, fps=fps)


def _summary_stats(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(arr.size) if arr.size > 0 else float("nan")
    return mean, std, ci95


def _latent_token_shape(vqvae_model) -> Tuple[int, int, int]:
    if not hasattr(vqvae_model, "latent_shape"):
        raise AttributeError("VQ-VAE model missing latent_shape.")
    latent_shape = vqvae_model.latent_shape
    ds_shape = latent_shape[:-1]
    if len(ds_shape) == 3:
        t_lat, h_lat, w_lat = ds_shape
    elif len(ds_shape) == 2:
        t_lat, h_lat, w_lat = 1, ds_shape[0], ds_shape[1]
    else:
        raise ValueError(f"Unexpected latent_shape: {latent_shape}")
    return int(t_lat), int(h_lat), int(w_lat)


@torch.no_grad()
def evaluate_random_tokens(
    trainer: WatermarkBankTrainer,
    vqvae_model,
    snr_points: List[float],
    num_clips: int,
    batch_size: int,
    csv_writer: csv.DictWriter,
) -> None:
    trainer.eval()
    vqvae_model.eval()
    bank = trainer.comm.normalize_power(trainer.comm.get_waveforms())
    t_lat, h_lat, w_lat = _latent_token_shape(vqvae_model)

    # Extract codebook embeddings for proper L2 distance computation
    codebook_emb = vqvae_model.codebook.embeddings  # (n_codebooks, K, D)
    codebook_flat = codebook_emb.reshape(-1, codebook_emb.shape[-1]).to(COMM_DEVICE)  # (K, D)

    # Compute l2_max for normalization (max pairwise distance in codebook)
    diff = codebook_flat.unsqueeze(0) - codebook_flat.unsqueeze(1)  # (K, K, D)
    pairwise_l2 = torch.norm(diff, p=2, dim=-1)  # (K, K)
    l2_max = pairwise_l2.max().item()

    for snr_db in snr_points:
        l2_vals: List[float] = []
        l2_norm_vals: List[float] = []
        acc_vals: List[float] = []
        remaining = num_clips
        pbar = tqdm(total=num_clips, desc=f"Eval SNR {snr_db:.1f}dB")
        while remaining > 0:
            bsz = min(batch_size, remaining)
            indices = torch.randint(
                0, NUM_TOKENS, (bsz, t_lat, h_lat, w_lat), device=COMM_DEVICE
            )
            snrs = torch.full((bsz,), float(snr_db), device=COMM_DEVICE)
            _, seq_results = trainer.simulate_batch_sequences(
                indices,
                snrs,
                trainer.eval_channels,
                add_awgn=EVAL_USE_AWGN,
                flat_channel=EVAL_FLAT_CHANNEL,
            )
            for rx_flat, targets, _ in seq_results:
                if rx_flat.numel() == 0:
                    continue
                rx_norm = trainer.comm.normalize_power(rx_flat)
                scores = trainer.comm.compute_similarity(
                    rx_norm, bank, metric=SIMILARITY_METRIC
                )
                pred_flat = torch.argmax(scores, dim=-1)
                target_tensor = targets.to(COMM_DEVICE)

                # Compute L2 distance in embedding space (not index space)
                pred_emb = codebook_flat[pred_flat]      # (N, D)
                target_emb = codebook_flat[target_tensor]  # (N, D)
                l2_per_token = torch.norm(pred_emb - target_emb, p=2, dim=-1)  # (N,)
                raw_l2 = l2_per_token.mean().item()
                norm_l2 = raw_l2 / l2_max

                l2_vals.append(raw_l2)
                l2_norm_vals.append(norm_l2)
                acc_vals.append(float((pred_flat == target_tensor).float().mean().item()))
            remaining -= bsz
            pbar.update(bsz)
        pbar.close()

        l2_mean, l2_std, l2_ci = _summary_stats(l2_vals)
        l2_norm_mean, l2_norm_std, l2_norm_ci = _summary_stats(l2_norm_vals)
        acc_mean, acc_std, acc_ci = _summary_stats(acc_vals)
        csv_writer.writerow(
            {
                "snr_db": float(snr_db),
                "n": len(l2_vals),
                "l2_mean": l2_mean,
                "l2_std": l2_std,
                "l2_ci95": l2_ci,
                "l2_norm_mean": l2_norm_mean,
                "l2_norm_std": l2_norm_std,
                "l2_norm_ci95": l2_norm_ci,
                "token_acc_mean": acc_mean,
                "token_acc_std": acc_std,
                "token_acc_ci95": acc_ci,
            }
        )


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def _create_run_dir(base_dir: Path, prefix: str = "") -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{prefix}_{timestamp}" if prefix else timestamp
    candidate = base_dir / name
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = base_dir / f"{name}_{counter:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate waveform bank with random tokens.")
    parser.add_argument("--vqvae_ckpt", required=True, help="VQ-VAE checkpoint path.")
    parser.add_argument("--bank_ckpt", required=True, help="Waveform bank checkpoint path.")
    parser.add_argument("--channel", required=True, help="Channel name for evaluation.")
    parser.add_argument("--waveform_len", type=int, required=True, help="Waveform length for bank.")
    parser.add_argument("--snr_min", type=float, default=EVAL_SNR_MIN)
    parser.add_argument("--snr_max", type=float, default=EVAL_SNR_MAX)
    parser.add_argument("--snr_step", type=float, default=EVAL_SNR_STEP)
    parser.add_argument("--num_clips", type=int, required=True, help="Number of random sequences per SNR.")
    parser.add_argument("--batch_size", type=int, default=VAL_BATCH_SIZE)
    parser.add_argument("--output_csv", default="", help="CSV path for results.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    global TRAIN_CHANNELS, EVAL_CHANNELS, WAVEFORM_LEN
    TRAIN_CHANNELS = [args.channel]
    EVAL_CHANNELS = [args.channel]
    WAVEFORM_LEN = args.waveform_len

    print(f"Loading VideoGPT VQ-VAE checkpoint: {args.vqvae_ckpt}")
    ckpt = torch.load(args.vqvae_ckpt, map_location=ACCEL_DEVICE)
    cond_types = ckpt.get("hp", {}).get("cond_types", None)
    vqvae_model, _ = videogpt_load_model(
        ckpt,
        device=ACCEL_DEVICE,
        freeze_model=True,
        cond_types=cond_types,
        foveated_loss=False,
    )
    vqvae_model = vqvae_model.to(ACCEL_DEVICE)
    vqvae_model.eval()

    emb = vqvae_model.codebook.embeddings  # (n_codebooks, K, D); here n_codebooks=1
    emb_flat = emb.reshape(-1, emb.shape[-1]).to(ACCEL_DEVICE)
    global NUM_TOKENS
    NUM_TOKENS = emb_flat.shape[0]
    relevance = compute_relevance_matrix(emb_flat)

    trainer = WatermarkBankTrainer(relevance, num_tokens=NUM_TOKENS)
    bank_state = torch.load(args.bank_ckpt, map_location=COMM_DEVICE)
    if isinstance(bank_state, dict) and "state_dict" in bank_state:
        bank_state = bank_state["state_dict"]
    trainer.comm.load_state_dict(bank_state)

    snr_points = _snr_points(args.snr_min, args.snr_max, args.snr_step)
    bank_dir = Path(args.bank_ckpt).resolve().parent
    csv_path = Path(args.output_csv) if args.output_csv else bank_dir / f"eval_random_{bank_dir.name}_snr_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "snr_db",
        "n",
        "l2_mean",
        "l2_std",
        "l2_ci95",
        "l2_norm_mean",
        "l2_norm_std",
        "l2_norm_ci95",
        "token_acc_mean",
        "token_acc_std",
        "token_acc_ci95",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        evaluate_random_tokens(
            trainer,
            vqvae_model,
            snr_points,
            args.num_clips,
            args.batch_size,
            writer,
        )

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
