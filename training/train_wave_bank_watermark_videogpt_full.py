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
from torch.utils.tensorboard import SummaryWriter
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
NUM_EPOCHS = 50
VAL_FREQ = 10
TRAIN_SNR_RANGE = (0.0, 30.0)
EVAL_SNR_RANGE = (15, 15.1)
TRAIN_CHANNELS = ["NCS1"]
EVAL_CHANNELS = ["NOF1"]
CHANNEL_BASE = Path("/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/input/channels")
TRAIN_RECORDING_MODE = "random"  # "first", "random", or "fixed"
EVAL_RECORDING_MODE = "random"  # "first", "random", or "fixed"
EVAL_RECORDING_SEED = 123
MAX_RECORDINGS_PER_CHANNEL = 0  # 0 to use all recordings
# Preamble/pilot controls
TRAIN_LENGTH = 0  # set to 0 to drop the unused training segment
PILOT_PERIOD = 4  # number of OFDM symbols per pilot block (1 pilot + rest data)
VIDEO_RESOLUTION = 128
VAL_MAX_LENGTH = 200
LOG_DIR_BASE = Path("/home/cps-tingcong/Documents/GitHub/wave/runs/watermark_videogpt")
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
        processed = preprocess(
            video,
            self.resolution,
            sequence_length=self.sequence_length,
            sample_every_n_frames=self.sample_every_n_frames,
        )
        return {"video": processed}


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
@torch.no_grad()
def run_validation(
    epoch: int,
    trainer: WatermarkBankTrainer,
    vqvae_model,
    val_loader: DataLoader,
    writer: SummaryWriter,
) -> Tuple[float, float]:
    trainer.eval()
    vqvae_model.eval()
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(ACCEL_DEVICE)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(
        ACCEL_DEVICE
    )
    psnr_metric_perfect = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(
        ACCEL_DEVICE
    )
    ssim_metric_perfect = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(
        ACCEL_DEVICE
    )
    bank = trainer.comm.normalize_power(trainer.comm.get_waveforms())
    total_token_acc = 0.0
    total_topk_acc = 0.0
    total_token_l2 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_psnr_perfect = 0.0
    total_ssim_perfect = 0.0
    total_samples = 0
    total_symbol_batches = 0
    logged_video = False

    for batch in tqdm(val_loader, desc="Eval"):
        # If using synthetic val loader, batch is tensor of token ids
        if isinstance(batch, dict):
            video = batch["video"].to(ACCEL_DEVICE)
            video_bcthw = video
            _, encodings = vqvae_model.encode(video_bcthw, no_flatten=True)
            indices = encodings.squeeze(-1)  # (b, t', h', w')
            B = video.shape[0]
        else:
            # Synthetic tokens
            token_batch = batch.to(COMM_DEVICE)
            B = token_batch.shape[0]
            if token_batch.dim() == 1:
                indices = token_batch.view(B, 1, 1, 1)
            else:
                indices = token_batch.unsqueeze(1)
            # Dummy video tensors for metrics (skip PSNR/SSIM)
            video = None
        snrs = torch.empty(B, device=COMM_DEVICE).uniform_(*EVAL_SNR_RANGE)
        frame_shape = indices.shape[2:]

        channel_name, seq_results = trainer.simulate_batch_sequences(
            indices.to(COMM_DEVICE),
            snrs,
            trainer.eval_channels,
            add_awgn=EVAL_USE_AWGN,
            flat_channel=EVAL_FLAT_CHANNEL,
        )

        for b, (rx_flat, targets, counts) in enumerate(seq_results):
            if rx_flat.numel() == 0:
                continue
            rx_norm = trainer.comm.normalize_power(rx_flat)
            scores = trainer.comm.compute_similarity(
                rx_norm, bank, metric=SIMILARITY_METRIC
            )
            pred_flat = torch.argmax(scores, dim=-1)
            target_tensor = targets.to(COMM_DEVICE)
            token_acc = (pred_flat == target_tensor).float().mean()
            topk_k = min(TOP_K_TARGETS, scores.shape[1])
            topk_idx = torch.topk(scores, k=topk_k, dim=1).indices
            topk_acc = (topk_idx == target_tensor.unsqueeze(1)).any(dim=1).float().mean()
            pred_waves = bank.index_select(0, pred_flat)
            target_waves = bank.index_select(0, target_tensor)
            token_l2 = (pred_waves - target_waves).pow(2).sum(dim=-1).mean().real
            total_token_acc += token_acc.item()
            total_topk_acc += topk_acc.item()
            total_token_l2 += token_l2.item()
            total_symbol_batches += 1

            cursor = 0
            predicted_frames: List[torch.Tensor] = []
            for count in counts:
                frame_pred = pred_flat[cursor : cursor + count]
                frame_pred = frame_pred.view(frame_shape)
                predicted_frames.append(frame_pred)
                cursor += count
            if video is not None:
                pred_indices = torch.stack(predicted_frames, dim=0).unsqueeze(0).to(ACCEL_DEVICE)
                recon = vqvae_model.decode(pred_indices.unsqueeze(-1))
                perfect_recon = vqvae_model.decode(indices[b : b + 1].unsqueeze(-1))
                real = video[b : b + 1]

                real_01 = ((real + 1) * 0.5).clamp(0, 1)
                recon_01 = ((recon + 1) * 0.5).clamp(0, 1)
                perfect_01 = ((perfect_recon + 1) * 0.5).clamp(0, 1)
                real_frames = real_01.permute(0, 2, 1, 3, 4).reshape(
                    -1, real_01.shape[1], real_01.shape[3], real_01.shape[4]
                )
                recon_frames = recon_01.permute(0, 2, 1, 3, 4).reshape(
                    -1, recon_01.shape[1], recon_01.shape[3], recon_01.shape[4]
                )
                perfect_frames = perfect_01.permute(0, 2, 1, 3, 4).reshape(
                    -1, perfect_01.shape[1], perfect_01.shape[3], perfect_01.shape[4]
                )
                # Align frame counts in case of mismatch
                min_len = min(
                    real_frames.shape[0],
                    recon_frames.shape[0],
                    perfect_frames.shape[0],
                )
                real_frames = real_frames[:min_len]
                recon_frames = recon_frames[:min_len]
                perfect_frames = perfect_frames[:min_len]
                psnr_val = psnr_metric(recon_frames, real_frames)
                ssim_val = ssim_metric(recon_frames, real_frames)
                psnr_perfect = psnr_metric_perfect(perfect_frames, real_frames)
                ssim_perfect = ssim_metric_perfect(perfect_frames, real_frames)
                total_psnr += psnr_val.item()
                total_ssim += ssim_val.item()
                total_psnr_perfect += psnr_perfect.item()
                total_ssim_perfect += ssim_perfect.item()
                total_samples += 1

                if not logged_video:
                    writer.add_video(
                        "Eval/Reconstruction",
                        make_grid_video_triplet(real, recon, perfect_recon),
                        epoch,
                        fps=VIS_FPS,
                    )
                    logged_video = True

    avg_psnr = total_psnr / max(total_samples, 1)
    avg_ssim = total_ssim / max(total_samples, 1)
    avg_psnr_perfect = total_psnr_perfect / max(total_samples, 1)
    avg_ssim_perfect = total_ssim_perfect / max(total_samples, 1)
    avg_token_acc = total_token_acc / max(total_symbol_batches, 1)
    avg_topk_acc = total_topk_acc / max(total_symbol_batches, 1)
    avg_token_l2 = total_token_l2 / max(total_symbol_batches, 1)
    if total_samples > 0:
        writer.add_scalar("Eval/PSNR", avg_psnr, epoch)
        writer.add_scalar("Eval/SSIM", avg_ssim, epoch)
        writer.add_scalar("Eval/PSNR_Perfect", avg_psnr_perfect, epoch)
        writer.add_scalar("Eval/SSIM_Perfect", avg_ssim_perfect, epoch)
    writer.add_scalar("Eval/TokenAcc", avg_token_acc, epoch)
    writer.add_scalar("Eval/TopKAcc", avg_topk_acc, epoch)
    writer.add_scalar("Eval/TokenL2", avg_token_l2, epoch)
    print(
        f"Epoch {epoch}: Eval PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, "
        f"Perfect PSNR={avg_psnr_perfect:.2f}, Perfect SSIM={avg_ssim_perfect:.4f}, "
        f"TokenAcc={avg_token_acc:.3f}, TopKAcc={avg_topk_acc:.3f}"
    )
    psnr_metric.reset()
    ssim_metric.reset()
    psnr_metric_perfect.reset()
    ssim_metric_perfect.reset()
    return avg_psnr, avg_ssim


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


def main(video_root: str, vqvae_ckpt: str, exp_prefix: str = "") -> None:
    log_dir = _create_run_dir(LOG_DIR_BASE, prefix=exp_prefix)
    writer = SummaryWriter(log_dir.as_posix())

    print(f"Loading VideoGPT VQ-VAE checkpoint: {vqvae_ckpt}")
    ckpt = torch.load(vqvae_ckpt, map_location=ACCEL_DEVICE)
    cond_types = ckpt.get("hp", {}).get("cond_types", None)
    vqvae_model, cfg = videogpt_load_model(
        ckpt,
        device=ACCEL_DEVICE,
        freeze_model=True,
        cond_types=cond_types,
        foveated_loss=False,
    )
    vqvae_model = vqvae_model.to(ACCEL_DEVICE)
    vqvae_model.eval()

    # Adapt sequence length / resolution to the checkpoint config (VideoGPT typically uses 16-frame clips)
    if "input_shape" in ckpt.get("hp", {}):
        try:
            # For this checkpoint, hp['input_shape'] is (T, H, W)
            t_in, h_in, w_in = ckpt["hp"]["input_shape"]
            global FRAME_SEQ_LEN, VIDEO_RESOLUTION
            FRAME_SEQ_LEN = t_in
            VIDEO_RESOLUTION = h_in
            print(f"Using sequence length {FRAME_SEQ_LEN} and resolution {VIDEO_RESOLUTION} from checkpoint.")
        except Exception:
            pass

    emb = vqvae_model.codebook.embeddings  # (n_codebooks, K, D); here n_codebooks=1
    emb_flat = emb.reshape(-1, emb.shape[-1]).to(COMM_DEVICE)
    global NUM_TOKENS
    NUM_TOKENS = emb_flat.shape[0]
    relevance = compute_relevance_matrix(emb_flat)

    trainer = WatermarkBankTrainer(relevance, num_tokens=NUM_TOKENS)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=LR, weight_decay=1e-4)
    token_shape: Tuple[int, ...] = ()
    if MIMIC_EVAL_TOKEN_GRID and hasattr(vqvae_model, "latent_shape"):
        latent_shape = vqvae_model.latent_shape
        if len(latent_shape) >= 4:
            token_shape = (int(latent_shape[1]), int(latent_shape[2]))
        elif len(latent_shape) == 3:
            token_shape = (int(latent_shape[0]), int(latent_shape[1]))
        if token_shape:
            print(f"Using token grid {token_shape[0]}x{token_shape[1]} for training.")
    train_loader = DataLoader(
        CodebookIndexDataset(num_tokens=NUM_TOKENS, token_shape=token_shape),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=0,
    )
    # Eval loader: switch between real video and synthetic token dataset
    
    if USE_SYNTHETIC_VAL:
        val_loader = DataLoader(
            CodebookIndexDataset(num_tokens=NUM_TOKENS, token_shape=token_shape),
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
    else:
        val_loader = DataLoader(
            VideoDataset(
                video_root,
                sequence_length=FRAME_SEQ_LEN,
                train=False,
                resolution=VIDEO_RESOLUTION,
                max_length=VAL_MAX_LENGTH,
            ),
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )

    best_psnr = -float("inf")
    best_ssim = -float("inf")
    step = 0

    def log_pred_plot(
        epoch_idx: int,
        batch_idx: int,
        soft_targets: torch.Tensor,
        pred_probs: torch.Tensor,
        snrs: torch.Tensor,
    ):
        with torch.no_grad():
            targets_np = soft_targets.detach().cpu().numpy()
            preds_np = pred_probs.detach().cpu().numpy()
            snr_np = snrs.detach().cpu().numpy() if snrs is not None else None
            bsz = targets_np.shape[0]
            max_plot = min(bsz, PLOT_MAX_SAMPLES)
            targets_np = targets_np[:max_plot]
            preds_np = preds_np[:max_plot]
            if snr_np is not None:
                snr_np = snr_np[:max_plot]
            bsz = max_plot
            residual_np = preds_np - targets_np
            pred_vmax = float(np.max(preds_np)) if preds_np.size else 1e-6
            target_vmax = float(np.max(targets_np)) if targets_np.size else 1e-6
            residual_vmax = float(np.max(np.abs(residual_np))) if residual_np.size else 1e-6
            residual_vmax = max(residual_vmax, 1e-6)

            fig, axes = plt.subplots(
                3, bsz, figsize=(max(8, 4 * bsz), 6), squeeze=False
            )
            for i in range(bsz):
                axes[0, i].imshow(
                    preds_np[i][None, :],
                    aspect="auto",
                    cmap="inferno",
                    vmin=0,
                    vmax=pred_vmax,
                )
                axes[1, i].imshow(
                    targets_np[i][None, :],
                    aspect="auto",
                    cmap="inferno",
                    vmin=0,
                    vmax=target_vmax,
                )
                axes[2, i].imshow(
                    residual_np[i][None, :],
                    aspect="auto",
                    cmap="coolwarm",
                    vmin=-residual_vmax,
                    vmax=residual_vmax,
                )
                if i == 0:
                    axes[0, i].set_ylabel("Pred")
                    axes[1, i].set_ylabel("Target")
                    axes[2, i].set_ylabel("Residual")
                for row in range(3):
                    axes[row, i].set_yticks([])
                    axes[row, i].set_xticks([])
                title = f"Ep {epoch_idx} B{batch_idx} SNR={snr_np[i]:.1f}dB" if snr_np is not None else f"Ep {epoch_idx} B{batch_idx}"
                axes[0, i].set_title(title)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            img_bytes = buf.getvalue()
            img_tensor = torch.frombuffer(img_bytes, dtype=torch.uint8).clone()  # make writable
            img = torchvision.io.decode_png(img_tensor)
            writer.add_image("Train/TargetPredHeatmap", img, epoch_idx)

    sample_shape = (BATCH_SIZE_TRAIN, *token_shape) if token_shape else (BATCH_SIZE_TRAIN,)
    sample_indices = torch.randint(0, NUM_TOKENS, sample_shape, device=COMM_DEVICE)
    trainer.comm.zero_grad()
    out = trainer.forward_train(sample_indices, detach_reference=True)
    out["loss"].backward()
    grad_norm = trainer.comm.freq_real.grad.norm().item()
    print(f"Gradient norm on waveform bank: {grad_norm:.4f}")

    for epoch in range(NUM_EPOCHS):
        trainer.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_records = []
        for batch_idx, indices in enumerate(pbar):
            optimizer.zero_grad()
            out = trainer.forward_train(indices.to(COMM_DEVICE))
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/WaveformMSE", out["mse"].item(), step)
            writer.add_scalar("Train/CosSim", out["cos_sim"].item(), step)
            writer.add_scalar("Train/TokenAcc", out["token_acc"].item(), step)
            writer.add_scalar("Train/TopKAcc", out["topk_acc"].item(), step)
            writer.add_scalar("Train/TokenL2", out["token_l2"].item(), step)
            writer.add_scalar("Train/ScoresMean", out["scores_mean"].item(), step)
            writer.add_scalar("Train/ScoresStd", out["scores_std"].item(), step)
            writer.add_histogram("Train/PredHist", out["pred_hist"], step)
            if batch_idx == 0:
                # Log per-sample scatter of target vs pred for the first batch of each epoch
                log_pred_plot(epoch, batch_idx, out["soft_targets"], out["pred_probs"], out["snrs"])
            epoch_records.append(
                (
                    step,
                    out["channel_name"],
                    loss.item(),
                    out["channel_meta"]["snr_mean"],
                    out["channel_meta"]["snr_min"],
                    out["channel_meta"]["snr_max"],
                    out["channel_meta"]["V0"],
                    out["channel_meta"]["fc_hz"],
                    out["channel_meta"]["fs_tau"],
                    out["channel_meta"]["fs_t"],
                    out["channel_meta"]["cp_length"],
                    out["channel_meta"]["pilot_period"],
                    out["channel_meta"]["oversample_q"],
                    out["channel_meta"]["num_carriers"],
                    out["channel_meta"]["num_ofdm_symbols"],
                    out["channel_meta"]["frame_count"],
                )
            )
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{out['mse'].item():.4e}",
                cos=f"{out['cos_sim'].item():.3f}",
                acc=f"{out['token_acc'].item():.3f}",
                topk=f"{out['topk_acc'].item():.3f}",
            )
            step += 1

        # write per-epoch loss/channel log as numpy binary
        if epoch_records:
            max_len = max(len(rec[1]) for rec in epoch_records)
            dtype = [
                ("step", np.int64),
                ("channel", f"U{max_len}"),
                ("loss", np.float32),
                ("snr_mean", np.float32),
                ("snr_min", np.float32),
                ("snr_max", np.float32),
                ("V0", np.float32),
                ("fc_hz", np.float32),
                ("fs_tau", np.float32),
                ("fs_t", np.float32),
                ("cp_length", np.int64),
                ("pilot_period", np.int64),
                ("oversample_q", np.int64),
                ("num_carriers", np.int64),
                ("num_ofdm_symbols", np.int64),
                ("frame_count", np.int64),
            ]
            arr = np.array(epoch_records, dtype=dtype)
            log_path = log_dir / f"loss_channels_epoch{epoch:04d}.npy"
            np.save(log_path, arr)

        if (epoch + 1) % VAL_FREQ == 0:
            psnr, ssim = run_validation(epoch, trainer, vqvae_model, val_loader, writer)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(trainer.comm.state_dict(), osp.join(log_dir, "best_psnr_bank.pth"))
                print(f"New best PSNR: {best_psnr:.2f}, model saved.")
            if ssim > best_ssim:
                best_ssim = ssim
                torch.save(trainer.comm.state_dict(), osp.join(log_dir, "best_ssim_bank.pth"))
                print(f"New best SSIM: {best_ssim:.4f}, model saved.")
            torch.save(trainer.comm.state_dict(), osp.join(log_dir, "latest_bank.pth"))
            print("Latest model saved.")

    

    writer.close()


if __name__ == "__main__":
    # Set your defaults here; adjust exp_prefix to tag runs.
    waveform_len_list = [5, 7, 15, 10, 13, 30, 9]
    for ele in waveform_len_list:
        WAVEFORM_LEN = ele
        VIDEO_DIR = "/home/cps-tingcong/Documents/GitHub/wave/ProcessedDataset"
        CKPT = "/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/VideoGPT Implementation/best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar"
        EXP_PREFIX = f"cross-entropy_waveform_len_{WAVEFORM_LEN}_videogpt_4_16_16_1024_train_NCS1_eval_NCS1_temperature_0.01_top_5_video_training_res_128"  # e.g., "l2_sweep1"
        main(VIDEO_DIR, CKPT, EXP_PREFIX)
