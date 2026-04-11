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
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from OmniTokenizer import OmniTokenizer_VQGAN
from pairwise_lambda import LambdaNDCGLoss2
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm

from python_replicate.channel_dataset import ChannelCollection
from python_replicate.frame_preparation import FramePrepConfig
from python_replicate.ofdm_mapper import OFDMConfig
from python_replicate.waveform_bank import ComplexWaveformSystem


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMM_DEVICE = device
ACCEL_DEVICE = device
NUM_TOKENS = 8192
FRAME_SEQ_LEN = 17
WAVEFORM_LEN = 40
BATCH_SIZE_TRAIN = 8
VAL_BATCH_SIZE = 4
LR = 1e-3
NUM_EPOCHS = 100
VAL_FREQ = 1
TRAIN_SNR_RANGE = (0.0, 30.0)
EVAL_SNR_RANGE = (0.0, 30.0)
TRAIN_CHANNELS = ["NCS1"]
EVAL_CHANNELS = ["NOF1"]
CHANNEL_BASE = Path(os.environ.get("E2E_WAVE_CHANNELS_DIR", "data/channels"))
# Preamble/pilot controls
TRAIN_LENGTH = 0  # set to 0 to drop the unused training segment
PILOT_PERIOD = 4  # number of OFDM symbols per pilot block (1 pilot + rest data)
VIDEO_RESOLUTION = 64
VAL_MAX_LENGTH = 200
LOG_DIR_BASE = Path(os.environ.get("E2E_WAVE_RUNS_DIR", "runs/watermark_replicate"))
VIS_FPS = 4
SIMILARITY_METRIC = "l2" # "dot", "si_l2", or "mlp"
TRAIN_USE_AWGN = True
TRAIN_FLAT_CHANNEL = False
EVAL_USE_AWGN = True
EVAL_FLAT_CHANNEL = False


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

    t, c, h, w = video.shape
    if sequence_length <= t:
        video = video[:sequence_length]
    else:
        while video.shape[0] < sequence_length:
            video = torch.cat([video, video], dim=0)
        video = video[:sequence_length]

    if sample_every_n_frames > 1:
        video = video[::sample_every_n_frames]

    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous()
    video = (video - 0.5) * 2.0
    return video


def _parent_dir(path: str) -> str:
    return osp.basename(osp.dirname(path))


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

    def __len__(self) -> int:
        total = self._clips.num_clips()
        if self.max_length is None:
            return total
        return min(total, self.max_length)

    def __getitem__(self, idx: int):
        if self.max_length is not None:
            idx = random.randint(0, self._clips.num_clips() - 1)
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
    def __init__(self, num_tokens: int = NUM_TOKENS) -> None:
        self.indices = torch.arange(num_tokens, dtype=torch.long)

    def __len__(self) -> int:
        return self.indices.numel()

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.indices[idx]


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


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class WatermarkBankTrainer(nn.Module):
    def __init__(self, fixed_relevance: torch.Tensor) -> None:
        super().__init__()
        self.comm = ComplexWaveformSystem(
            num_tokens=NUM_TOKENS,
            output_seq_len=WAVEFORM_LEN,
            use_temperature=False,
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
        )
        self.eval_channels = ChannelCollection(
            EVAL_CHANNELS,
            CHANNEL_BASE,
            frame_config=self.frame_config,
            ofdm_config=self.ofdm_config,
            device=COMM_DEVICE,
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
        pipeline = channel_collection.pipelines[channel_name]
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
        pipeline = self.train_channels.pipelines[channel_name]
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
        targets: List[int] = []
        mse_terms: List[torch.Tensor] = []

        for frame_idx, (rx_frame, tx_frame) in enumerate(
            zip(rx_sequences, result.tx_waveforms)
        ):
            tx_waveform = tx_frame.reshape(-1, self.comm.output_seq_len)
            if tx_waveform.numel() == 0:
                continue
            rx_waveform = rx_frame.reshape(-1, self.comm.output_seq_len)
            if rx_waveform.numel() == 0:
                continue
            tx_vec = tx_waveform[0].to(COMM_DEVICE, dtype=self.comm.get_waveforms().dtype)
            rx_vec = rx_waveform[0].to(COMM_DEVICE, dtype=self.comm.get_waveforms().dtype)
            rx_vec = self.comm.normalize_power(rx_vec.unsqueeze(0)).squeeze(0)
            tx_vec = self.comm.normalize_power(tx_vec.unsqueeze(0)).squeeze(0)
            rx_vec = self._pad_to_len(rx_vec)
            tx_vec = self._pad_to_len(tx_vec)
            rx_features.append(rx_vec)
            tx_refs.append(tx_vec)
            targets.append(indices[frame_idx].item())
            mse_terms.append(
                F.mse_loss(torch.view_as_real(rx_vec), torch.view_as_real(tx_vec))
            )

        if not rx_features:
            raise RuntimeError("All training samples failed through the channel.")

        rx_batch = torch.stack(rx_features, dim=0)
        tx_batch = torch.stack(tx_refs, dim=0)
        bank = self.comm.normalize_power(self.comm.get_waveforms())
        ref_bank = bank.detach() if detach_reference else bank
        scores = self.comm.compute_similarity(
            rx_batch, ref_bank, metric=SIMILARITY_METRIC
        )
        target_tensor = torch.tensor(targets, dtype=torch.long, device=COMM_DEVICE)
        relevance = torch.index_select(self.fixed_relevance, 0, target_tensor)
        counts = torch.full(
            (scores.shape[0],), NUM_TOKENS, dtype=torch.long, device=COMM_DEVICE
        )
        loss = self.lambda_loss(scores, relevance, counts).mean()
        mse_value = torch.stack(mse_terms).mean()
        cos_sim = self._complex_cos_similarity(rx_batch, tx_batch).mean()
        pred = torch.argmax(scores, dim=-1)
        token_acc = (pred == target_tensor).float().mean()
        pred_waves = bank.index_select(0, pred)
        target_waves = bank.index_select(0, target_tensor)
        token_l2 = (pred_waves - target_waves).pow(2).sum(dim=-1).mean().real
        return loss, mse_value, cos_sim, token_acc, token_l2, channel_name, channel_meta

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
        pipeline = channel_collection.pipelines[channel_name]
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
        pipeline = channel_collection.pipelines[channel_name]
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
    omni_model: OmniTokenizer_VQGAN,
    val_loader: DataLoader,
    writer: SummaryWriter,
) -> Tuple[float, float]:
    trainer.eval()
    omni_model.eval()
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(ACCEL_DEVICE)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(
        ACCEL_DEVICE
    )
    bank = trainer.comm.normalize_power(trainer.comm.get_waveforms())
    total_token_acc = 0.0
    total_token_l2 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    total_symbol_batches = 0
    logged_video = False

    for batch in tqdm(val_loader, desc="Eval"):
        video = batch["video"].to(ACCEL_DEVICE)
        indices = omni_model.encode(video, is_image=False)
        B = video.shape[0]
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
            pred_waves = bank.index_select(0, pred_flat)
            target_waves = bank.index_select(0, target_tensor)
            token_l2 = (pred_waves - target_waves).pow(2).sum(dim=-1).mean().real
            total_token_acc += token_acc.item()
            total_token_l2 += token_l2.item()
            total_symbol_batches += 1

            cursor = 0
            predicted_frames: List[torch.Tensor] = []
            for count in counts:
                frame_pred = pred_flat[cursor : cursor + count]
                frame_pred = frame_pred.view(frame_shape)
                predicted_frames.append(frame_pred)
                cursor += count
            pred_indices = torch.stack(predicted_frames, dim=0).unsqueeze(0).to(ACCEL_DEVICE)
            recon = omni_model.decode(pred_indices, is_image=False)
            real = video[b : b + 1]

            real_01 = ((real + 1) * 0.5).clamp(0, 1)
            recon_01 = ((recon + 1) * 0.5).clamp(0, 1)
            real_frames = real_01.permute(0, 2, 1, 3, 4).reshape(
                -1, real_01.shape[1], real_01.shape[3], real_01.shape[4]
            )
            recon_frames = recon_01.permute(0, 2, 1, 3, 4).reshape(
                -1, recon_01.shape[1], recon_01.shape[3], recon_01.shape[4]
            )
            psnr_val = psnr_metric(recon_frames, real_frames)
            ssim_val = ssim_metric(recon_frames, real_frames)
            total_psnr += psnr_val.item()
            total_ssim += ssim_val.item()
            total_samples += 1

            if not logged_video:
                writer.add_video(
                    "Eval/Reconstruction", make_grid_video(real, recon), epoch, fps=VIS_FPS
                )
                logged_video = True

    avg_psnr = total_psnr / max(total_samples, 1)
    avg_ssim = total_ssim / max(total_samples, 1)
    avg_token_acc = total_token_acc / max(total_symbol_batches, 1)
    avg_token_l2 = total_token_l2 / max(total_symbol_batches, 1)
    writer.add_scalar("Eval/PSNR", avg_psnr, epoch)
    writer.add_scalar("Eval/SSIM", avg_ssim, epoch)
    writer.add_scalar("Eval/TokenAcc", avg_token_acc, epoch)
    writer.add_scalar("Eval/TokenL2", avg_token_l2, epoch)
    print(f"Epoch {epoch}: Eval PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, TokenAcc={avg_token_acc:.3f}")
    psnr_metric.reset()
    ssim_metric.reset()
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


def main(video_root: str, omni_ckpt: str, exp_prefix: str = "") -> None:
    log_dir = _create_run_dir(LOG_DIR_BASE, prefix=exp_prefix)
    writer = SummaryWriter(log_dir.as_posix())

    print("Loading OmniTokenizer checkpoint...")
    omni_model = OmniTokenizer_VQGAN.load_from_checkpoint(
        omni_ckpt, strict=False, weights_only=False
    ).to(ACCEL_DEVICE)
    omni_model.eval()
    codebook = omni_model.codebook.embeddings.detach().to(COMM_DEVICE)
    relevance = compute_relevance_matrix(codebook)

    trainer = WatermarkBankTrainer(relevance)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=LR, weight_decay=1e-4)
    train_loader = DataLoader(
        CodebookIndexDataset(), batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=0
    )
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

    sample_indices = torch.randint(0, NUM_TOKENS, (BATCH_SIZE_TRAIN,), device=COMM_DEVICE)
    trainer.comm.zero_grad()
    loss, _, _, _, _, _, _ = trainer.forward_train(sample_indices, detach_reference=True)
    loss.backward()
    grad_norm = trainer.comm.freq_real.grad.norm().item()
    print(f"Gradient norm on waveform bank: {grad_norm:.4f}")

    for epoch in range(NUM_EPOCHS):
        trainer.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_records = []
        for indices in pbar:
            optimizer.zero_grad()
            (
                loss,
                waveform_mse,
                cos_sim,
                token_acc,
                token_l2,
                channel_name,
                channel_meta,
            ) = trainer.forward_train(indices.to(COMM_DEVICE))
            loss.backward()
            optimizer.step()
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/WaveformMSE", waveform_mse.item(), step)
            writer.add_scalar("Train/CosSim", cos_sim.item(), step)
            writer.add_scalar("Train/TokenAcc", token_acc.item(), step)
            writer.add_scalar("Train/TokenL2", token_l2.item(), step)
            epoch_records.append(
                (
                    step,
                    channel_name,
                    loss.item(),
                    channel_meta["snr_mean"],
                    channel_meta["snr_min"],
                    channel_meta["snr_max"],
                    channel_meta["V0"],
                    channel_meta["fc_hz"],
                    channel_meta["fs_tau"],
                    channel_meta["fs_t"],
                    channel_meta["cp_length"],
                    channel_meta["pilot_period"],
                    channel_meta["oversample_q"],
                    channel_meta["num_carriers"],
                    channel_meta["num_ofdm_symbols"],
                    channel_meta["frame_count"],
                )
            )
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{waveform_mse.item():.4e}",
                cos=f"{cos_sim.item():.3f}",
                acc=f"{token_acc.item():.3f}",
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
            psnr, ssim = run_validation(epoch, trainer, omni_model, val_loader, writer)
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
    VIDEO_DIR = os.environ.get("E2E_WAVE_VIDEO_TRAIN_DIR", "data/ProcessedDataset")
    CKPT = os.environ.get(
        "E2E_WAVE_OMNITOKENIZER_CKPT",
        "checkpoints/omnitokenizer/imagenet_k600_fixed.ckpt",
    )
    EXP_PREFIX = "l2_similarity_PLL_AWGN_0_30db_testing(useless)"  # e.g., "l2_sweep1"
    main(VIDEO_DIR, CKPT, EXP_PREFIX)
