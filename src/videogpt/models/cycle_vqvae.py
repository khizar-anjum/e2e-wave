"""
CycleVQVAE: Vector Quantized VAE with Cycle Consistency Loss

This module implements a novel VQVAE training approach that replaces the standard
straight-through estimator and EMA codebook updates with cycle consistency loss.

Key Idea:
    Instead of using straight-through gradients from reconstruction loss to train
    the encoder, we use a cycle consistency loss:

    1. Forward pass: video -> encoder -> z1 -> quantize -> decoder -> recon_video
    2. Cycle pass: recon_video -> encoder -> z2 -> quantize -> compare with z1's assignment

    The encoder is trained to produce representations that, when reconstructed and
    re-encoded, map to the same codebook entries.

Two Versions:
    - Hard Cycle Loss (Version 1): Uses cross-entropy with hard indices as targets
    - Soft Cycle Loss (Version 2): Uses distribution matching + entropy minimization

Gradient Flow:
    - Encoder: trained by cycle_loss + commitment_loss
    - Decoder: trained by reconstruction_loss only
    - Codebook: trained by codebook_loss (gradient-based, no EMA)

Authors: [Your Name]
Date: 2024
"""

from collections import OrderedDict
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from videogpt.layers.pos_embd import BroadcastPosEmbedND
from videogpt.layers.vqvae import Encoder, Decoder
from videogpt.layers.utils import SamePadConvNd, shift_dim


# =============================================================================
# Utility Functions
# =============================================================================

def create_foveated_weights(height, width, sigma=0.5, device=None, dtype=None):
    """Create 2D Gaussian foveated weights centered at the frame center.

    Args:
        height: frame height
        width: frame width
        sigma: standard deviation as fraction of frame size (0.5 = gentle falloff)
        device: torch device
        dtype: torch dtype

    Returns:
        weights: tensor of shape (1, 1, 1, height, width) for broadcasting
    """
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    dist_sq = xx**2 + yy**2
    weights = torch.exp(-dist_sq / (2 * sigma**2))
    weights = weights / weights.mean()

    return weights.view(1, 1, 1, height, width)


def compute_psnr(x_recon, x, data_range=1.0):
    """Compute PSNR between reconstruction and original."""
    mse = F.mse_loss(x_recon, x)
    psnr = 10 * torch.log10(data_range ** 2 / (mse + 1e-8))
    return psnr


def compute_ssim(x_recon, x, window_size=11, data_range=1.0, return_map=False):
    """Compute SSIM between reconstruction and original.

    Args:
        x_recon: reconstructed tensor (b, c, t, h, w)
        x: original tensor (b, c, t, h, w)
        window_size: size of the gaussian window
        data_range: the range of the data (1.0 for [-0.5, 0.5])
        return_map: if True, return the full SSIM map instead of mean

    Returns:
        ssim: scalar tensor (mean SSIM) or SSIM map if return_map=True
    """
    b, c, t, h, w = x_recon.shape
    x_recon_2d = x_recon.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

    window = gaussian_window(window_size).to(x_2d.device).to(x_2d.dtype)
    window = window.expand(c, 1, window_size, window_size)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = F.conv2d(x_2d, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(x_recon_2d, window, padding=window_size // 2, groups=c)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x_2d ** 2, window, padding=window_size // 2, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(x_recon_2d ** 2, window, padding=window_size // 2, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(x_2d * x_recon_2d, window, padding=window_size // 2, groups=c) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    if return_map:
        return ssim_map
    return ssim_map.mean()


def ssim_loss(x_recon, x, window_size=11, data_range=1.0):
    """Compute SSIM loss (1 - SSIM) for training.

    SSIM ranges from -1 to 1, where 1 means identical images.
    We use 1 - SSIM as loss so that minimizing loss maximizes SSIM.

    Args:
        x_recon: reconstructed tensor (b, c, t, h, w)
        x: original tensor (b, c, t, h, w)
        window_size: size of the gaussian window
        data_range: the range of the data (1.0 for [-0.5, 0.5])

    Returns:
        loss: scalar tensor (1 - mean SSIM)
    """
    ssim_val = compute_ssim(x_recon, x, window_size, data_range)
    return 1.0 - ssim_val


# =============================================================================
# CycleQuantize: Gradient-based Codebook (No EMA)
# =============================================================================

class CycleQuantize(nn.Module):
    """
    Vector Quantization layer with gradient-based codebook learning.

    Unlike standard VQ-VAE which uses EMA to update codebook, this layer
    treats the codebook as a learnable parameter updated via gradients.

    Includes dead code replacement to prevent codebook collapse.

    Args:
        n_codebooks: Number of separate codebooks (for product quantization)
        codes_per_book: Number of entries in each codebook
        embedding_dim: Dimension of each codebook entry
        commitment_cost: Weight for commitment loss (encoder -> codebook)
        codebook_cost: Weight for codebook loss (codebook -> encoder)
        reset_threshold: Usage threshold below which codes are replaced (default: 1.0)
        reset_decay: Decay factor for usage tracking (default: 0.99)

    Shapes:
        Input: (batch, embedding_dim * n_codebooks, t, h, w)
        Output dict contains:
            - quantized: (batch, embedding_dim * n_codebooks, t, h, w) or
                        (batch, embedding_dim, t, h, w, n_codebooks) if no_flatten=True
            - encodings: (batch, t, h, w, n_codebooks) - integer indices
            - distances: (n_codebooks, batch*t*h*w, codes_per_book) - for cycle loss
            - commitment_loss: scalar
            - codebook_loss: scalar
            - perplexity: scalar - measure of codebook utilization
    """

    def __init__(self, n_codebooks, codes_per_book, embedding_dim, 
                 commitment_cost=0.25, codebook_cost=0.25,
                 reset_threshold=1.0, reset_decay=0.99,
                 cycle_loss_type="hard"):
        super().__init__()

        # Codebook as learnable parameter (NOT buffer like in EMA version)
        self.embeddings = nn.Parameter(
            torch.randn(n_codebooks, codes_per_book, embedding_dim)
        )

        # Initialize with better distribution
        nn.init.uniform_(self.embeddings, -1.0 / codes_per_book, 1.0 / codes_per_book)

        self.n_codebooks = n_codebooks
        self.codes_per_book = codes_per_book
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.codebook_cost = codebook_cost

        # Dead code replacement parameters
        self.reset_threshold = reset_threshold
        self.reset_decay = reset_decay

        # cycle loss type
        self.cycle_loss_type = cycle_loss_type

        # Track codebook usage with exponential moving average
        self.register_buffer('usage_ema', torch.ones(n_codebooks, codes_per_book))
        self.register_buffer('usage_count', torch.zeros(n_codebooks, codes_per_book))
        self._need_init = True

    def _init_from_data(self, z):
        """Initialize codebook from first batch of encoder outputs."""
        self._need_init = False

        # z: (b, c, t, h, w) where c = embedding_dim * n_codebooks
        flat_inputs = shift_dim(z, 1, -1).view(-1, self.n_codebooks, self.embedding_dim)
        flat_inputs = flat_inputs.permute(1, 0, 2).contiguous()  # (n_codebooks, N, embedding_dim)

        N = flat_inputs.shape[1]

        # Sample random encoder outputs for initialization
        with torch.no_grad():
            for c in range(self.n_codebooks):
                if N >= self.codes_per_book:
                    # Random sample without replacement
                    perm = torch.randperm(N, device=z.device)[:self.codes_per_book]
                    self.embeddings.data[c] = flat_inputs[c, perm].clone()
                else:
                    # Tile if not enough samples
                    repeats = (self.codes_per_book + N - 1) // N
                    tiled = flat_inputs[c].repeat(repeats, 1)[:self.codes_per_book]
                    noise = torch.randn_like(tiled) * 0.01
                    self.embeddings.data[c] = tiled + noise

    def compute_distances(self, z):
        """
        Compute squared L2 distances from encoder outputs to all codebook entries.

        Args:
            z: (batch, embedding_dim * n_codebooks, t, h, w)

        Returns:
            distances: (n_codebooks, batch*t*h*w, codes_per_book)
            flat_inputs: (n_codebooks, batch*t*h*w, embedding_dim)
        """
        # Reshape: (b, c, t, h, w) -> (n_codebooks, N, embedding_dim)
        flat_inputs = shift_dim(z, 1, -1).view(-1, self.n_codebooks, self.embedding_dim)
        flat_inputs = flat_inputs.permute(1, 0, 2).contiguous()

        # Compute squared L2 distances: ||z - e||^2 = ||z||^2 - 2*z*e + ||e||^2
        # flat_inputs: (n_codebooks, N, embedding_dim)
        # embeddings: (n_codebooks, codes_per_book, embedding_dim)

        z_sq = (flat_inputs ** 2).sum(dim=2, keepdim=True)  # (n_codebooks, N, 1)
        e_sq = (self.embeddings ** 2).sum(dim=2, keepdim=True)  # (n_codebooks, codes_per_book, 1)

        # (n_codebooks, N, codes_per_book)
        distances = z_sq - 2 * torch.bmm(flat_inputs, self.embeddings.transpose(1, 2)) + e_sq.transpose(1, 2)

        return distances, flat_inputs

    def forward(self, z, no_flatten=False):
        """
        Forward pass: quantize encoder outputs.

        Args:
            z: (batch, embedding_dim * n_codebooks, t, h, w)
            no_flatten: if True, keep codebooks dimension separate

        Returns:
            dict with: quantized, encodings, distances, commitment_loss,
                      codebook_loss, perplexity
        """
        # Initialize codebook from data on first forward pass
        if self._need_init and self.training:
            self._init_from_data(z)

        batch_size = z.shape[0]
        spatial_shape = z.shape[2:]  # (t, h, w)

        # Compute distances to all codebook entries
        distances, flat_inputs = self.compute_distances(z)  # (n_codebooks, N, codes_per_book)

        # Hard assignment: find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=2)  # (n_codebooks, N)

        # Create one-hot encodings for perplexity calculation
        encode_onehot = F.one_hot(encoding_indices, self.codes_per_book).float()  # (n_codebooks, N, codes_per_book)

        # Reshape indices for output: (n_codebooks, N) -> (batch, t, h, w, n_codebooks)
        encoding_indices_reshaped = encoding_indices.view(
            self.n_codebooks, batch_size, *spatial_shape
        )  # (n_codebooks, batch, t, h, w)
        encodings_out = shift_dim(encoding_indices_reshaped, 0, -1)  # (batch, t, h, w, n_codebooks)

        # Look up quantized vectors
        # quantized: (batch, t, h, w, n_codebooks, embedding_dim)
        quantized = torch.stack([
            F.embedding(encoding_indices_reshaped[i], self.embeddings[i])
            for i in range(self.n_codebooks)
        ], dim=-2)

        # Reshape z for loss computation
        # if no_flatten:
        #     z_reshaped = shift_dim(z, 1, -1)  # (batch, t, h, w, n_codebooks * embedding_dim)
        #     z_reshaped = z_reshaped.view(*z_reshaped.shape[:-1], self.n_codebooks, self.embedding_dim)
        #     z_for_loss = z_reshaped  # (batch, t, h, w, n_codebooks, embedding_dim)
        # else:
        #     # Need to reshape z to match quantized for loss
        #     z_for_loss = shift_dim(z, 1, -1).view(
        #         batch_size, *spatial_shape, self.n_codebooks, self.embedding_dim
        #     )

        # we do not need commitment and codebook losses for now. 
        # Commitment loss: push encoder outputs toward codebook
        # Encoder learns to produce outputs close to codebook entries
        # commitment_loss = self.commitment_cost * F.mse_loss(z_for_loss, quantized.detach())

        # Codebook loss: push codebook toward encoder outputs
        # Codebook entries move toward assigned encoder outputs
        # codebook_loss = self.codebook_cost * F.mse_loss(z_for_loss.detach(), quantized)

        # Flatten for output if needed
        if not no_flatten:
            quantized = quantized.flatten(start_dim=-2)  # (batch, t, h, w, n_codebooks * embedding_dim)

        quantized = shift_dim(quantized, -1, 1)  # (batch, n_codebooks * embedding_dim, t, h, w)

        # Straight-through estimator for quantized output
        # Forward: use quantized, Backward: gradient flows to z
        quantized_st = z + (quantized - z).detach()

        # Compute perplexity (codebook utilization metric)
        avg_probs = encode_onehot.mean(dim=1)  # (n_codebooks, codes_per_book)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=1))
        perplexity = perplexity.mean()

        # Update usage tracking and perform dead code replacement
        if self.training:
            with torch.no_grad():
                # Update usage count
                batch_usage = encode_onehot.sum(dim=1)  # (n_codebooks, codes_per_book)
                self.usage_count += batch_usage

                # Update usage EMA
                self.usage_ema.mul_(self.reset_decay).add_(batch_usage, alpha=1 - self.reset_decay)

                # Dead code replacement: replace codes with usage below threshold
                # with random encoder outputs from current batch
                dead_codes = self.usage_ema < self.reset_threshold  # (n_codebooks, codes_per_book)

                if dead_codes.any():
                    # Get random encoder outputs for replacement
                    N = flat_inputs.shape[1]
                    for c in range(self.n_codebooks):
                        dead_indices = dead_codes[c].nonzero(as_tuple=True)[0]
                        n_dead = len(dead_indices)
                        if n_dead > 0:
                            # Sample random encoder outputs
                            random_indices = torch.randint(0, N, (n_dead,), device=z.device)
                            # Replace dead codes with encoder outputs + small noise
                            new_codes = flat_inputs[c, random_indices].clone()
                            noise = torch.randn_like(new_codes) * 0.01
                            self.embeddings.data[c, dead_indices] = new_codes + noise
                            # Reset usage for replaced codes
                            self.usage_ema[c, dead_indices] = 1.0
        if self.cycle_loss_type == "hard":
            return {
                'quantized': quantized_st,
                'distances': distances.detach(),  # Important for cycle loss, only need as labels
                'perplexity': perplexity,
            }
        else:
            return {
                'quantized': quantized_st,
                'distances': distances,  # Important for cycle loss, grad through in soft version
                'perplexity': perplexity,
            }

    def dictionary_lookup(self, encodings):
        """
        Look up codebook entries given indices.

        Args:
            encodings: (batch, t, h, w, n_codebooks) integer indices

        Returns:
            quantized: (batch, n_codebooks * embedding_dim, t, h, w)
        """
        # (n_codebooks, batch, t, h, w)
        encodings_transposed = shift_dim(encodings, -1, 0)

        # Look up and stack
        quantized = torch.stack([
            F.embedding(encodings_transposed[i], self.embeddings[i])
            for i in range(self.n_codebooks)
        ], dim=-2)  # (batch, t, h, w, n_codebooks, embedding_dim)

        quantized = quantized.flatten(start_dim=-2)  # (batch, t, h, w, n_codebooks * embedding_dim)
        quantized = shift_dim(quantized, -1, 1)  # (batch, n_codebooks * embedding_dim, t, h, w)

        return quantized


# =============================================================================
# Cycle Loss Functions
# =============================================================================

def hard_cycle_loss(distances1, distances2, temperature=1.0, entropy_weight=0.1):
    """
    Version 1: Hard Cycle Loss using cross-entropy.

    Uses distances from pass 1 as classification targets for pass 2.

    Args:
        distances1: (n_codebooks, N, codes_per_book) - distances from pass 1
        distances2: (n_codebooks, N, codes_per_book) - hard indices from pass 2

    Returns:
        loss: scalar tensor

    Gradient Flow:
        - Gradients flow through distances2 -> encoder
        - distances1 is treated as target labels (no gradient)
    """
    n_codebooks = distances2.shape[0]

    total_loss = 0.0
    for c in range(n_codebooks):
        # Logits: negative distances (smaller distance = higher logit)
        log_prob1 = F.log_softmax(-distances1[c] / temperature, dim=-1)  # (n_codebooks, N, codes_per_book)
        log_prob2 = F.log_softmax(-distances2[c] / temperature, dim=-1)
        prob1 = torch.exp(log_prob1)
        prob2 = torch.exp(log_prob2)

        entropy2 = -torch.sum(prob2 * log_prob2, dim=-1).mean()
        cross_entropy_loss = F.cross_entropy(prob1, prob2)

        # logits = F.softmax(-distances2[c], dim =-1)  # (N, codes_per_book)
        # targets = F.softmax(-distances1[c], dim=-1)     # (N, codes_per_book)

        total_loss += cross_entropy_loss + entropy_weight * entropy2

    total_loss = total_loss / n_codebooks
    
    metrics = {
        'cross_entropy_loss': cross_entropy_loss.detach(),
        'entropy2': entropy2.detach(),
    }

    return total_loss, metrics


def soft_cycle_loss(distances1, distances2, temperature=1.0, entropy_weight=0.1):
    """
    Version 2: Soft Cycle Loss with distribution matching and entropy minimization.

    Converts distances to probability distributions and:
    1. Matches distribution from pass 2 to pass 1 (cycle consistency)
    2. Minimizes entropy of both distributions (encourages sharp assignments)

    Args:
        distances1: (n_codebooks, N, codes_per_book) - distances from pass 1
        distances2: (n_codebooks, N, codes_per_book) - distances from pass 2
        temperature: softmax temperature (lower = sharper distributions)
        entropy_weight: weight for entropy minimization term

    Returns:
        loss: scalar tensor
        metrics: dict with 'distribution_loss', 'entropy1', 'entropy2'

    Gradient Flow:
        - distribution_loss: gradients flow through distances2 -> encoder
        - entropy1: gradients flow through distances1 -> encoder (pass 1)
        - entropy2: gradients flow through distances2 -> encoder (pass 2)
    """
    n_codebooks = distances1.shape[0]

    total_loss = 0.0
    for c in range(n_codebooks):
        # Convert to probability distributions
        # Negative distances because smaller distance = more likely
        log_prob1 = F.log_softmax(-distances1[c] / temperature, dim=-1)  # (n_codebooks, N, codes_per_book)
        log_prob2 = F.log_softmax(-distances2[c] / temperature, dim=-1)
        prob1 = torch.exp(log_prob1)
        prob2 = torch.exp(log_prob2)

        # cross entropy loss
        cross_entropy_loss = F.cross_entropy(prob1, prob2)

        # Entropy minimization: H(p) = -sum(p * log(p))
        # Low entropy = confident/peaky distribution (good for discrete-like behavior)
        entropy1 = -torch.sum(prob1 * log_prob1, dim=-1).mean()
        entropy2 = -torch.sum(prob2 * log_prob2, dim=-1).mean()

        total_loss += cross_entropy_loss + entropy_weight * (entropy1 + entropy2)

    total_loss = total_loss / n_codebooks

    metrics = {
        'cross_entropy_loss': cross_entropy_loss.detach(),
        'entropy1': entropy1.detach(),
        'entropy2': entropy2.detach(),
    }

    return total_loss, metrics


# =============================================================================
# CycleVQVAE Model
# =============================================================================

class CycleVQVAE(nn.Module):
    """
    Vector Quantized VAE with Cycle Consistency Loss.

    This model replaces the standard VQ-VAE training approach:
    - NO straight-through estimator for reconstruction gradients to encoder
    - NO EMA codebook updates

    Instead uses:
    - Cycle consistency loss to train encoder
    - Gradient-based codebook learning
    - Clean separation of training signals

    Architecture:
        video -> Encoder -> pre_vq_conv -> CycleQuantize -> Decoder -> reconstruction
                                              |
                              reconstruction -> Encoder -> CycleQuantize -> cycle loss

    Training Signals:
        - Encoder: cycle_loss + commitment_loss
        - Decoder: reconstruction_loss (MSE)
        - Codebook: codebook_loss

    Args:
        embedding_dim: Dimension of codebook entries
        codes_per_book: Number of entries per codebook
        n_codebooks: Number of codebooks (product quantization)
        input_shape: (T, H, W) shape of input video frames
        downsample: (t_ds, h_ds, w_ds) downsampling factors
        num_hiddens: Hidden dimension in encoder/decoder
        num_residual_layers: Number of residual blocks
        num_residual_hiddens: Hidden dim in residual blocks
        use_attn: Whether to use attention in residual blocks
        attn_n_heads: Number of attention heads
        commitment_cost: Weight for commitment loss
        codebook_cost: Weight for codebook loss
        cycle_loss_type: 'hard' or 'soft'
        cycle_loss_weight: Weight for cycle consistency loss
        temperature: Softmax temperature for soft cycle loss
        entropy_weight: Entropy term weight for soft cycle loss
        ssim_loss_weight: Weight for SSIM loss in reconstruction (0.0 to disable)
        cond_types: Conditioning types (unused, for compatibility)
        foveated_loss: Whether to use foveated reconstruction loss
        foveated_sigma: Sigma for foveated weighting
    """

    def __init__(
        self,
        embedding_dim: int,
        codes_per_book: int,
        n_codebooks: int,
        input_shape: tuple,
        downsample: tuple,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        use_attn: bool,
        attn_n_heads: int,
        commitment_cost: float = 0.25,
        codebook_cost: float = 0.25,
        cycle_loss_type: str = 'hard',
        cycle_loss_weight: float = 1.0,
        temperature: float = 1.0,
        entropy_weight: float = 0.1,
        ssim_loss_weight: float = 0.1,
        cond_types=tuple(),
        foveated_loss: bool = False,
        foveated_sigma: float = 0.5,
        decay: float = 0.99,  # Unused, kept for compatibility
    ):
        super().__init__()

        assert len(input_shape) == len(downsample)
        assert cycle_loss_type in ('hard', 'soft'), f"cycle_loss_type must be 'hard' or 'soft', got {cycle_loss_type}"

        # Store config
        self.cycle_loss_type = cycle_loss_type
        self.cycle_loss_weight = cycle_loss_weight
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.ssim_loss_weight = ssim_loss_weight
        self.foveated_loss = foveated_loss
        self.foveated_sigma = foveated_sigma
        self._foveated_weights = None

        # Compute latent shape
        ds_shape = tuple([s // d for s, d in zip(input_shape, downsample)])

        # Position embedding for attention
        if use_attn:
            self.pos_embd = BroadcastPosEmbedND(shape=ds_shape, embd_dim=num_hiddens)
        else:
            self.pos_embd = None

        n_dim = len(input_shape)
        embedding_channels = embedding_dim * n_codebooks

        # Encoder
        self.encoder = Encoder(
            input_shape, num_hiddens, num_residual_layers, num_residual_hiddens,
            attn_n_heads, downsample, use_attn, pos_embd=self.pos_embd
        )

        # Decoder
        self.decoder = Decoder(
            ds_shape, embedding_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens, attn_n_heads, downsample, use_attn, pos_embd=self.pos_embd
        )

        # Pre-quantization projection
        self.pre_vq_conv = SamePadConvNd(
            n_dim, in_channels=num_hiddens, out_channels=embedding_channels,
            kernel_size=1, stride=1
        )

        # Quantization layer (gradient-based, no EMA)
        self.codebook = CycleQuantize(
            n_codebooks, codes_per_book, embedding_dim,
            commitment_cost=commitment_cost, codebook_cost=codebook_cost,
            cycle_loss_type=cycle_loss_type
        )

        # Store shapes for external access
        self.input_shape = input_shape
        self.latent_shape = (*ds_shape, n_codebooks)
        self.quantized_shape = (embedding_dim, *ds_shape, n_codebooks)

    @property
    def metrics(self):
        """List of metrics to track during training."""
        base = ['loss', 'recon', 'mse', 'ssim_loss', 'cycle', 'perplexity', 'psnr', 'ssim']
        if self.cycle_loss_type == 'soft':
            base.extend(['cross_entropy_loss', 'entropy1', 'entropy2'])
        else:
            base.extend(['cross_entropy_loss', 'entropy2'])
        return base

    @property
    def metrics_fmt(self):
        """Format strings for metrics display."""
        base = [':6.4f', ':6.4f', ':6.4f', ':6.4f', ':6.4f', ':6.2f', ':6.2f', ':6.4f']
        if self.cycle_loss_type == 'soft':
            base.extend([':6.4f', ':6.4f', ':6.4f'])
        else:
            base.extend([':6.4f', ':6.4f',])
        return base

    def get_foveated_weights(self, x):
        """Get or create cached foveated weights."""
        h, w = x.shape[-2], x.shape[-1]
        if self._foveated_weights is None or self._foveated_weights.shape[-2:] != (h, w):
            self._foveated_weights = create_foveated_weights(
                h, w, sigma=self.foveated_sigma, device=x.device, dtype=x.dtype
            )
        return self._foveated_weights

    def encode(self, x):
        """
        Encode input video to latent space.

        Args:
            x: (batch, channels, t, h, w)

        Returns:
            z: (batch, embedding_dim * n_codebooks, t', h', w')
        """
        return self.pre_vq_conv(self.encoder(x))

    def forward(self, x):
        """
        Forward pass with cycle consistency.

        The key insight: we do TWO encoder passes.
        Pass 1: Encode original video
        Pass 2: Encode reconstructed video, should give same codes

        Args:
            x: (batch, channels, t, h, w) input video

        Returns:
            dict with loss, metrics, and intermediate values
        """
        return_dict = OrderedDict()

        # =====================================================================
        # Pass 1: Encode original video
        # =====================================================================
        z1 = self.encode(x)
        vq_output1 = self.codebook(z1)

        z_q1 = vq_output1['quantized']
        distances1 = vq_output1['distances']

        # =====================================================================
        # Decode (so gradients also pass back to encoder!)
        # =====================================================================
        x_recon = self.decoder(z_q1)

        # =====================================================================
        # Pass 2: Encode reconstruction (cycle pass)
        # =====================================================================
        if self.cycle_loss_type == "hard":
            z2 = self.encode(x_recon.detach())
            # We only need distances for cycle loss, not full quantization
            distances2, _ = self.codebook.compute_distances(z2)
        else: # soft
            z2 = self.encode(x_recon)
            # We only need distances for cycle loss, not full quantization
            distances2, _ = self.codebook.compute_distances(z2.detach())

        # =====================================================================
        # Compute losses
        # =====================================================================

        # 1. Reconstruction loss (trains decoder only)
        # Composed of MSE loss and optional SSIM loss
        if self.foveated_loss:
            fov_weights = self.get_foveated_weights(x)
            squared_error = (x_recon - x) ** 2
            mse_loss_val = (fov_weights * squared_error).mean() / 0.06
        else:
            mse_loss_val = F.mse_loss(x_recon, x) / 0.06

        # SSIM loss: 1 - SSIM (so minimizing loss maximizes SSIM)
        ssim_loss_val = ssim_loss(x_recon, x)

        # Combined reconstruction loss
        recon_loss = mse_loss_val + self.ssim_loss_weight * ssim_loss_val

        # 2. Cycle consistency loss (trains encoder)
        # Convert indices to format needed for cycle loss
        # indices1: (batch, t, h, w, n_codebooks) -> (n_codebooks, N)
        # indices1_flat = shift_dim(indices1, -1, 0).flatten(start_dim=1)

        if self.cycle_loss_type == 'hard':
            cycle_loss, metrics = hard_cycle_loss(
                distances1, distances2,
                temperature=self.temperature,
                entropy_weight=self.entropy_weight
            )
            return_dict.update({
                'cross_entropy_loss': metrics['cross_entropy_loss'],
                'entropy2': metrics['entropy2'],
            })
        else:  # soft
            cycle_loss, metrics = soft_cycle_loss(
                distances1, distances2,
                temperature=self.temperature,
                entropy_weight=self.entropy_weight
            )
            return_dict.update({
                'cross_entropy_loss': metrics['cross_entropy_loss'],
                'entropy1': metrics['entropy1'],
                'entropy2': metrics['entropy2'],
            })

        # Total loss
        total_loss = (
            recon_loss +
            self.cycle_loss_weight * cycle_loss
        )

        # =====================================================================
        # Compute metrics (no gradients needed)
        # =====================================================================
        with torch.no_grad():
            psnr = compute_psnr(x_recon, x)
            ssim = compute_ssim(x_recon, x)

        # Build return dict
        return_dict.update({
            'loss': total_loss,
            'recon': recon_loss,
            'mse': mse_loss_val,
            'ssim_loss': ssim_loss_val,
            'cycle': cycle_loss.detach() if isinstance(cycle_loss, torch.Tensor) else cycle_loss,
            'perplexity': vq_output1['perplexity'],
            'psnr': psnr,
            'ssim': ssim,
        })

        return return_dict

    def get_reconstruction(self, x):
        """
        Get reconstruction for visualization (no cycle pass).

        Args:
            x: (batch, channels, t, h, w)

        Returns:
            x_recon: (batch, channels, t, h, w)
        """
        z = self.encode(x)
        vq_output = self.codebook(z)
        return self.decoder(vq_output['quantized'])

    def decode(self, encodings):
        """
        Decode from discrete encodings.

        Args:
            encodings: (batch, t, h, w, n_codebooks) integer indices

        Returns:
            x_recon: (batch, channels, t, h, w)
        """
        z_q = self.codebook.dictionary_lookup(encodings)
        return self.decoder(z_q)

    def no_need_init(self):
        """Skip codebook initialization from data."""
        self.codebook._need_init = False
