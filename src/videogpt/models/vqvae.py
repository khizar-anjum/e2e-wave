from collections import OrderedDict
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from videogpt.layers.pos_embd import BroadcastPosEmbedND
from videogpt.layers.vqvae import Encoder, Decoder, Quantize
from videogpt.layers.utils import SamePadConvNd
from videogpt.layers.utils import shift_dim


def create_foveated_weights(height, width, sigma=0.5, device=None, dtype=None):
    """Create 2D Gaussian foveated weights centered at the frame center.

    Args:
        height: frame height
        width: frame width
        sigma: standard deviation as fraction of frame size (0.5 = gentle falloff, 0.2 = sharp)
        device: torch device
        dtype: torch dtype

    Returns:
        weights: tensor of shape (1, 1, 1, height, width) for broadcasting with (b, c, t, h, w)
    """
    # Create coordinate grids normalized to [-1, 1]
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Compute Gaussian weights based on distance from center
    # sigma controls the spread (larger = gentler falloff)
    dist_sq = xx**2 + yy**2
    weights = torch.exp(-dist_sq / (2 * sigma**2))

    # Normalize so mean weight is 1.0 (preserves loss scale)
    weights = weights / weights.mean()

    # Shape: (1, 1, 1, H, W) for broadcasting with (B, C, T, H, W)
    return weights.view(1, 1, 1, height, width)


def compute_psnr(x_recon, x, data_range=1.0, weights=None):
    """Compute PSNR between reconstruction and original.

    Args:
        x_recon: reconstructed tensor (b, c, t, h, w) in [-0.5, 0.5]
        x: original tensor (b, c, t, h, w) in [-0.5, 0.5]
        data_range: the range of the data (1.0 for [-0.5, 0.5])
        weights: optional spatial weights (1, 1, 1, h, w) for foveated PSNR

    Returns:
        psnr: scalar tensor
    """
    if weights is not None:
        # Weighted MSE: mean of (weight * squared_error)
        squared_error = (x_recon - x) ** 2
        mse = (weights * squared_error).mean()
    else:
        mse = F.mse_loss(x_recon, x)
    psnr = 10 * torch.log10(data_range ** 2 / (mse + 1e-8))
    return psnr


def compute_ssim(x_recon, x, window_size=11, data_range=1.0, weights=None):
    """Compute SSIM between reconstruction and original.

    Computes SSIM per frame and averages across batch, time, and channels.

    Args:
        x_recon: reconstructed tensor (b, c, t, h, w) in [-0.5, 0.5]
        x: original tensor (b, c, t, h, w) in [-0.5, 0.5]
        window_size: size of the gaussian window
        data_range: the range of the data (1.0 for [-0.5, 0.5])
        weights: optional spatial weights (1, 1, 1, h, w) for foveated SSIM

    Returns:
        ssim: scalar tensor
    """
    # Reshape to (b*t, c, h, w) to compute SSIM per frame
    b, c, t, h, w = x_recon.shape
    x_recon_2d = x_recon.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

    # Create gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

    window = gaussian_window(window_size).to(x_2d.device).to(x_2d.dtype)
    window = window.expand(c, 1, window_size, window_size)

    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Compute means
    mu_x = F.conv2d(x_2d, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(x_recon_2d, window, padding=window_size // 2, groups=c)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Compute variances and covariance
    sigma_x_sq = F.conv2d(x_2d ** 2, window, padding=window_size // 2, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(x_recon_2d ** 2, window, padding=window_size // 2, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(x_2d * x_recon_2d, window, padding=window_size // 2, groups=c) - mu_xy

    # SSIM formula
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    if weights is not None:
        # Reshape weights from (1, 1, 1, h, w) to (1, 1, h, w) for 2D SSIM map
        weights_2d = weights.squeeze(2)  # (1, 1, h, w)
        # Weight the SSIM map before averaging
        weighted_ssim = (weights_2d * ssim_map).mean()
        return weighted_ssim
    else:
        return ssim_map.mean()


class VQVAE(nn.Module):
    def __init__(self, embedding_dim: int, codes_per_book: int, n_codebooks: int,
                 input_shape: tuple, downsample: tuple, num_hiddens: int, num_residual_layers: int,
                 num_residual_hiddens: int, use_attn: bool, attn_n_heads: int,
                 commitment_cost: float, decay: float, cond_types,
                 foveated_loss: bool = False, foveated_sigma: float = 0.5):
        super().__init__()
        assert len(input_shape) == len(downsample), ('input shape', input_shape, 'ds', downsample)

        # Foveated loss settings
        self.foveated_loss = foveated_loss
        self.foveated_sigma = foveated_sigma
        self._foveated_weights = None  # Cached weights

        assert all([int(math.log2(d)) == math.log2(d)] for d in downsample), f'downsample must be powers of 2'
        ds_shape = tuple([s // d for s, d in zip(input_shape, downsample)])

        if use_attn:
            # share embedding layer between encoder and decoder
            self.pos_embd = BroadcastPosEmbedND(
                shape=ds_shape, embd_dim=num_hiddens
            )
        else:
            self.pos_embd = None
        n_dim = len(input_shape)

        embedding_channels = embedding_dim * n_codebooks
        self.encoder = Encoder(input_shape, num_hiddens,
                               num_residual_layers,
                               num_residual_hiddens,
                               attn_n_heads, downsample,
                               use_attn, pos_embd=self.pos_embd)
        self.decoder = Decoder(ds_shape, embedding_channels,
                               num_hiddens, num_residual_layers,
                               num_residual_hiddens,
                               attn_n_heads, downsample,
                               use_attn, pos_embd=self.pos_embd)

        self.pre_vq_conv1 = SamePadConvNd(
            n_dim,
            in_channels=num_hiddens,
            out_channels=embedding_channels,
            kernel_size=1,
            stride=1)

        self.codebook = Quantize(n_codebooks, codes_per_book,
                                 embedding_dim, commitment_cost,
                                 decay=decay)
        self.input_shape = input_shape

        self.latent_shape = (*ds_shape, n_codebooks)
        self.quantized_shape = (embedding_dim, *ds_shape, n_codebooks)

    @property
    def metrics(self):
        base_metrics = ['loss', 'commitment', 'perplexity', 'recon', 'psnr', 'ssim']
        if self.foveated_loss:
            base_metrics.extend(['fov_psnr', 'fov_ssim'])
        return base_metrics

    @property
    def metrics_fmt(self):
        base_fmt = [':6.4f', ':6.4f', ':6.4f', ':6.4f', ':6.2f', ':6.4f']
        if self.foveated_loss:
            base_fmt.extend([':6.2f', ':6.4f'])
        return base_fmt

    def get_foveated_weights(self, x):
        """Get or create cached foveated weights matching input spatial dims."""
        h, w = x.shape[-2], x.shape[-1]
        if self._foveated_weights is None or self._foveated_weights.shape[-2:] != (h, w):
            self._foveated_weights = create_foveated_weights(
                h, w, sigma=self.foveated_sigma, device=x.device, dtype=x.dtype
            )
        return self._foveated_weights

    def no_need_init(self):
        assert self.codebook._need_init
        self.codebook._need_init = False

    def forward(self, x):
        """
        :param x: torch.Tensor with shape (b, c, t, h, w)
        """
        return_dict = OrderedDict()
        z = self.pre_vq_conv1(self.encoder(x=x))

        vq_output = self.codebook(z, no_flatten=True)
        dec_inp = vq_output['quantized']
        dec_inp = shift_dim(dec_inp, -1, 1).flatten(1, 2)  # -> (b, l, d, t', h', w') -> (b, l*d, t', h', w')
        x_recon = self.decoder(x=dec_inp)

        commitment_loss = vq_output['commitment_loss']

        # Compute reconstruction loss (optionally foveated)
        if self.foveated_loss:
            fov_weights = self.get_foveated_weights(x)
            # Weighted MSE loss
            squared_error = (x_recon - x) ** 2
            recon_loss = (fov_weights * squared_error).mean() / 0.06
        else:
            recon_loss = F.mse_loss(x_recon, x) / 0.06

        loss = commitment_loss + recon_loss

        # Compute PSNR and SSIM (no grad needed for metrics)
        with torch.no_grad():
            # Always compute standard (uniform) metrics
            psnr = compute_psnr(x_recon, x)
            ssim = compute_ssim(x_recon, x)

            return_dict.update(loss=loss,
                               commitment=commitment_loss,
                               recon=recon_loss,
                               perplexity=vq_output['perplexity'],
                               psnr=psnr,
                               ssim=ssim)

            # Compute foveated metrics if enabled
            if self.foveated_loss:
                fov_weights = self.get_foveated_weights(x)
                fov_psnr = compute_psnr(x_recon, x, weights=fov_weights)
                fov_ssim = compute_ssim(x_recon, x, weights=fov_weights)
                return_dict.update(fov_psnr=fov_psnr, fov_ssim=fov_ssim)

        return return_dict

    def encode(self, x, no_flatten=False):
        """
        Must be in eval mode.
        :param x: (b, c, t, h, w)
        :param no_flatten:
        :return:
            quantize: (b, d*l, t', h', w') if no_flatten = False (default);
            else: (b, d, t', h', w', l)
            encodings: (b, t', h', w', l)
        """
        z = self.pre_vq_conv1(self.encoder(x=x))
        vq_output = self.codebook(z, no_flatten=no_flatten)
        return vq_output['quantized'], vq_output['encodings']

    def decode(self, x):
        x = self.codebook.dictionary_lookup(x)
        return self.decoder(x)

    def get_reconstruction(self, x):
        _, encodings = self.encode(x=x)
        return self.decode(encodings)
