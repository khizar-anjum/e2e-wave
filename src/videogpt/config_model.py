import videogpt.models as models
import copy


# BAIR / ViZDoom / RoboNet
# 16 x 64 x 64 -> 8 x 32 x 32
vae_res64_ds222 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(2, 2, 2)
)

# TGIF / UCF 64 x 64
# 16 x 64 x 64 -> 4 x 32 x 32
vae_res64_ds422 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 2, 2)
)

# UCF 128 x 128
# 16 x 128 x 128 -> 4 x 32 x 32
vae_res128_ds444 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 4, 4)
)

# 256 x 256
# 16 x 256 x 256 -> 4 x 32 x 32
vae_res256_ds488 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 8, 8)
)

# 256 x 256 with larger latent space
# 16 x 256 x 256 -> 4 x 64 x 64
vae_res256_ds444 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 4, 4)
)

# 512 x 512
# 16 x 512 x 512 -> 4 x 32 x 32
vae_res512_ds4_16_16 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 16, 16)
)

# ============================================================================
# Resolution-agnostic downsample configs
# Use with any --resolution value
# Latent shape: (n_frames/t, resolution/h, resolution/w)
# ============================================================================

# Downsample (4, 2, 2) - minimal spatial compression
# 64x64 -> 32x32, 128x128 -> 64x64, 256x256 -> 128x128, 512x512 -> 256x256
vae_ds422 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 2, 2)
)

# Downsample (4, 4, 4) - moderate compression
# 64x64 -> 16x16, 128x128 -> 32x32, 256x256 -> 64x64, 512x512 -> 128x128
vae_ds444 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 4, 4)
)

# Downsample (4, 8, 8) - high compression
# 64x64 -> 8x8, 128x128 -> 16x16, 256x256 -> 32x32, 512x512 -> 64x64
vae_ds488 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 8, 8)
)

# Downsample (4, 16, 16) - very high compression
# 64x64 -> 4x4, 128x128 -> 8x8, 256x256 -> 16x16, 512x512 -> 32x32
vae_ds4_16_16 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 16, 16)
)

# Downsample (8, 8, 8) - very high compression with more temporal
# 64x64 -> 8x8, 128x128 -> 16x16, 256x256 -> 32x32, 512x512 -> 64x64
# (also halves temporal frames compared to ds4xx)
vae_ds888 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(8, 8, 8)
)

# Downsample (8, 16, 16) - extreme compression
# 64x64 -> 4x4, 128x128 -> 8x8, 256x256 -> 16x16, 512x512 -> 32x32
vae_ds8_16_16 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(8, 16, 16)
)

# =============================================================================
# CycleVQVAE Configurations
# Uses cycle consistency loss instead of straight-through estimator
# Gradient-based codebook learning instead of EMA
# =============================================================================

# CycleVQVAE with HARD cycle loss (Version 1)
# Uses cross-entropy with hard indices as targets
cycle_vae_hard_ds444 = dict(
    model_cls='CycleVQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, codebook_cost=0.25,
    cycle_loss_type='hard',
    cycle_loss_weight=1.0,
    ssim_loss_weight=0.1,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 4, 4)
)

cycle_vae_hard_ds422 = dict(
    model_cls='CycleVQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, codebook_cost=0.25,
    cycle_loss_type='hard',
    cycle_loss_weight=1.0,
    entropy_weight=0.5,
    temperature=1.0,
    ssim_loss_weight=0.1,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 2, 2)
)

cycle_vae_hard_ds488 = dict(
    model_cls='CycleVQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, codebook_cost=0.25,
    cycle_loss_type='hard',
    cycle_loss_weight=1.0,
    ssim_loss_weight=0.1,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 8, 8)
)

# CycleVQVAE with SOFT cycle loss (Version 2)
# Uses distribution matching + entropy minimization
cycle_vae_soft_ds444 = dict(
    model_cls='CycleVQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, codebook_cost=0.25,
    cycle_loss_type='soft',
    cycle_loss_weight=1.0,
    temperature=1.0,
    entropy_weight=0.1,
    ssim_loss_weight=0.1,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 4, 4)
)

cycle_vae_soft_ds422 = dict(
    model_cls='CycleVQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, codebook_cost=0.25,
    cycle_loss_type='soft',
    cycle_loss_weight=1.0,
    temperature=1.0,
    entropy_weight=0.5,
    ssim_loss_weight=0.5,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 2, 2)
)

cycle_vae_soft_ds488 = dict(
    model_cls='CycleVQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, codebook_cost=0.25,
    cycle_loss_type='soft',
    cycle_loss_weight=1.0,
    temperature=1.0,
    entropy_weight=0.1,
    ssim_loss_weight=0.1,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 8, 8)
)

# BAIR / ViZDoom / RoboNet
gpt_small = dict(
    model_cls='ImageGPT',
    out_features=512,
    proj_dim=128,
    n_head=4, n_layer=8,
    ff_mult=4,
    dropout=0.2,
    checkpoint=False,
    attn_type='full',
    attn_kwargs=dict(attn_dropout=0.),
)

# TGIF / UCF
gpt_large = dict(
    model_cls='ImageGPT',
    out_features=1024,
    proj_dim=128,
    n_head=8, n_layer=20,
    ff_mult=4,
    dropout=0.2,
    checkpoint=True,
    attn_type='full',
    attn_kwargs=dict(attn_dropout=0.),
)


configs_str_to_configs = {
    # Legacy resolution-specific configs
    'vae_res64_ds222': vae_res64_ds222,
    'vae_res64_ds422': vae_res64_ds422,
    'vae_res128_ds444': vae_res128_ds444,
    'vae_res256_ds488': vae_res256_ds488,
    'vae_res256_ds444': vae_res256_ds444,
    'vae_res512_ds4_16_16': vae_res512_ds4_16_16,

    # Resolution-agnostic downsample configs (use with any --resolution)
    'vae_ds422': vae_ds422,
    'vae_ds444': vae_ds444,
    'vae_ds488': vae_ds488,
    'vae_ds4_16_16': vae_ds4_16_16,
    'vae_ds888': vae_ds888,
    'vae_ds8_16_16': vae_ds8_16_16,

    # CycleVQVAE configs - Hard cycle loss (Version 1)
    'cycle_vae_hard_ds422': cycle_vae_hard_ds422,
    'cycle_vae_hard_ds444': cycle_vae_hard_ds444,
    'cycle_vae_hard_ds488': cycle_vae_hard_ds488,

    # CycleVQVAE configs - Soft cycle loss (Version 2)
    'cycle_vae_soft_ds422': cycle_vae_soft_ds422,
    'cycle_vae_soft_ds444': cycle_vae_soft_ds444,
    'cycle_vae_soft_ds488': cycle_vae_soft_ds488,

    'gpt_small': gpt_small,
    'gpt_large': gpt_large,

    '': dict(),
}


def config_model(*, configs_str, cond_types, **override_kwargs):
    configs = copy.deepcopy(configs_str_to_configs[configs_str])
    configs.update(override_kwargs)

    model_cls = configs.pop('model_cls')
    model = getattr(models, model_cls)(**configs, cond_types=cond_types)

    configs_to_log = {**configs, 'model_cls': model_cls}
    return model, configs_to_log
