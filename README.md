# E2E-WAVE

Reproducibility bundle for **E2E-WAVE: End-to-End Waveform Adaptive Encoding for
Underwater Video Multicasting** (SECON 2026).

E2E-WAVE combines VideoGPT tokenization with a trainable complex-valued waveform
bank and a differentiable OFDM channel to transmit low-bitrate video over harsh
underwater acoustic links. This repo contains the training / evaluation code,
the E2E-WAVE eval results used in the paper, and plotting scripts that
regenerate the quantitative figures from those results.

## Repository layout

```
e2e-wave/
├── src/
│   ├── videogpt/            # VQ-VAE + transformer stack (i3d/FVD weights stripped)
│   ├── python_replicate/    # Differentiable OFDM + Watermark channel replay
│   └── pairwise_lambda.py   # LambdaNDCG loss used by the waveform bank
├── training/
│   ├── train_vqvae.py                          # VideoGPT VQ-VAE backbone training
│   ├── test_vqvae.py                           # VQ-VAE sanity eval
│   └── train_wave_bank_watermark_videogpt_full.py
├── eval/
│   ├── e2e_wave/            # Wavebank eval (PSNR / SSIM / L2)
│   │   ├── eval_wave_bank_watermark_videogpt_full.py
│   │   ├── eval_wave_bank_watermark_videogpt_full_random_tokens.py
│   │   ├── process_eval_results.py
│   │   └── run_*.sh
│   └── baselines/           # VQ-VAE+digital, SoftCast, H.265 baselines
├── figures/
│   ├── plot_psnr_ssim.py        # → psnr_bpsk.pdf, ssim_bpsk.pdf, psnr_qpsk.pdf, ssim_qpsk.pdf
│   ├── plot_l2_comparison.py    # → l2_comparison_raw.pdf
│   ├── plot_ber_comparison.py   # → ber_comparison.pdf
│   ├── ber_simo_combined.py     # BER measurement (SIMO OFDM replay)
│   ├── ber_kau1.py              # BER measurement (KAU1 SISO variant)
│   └── consolidate_runs.py      # Builds results/plots/manifest.json
├── results/
│   └── wavebank/            # E2E-WAVE eval CSVs (shipped; see DATA.md to regenerate)
├── checkpoints/
│   ├── vqvae/               # VQ-VAE pretrained backbone (147 MB, Git LFS)
│   └── trained_banks/       # Trained waveform banks for L ∈ {5,7,9,10,13,15,30}
├── scripts/
│   └── env.example.sh       # Copy → env.sh, edit, and `source env.sh`
├── docs/
├── DATA.md                  # Where to get the underlying datasets + channels
├── requirements.txt
└── README.md
```

## Configuration (`env.sh`)

The scripts read paths from environment variables instead of hard-coded
locations. Copy the example and point it at your dataset / checkpoint roots:

```bash
cp scripts/env.example.sh env.sh
$EDITOR env.sh          # set UVE-38K, Watermark channels, etc.
source env.sh
```

The variables used throughout the codebase:

| Variable | Purpose |
|---|---|
| `E2E_WAVE_UVE_DIR` | UVE-38K 10-second clips root (contains `clear/`, `turbid/`) |
| `E2E_WAVE_CHANNELS_DIR` | Watermark channels root (contains `NOF1/`, `NCS1/`, `BCH1/`, ...) |
| `E2E_WAVE_VIDEO_TRAIN_DIR` | Training video corpus (for training a new VQ-VAE or wavebank) |
| `E2E_WAVE_VQVAE_CKPT` | VQ-VAE `.pth.tar` checkpoint (repo ships one) |
| `E2E_WAVE_TRAINED_BANKS_DIR` | Directory holding `cross-entropy_waveform_len_*` bank checkpoints |
| `E2E_WAVE_RESULTS_DIR` | Where new eval CSVs are written (default: `results/wavebank`) |
| `E2E_WAVE_RUNS_DIR` | Training run logs (default: `runs/watermark_videogpt`) |
| `PYTHONPATH` | Must include `src/` and `eval/baselines/` (the example env.sh handles this) |

`env.sh` is gitignored — keep your local paths out of the repo.

## Quickstart — regenerate the figures

Only the E2E-WAVE CSVs are shipped in this repo. The PSNR/SSIM and BER
comparison figures also include baselines (VQ-VAE+digital, SoftCast, H.265)
that you need to rerun yourself — see [Reproducing the baselines](#reproducing-the-baselines)
below.

```bash
# 1. Install deps and set up paths
pip install -r requirements.txt
source env.sh   # see "Configuration" above

# 2. L2 comparison (uses only E2E-WAVE data; reproducible out of the box)
python figures/plot_l2_comparison.py \
    --csv-dir results/wavebank \
    --out figures/l2_comparison_raw.pdf

# 3. PSNR / SSIM figures: build a consolidated manifest, then plot
python figures/consolidate_runs.py \
    --input-dir results \
    --output-dir results/plots
python figures/plot_psnr_ssim.py \
    --plots-dir results/plots \
    --output-dir figures
#  → figures/psnr_clear_bpsk.pdf, psnr_turbid_bpsk.pdf, ssim_*,
#    and the QPSK variants.

# 4. BER comparison
#    ber_simo_combined.py reads channel .mat files from $E2E_WAVE_CHANNELS_DIR
#    and prints BER per channel/modulation; convert its stdout into a CSV
#    matching the schema in figures/plot_ber_comparison.py's docstring.
python figures/ber_simo_combined.py > ber_raw.txt
# ... convert ber_raw.txt → ber_results.csv (channel,modulation,fec,snr_db,ber) ...
python figures/plot_ber_comparison.py --csv ber_results.csv \
    --out figures/ber_comparison.pdf
```

## Reproducing the E2E-WAVE eval results

If you want to regenerate the CSVs under `results/wavebank/` from scratch
(`source env.sh` first):

```bash
bash eval/e2e_wave/run_command.sh                  # raw PSNR/SSIM CSVs
bash eval/e2e_wave/run_eval_process_command.sh     # *_summary.csv files
bash eval/e2e_wave/run_relevance_eval_command.sh   # *_l2_relevance.csv files
```

All three scripts pick up `$E2E_WAVE_UVE_DIR`, `$E2E_WAVE_CHANNELS_DIR`,
`$E2E_WAVE_VQVAE_CKPT`, `$E2E_WAVE_TRAINED_BANKS_DIR`, and write to
`$E2E_WAVE_RESULTS_DIR`.

## Reproducing the baselines

`eval/baselines/` contains the scripts used to produce the VQ-VAE+digital,
SoftCast, and H.265 comparison curves. These are **not included as
pre-computed CSVs** — rerun them against the UVE-38K test set:

```bash
bash eval/baselines/run_baselines.sh
```

This produces sweep runs under `results/vqvae/`, `results/mpeg4_uve/`, and
`results/softcast_uve/`. Then rerun `figures/consolidate_runs.py` to fold
them into the manifest — `figures/plot_psnr_ssim.py` will include them
automatically on the next invocation.

## Training a new VQ-VAE backbone

The VQ-VAE checkpoint we ship under `checkpoints/vqvae/` was trained with
`training/train_vqvae.py` on the UVE-38K corpus at 128×128 resolution with a
1024-entry codebook (4×16×16 latent). To reproduce it from scratch:

```bash
python training/train_vqvae.py \
    --data_path /path/to/uve38k/train \
    --resolution 128 --n_codes 1024 \
    --output_dir runs/vqvae_41616_128x128
```

See the script's `--help` for the full argument list. `training/test_vqvae.py`
runs a quick reconstruction sanity check on a trained checkpoint.

## Training a new waveform bank

```bash
python training/train_wave_bank_watermark_videogpt_full.py \
    --vqvae_ckpt checkpoints/vqvae/vqvae_41616_model_best_128x128.pth_1024.tar \
    --channel NCS1 --waveform_len 9 \
    --video_training_res 128
```

See the script's `--help` for the full argument list. Trained banks for the
seven wavelengths reported in the paper are already under
`checkpoints/trained_banks/`.

## Figures covered

| Paper figure | Script | Status |
|---|---|---|
| `psnr_bpsk.pdf`, `ssim_bpsk.pdf`, `psnr_qpsk.pdf`, `ssim_qpsk.pdf` | `figures/plot_psnr_ssim.py` | Regenerates fully if baselines are rerun; E2E-WAVE curves work out of the box |
| `l2_comparison_raw.pdf` | `figures/plot_l2_comparison.py` | Works out of the box (E2E-WAVE only); digital baselines TODO if needed |
| `ber_comparison.pdf` | `figures/plot_ber_comparison.py` + `figures/ber_simo_combined.py` | Requires Watermark channel .mat files (see DATA.md) |
| `qual_eval.pdf` | — | Out of scope — requires raw reconstructed videos, not staged here |
| `E2E_Wave_Figure2.pdf`, `overview.pdf`, architecture/pipeline diagrams | — | Hand-drawn, not produced by this repo |

## Citation

If you use this code, the trained waveform banks, or the eval results in
your own work, please cite:

```bibtex
@inproceedings{e2ewave2026,
  title     = {E2E-WAVE: End-to-End Waveform Adaptive Encoding for
               Underwater Video Multicasting},
  author    = {Anjum, Khizar and Jiang, Tingcong and Pompili, Dario},
  booktitle = {Proceedings of the 23rd IEEE International Conference on
               Sensing, Communication, and Networking (SECON)},
  year      = {2026}
}
```

Please also cite the Watermark benchmark whose channel soundings this work
depends on:

```bibtex
@article{van2017watermark,
  title   = {The watermark benchmark for underwater acoustic modulation schemes},
  author  = {van Walree, Paul A and Socheleau, Fran{\c{c}}ois-Xavier and Otnes, Roald and Jenserud, Trond},
  journal = {IEEE Journal of Oceanic Engineering},
  volume  = {42},
  number  = {4},
  pages   = {1007--1018},
  year    = {2017},
  publisher = {IEEE}
}
```

and the UVE-38K video corpus: <https://github.com/TrentQiQ/UVE-38K>.

## License

See `LICENSE` (to be added before publication).
