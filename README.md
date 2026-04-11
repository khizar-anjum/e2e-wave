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
├── docs/
├── DATA.md                  # Where to get the underlying datasets + channels
├── requirements.txt
└── README.md
```

## Quickstart — regenerate the figures

Only the E2E-WAVE CSVs are shipped in this repo. The PSNR/SSIM and BER
comparison figures also include baselines (VQ-VAE+digital, SoftCast, H.265)
that you need to rerun yourself — see [Reproducing the baselines](#reproducing-the-baselines)
below.

```bash
# 1. Install deps
pip install -r requirements.txt
pip install -e .     # not required; add src/ to PYTHONPATH instead
export PYTHONPATH="$PWD/src:$PYTHONPATH"

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
#    Step 4a: run BER sweeps on the Watermark channels (see below for
#             prerequisites). ber_simo_combined.py prints BER per
#             channel/modulation; redirect to a CSV matching the schema in
#             figures/plot_ber_comparison.py's docstring.
python figures/ber_simo_combined.py > ber_raw.txt
# ... convert ber_raw.txt into ber_results.csv (channel,modulation,fec,snr_db,ber) ...
python figures/plot_ber_comparison.py --csv ber_results.csv \
    --out figures/ber_comparison.pdf
```

## Reproducing the E2E-WAVE eval results

If you want to regenerate the CSVs under `results/wavebank/` from scratch:

```bash
# Edit eval/e2e_wave/run_command.sh to point at your local paths:
#   VIDEO_DIR    — UVE-38K 10-second clips (see DATA.md)
#   VQVAE_CKPT   — checkpoints/vqvae/vqvae_41616_model_best_128x128.pth_1024.tar
#   BANK_CKPT    — checkpoints/trained_banks/cross-entropy_waveform_len_${L}_.../best_ssim_bank.pth
#   BASE_OUTPUT_DIR → results/wavebank
bash eval/e2e_wave/run_command.sh
bash eval/e2e_wave/run_eval_process_command.sh         # builds *_summary.csv files
bash eval/e2e_wave/run_relevance_eval_command.sh       # builds *_l2_relevance.csv files
```

## Reproducing the baselines

`eval/baselines/` contains the scripts used to produce the VQ-VAE+digital,
SoftCast, and H.265 comparison curves. These are **not included as
pre-computed CSVs** in this repo — rerun them against the UVE-38K test set
(DATA.md) to obtain:

- `results/vqvae/uve_eval_{NOF1,NCS1,BCH1}_dvbs2_ldpc_snr_sweep_.../sweep_summary.json`
- `results/mpeg4_uve/uve_mpeg4_{NOF1,NCS1,BCH1}_.../sweep_summary.json`
- `results/softcast_uve/uve_softcast_{NOF1,NCS1,BCH1}_.../sweep_summary.json`

Then rerun `figures/consolidate_runs.py` to pick them up into the manifest.
`figures/plot_psnr_ssim.py` will include them automatically.

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
