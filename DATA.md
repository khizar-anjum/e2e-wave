# Datasets and channel data

This repo ships the E2E-WAVE evaluation CSVs and the trained model
checkpoints, but **not** the underlying video corpus or the Watermark channel
recordings — those have to be downloaded separately due to size and licensing.

## 1. Watermark channel recordings

The Watermark benchmark dataset (Socheleau et al., 2016) provides
Time-Varying Impulse Responses from at-sea underwater acoustic measurements.
We use five channels: `NOF1`, `NCS1`, `BCH1`, `KAU1`, `KAU2`.

- **Source:** Van Walree et al., *The Watermark Benchmark for Underwater
  Acoustic Modulation Schemes*, IEEE J. Oceanic Engineering 42(4):1007–1018,
  2017. The `.mat` channel soundings are distributed by FFI / CMRE — see
  the paper for the current download instructions.
- **Citation (BibTeX):**
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
- **Expected layout** after download (what `python_replicate` expects):
  ```
  channels/
  ├── NOF1/
  │   ├── mat/NOF1_001.mat
  │   └── png/...
  ├── NCS1/ ...
  ├── BCH1/ ...
  ├── KAU1/ ...
  └── KAU2/ ...
  ```
- **Size:** ~750 MB total.
- **Used by:** `eval/e2e_wave/*.py`, `figures/ber_simo_combined.py`,
  `figures/ber_kau1.py`, `training/train_wave_bank_watermark_videogpt_full.py`.

Point the eval / training scripts at the root of this tree via the
`--channel` argument (the scripts currently hard-code the base path — see
`src/python_replicate/channel_replay.py::load_channel_sounding`).

## 2. UVE-38K video corpus (eval set)

10-second underwater video clips used for PSNR/SSIM/L2 evaluation.

- **Source:** <https://github.com/TrentQiQ/UVE-38K> (follow the instructions
  in that repo to obtain the 10-second clips).
- **Expected layout:**
  ```
  10_sec_clips/
  ├── clear/
  │   ├── coral_clip.mp4
  │   └── ...
  └── turbid/
      ├── reef_clip.mp4
      └── ...
  ```
- **Used by:** `eval/e2e_wave/eval_wave_bank_watermark_videogpt_full.py` via
  the `--video_dir` argument, and all baseline evaluators in
  `eval/baselines/`.

## 3. Training corpus

For training a new waveform bank, we use the same UVE-38K video corpus. No
separate dataset is required.

## Re-pointing the scripts

Most scripts accept CLI flags for the relevant paths:

```
eval/e2e_wave/run_command.sh                  # edit VIDEO_DIR, VQVAE_CKPT, BANK_CKPT, BASE_OUTPUT_DIR
eval/e2e_wave/run_eval_process_command.sh     # edit BASE_OUTPUT_DIR
eval/e2e_wave/run_relevance_eval_command.sh   # edit BASE_OUTPUT_DIR, VQVAE_CKPT, BANK_CKPT
```

For the BER scripts in `figures/`, the channel path is currently hard-coded
near the top of each file (`CHANNEL_PATH = ...`). Edit to match your local
layout before running.
