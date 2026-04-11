#!/bin/bash

# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73

# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5

# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/gnb/khizar/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/gnb/khizar/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/gnb/khizar/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33

# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5

# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73

# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33
# python eval_vqvae_uve.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33

# python eval_softcast_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel NOF1 --snr-min 0 --snr-max 30 --snr-step 5 
# python eval_softcast_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel BCH1 --snr-min 0 --snr-max 30 --snr-step 5 
# python eval_softcast_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel NCS1 --snr-min 0 --snr-max 30 --snr-step 5 

# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fps 16
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fps 16
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fps 16

# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73 --fps 11
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73 --fps 11
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73 --fps 11

# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33 --fps 5
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33 --fps 5
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33 --fps 5

# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fps 16
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fps 16
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fps 16

# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73 --fps 11
# python eval_mpeg4_uve.py --uve-path /home/khizar /Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73 --fps 11
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73 --fps 11

# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NOF1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33 --fps 5
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel BCH1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33 --fps 5
# python eval_mpeg4_uve.py --uve-path /home/khizar/Datasets/UVE38K/raw/10_sec_clips/ --channel-type uwa --channel NCS1 --modulation QPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33 --fps 5

python test_random_channel.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --channel-type uwa --channel NOF1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33 
python test_random_channel.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --channel-type uwa --channel BCH1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33 
python test_random_channel.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --channel-type uwa --channel NCS1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.33 

python test_random_channel.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --channel-type uwa --channel NOF1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73 
python test_random_channel.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --channel-type uwa --channel BCH1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73 
python test_random_channel.py --ckpt best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar --channel-type uwa --channel NCS1 --modulation BPSK --snr-min 0 --snr-max 30 --snr-step 5 --fec dvbs2_ldpc --fec-rate 0.73 
