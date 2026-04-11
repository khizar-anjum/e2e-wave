WAVEFORM_LENS=(5 7 15 10 13 30 9)
CHANNELS=("NOF1" "BCH1" "NCS1")

# Define base directories (these remain constant)
BASE_OUTPUT_DIR="/home/cps-tingcong/Documents/GitHub/wave/Eval_Results"
VIDEO_DIR="/home/cps-tingcong/10_sec_clips"
VQVAE_CKPT="/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/VideoGPT Implementation/best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar"
SNR_MIN=0
SNR_MAX=30
SNR_STEP=5

# Loop over channels
for CHANNEL in "${CHANNELS[@]}"; do
    # Loop over waveform lengths
    for WAVEFORM_LEN in "${WAVEFORM_LENS[@]}"; do
        echo "=========================================="
        echo "Processing: Channel=${CHANNEL}, Waveform_Len=${WAVEFORM_LEN}"
        echo "=========================================="
        
        # Construct paths based on current channel and waveform_len
        OUTPUT_CSV="${BASE_OUTPUT_DIR}/${CHANNEL}_waveform_len_${WAVEFORM_LEN}_l2_relevance.csv"
        BANK_CKPT="/home/cps-tingcong/Documents/GitHub/wave/runs/watermark_videogpt/cross-entropy_waveform_len_${WAVEFORM_LEN}_videogpt_4_16_16_1024_train_NCS1_eval_NCS1_temperature_0.01_top_5_video_training_res_128/best_ssim_bank.pth"
        
        # Run the Python script
        /home/cps-tingcong/anaconda3/envs/wave/bin/python \
            "/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/VideoGPT Implementation/eval_wave_bank_watermark_videogpt_full_random_tokens.py" \
            --vqvae_ckpt "${VQVAE_CKPT}" \
	    --bank_ckpt "${BANK_CKPT}" \
	    --channel "${CHANNEL}" \
	    --waveform_len ${WAVEFORM_LEN} \
	    --snr_min ${SNR_MIN} --snr_max ${SNR_MAX} --snr_step ${SNR_STEP} \
	    --num_clips 500 \
	    --batch_size 4 \
	    --output_csv "${OUTPUT_CSV}"


        
        # Check if the Python script succeeded
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed: Channel=${CHANNEL}, Waveform_Len=${WAVEFORM_LEN}"
        else
            echo "✗ Error occurred for: Channel=${CHANNEL}, Waveform_Len=${WAVEFORM_LEN}"
        fi
        
        echo ""
    done
done

echo "=========================================="
echo "All processing complete!"
echo "=========================================="
