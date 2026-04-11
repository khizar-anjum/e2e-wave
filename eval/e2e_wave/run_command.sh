## waveform_len_list = [5, 7, 15, 10, 13, 30, 9]
#!/bin/bash
# Define base directories
# BASE_OUTPUT_DIR="/home/cps-tingcong/Documents/GitHub/wave/Eval_Results"
# VIDEO_DIR="/home/cps-tingcong/10_sec_clips"
# VQVAE_CKPT="/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/VideoGPT Implementation/best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar"


# # Set waveform_len and channel as a variable
# WAVEFORM_LEN=5
# CHANNEL="NCS1"

# # Construct paths based on waveform_len
# BANK_CKPT="/home/cps-tingcong/Documents/GitHub/wave/runs/watermark_videogpt/cross-entropy_waveform_len_${WAVEFORM_LEN}_videogpt_4_16_16_1024_train_NCS1_eval_NCS1_temperature_0.01_top_5_video_training_res_128/best_ssim_bank.pth"
# OUTPUT_CSV="${BASE_OUTPUT_DIR}/${CHANNEL}_waveform_len_${WAVEFORM_LEN}.csv"
# OUTPUT_DIR="${BASE_OUTPUT_DIR}/videos/${CHANNEL}_waveform_len_${WAVEFORM_LEN}_videos"

# # # Create output directory if it doesn't exist
# # mkdir -p "${OUTPUT_DIR}"

# # Run the Python script
# /home/cps-tingcong/anaconda3/envs/wave/bin/python \
#     "/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/VideoGPT Implementation/eval_wave_bank_watermark_videogpt_full.py" \
#     --video_dir "${VIDEO_DIR}" \
#     --vqvae_ckpt "${VQVAE_CKPT}" \
#     --bank_ckpt "${BANK_CKPT}" \
#     --channel "${CHANNEL}" \
#     --waveform_len ${WAVEFORM_LEN} \
#     --output_csv "${OUTPUT_CSV}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --snr_min 0 \
#     --snr_max 30 \
#     --snr_step 5 \
#     --batch_size 8 \
#     --max_clips 4000

# WAVEFORM_LENS=(5 7 15 10 13 30 9)
WAVEFORM_LENS=(9)
# CHANNELS=("NOF1" "BCH1" "NCS1")
CHANNELS=("NOF1")

# Define base directories (these remain constant)
BASE_OUTPUT_DIR="/home/cps-tingcong/Documents/GitHub/wave/Eval_Results"
VIDEO_DIR="/home/cps-tingcong/10_sec_clips/turbid"
VQVAE_CKPT="/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/VideoGPT Implementation/best_ckpts/vqvae_41616_model_best_128x128.pth_1024.tar"

# Loop over channels
for CHANNEL in "${CHANNELS[@]}"; do
    # Loop over waveform lengths
    for WAVEFORM_LEN in "${WAVEFORM_LENS[@]}"; do
        echo "=========================================="
        echo "Processing: Channel=${CHANNEL}, Waveform_Len=${WAVEFORM_LEN}"
        echo "=========================================="
        
        # Construct paths based on current channel and waveform_len
        BANK_CKPT="/home/cps-tingcong/Documents/GitHub/wave/runs/watermark_videogpt/cross-entropy_waveform_len_${WAVEFORM_LEN}_videogpt_4_16_16_1024_train_NCS1_eval_NCS1_temperature_0.01_top_5_video_training_res_128/best_ssim_bank.pth"
        # OUTPUT_CSV="${BASE_OUTPUT_DIR}/${CHANNEL}_waveform_len_${WAVEFORM_LEN}.csv"
        OUTPUT_CSV="${BASE_OUTPUT_DIR}/useless.csv"
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/videos/${CHANNEL}_waveform_len_${WAVEFORM_LEN}_videos"
        
        # Create output directory if it doesn't exist
        mkdir -p "${OUTPUT_DIR}"
        
        # Check if bank checkpoint exists
        if [ ! -f "${BANK_CKPT}" ]; then
            echo "WARNING: Bank checkpoint not found: ${BANK_CKPT}"
            echo "Skipping this configuration..."
            continue
        fi
        
        # Run the Python script
        /home/cps-tingcong/anaconda3/envs/wave/bin/python \
            "/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/VideoGPT Implementation/eval_wave_bank_watermark_videogpt_full.py" \
            --video_dir "${VIDEO_DIR}" \
            --vqvae_ckpt "${VQVAE_CKPT}" \
            --bank_ckpt "${BANK_CKPT}" \
            --channel "${CHANNEL}" \
            --waveform_len ${WAVEFORM_LEN} \
            --output_csv "${OUTPUT_CSV}" \
            --output_dir "${OUTPUT_DIR}" \
            --snr_min 30 \
            --snr_max 30 \
            --snr_step 5 \
            --batch_size 4 \
            --max_clips 300 \
            --save_videos
        
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