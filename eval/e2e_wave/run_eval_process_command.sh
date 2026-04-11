# WAVEFORM_LENS=(5 7 15 10 13 9 30)
WAVEFORM_LENS=(30)
# CHANNELS=("NOF1" "BCH1" "NCS1")
CHANNELS=("NOF1" "BCH1" "NCS1")

# Define base directories (these remain constant)
BASE_OUTPUT_DIR="/home/cps-tingcong/Documents/GitHub/wave/Eval_Results"
# Loop over channels
for CHANNEL in "${CHANNELS[@]}"; do
    # Loop over waveform lengths
    for WAVEFORM_LEN in "${WAVEFORM_LENS[@]}"; do
        echo "=========================================="
        echo "Processing: Channel=${CHANNEL}, Waveform_Len=${WAVEFORM_LEN}"
        echo "=========================================="
        
        # Construct paths based on current channel and waveform_len
        OUTPUT_CSV="${BASE_OUTPUT_DIR}/${CHANNEL}_waveform_len_${WAVEFORM_LEN}.csv"
        
        # Run the Python script
        /home/cps-tingcong/anaconda3/envs/wave/bin/python \
            "/home/cps-tingcong/Documents/GitHub/wave/WaterMark/Watermark/scripts/process_eval_results.py" \
            --input_csv "${OUTPUT_CSV}" \
            --save_plots \

        
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
