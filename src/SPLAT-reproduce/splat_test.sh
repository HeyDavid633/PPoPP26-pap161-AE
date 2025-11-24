#!/bin/bash
BATCH_SIZES=(1 8 16)
SEQ_LENS=(128 256 512 1024 2048 4096)

# Output file
OUTPUT_FILE="../../data/MHA_Performance/splat_benchmark_results_fp16.txt"
EXECUTABLE="./splat_my_test_fp16_exec"

# Check if executable file exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable file $EXECUTABLE does not exist"
    echo "Please compile the program first: make"
    exit 1
fi

echo "=== SPLAT Performance Test Results ===" > $OUTPUT_FILE
echo "Test Time: $(date)" >> $OUTPUT_FILE
echo "==========================================" >> $OUTPUT_FILE
echo "Batch Size | Seq Length | Time (ms/iter)" >> $OUTPUT_FILE
echo "-----------|------------|----------------" >> $OUTPUT_FILE

# Counter
total_tests=$(( ${#BATCH_SIZES[@]} * ${#SEQ_LENS[@]} ))
current_test=0

echo "[SPLAT] Starting performance testing..."
echo "Total test configurations: $total_tests"
echo ""

# Iterate through all configuration combinations
for batch in "${BATCH_SIZES[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        current_test=$((current_test + 1))
        echo "[$current_test/$total_tests] Testing batch_size=$batch, seq_len=$seq_len ..."
        
        # Run test and capture output
        output=$($EXECUTABLE $batch $seq_len 2>&1)
        
        # Append output to results file
        echo "$output" >> $OUTPUT_FILE

    done
done