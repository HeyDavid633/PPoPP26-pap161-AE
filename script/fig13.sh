# #!/bin/bash

models=("bert_base" "bert_large" "gpt" "llama_base" "t5" "vit_base")

cd ../src

for model in "${models[@]}"; do
    script -q -a -c "python overhead_analysis.py \
        --batch_size=1 \
        --seq_len=128 \
        --model="$model" " \
        "../data/Overhead_Analysis/overhead_analysis_raw.csv"  
done

for model in "${models[@]}"; do
    script -q -a -c "python overhead_analysis.py \
        --batch_size=8 \
        --seq_len=512 \
        --model="$model" " \
        "../data/Overhead_Analysis/overhead_analysis_raw.csv"  
done

for model in "${models[@]}"; do
    script -q -a -c "python overhead_analysis.py \
        --batch_size=16 \
        --seq_len=2048 \
        --model="$model" " \
        "../data/Overhead_Analysis/overhead_analysis_raw.csv"  
done


cd ../plot/fig13
python process_data.py
python fig13.py --file_path1="../../data/Overhead_Analysis/Overhead_analysis.csv"