#!/bin/bash

cd ../src

models=("bert_base" "bert_large" "gpt" "llama_base" "t5" "vit_base")


for model in "${models[@]}"; do
    script -q -a -c "python ablation_onlyMHA.py \
        --batch_size=1 --seq_len=128 \
        --model="$model" \
        --method="STOF" " \
        "../data/Ablation_Study/Ablation-base.txt"
done


for model in "${models[@]}"; do
    script -q -a -c "python ablation_onlyMHA.py \
        --batch_size=8 --seq_len=512 \
        --model="$model" \
        --method="STOF" " \
        "../data/Ablation_Study/Ablation-base.txt"
done


for model in "${models[@]}"; do
    script -q -a -c "python ablation_onlyMHA.py \
        --batch_size=16 --seq_len=2048 \
        --model="$model" \
        --method="STOF" "\
        "../data/Ablation_Study/Ablation-base.txt"

done


for model in "${models[@]}"; do
    script -q -a -c "python ablation_onlyfusion.py \
        --batch_size=1 --seq_len=128 \
        --model="$model" \
        --method="STOF-Compiled" " \
        "../data/Ablation_Study/Ablation-base.txt"        
done

for model in "${models[@]}"; do
    script -q -a -c "python ablation_onlyfusion.py \
        --batch_size=8 --seq_len=512 \
        --model="$model" \
        --method="STOF-Compiled" " \
        "../data/Ablation_Study/Ablation-base.txt"    
done

for model in "${models[@]}"; do
    script -q -a -c "python ablation_onlyfusion.py \
        --batch_size=16 --seq_len=2048 \
        --model="$model" \
        --method="STOF-Compiled" "\
        "../data/Ablation_Study/Ablation-base.txt"    
done

for model in "${models[@]}"; do
    script -q -a -c "python MHA_boundary.py \
        --model="$model" " \
        "../data/Ablation_Study/Ablation-base.txt"
done



cd ../plot/fig12
python fig12.py --file_path0="../../data/End2End_Performance/fig11_all_data.txt" \
                --file_path1="../../data/Ablation_Study/Ablation-base.txt"