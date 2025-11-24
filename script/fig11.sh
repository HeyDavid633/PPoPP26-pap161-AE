#!/bin/bash
cd ../src


models=("bert_base" "bert_large" "gpt" "t5"  "llama_base" "vit_base")
methods=("TorchNative" "TorchCompile" "ByteTrans" "STOF" "MCFuser")
batch_seq_pairs=("1 128" "8 512" "16 2048")



for bs_seq in "${batch_seq_pairs[@]}"; do
    bs=$(echo $bs_seq | cut -d' ' -f1)
    seq=$(echo $bs_seq | cut -d' ' -f2)
    
    for model in "${models[@]}"; do
        for method in "${methods[@]}"; do
            script -q -a -c "python benchmk_end2end.py \
                --batch_size=$bs \
                --seq_len=$seq \
                --model=$model \
                --method=$method" \
                "../data/End2End_Performance/fig11_all_data.txt"
        done
    done
done
    

cd ../plot/fig11
python fig11_single_device.py \
    --all_data_device="../../data/End2End_Performance/fig11_all_data.txt" \


