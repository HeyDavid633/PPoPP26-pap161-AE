
models=("bert_base" "bert_large" "gpt" "llama_base" "t5" "vit_base")

cd ../src

for model in "${models[@]}"; do
    script -q -a -c "python tuning_STOF_cost.py \
        --method="STOF" \
        --batch_size=1 \
        --seq_len=128 \
        --model="$model" " \
        "../data/Tuning_Cost/Tuning_cost.csv"
done

for model in "${models[@]}"; do
    script -q -a -c "python tuning_STOF_cost.py \
        --method="STOF" \
        --batch_size=8 \
        --seq_len=512 \
        --model="$model" "\
        "../data/Tuning_Cost/Tuning_cost.csv"
done

for model in "${models[@]}"; do
    script -q -a -c "python tuning_STOF_cost.py \
        --method="STOF" \
        --batch_size=16 \
        --seq_len=2048 \
        --model="$model" "\
        "../data/Tuning_Cost/Tuning_cost.csv"
done