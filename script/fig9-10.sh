# #!/bin/bash
cd ../src

# mask_ids=(0)
mask_ids=(0 1 2 3)
for mask_id in "${mask_ids[@]}"; do
    script -c "python benchmk_attn_unified.py --mask_id=$mask_id"   \
        ../data/MHA_Performance/fig_9_10_mask_${mask_id}.txt
done

cd SPLAT-reproduce
bash splat_test.sh
cd ../

cd ../plot/fig9-10
python fig9-10.py --file_path1="../../data/MHA_Performance/fig_9_10_mask_0.txt" \
                    --file_path2="../../data/MHA_Performance/fig_9_10_mask_1.txt" \
                    --file_path3="../../data/MHA_Performance/fig_9_10_mask_2.txt" \
                    --file_path4="../../data/MHA_Performance/fig_9_10_mask_3.txt" \
                    --splat_file_path="../../data/MHA_Performance/splat_benchmark_results_fp16.txt"