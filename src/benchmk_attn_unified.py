
import sys
import os
import argparse
import torch
import math
from einops import rearrange, repeat
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel                 #  FlashAttn2
from torch.nn.attention.flex_attention import flex_attention           #  FlexAttn
from ops.package_op import binding_attn_func  # Our kernel

from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, time_stamp_cudasync, transpose_for_scores
from util.masks import generate_causal_mask, generate_dilated_mask, generate_sliding_mask, generate_longformer_mask, generate_bigbird_mask, get_sparse_storage, get_OuterTile_storage
from util.masks import create_block_mask_cached, flex_bigbird_mask, flex_causal_mask, flex_sliding_window_mask, flex_longformer_mask
import random


import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
from functools import partial

def torch_attn_std(q, k, v, dropout_p=0.0, causal=True):
    """
    Arguments:
        q, k, v: (batch_size, seq_len, head_num, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seq_len, head_num, head_dim)
    """
    batch_size, seq_len, head_num, d = q.shape
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    
    scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=head_num)
    
    if causal:
        causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=q.dtype)


def check_tensor(other_output, torch_output):
    max_diff = (other_output - torch_output).abs().max().item()
    mean_diff = (other_output - torch_output).abs().mean().item()
    return max_diff, mean_diff



import itertools


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bhkd->bhqd', attention_weights, V)
    return output

class GPUParams:
    def __init__(self, gpu_type="4090"):
        if gpu_type == "A100":
            self.sm_count = 108
            self.sm_shared = 164 * 1024 
            self.max_threads_per_sm = 2048
            self.max_warps_per_sm = 64
            self.l2_cache = 40 * 1024 * 1024
            self.tensor_core_count = self.sm_count * 4
        else:  
            self.sm_count = 128
            self.sm_shared = 100 * 1024
            self.max_threads_per_sm = 1536
            self.max_warps_per_sm = 48
            self.l2_cache = 72 * 1024 * 1024 
            self.tensor_core_count = self.sm_count * 4 

        self.register_file_size = 256 * 1024
        self.max_blocks_per_sm = 32 


BLOCK_WISE_CONFIGS =  [
    # num_warps=1  11
    {'num_warps': 1, 'block_m': 16, 'block_n': 16},
    {'num_warps': 1, 'block_m': 32, 'block_n': 16},
    {'num_warps': 1, 'block_m': 64, 'block_n': 16},
    {'num_warps': 1, 'block_m': 16, 'block_n': 32},
    {'num_warps': 1, 'block_m': 32, 'block_n': 32},
    {'num_warps': 1, 'block_m': 64, 'block_n': 32},
    {'num_warps': 1, 'block_m': 16, 'block_n': 64},
    {'num_warps': 1, 'block_m': 32, 'block_n': 64},
    {'num_warps': 1, 'block_m': 64, 'block_n': 64},
    {'num_warps': 1, 'block_m': 16, 'block_n': 128},
    {'num_warps': 1, 'block_m': 32, 'block_n': 128},
    # num_warps=2  10
    {'num_warps': 2, 'block_m': 32, 'block_n': 16},
    {'num_warps': 2, 'block_m': 64, 'block_n': 16},
    {'num_warps': 2, 'block_m': 16, 'block_n': 32},
    {'num_warps': 2, 'block_m': 32, 'block_n': 32},
    {'num_warps': 2, 'block_m': 64, 'block_n': 32},
    {'num_warps': 2, 'block_m': 16, 'block_n': 64},
    {'num_warps': 2, 'block_m': 32, 'block_n': 64},
    {'num_warps': 2, 'block_m': 64, 'block_n': 64},
    {'num_warps': 2, 'block_m': 16, 'block_n': 128},
    {'num_warps': 2, 'block_m': 32, 'block_n': 128},
    # num_warps=4  8 
    {'num_warps': 4, 'block_m': 64, 'block_n': 16},
    {'num_warps': 4, 'block_m': 32, 'block_n': 32},
    {'num_warps': 4, 'block_m': 64, 'block_n': 32},
    {'num_warps': 4, 'block_m': 16, 'block_n': 64},
    {'num_warps': 4, 'block_m': 32, 'block_n': 64},
    {'num_warps': 4, 'block_m': 64, 'block_n': 64},
    {'num_warps': 4, 'block_m': 16, 'block_n': 128},
    {'num_warps': 4, 'block_m': 32, 'block_n': 128},
    # num_warps=8  5
    {'num_warps': 8, 'block_m': 64, 'block_n': 32},
    {'num_warps': 8, 'block_m': 32, 'block_n': 64},
    {'num_warps': 8, 'block_m': 64, 'block_n': 64},
    {'num_warps': 8, 'block_m': 16, 'block_n': 128},
    {'num_warps': 8, 'block_m': 32, 'block_n': 128},
    # num_warps=16  2
    {'num_warps': 16, 'block_m': 64, 'block_n': 64},
    {'num_warps': 16, 'block_m': 32, 'block_n': 128},
]



def phase1_decision(load_row_ptr, seq_len):
    base_block = 16
    total_blocks = (seq_len // base_block) ** 2
    load_blocks = load_row_ptr[-1].item() 
    
    rho = load_blocks / total_blocks
    tau = 1 / (math.log2(seq_len / 16) ** 2)
    
    print(f"valid block ratio: {rho:.3f}, threadhold: {tau:.3f}")
    return rho >= tau


def compute_shared_mem(config):
    blk_m, blk_n = config['block_m'], config['block_n']
    padding = 16 
    
    q   = blk_m * (64 + padding) * 2     
    kv  = blk_n * (64 + padding) * 2
    acc = blk_m * (blk_n + padding) * 2
    res = blk_m * (64 + padding) * 2
    meta = 4 * blk_m * 4
    
    return q + kv + acc + res + meta


def phase2_selection(configs, gpu, seq_len, batch_size, head_num):
    WARP_SIZE = 32
    valid_configs = []
    
    for cfg in configs:
        req_smem = compute_shared_mem(cfg)
        if req_smem > gpu.sm_shared:
            continue
                    

        blocks_by_smem = gpu.sm_shared // req_smem
        blocks_by_threads = gpu.max_warps_per_sm // cfg['num_warps']
        blocks_per_sm = min(blocks_by_smem, blocks_by_threads)
        
        active_warps = blocks_per_sm * cfg['num_warps']
        occupancy = active_warps / gpu.max_warps_per_sm
        
        tc_util = (cfg['block_m']//16) * (cfg['block_n']//16) 
        block_area = cfg['block_m'] * cfg['block_n']
        compute_density = tc_util / (block_area ** 1.5)
        
        grid_size = gpu.sm_count * (seq_len//cfg['block_m']) * batch_size / block_area
        parallel_potential = batch_size * seq_len * head_num * math.sqrt(grid_size)
        
        score = occupancy * parallel_potential
        
        valid_configs.append( (cfg, score) )
    
    valid_configs.sort(key=lambda x: x[1], reverse=True)
    return valid_configs[0][0] if valid_configs else None


def select_operator(gpu_params, load_row_ptr, batch_size, seq_len, head_num):
    if phase1_decision(load_row_ptr, seq_len):
        best_config = phase2_selection(BLOCK_WISE_CONFIGS, gpu_params, seq_len, batch_size, head_num)
        
        if best_config:
            return f"block-wise warp{best_config['num_warps']}m{best_config['block_m']}n{best_config['block_n']}"
    return "row-wise"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--head_num', type=int, default=12, help='Number of heads (default: 12)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length (default: 256)')
    parser.add_argument('--mask_id', type=int, default=0, help='Mask type: 1-Sliding | 2-Longformer | 3-BigBird | 4-Dilated (default: 1)')
    parser.add_argument('--block_m', type=int, default=64, help='Block Size of M (default:16)')
    parser.add_argument('--block_n', type=int, default=64, help='Block Size of N (default:16)')
    parser.add_argument('--print_flag', type=bool, default=True, help='wether to print info')
    parser.add_argument('--num_warps', type=int, default=1, help='Warp Num to launch (default:1)')
    parser.add_argument('--config_path', type=str, default='./', help='Path to the input txt file.')
    args = parser.parse_args() 
    
    batch_size = args.batch_size
    head_num   = args.head_num
    head_size  = args.head_size
    seq_len    = args.seq_len
    mask_id    = args.mask_id
    BLOCK_M    = args.block_m
    BLOCK_N    = args.block_n
    print_flag = args.print_flag
    num_warps  = args.num_warps
    config_path = args.config_path

    if(num_warps > (BLOCK_M/16) * (BLOCK_N/16)):
        print(f"num_warps: {num_warps}, (BLOCK_M/16) * (BLOCK_N/16): {int(BLOCK_M / 16) * int(BLOCK_N / 16)}")
        print("Error! Here should be: num_warps <= (BLOCK_M/16) * (BLOCK_N/16) !")
        exit(0)
        
        
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    device = torch_cuda_identify(print_info = print_flag)
    torch._dynamo.config.cache_size_limit = 64
    
    
    is_4080_laptop = False
    is_4090 = False
    is_A100 = False
    gpu_name = torch.cuda.get_device_name()
    if "NVIDIA GeForce RTX 4080 Laptop GPU" in gpu_name:
        is_4080_laptop = True
    if "NVIDIA GeForce RTX 4090" in gpu_name:
        is_4090 = True
    if "NVIDIA A100-PCIE-40GB" in gpu_name:
        is_A100 = True
        
        
    if(mask_id == 1):
        mask_name = 'Sliding_Mask'
    elif(mask_id == 2):
        mask_name = 'Longformer_Mask'
    elif(mask_id == 3):
        mask_name = 'BigBird_Mask'
    elif(mask_id == 4):
        mask_name = 'Dilated_Mask'
    elif(mask_id == 0):
        mask_name = 'Causal_Mask'
        
    if print_flag:
        print(f' [Benchmark] Attention unified benchmark for {mask_name}')
    
    
    torch.set_printoptions(profile="default")
    torch.set_printoptions(precision=3, sci_mode=False) 
    
    data_type  = torch.float16
    running_device = "cuda"
    sqrt_seq_len = int(math.sqrt(seq_len))
    fill_rate    = 0.1
    dropout_p = 0.0
    
    warmup_iters = 10
    running_iters = 20
    
    

    for loop1 in [1, 8, 16]:
        for loop2 in [128, 256, 512, 1024, 2048, 4096]:
            
            torch.cuda.empty_cache()
            
            batch_size = loop1
            seq_len = loop2
    
            test_FlexAttn  = True
            test_TVM       = False
            test_FlashAttn = True
            test_Torch     = True
            test_ByteTrans = True
            test_tilelang  = True
             
           
            if is_4080_laptop == True:
                if(batch_size == 8 and seq_len >= 4096): 
                    print("4080-laptop unsupport ! error")
                    continue
                if(batch_size == 16 and seq_len >= 2048):
                    print("4080-laptop unsupport ! error")
                    continue
                test_FlexAttn = False
                test_TVM      = False
            


            if(test_ByteTrans):
                hidden_dim=head_num*head_size
                dtype="fp16"
                hidden_states  = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
                qkv          = set_dtype(torch.zeros(batch_size, seq_len, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) 
                qkv_bias       = set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) 

            qkv_chunk = qkv + qkv_bias
            query, key, value = qkv_chunk.chunk(3, dim=-1)
            query1 = transpose_for_scores(query, head_num, head_size)  
            key1 = transpose_for_scores(key, head_num, head_size)  
            value1 = transpose_for_scores(value, head_num, head_size)
            
            query = query1.permute(0, 2, 1, 3)
            key   = key1.permute(0, 2, 1, 3)
            value = value1.permute(0, 2, 1, 3) 

            
            avg_seq_len = seq_len
            low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
            input_lens = torch.randint(low=low, high=high, size=(batch_size,))
            seqlen_mask = seqlen_to_mask(input_lens, seq_len)
            attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")
            
            mask_mod = None
            score_mod = None
            
            if(mask_id == 0):
                is_causal = True
                mask_name = 'Causal_Mask'
                mask_mod = flex_causal_mask
                mask = generate_causal_mask(attr_mask).cuda()
            elif(mask_id == 1):
                is_causal = True
                mask_name = 'Sliding_Mask'
                mask_mod = flex_sliding_window_mask
                mask = generate_sliding_mask(attr_mask, bandwidth=BLOCK_M, is_cauasl=True).cuda()
            elif(mask_id == 2):
                is_causal = False
                mask_name = 'Longformer_Mask'
                mask_mod = flex_longformer_mask
                mask = generate_longformer_mask(attr_mask, globalwidth=32, bandwidth=32, is_cauasl=is_causal).cuda()
            elif(mask_id == 3):
                is_causal = False
                mask_name = 'BigBird_Mask'
                mask_mod = flex_bigbird_mask
                mask = generate_bigbird_mask(attr_mask, globalwidth=32, bandwidth=32, fill_rate=fill_rate, is_cauasl=is_causal).cuda()
            elif(mask_id == 4):
                is_causal = False
                mask_name = 'Dilated_Mask'
                mask_mod = flex_sliding_window_mask
                mask = generate_dilated_mask(attr_mask, bandwidth=BLOCK_M, dilation_rate=1, is_cauasl=True).cuda()

            nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, BLOCK_M, BLOCK_N)

            
            
            # FlashAttn2  ----------------------------------------
            if(test_FlashAttn):
                for i in range(warmup_iters + running_iters):
                    if i == warmup_iters:    
                        t2_start = time_stamp_cudasync()
                    
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        FA2_output = torch.nn.functional.scaled_dot_product_attention(query1, key1, value1, is_causal=is_causal)
                    
                    h = FA2_output.permute(0, 2, 1, 3).contiguous() 

                t2_end = time_stamp_cudasync()
                Flashattn_time = (t2_end - t2_start) * 1000 / running_iters
                print(" bs:{} | h_num:{} | seq:{}  |  FlashAttn2  : {:.3f} ms / iter".format(batch_size, head_num, seq_len, Flashattn_time)) 
            
        

            # PyTorch Naive  ---------------------------------------
            if(test_Torch):
                for i in range(warmup_iters + running_iters):
                    if i == warmup_iters:    
                        t0_start = time_stamp_cudasync()
                        
                    torch_output = torch_attn_std(query, key, value, dropout_p=dropout_p, causal=is_causal)

                t0_end = time_stamp_cudasync()
                base_time = (t0_end - t0_start) * 1000 / running_iters
                print(" bs:{} | h_num:{} | seq:{}  |  Torch Naive : {:.3f} ms / iter".format(batch_size, head_num, seq_len, base_time)) 
            

            # FlexAttn  ---------------------------------------
            if(test_FlexAttn):
                compiled_flex_attention = torch.compile(flex_attention, mode="default", dynamic=False)
                block_mask = create_block_mask_cached(mask_mod, 1, 1, seq_len, seq_len, device=query.device)
                for i in range(warmup_iters + running_iters):
                    if i == warmup_iters:    
                        t3_start = time_stamp_cudasync()
            
                    flex_output = compiled_flex_attention(query1, key1, value1, score_mod=score_mod, block_mask=block_mask)
                    flex_output1 = flex_output.permute(0, 2, 1, 3)
                    
                t3_end = time_stamp_cudasync()
                flexattn_time = (t3_end - t3_start) * 1000 / running_iters
                print(" bs:{} | h_num:{} | seq:{}  |   FlexAttn   : {:.3f} ms / iter".format(batch_size, head_num, seq_len, flexattn_time)) 
                
                            
            # ByteTransformer --------------------------------------- 
            if(test_ByteTrans):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, os.path.join(current_dir, "ByteTransformer"))
                from ByteTransformer.ops.package_op import bytetr_attn_op, bytetr_longattn_op
                mask=mask.half()
                bytetr_attn_pybind_op = None
                    
                if seq_len<=256:
                    bytetr_attn_pybind_op = bytetr_attn_op
                elif 256<seq_len<=1024:
                    bytetr_attn_pybind_op = bytetr_longattn_op
                else:
                    test_ByteTrans = False
                
                if(test_ByteTrans):
                    for i in range(warmup_iters + running_iters):                
                        if i == warmup_iters:
                            t_byte_start = time_stamp_cudasync()
                            
                        ByteTransformer_output = bytetr_attn_pybind_op(qkv,qkv_bias,mask,head_num)
                        ByteTransformer_output_4d =  ByteTransformer_output.view(batch_size, seq_len, head_num, head_size).permute(0, 2, 1, 3).contiguous()
                        
                    t_byte_end = time_stamp_cudasync()
                
                    bytekernel_time = (t_byte_end - t_byte_start) * 1000 / running_iters
                    print(" bs:{} | h_num:{} | seq:{}  |  ByteTrans   : {:.3f} ms / iter".format(batch_size, head_num, seq_len, bytekernel_time))
                            
            
            # # Our Kernel ------------------------------------
            for i in range(warmup_iters + running_iters):
                if i == warmup_iters:    
                    t_start = time_stamp_cudasync()
                binding_out = binding_attn_func(query, key, value,
                                        full_row_ptr, full_col_idx, 
                                        part_row_ptr, part_col_idx, inner_bitmaps, 
                                        load_row_ptr, load_col_idx, 
                                        dropout_p=dropout_p, causal=is_causal)     
            t_end = time_stamp_cudasync()
            ourkernel_time = (t_end - t_start) * 1000 / running_iters
            print(" bs:{} | h_num:{} | seq:{}  |  Our Kernel  : {:.3f} ms / iter\n".format(batch_size, head_num, seq_len, ourkernel_time))
            
