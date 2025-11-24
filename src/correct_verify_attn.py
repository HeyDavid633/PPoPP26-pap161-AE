
import sys
import os
import argparse
import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from ops.package_op import binding_attn_func                  # Our kernel
from torch.nn.attention import SDPBackend, sdpa_kernel        # FlashAttn2
# import ct_mcfuser_mask                                        # MCFuser
from torch.nn.attention.flex_attention import flex_attention  # FlexAttn

from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, transpose_for_scores
from util.masks import generate_causal_mask, get_sparse_storage, get_OuterTile_storage
from util.masks import create_block_mask_cached, flex_causal_mask

def torch_attn_std(q, k, v, dropout_p=0.0, causal=True):
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


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    device = torch_cuda_identify(print_info = True)
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
   
    parser = argparse.ArgumentParser(description="Give the parameters for the attention test (with Mask)")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--head_num', type=int, default=12, help='Number of heads (default: 12)')
    parser.add_argument('--head_size', type=int, default=64, help='Head size (default: 64)')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length (default: 256)')
    parser.add_argument('--mask_id', type=int, default=0, help='Mask type: 0-Casual (default: 0)')
    parser.add_argument('--block_m', type=int, default=32, help='Block Size of M (default:32)')
    parser.add_argument('--block_n', type=int, default=32, help='Block Size of N (default:32)')
    parser.add_argument('--num_warps', type=int, default=1, help='Warp Num to launch (default:1)')
    args = parser.parse_args() 
    
    batch_size = args.batch_size
    head_num   = args.head_num
    head_size  = args.head_size
    seq_len    = args.seq_len
    mask_id    = args.mask_id
    BLOCK_M    = args.block_m
    BLOCK_N    = args.block_n
    num_warps  = args.num_warps
    dropout_p  = 0.0
    
    if(num_warps > (BLOCK_M/16) * (BLOCK_N/16)):
        print(f"num_warps: {num_warps}, (BLOCK_M/16) * (BLOCK_N/16): {int(BLOCK_M / 16) * int(BLOCK_N / 16)}")
        print("Error! Here should be: num_warps <= (BLOCK_M/16) * (BLOCK_N/16) !")
        exit(0)
    
    data_type  = torch.float16
    running_device = "cuda"
    hidden_dim=head_num*head_size
    dtype="fp16"
    
    test_FlexAttn  = True
    test_FlashAttn = True
    test_Torch     = True
    test_ByteTrans = True
    
    if(is_4080_laptop):
        test_FlexAttn = False
    
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
    seq_len_mask = seqlen_to_mask(input_lens, seq_len)
    attr_mask   = set_dtype(torch.tile(seq_len_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")
    
    mask_mod = None
    score_mod = None
    
    if(mask_id == 0):
        is_causal = True
        mask_name = 'Causal_Mask'
        mask_mod = flex_causal_mask
        mask = generate_causal_mask(attr_mask).cuda()
        
    
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, part_block_mask, load_row_ptr, load_col_idx = get_sparse_storage(mask, BLOCK_M, BLOCK_N)
    
    print(f"Correctness Verification for Attention with Mask ({mask_name}) ... ...")
    
    # PyTorch Naive  ---------------------------------------
    torch_output = torch_attn_std(query, key, value, dropout_p=dropout_p, causal=is_causal)
      
    # Binding FlashAttn2  --------------------------------
    if(test_FlashAttn):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            FA2_output = torch.nn.functional.scaled_dot_product_attention(query1, key1, value1, is_causal=is_causal)
            FA2_output = FA2_output.permute(0, 2, 1, 3)
        max_diff, mean_diff = check_tensor(FA2_output, torch_output)
        print(f"[CHECK]  FlashAttn2\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )
    
    # FlexAttn  ---------------------------------------
    if(test_FlexAttn):   
        compiled_flex_attention = torch.compile(flex_attention, mode="default", dynamic=False)
        block_mask = create_block_mask_cached(mask_mod, 1, 1, seq_len, seq_len, device=query.device)
        flex_output = compiled_flex_attention(query1, key1, value1, score_mod=score_mod, block_mask=block_mask)
        flex_output1 = flex_output.permute(0, 2, 1, 3)
        max_diff, mean_diff = check_tensor(flex_output1, torch_output)
        print(f"[CHECK]  FlexAttn\t  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )
            
    # ByteTransformer --------------------------------------- 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, "ByteTransformer"))
    from ByteTransformer.ops.package_op import bytetr_attn_op, bytetr_longattn_op

    
    if seq_len<=256:
        mask=mask.half()
        ByteTransformer_output = bytetr_attn_op(qkv,qkv_bias,mask,head_num)
        ByteTransformer_output_4d =  ByteTransformer_output.view(batch_size, seq_len, head_num, head_size)
        max_diff, mean_diff = check_tensor(ByteTransformer_output_4d,torch_output)
        print(f"[CHECK]  ByteTransformer  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}")
    elif seq_len<=1024:
        mask=mask.half()
        ByteTransformer_output = bytetr_longattn_op(qkv,qkv_bias,mask,head_num)
        ByteTransformer_output_4d = ByteTransformer_output.view(batch_size, seq_len, head_num, head_size)
        max_diff, mean_diff = check_tensor(ByteTransformer_output_4d,torch_output)
        print(f"[CHECK]  ByteTransformer  max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}")
    else:
        print("ByteTransformer unsurpported for seq_len > 1024 !")

    dropout_p = 0.0
    BLOCK_M    = 64
    BLOCK_N    = 64
    num_warps  = 1
    nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, BLOCK_M, BLOCK_N)

    STOF_output = binding_attn_func(query, key, value,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, inner_bitmaps, 
                                load_row_ptr, load_col_idx, 
                                dropout_p=dropout_p, causal=is_causal)     
    max_diff, mean_diff = check_tensor(STOF_output, torch_output)
    print(f"[CHECK]  STOF\t          max_diff:{max_diff:.4f}  mean_diff:{mean_diff:.4f}" )

        
   

