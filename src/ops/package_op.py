import torch
from torch.autograd import Function

from typing import Optional, Sequence, Tuple, Union

import block_attn_mask
import binding_attn

__all__ = ['block_attn_mask_func', 'bining_attn_func']


# result.shape = {batch_size, seq_len, head_num * head_size}        
class Block_Attn_Mask(Function):
    @staticmethod
    def forward(ctx, q, k, v, 
                full_row_ptr, full_col_idx, 
                part_row_ptr, part_col_idx, part_block_mask,
                load_row_ptr, load_col_idx,
                BLOCK_M, BLOCK_N, num_warps):  
        
        block_attn_mask.forward(
            q.contiguous(), 
            k.contiguous(), 
            v.contiguous(),
            full_row_ptr.contiguous(), full_col_idx.contiguous(), 
            part_row_ptr.contiguous(), part_col_idx.contiguous(), part_block_mask.contiguous(), 
            load_row_ptr.contiguous(), load_col_idx.contiguous(),
            BLOCK_M, BLOCK_N, num_warps)

        ctx.mark_non_differentiable(q)
        return q
    
block_attn_mask_func =  Block_Attn_Mask.apply


def _binding_attn_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    full_row_ptr: torch.Tensor, full_col_idx: torch.Tensor, 
    part_row_ptr: torch.Tensor, part_col_idx: torch.Tensor, inner_bitmaps: torch.Tensor,
    load_row_ptr: torch.Tensor, load_col_idx: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int, window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    
    out, softmax_lse, S_dmask, rng_state = binding_attn.forward(
        q, k, v,
        full_row_ptr, full_col_idx, 
        part_row_ptr, part_col_idx, inner_bitmaps,
        load_row_ptr, load_col_idx,
        None, alibi_slopes, dropout_p, softmax_scale, causal,
        window_size_left, window_size_right, softcap, return_softmax, None,
    )
    return out, softmax_lse, S_dmask, rng_state
    
class BindingAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        full_row_ptr, full_col_idx, 
        part_row_ptr, part_col_idx, inner_bitmaps,
        load_row_ptr, load_col_idx,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _binding_attn_forward(
            q, k, v,
            full_row_ptr, full_col_idx, 
            part_row_ptr, part_col_idx, inner_bitmaps,
            load_row_ptr, load_col_idx,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)


#  q, k, v: (batch_size, seqlen, nheads, headdim)
#      out: (batch_size, seqlen, nheads, headdim)
def binding_attn_func(
    q, k, v,
    full_row_ptr, full_col_idx, 
    part_row_ptr, part_col_idx, inner_bitmaps,
    load_row_ptr, load_col_idx,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    return BindingAttnFunc.apply(
        q, k, v,
        full_row_ptr, full_col_idx, 
        part_row_ptr, part_col_idx, inner_bitmaps,
        load_row_ptr, load_col_idx,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        False,
    )

