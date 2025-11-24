
import sys
import os

import argparse
import torch
import torch._dynamo
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.attention.flex_attention import flex_attention  
from ops.package_op import binding_attn_func  
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, time_stamp_cudasync, transpose_for_scores
from util.utils import set_dtype, seqlen_to_mask, torch_cuda_identify, time_stamp_cudasync, transpose_for_scores
from util.masks import generate_causal_mask, generate_dilated_mask, generate_sliding_mask, generate_longformer_mask, generate_bigbird_mask, get_sparse_storage, get_OuterTile_storage
from util.masks import create_block_mask_cached, flex_bigbird_mask, flex_causal_mask, flex_sliding_window_mask, flex_longformer_mask,  get_OuterTile_storagevit
import random

import warnings
from torch.jit import TracerWarning
warnings.filterwarnings("ignore", category=TracerWarning)

import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from MCFuser.ct_mcfuser_mask  import mcfuser_attn


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


def new_gelu(input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# Bert-small | Bert-base | Bert-large ----------------------------------------
# ----------------------------------------------------------------------------
def bert_fwd_std(mask):
    with torch.no_grad():
        hidden_states = input_from_tensor
        for layer in range(layer_num):
            input_tensor = hidden_states

            qkv = qkv_kernel[layer]  + qkv_bias[layer]
            q1, k1, v1 = qkv.chunk(3, dim=-1)
            q1 = transpose_for_scores(q1, head_num, head_size)  
            k1 = transpose_for_scores(k1, head_num, head_size)  
            v1 = transpose_for_scores(v1, head_num, head_size)
            
            q = q1.permute(0, 2, 1, 3).contiguous()
            k = k1.permute(0, 2, 1, 3).contiguous()
            v = v1.permute(0, 2, 1, 3).contiguous() 
            h= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
            h2= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)

        
            # ------------------------------------------------------------- Attention start
            if attention_type=="torch_attention":
                q = rearrange(q, 'b t h d -> (b h) t d')
                k = rearrange(k, 'b s h d -> (b h) d s')
                softmax_scale = 1.0 / math.sqrt(head_size)
                
                scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
                scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                                   '(b h) t s -> b h t s', h=head_num)
                if is_causal:
                    causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
                    scores = scores + causal_mask.to(dtype=scores.dtype)
                
                attention = torch.softmax(scores, dim=-1)
                attention_drop = F.dropout(attention, dropout_p)
                h = torch.einsum('bhts,bshd->bthd', attention_drop , v).to(dtype=q.dtype)

            elif (attention_type == 'ByteTransformer'):
                mask=mask.half()
                if seq_len<=256:
                    result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
                else:
                    result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
                h = result.view(batch_size, seq_len, head_num, head_size).contiguous()
  
            elif (attention_type == 'STOF_attention'):
                h = binding_attn_func(q, k, v,
                                full_row_ptr, full_col_idx, 
                                part_row_ptr, part_col_idx, inner_bitmaps, 
                                load_row_ptr, load_col_idx, 
                                dropout_p=dropout_p, causal=is_causal) 
         
            

            elif (attention_type == 'MCFuser'):
                mask_4d = mask.unsqueeze(1).repeat(1, head_num, 1, 1)
                mask_4d_modified = torch.where(mask_4d == 0, torch.tensor(-6000.0, device=mask_4d.device), torch.tensor(0.0, device=mask_4d.device))
                h=mcfuser_attn(batch_size,head_num,seq_len,head_size,mask_4d_modified,q1,k1,v1)
                h = h.permute(0, 2, 1, 3) 

            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            hidden_states = h.reshape(new_context_layer_shape)
                    
            # ------------------------------------------------------------ Attention End
            hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer]) + attr_output_bias[layer]
            hidden_states = hidden_states + input_tensor
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),
                                        weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
            residual = hidden_states
        
            hidden_states = torch.matmul(hidden_states, inter_kernel[layer]) + inter_bias[layer] 
            hidden_states = F.gelu(hidden_states) 
            hidden_states = torch.matmul(hidden_states, output_kernel[layer]) + output_bias[layer]
            hidden_states = hidden_states + residual 
            hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  
                                        weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
        
            transformer_output[layer] = hidden_states


# GPT2 ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
def gpt_base_fwd_std(mask):
    hidden_states = input_from_tensor
    for layer in range(layer_num):
        input_tensor = hidden_states

        qkv = qkv_kernel[layer]  + qkv_bias[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q1 = transpose_for_scores(q, head_num, head_size).contiguous()  
        k1 = transpose_for_scores(k, head_num, head_size).contiguous()  
        v1 = transpose_for_scores(v, head_num, head_size).contiguous()
        
        q = q1.permute(0, 2, 1, 3)
        k = k1.permute(0, 2, 1, 3)
        v = v1.permute(0, 2, 1, 3) 
        h= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
        h2= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)

    
        # ------------------------------------------------------------- Attention start
        if attention_type=="torch_attention":
            q = rearrange(q, 'b t h d -> (b h) t d')
            k = rearrange(k, 'b s h d -> (b h) d s')
            softmax_scale = 1.0 / math.sqrt(head_size)
            
            scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
            scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                               '(b h) t s -> b h t s', h=head_num)
            if is_causal:
                causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
                scores = scores + causal_mask.to(dtype=scores.dtype)
            
            attention = torch.softmax(scores, dim=-1)
            attention_drop = F.dropout(attention, dropout_p)
            h = torch.einsum('bhts,bshd->bthd', attention_drop , v).to(dtype=q.dtype)

        elif (attention_type == 'ByteTransformer'):
            mask=mask.half()
            if seq_len<=256:
                result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            else:
                result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            h = result.view(batch_size, seq_len, head_num, head_size).contiguous()

        elif (attention_type == 'STOF_attention'):
            h = binding_attn_func(q, k, v,
                            full_row_ptr, full_col_idx, 
                            part_row_ptr, part_col_idx, inner_bitmaps, 
                            load_row_ptr, load_col_idx, 
                            dropout_p=dropout_p, causal=is_causal) 
        
        elif (attention_type == 'MCFuser'):
            mask_4d = mask.unsqueeze(1).repeat(1, head_num, 1, 1)
            mask_4d_modified = torch.where(mask_4d == 0, torch.tensor(-6000.0, device=mask_4d.device), torch.tensor(0.0, device=mask_4d.device))
            h=mcfuser_attn(batch_size,head_num,seq_len,head_size,mask_4d_modified,q1,k1,v1)
            h = h.permute(0, 2, 1, 3) 

        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        hidden_states = h.reshape(new_context_layer_shape)
                
        # ------------------------------------------------------------ Attention End
                
        hidden_states = torch.matmul(hidden_states, attr_output_kernel[layer])
        hidden_states = hidden_states + attr_output_bias[layer]
        hidden_states = hidden_states + input_tensor
        hidden_states = F.layer_norm(hidden_states, (hidden_dim, ), weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
        residual = hidden_states 
        hidden_states = torch.matmul(hidden_states, inter_kernel[layer])
        hidden_states = hidden_states + inter_bias[layer] 
        hidden_states = new_gelu(hidden_states)  
        hidden_states = torch.matmul(hidden_states, output_kernel[layer]) 
        hidden_states = hidden_states + output_bias[layer]  
        hidden_states = hidden_states + residual
        hidden_states = F.layer_norm(hidden_states, (hidden_dim, ),  weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
        transformer_output[layer] = hidden_states


# T5 ------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def T5_base_fwd_std(mask):
    encoder_hidden_states = input_from_tensor
    
    # Encoder ---------------------------------------------------
    for layer in range(layer_num):
        input_tensor = encoder_hidden_states
        qkv = torch.matmul(encoder_hidden_states, qkv_kernel_raw[layer])
        qkv = qkv + qkv_bias[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q1 = transpose_for_scores(q, head_num, head_size)  
        k1 = transpose_for_scores(k, head_num, head_size)  
        v1 = transpose_for_scores(v, head_num, head_size)
        
        q = q1.permute(0, 2, 1, 3)
        k = k1.permute(0, 2, 1, 3)
        v = v1.permute(0, 2, 1, 3) 
        h= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
        h2= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
    
        # ------------------------------------------------------------- Attention start
        if attention_type=="torch_attention":
            q = rearrange(q, 'b t h d -> (b h) t d')
            k = rearrange(k, 'b s h d -> (b h) d s')
            softmax_scale = 1.0 / math.sqrt(head_size)
            
            scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
            scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                               '(b h) t s -> b h t s', h=head_num)
            if is_causal:
                causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
                scores = scores + causal_mask.to(dtype=scores.dtype)
            
            attention = torch.softmax(scores, dim=-1)
            attention_drop = F.dropout(attention, dropout_p)
            h = torch.einsum('bhts,bshd->bthd', attention_drop , v).to(dtype=q.dtype)

        elif (attention_type == 'ByteTransformer'):
            mask=mask.half()
            if seq_len<=256:
                result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            else:
                result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            h = result.view(batch_size, seq_len, head_num, head_size).contiguous()


        elif (attention_type == 'STOF_attention'):
            h = binding_attn_func(q, k, v,
                            full_row_ptr, full_col_idx, 
                            part_row_ptr, part_col_idx, inner_bitmaps, 
                            load_row_ptr, load_col_idx, 
                            dropout_p=dropout_p, causal=is_causal) 
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            encoder_hidden_states = h.view(new_context_layer_shape)
           
        elif (attention_type == 'MCFuser'):
            mask_4d = mask.unsqueeze(1).repeat(1, head_num, 1, 1)
            mask_4d_modified = torch.where(mask_4d == 0, torch.tensor(-6000.0, device=mask_4d.device), torch.tensor(0.0, device=mask_4d.device))
            h=mcfuser_attn(batch_size,head_num,seq_len,head_size,mask_4d_modified,q1,k1,v1)
            h = h.permute(0, 2, 1, 3) 

        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        encoder_hidden_states = h.reshape(new_context_layer_shape)
                
        # ------------------------------------------------------------ Attention End
    
        encoder_hidden_states = torch.matmul(encoder_hidden_states, attr_output_kernel[layer])
        encoder_hidden_states = encoder_hidden_states + attr_output_bias[layer]
        encoder_hidden_states = encoder_hidden_states + input_tensor
        encoder_hidden_states = F.layer_norm(encoder_hidden_states, (hidden_dim, ), weight=attr_output_layernorm_gamma[layer], bias=attr_output_layernorm_beta[layer])
        encoder_residual = encoder_hidden_states 
        encoder_hidden_states = torch.matmul(encoder_hidden_states, inter_kernel[layer])
        encoder_hidden_states = encoder_hidden_states + inter_bias[layer] 
        encoder_hidden_states = F.relu(encoder_hidden_states)  
        encoder_hidden_states = torch.matmul(encoder_hidden_states, output_kernel[layer]) 
        encoder_hidden_states = encoder_hidden_states + output_bias[layer]  
        encoder_hidden_states = encoder_hidden_states + encoder_residual 
        encoder_hidden_states = F.layer_norm(encoder_hidden_states, (hidden_dim, ), weight=output_layernorm_gamma[layer], bias=output_layernorm_beta[layer])  
        
    Encoder_qkv = torch.matmul(encoder_hidden_states, qkv_kernel_raw[layer])
    Encoder_qkv = Encoder_qkv + qkv_bias[layer]
    encoder_q, encoder_k, encoder_v = Encoder_qkv.chunk(3, dim=-1)
    encoder_q = transpose_for_scores(encoder_q, head_num, head_size)
    encoder_k = transpose_for_scores(encoder_k, head_num, head_size)
    
        
    # Decoder ---------------------------------------------------
    for layer in range(layer_num):
        
        qkv = qkv_kernel[layer]  + qkv_bias[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q1 = transpose_for_scores(q, head_num, head_size)  
        k1 = transpose_for_scores(k, head_num, head_size)  
        v1 = transpose_for_scores(v, head_num, head_size)
        
        q = q1.permute(0, 2, 1, 3)
        k = k1.permute(0, 2, 1, 3)
        v = v1.permute(0, 2, 1, 3) 
        h= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
        h2= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)

    

        if attention_type=="torch_attention":
            q = rearrange(q, 'b t h d -> (b h) t d')
            k = rearrange(k, 'b s h d -> (b h) d s')
            softmax_scale = 1.0 / math.sqrt(head_size)
            
            scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
            scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                               '(b h) t s -> b h t s', h=head_num)
            if is_causal:
                causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
                scores = scores + causal_mask.to(dtype=scores.dtype)
            
            attention = torch.softmax(scores, dim=-1)
            attention_drop = F.dropout(attention, dropout_p)
            h = torch.einsum('bhts,bshd->bthd', attention_drop , v).to(dtype=q.dtype)

        elif (attention_type == 'ByteTransformer'):
            mask=mask.half()
            if seq_len<=256:
                result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            else:
                result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            h = result.view(batch_size, seq_len, head_num, head_size).contiguous()

        elif (attention_type == 'STOF_attention'):
            h = binding_attn_func(q, k, v,
                            full_row_ptr, full_col_idx, 
                            part_row_ptr, part_col_idx, inner_bitmaps, 
                            load_row_ptr, load_col_idx, 
                            dropout_p=dropout_p, causal=is_causal) 
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            decoder_hidden_states = h.view(new_context_layer_shape)

        elif (attention_type == 'MCFuser'):
            mask_4d = mask.unsqueeze(1).repeat(1, head_num, 1, 1)
            mask_4d_modified = torch.where(mask_4d == 0, torch.tensor(-6000.0, device=mask_4d.device), torch.tensor(0.0, device=mask_4d.device))
            h=mcfuser_attn(batch_size,head_num,seq_len,head_size,mask_4d_modified,q1,k1,v1)
            h = h.permute(0, 2, 1, 3) 

        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        decoder_hidden_states = h.reshape(new_context_layer_shape)
        
        decoder_hidden_states = torch.matmul(decoder_hidden_states, attr_output_kernel_2[layer])
        decoder_hidden_states = decoder_hidden_states + attr_output_bias_2[layer]
        decoder_hidden_states = decoder_hidden_states + output_from_tensor

        decoder_hidden_states = F.layer_norm(decoder_hidden_states, (hidden_dim, ), weight=attn_lynorm_gamma_2[layer], bias=attn_lynorm_beta_2[layer])
        decoder_residual = decoder_hidden_states 
        
    
        qkv = torch.matmul(decoder_hidden_states, qkv_kernel_3[layer])
        qkv = qkv + qkv_bias_3[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q1 = transpose_for_scores(q, head_num, head_size)  
        k1 = transpose_for_scores(k, head_num, head_size)  
        v1 = transpose_for_scores(v, head_num, head_size)
        
        q = q1.permute(0, 2, 1, 3)
        k = k1.permute(0, 2, 1, 3)
        v = v1.permute(0, 2, 1, 3) 
        h= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
        h2= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)

    
        # ------------------------------------------------------------- Attention start
        if attention_type=="torch_attention":
            q = rearrange(q, 'b t h d -> (b h) t d')
            k = rearrange(k, 'b s h d -> (b h) d s')
            softmax_scale = 1.0 / math.sqrt(head_size)
            
            scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
            scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                               '(b h) t s -> b h t s', h=head_num)
            if is_causal:
                causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
                scores = scores + causal_mask.to(dtype=scores.dtype)
            
            attention = torch.softmax(scores, dim=-1)
            attention_drop = F.dropout(attention, dropout_p)
            h = torch.einsum('bhts,bshd->bthd', attention_drop , v).to(dtype=q.dtype)

        elif (attention_type == 'ByteTransformer'):
            mask=mask.half()
            if seq_len<=256:
                result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            else:
                result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            h = result.view(batch_size, seq_len, head_num, head_size).contiguous()
  
        elif (attention_type == 'STOF_attention'):
            h = binding_attn_func(q, k, v,
                            full_row_ptr, full_col_idx, 
                            part_row_ptr, part_col_idx, inner_bitmaps, 
                            load_row_ptr, load_col_idx, 
                            dropout_p=dropout_p, causal=is_causal) 


        elif (attention_type == 'MCFuser'):
            mask_4d = mask.unsqueeze(1).repeat(1, head_num, 1, 1)
            mask_4d_modified = torch.where(mask_4d == 0, torch.tensor(-6000.0, device=mask_4d.device), torch.tensor(0.0, device=mask_4d.device))
            h=mcfuser_attn(batch_size,head_num,seq_len,head_size,mask_4d_modified,q1,k1,v1)
            h = h.permute(0, 2, 1, 3) 

        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        decoder_hidden_states = h.reshape(new_context_layer_shape)

        # ------------------------------------------------------------ Attention End
            
        decoder_hidden_states = torch.matmul(decoder_hidden_states, crattr_output_kernel_2[layer])
        decoder_hidden_states = decoder_hidden_states + crattr_output_bias_2[layer]
        decoder_hidden_states = decoder_hidden_states + decoder_residual
        decoder_hidden_states = F.layer_norm(decoder_hidden_states, (hidden_dim, ), weight=crattn_lynorm_gamma_2[layer], bias=crattn_lynorm_beta_2[layer])
        decoder_residual = decoder_hidden_states
        
        decoder_hidden_states = torch.matmul(decoder_hidden_states, inter_kernel_2[layer])
        decoder_hidden_states = decoder_hidden_states + inter_bias_2[layer] 
        decoder_hidden_states = F.relu(decoder_hidden_states)  
        decoder_hidden_states = torch.matmul(decoder_hidden_states, output_kernel_2[layer]) 
        decoder_hidden_states = decoder_hidden_states + output_bias_2[layer]  
        decoder_hidden_states = decoder_hidden_states + decoder_residual
        encoder_hidden_states = F.layer_norm(decoder_hidden_states, (hidden_dim, ), weight=lynorm_gamma_2[layer], bias=lynorm_beta_2[layer])  
        
        transformer_output[layer] = encoder_hidden_states


from transformers.cache_utils import DynamicCache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.activations import ACT2FN

def llama3_base_fwd_std(mask):
    hidden_states = input_from_tensor

    past_key_values = DynamicCache()
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position: torch.Tensor = torch.arange(past_seen_tokens, past_seen_tokens + input_from_tensor.shape[1], device=input_from_tensor.device)
    position_ids = cache_position.unsqueeze(0)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_size, 2, dtype=torch.int64).to(device=hidden_states.device, dtype=torch.float) / head_num))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(hidden_states.device)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = hidden_states.device.type if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * 1.0
        sin = emb.sin() * 1.0
    position_embeddings = (cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype))
    

    for layer in range(layer_num):
        residual = hidden_states
        
        hidden_size = hidden_states.shape[-1]
        weight = torch.ones(hidden_size, device=hidden_states.device)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        hidden_statea = weight * hidden_states.to(input_dtype)

        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_num)
    
        qkv = qkv_kernel[layer]  + qkv_bias[layer]
        q, k, v = qkv.chunk(3, dim=-1)
        q1 = transpose_for_scores(q, head_num, head_size)  
        k1 = transpose_for_scores(k, head_num, head_size)  
        v1 = transpose_for_scores(v, head_num, head_size)
            
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q1 = (q1 * cos) + (torch.cat((-q1[..., q1.shape[-1] // 2 :], q1[..., : q1.shape[-1] // 2]), dim = -1) * sin)
        k1 = (k1 * cos) + (torch.cat((-k1[..., k1.shape[-1] // 2 :], k1[..., : k1.shape[-1] // 2]), dim = -1) * sin)

        q = q1.permute(0, 2, 1, 3)
        k = k1.permute(0, 2, 1, 3)
        v = v1.permute(0, 2, 1, 3)
        h= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
        h2= torch.empty((batch_size, head_num, seq_len, head_size), device='cuda',dtype=torch.float16)
        # ------------------------------------------------------------- Attention start
        if attention_type=="torch_attention":
            q = rearrange(q, 'b t h d -> (b h) t d')
            k = rearrange(k, 'b s h d -> (b h) d s')
            softmax_scale = 1.0 / math.sqrt(head_size)
            
            scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
            scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                               '(b h) t s -> b h t s', h=head_num)
            if is_causal:
                causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
                scores = scores + causal_mask.to(dtype=scores.dtype)
            
            attention = torch.softmax(scores, dim=-1)
            attention_drop = F.dropout(attention, dropout_p)
            h = torch.einsum('bhts,bshd->bthd', attention_drop , v).to(dtype=q.dtype)

        elif (attention_type == 'ByteTransformer'):
            mask=mask.half()
            if seq_len<=256:
                result = bytetr_attn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            else:
                result = bytetr_longattn_op(qkv_kernel[layer],qkv_bias[layer],mask,head_num)
            h = result.view(batch_size, seq_len, head_num, head_size).contiguous()
  
        elif (attention_type == 'STOF_attention'):

            h = binding_attn_func(q, k, v,
                            full_row_ptr, full_col_idx, 
                            part_row_ptr, part_col_idx, inner_bitmaps, 
                            load_row_ptr, load_col_idx, 
                            dropout_p=dropout_p, causal=is_causal) 


        elif (attention_type == 'MCFuser'):
            mask_4d = mask.unsqueeze(1).repeat(1, head_num, 1, 1)
            mask_4d_modified = torch.where(mask_4d == 0, torch.tensor(-6000.0, device=mask_4d.device), torch.tensor(0.0, device=mask_4d.device))
            h=mcfuser_attn(batch_size,head_num,seq_len,head_size,mask_4d_modified,q1,k1,v1)
            h = h.permute(0, 2, 1, 3) 

        new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
        hidden_states = h.reshape(new_context_layer_shape)
                
        # ------------------------------------------------------------ Attention End

        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_size = hidden_states.shape[-1]
        weight = torch.ones(hidden_size, device=hidden_states.device)

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        hidden_states = weight *hidden_states.to(input_dtype)

        act_fn = ACT2FN["silu"]
        hidden_states = act_fn(hidden_states) * hidden_states
        hidden_states = residual + hidden_states

        transformer_output[layer] = hidden_states
        


from torch import nn
from transformers.utils import torch_int
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

def interpolate_pos_encoding(embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
    num_patches = embeddings.shape[1] - 1
    num_positions = embeddings.shape[1] - 1
    # always interpolate when tracing to ensure the exported model works for dynamic input shapes
    if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
        return embeddings
    class_pos_embed = embeddings[:, :1]
    patch_pos_embed = embeddings[:, 1:]
    dim = embeddings.shape[-1]
    new_height = height // 16
    new_width = width // 16
    sqrt_num_positions = torch_int(num_positions**0.5)
    patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


def vit_base_fwd_std(mask):
    #Embedding
    batch_size, num_channels, height, width = pixel_values.shape
    hidden_size=head_size*head_num

    projection = nn.Conv2d(num_channels, hidden_size, kernel_size=16, stride=16).to(device=device)
    embeddings = projection(pixel_values.to(torch.float32)).flatten(2).transpose(1, 2).half()
    cls_token = nn.Parameter(torch.randn(1, 1, hidden_size)).to(device=device)
    cls_tokens = cls_token.expand(batch_size, -1, -1)
    embeddings = torch.cat((cls_tokens, embeddings), dim=1)
    embeddings = embeddings + interpolate_pos_encoding(embeddings, height, width)
    embedding_output = F.dropout(embeddings,p=0.0, training=False)


    #encoder
    for layer in range(layer_num):
        hidden_states = embedding_output
        layernorm_before = nn.LayerNorm(head_num * head_size, eps=1e-12).to(device=device)
        hidden_states = layernorm_before(hidden_states)
        
        layer_head_mask = vit_mask
        batch_size, seq_len, _ = hidden_states.shape

        qkv = hidden_states 
        q1 = transpose_for_scores(qkv, head_num, head_size)  
        k1 = transpose_for_scores(qkv, head_num, head_size)  
        v1 = transpose_for_scores(qkv, head_num, head_size)
        
        q = q1.permute(0, 2, 1, 3).half()
        k = k1.permute(0, 2, 1, 3).half()
        v = v1.permute(0, 2, 1, 3).half() 
    
        # ------------------------------------------------------------- Attention start
        if attention_type=="torch_attention":
            q = rearrange(q, 'b t h d -> (b h) t d')
            k = rearrange(k, 'b s h d -> (b h) d s')
            softmax_scale = 1.0 / math.sqrt(head_size)
            
            scores = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
            scores = rearrange(torch.baddbmm(scores, q, k, beta=1.0, alpha=softmax_scale),
                               '(b h) t s -> b h t s', h=head_num)
            if is_causal:
                causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
                scores = scores + causal_mask.to(dtype=scores.dtype)
            
            attention = torch.softmax(scores, dim=-1)
            attention_drop = F.dropout(attention, dropout_p)
            h = torch.einsum('bhts,bshd->bthd', attention_drop , v).to(dtype=q.dtype)
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            attn_output = h.reshape(new_context_layer_shape).float()

        elif (attention_type == 'ByteTransformer'):
            mask=mask.half()
            if seq_len<=256:
                result = bytetr_attn_op(vit_qkv_kernel[layer],vit_qkv_bias[layer],layer_head_mask,head_num)
            else:
                result = bytetr_longattn_op(vit_qkv_kernel[layer],vit_qkv_bias[layer],layer_head_mask,head_num)
            h = result.view(batch_size, seq_len, head_num, head_size).contiguous()
  
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            attn_output = h.view(new_context_layer_shape).float()

        elif (attention_type == 'STOF_attention'):
            h = binding_attn_func(q, k, v,
                            full_row_ptr, full_col_idx, 
                            part_row_ptr, part_col_idx, inner_bitmaps, 
                            load_row_ptr, load_col_idx, 
                            dropout_p=dropout_p, causal=is_causal) 
            new_context_layer_shape = h.size()[:-2] + (hidden_dim, )
            attn_output = h.view(new_context_layer_shape).float()
        # ------------------------------------------------------------ Attention End
        attn_weights = torch.empty(batch_size * head_num, seq_len, seq_len, dtype=q.dtype, device=q.device)
        self_outputs = (attn_output, attn_weights) 
        dense = nn.Linear(head_size * head_num, head_size * head_num).to(device=device).to(attn_output.dtype)
        attention_output0 = dense(self_outputs[0])
        attention_output0 = F.dropout(attention_output0,p=0.0, training=False)
        self_attention_outputs = (attention_output0,) + self_outputs[1:]
        
        
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        hidden_states = attention_output + hidden_states
        layernorm_after = nn.LayerNorm(head_size * head_num, eps=1e-12).to(device=device)
        dense = nn.Linear(head_size * head_num, 3072).to(device=device).to(attn_output.dtype)
        intermediate_act_fn = ACT2FN["gelu"]
        layer_output = layernorm_after(hidden_states)
        layer_output = dense(layer_output)
        layer_output = intermediate_act_fn(layer_output)
        
        dense = nn.Linear(3072, head_num * head_size).to(device=device)
        layer_output = dense(layer_output)
        layer_output = F.dropout(layer_output,p=0.0, training=False)
        layer_output = layer_output + hidden_states
        layer_outputs = (layer_output,) + outputs
        
        hidden_states = layer_outputs[0]

    encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )
    sequence_output = encoder_outputs[0]
    layernorm = nn.LayerNorm((head_size * head_num), eps=1e-12).to(device=device)
    sequence_output = layernorm(sequence_output)
    first_token_tensor = sequence_output[:, 0]
    dense = nn.Linear(head_num * head_size, head_num * head_size).to(device=device)
    activation = ACT2FN["tanh"]
    pooled_output = dense(first_token_tensor)
    pooled_output = activation(pooled_output)

    transformer__output = BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
 
        
        
        
        
        
        
        

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    device = torch_cuda_identify(print_info = False)
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
    parser.add_argument('--mask_id', type=int, default=3, help='Mask type: 1-Sliding | 2-Longformer | 3-BigBird (default: 0)')
    parser.add_argument('--block_m', type=int, default=64, help='Block Size of M (default:32)')
    parser.add_argument('--block_n', type=int, default=64, help='Block Size of N (default:32)')
    parser.add_argument('--num_warps', type=int, default=1, help='Warp Num to launch (default:4)')
    
    parser.add_argument('--method', type=str, default="STOF", help='TorchNative, TorchCompile, ByteTrans, STOF')
    parser.add_argument('--model', type=str, default="bert_base", help='Sequence length (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length (default: 256)')
    args = parser.parse_args() 

    mask_id    = args.mask_id
    BLOCK_M    = args.block_m
    BLOCK_N    = args.block_n
    num_warps  = args.num_warps
    model_selection  = args.model
    method_selection = args.method
    
    head_size = 64
    seq_len   = args.seq_len
    batch_size = args.batch_size
    
    data_type  = torch.float16
    dtype = "fp16"
    running_device = "cuda"
    attention_type="torch_attention"
    if(model_selection == "vit_base" and seq_len >= 2048):
        seq_len = 512
    
    
    if(num_warps > (BLOCK_M/16) * (BLOCK_N/16)):
        print(f"num_warps: {num_warps}, (BLOCK_M/16) * (BLOCK_N/16): {int(BLOCK_M / 16) * int(BLOCK_N / 16)}")
        print("Error! Here should be: num_warps <= (BLOCK_M/16) * (BLOCK_N/16) !")
        exit(0)

                
    data_type  = torch.float16
    running_device = "cuda"
    sqrt_seq_len = int(math.sqrt(seq_len))
    fill_rate    = 0.1
    dropout_p = 0.0
    layer_num=1
    num_channels=3
    
    warmup_iters = 10
    running_iters = 10
    if model_selection == "bert_small":
        inference_model=bert_fwd_std
        head_num=8
        layer_num=6
        head_size = 64
    elif model_selection == "bert_base":
        inference_model=bert_fwd_std
        head_num=12
        layer_num=12
        head_size = 64
    elif model_selection == "bert_large":
        inference_model=bert_fwd_std   
        head_num=16
        layer_num=24
        head_size = 64   
    elif model_selection == "gpt":
        inference_model=gpt_base_fwd_std
        head_num=12
        layer_num=12
        head_size = 64
    elif model_selection == "t5":
        inference_model= T5_base_fwd_std
        head_num=12
        layer_num=12
        head_size = 64
    elif model_selection == "llama_small":
        inference_model = llama3_base_fwd_std
        head_num = 8
        layer_num=16
        head_size = 64
    elif model_selection == "llama_base":
        inference_model = llama3_base_fwd_std
        head_num = 16
        layer_num=24
        head_size = 64
    elif model_selection == "vit_base":
        inference_model = vit_base_fwd_std
        head_num = 12
        layer_num=12
        head_size = 64
    hidden_dim = head_num * head_size 

    vit_avg_seq_len = (seq_len // 16) * (seq_len // 16) + 1
    noattention = torch.randn(batch_size,  seq_len, head_num, head_size, device=running_device, dtype=data_type)
    vit_noattention = torch.randn(batch_size,  vit_avg_seq_len, head_num, head_size, device=running_device, dtype=data_type)

    test_Torch           = False
    test_ByteTransformer = False
    test_Torch_Compile   = False
    test_STOF            = False
    test_MCFuser         = True

    
    
    if method_selection == "TorchNative":
        test_Torch           = True
    elif(method_selection == "TorchCompile"):
        test_Torch_Compile   = True
        if(is_4080_laptop): 
            test_Torch_Compile = False
    elif(method_selection == "ByteTrans"):
        test_ByteTransformer = True
        if seq_len > 1024:
            print("ByteTransformer unsupported for seq_len > 1024")
            test_ByteTransformer = False
    elif(method_selection == "STOF"):
        test_STOF    = True
    elif method_selection == "MCFuser":
        test_MCFuser = True
    
        
    if (model_selection == "vit_base"):
        vit_avg_seq_len = (seq_len // 16) * (seq_len // 16) + 1
        vit_low, vit_high = (2 * vit_avg_seq_len - vit_avg_seq_len, vit_avg_seq_len + 1)
        vit_input_lens = torch.randint(low=vit_low, high=vit_high, size=(batch_size,))
        vit_seqlen_mask = seqlen_to_mask(vit_input_lens, vit_avg_seq_len)
        vit_attr_mask   = set_dtype(torch.tile(vit_seqlen_mask, dims=(vit_avg_seq_len,)).reshape(batch_size, vit_avg_seq_len, vit_avg_seq_len).cuda(), "fp16")
    else:
        avg_seq_len = seq_len 
        low, high = (2 * avg_seq_len - seq_len, seq_len + 1)
        input_lens = torch.randint(low=low, high=high, size=(batch_size,))
        seqlen_mask = seqlen_to_mask(input_lens, seq_len)
        attr_mask   = set_dtype(torch.tile(seqlen_mask, dims=(seq_len,)).reshape(batch_size, seq_len, seq_len).cuda(), "fp16")
    
    mask_mod = None
    score_mod = None
    mask=None
    
    if(mask_id == 0):
        is_causal = True
        mask_name = 'Causal_Mask'
        mask_mod = flex_causal_mask
        if (model_selection == "vit_base"):
            vit_mask = generate_causal_mask(vit_attr_mask).cuda()
        else:
            mask = generate_causal_mask(attr_mask).cuda()
    elif(mask_id == 1):
        is_causal = True
        mask_name = 'Sliding_Mask'
        mask_mod = flex_sliding_window_mask
        if (model_selection == "vit_base"):
            vit_mask = generate_sliding_mask(vit_attr_mask, bandwidth=BLOCK_M, is_cauasl=True).cuda()
        else:
            mask = generate_sliding_mask(attr_mask, bandwidth=BLOCK_M, is_cauasl=True).cuda()
    elif(mask_id == 2):
        is_causal = False
        mask_name = 'Longformer_Mask'
        mask_mod = flex_longformer_mask
        if (model_selection == "vit_base"):
            vit_mask = generate_longformer_mask(vit_attr_mask, globalwidth=32, bandwidth=32, is_cauasl=is_causal).cuda()
        else:
            mask = generate_longformer_mask(attr_mask, globalwidth=32, bandwidth=32, is_cauasl=is_causal).cuda()
    elif(mask_id == 3):
        is_causal = False
        mask_name = 'BigBird_Mask'
        mask_mod = flex_bigbird_mask
        if (model_selection == "vit_base"):
            vit_mask = generate_bigbird_mask(vit_attr_mask, globalwidth=32, bandwidth=32, fill_rate=fill_rate, is_cauasl=is_causal).cuda()
        else:
            mask = generate_bigbird_mask(attr_mask, globalwidth=32, bandwidth=32, fill_rate=fill_rate, is_cauasl=is_causal).cuda()
    elif(mask_id == 4):
        is_causal = False
        mask_name = 'Dilated_Mask'
        mask_mod = flex_sliding_window_mask
        if (model_selection == "vit_base"):
            vit_mask = generate_dilated_mask(vit_attr_mask, bandwidth=BLOCK_M, dilation_rate=1, is_cauasl=True).cuda()
        else:
            mask = generate_dilated_mask(attr_mask, bandwidth=BLOCK_M, dilation_rate=1, is_cauasl=True).cuda()
        
    qkv_kernel_raw              = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_kernel          = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_bias            = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_gamma = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    attr_output_layernorm_beta  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_kernel                = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    inter_bias                  = [set_dtype(torch.zeros(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_kernel               = [set_dtype(torch.zeros(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_bias                 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_gamma      = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    output_layernorm_beta       = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    
    if (model_selection == "vit_base"):
        pixel_values = set_dtype(torch.empty(batch_size, num_channels, seq_len, seq_len).uniform_(-0.4, 0.4).cuda(), dtype)
        vit_qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        vit_qkv_kernel                  = [set_dtype(torch.zeros(batch_size, (seq_len//16) * (seq_len // 16) + 1, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
    else:
        input_from_tensor           = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
        qkv_bias                    = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        qkv_kernel                  = [set_dtype(torch.zeros(batch_size, seq_len, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]

    if (model_selection == "t5"):
        output_from_tensor = set_dtype(torch.empty(batch_size, seq_len, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype)
        qkv_kernel_2 = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        qkv_bias_2 = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        qkv_kernel_3 = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        qkv_bias_3 = [set_dtype(torch.zeros(hidden_dim * 3).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        
        attr_output_kernel_2 = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        attr_output_bias_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        crattr_output_kernel_2 = [set_dtype(torch.zeros(hidden_dim, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        crattr_output_bias_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        
        attn_lynorm_gamma_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        attn_lynorm_beta_2  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        crattn_lynorm_gamma_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        crattn_lynorm_beta_2  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        
        inter_kernel_2 = [set_dtype(torch.zeros(hidden_dim, hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        inter_bias_2   = [set_dtype(torch.zeros(hidden_dim * 4).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        output_kernel_2 = [set_dtype(torch.zeros(hidden_dim * 4, hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        output_bias_2   = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        lynorm_gamma_2 = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]
        lynorm_beta_2  = [set_dtype(torch.zeros(hidden_dim).uniform_(-0.4, 0.4).cuda(), dtype) for _ in range(layer_num)]   
   
   
    transformer_output = [None for _ in range(layer_num)]

    if model_selection == "vit_base":
        nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storagevit(vit_mask, BLOCK_M, BLOCK_N)
    else:
        nnz, full_row_ptr, full_col_idx, part_row_ptr, part_col_idx, load_row_ptr, load_col_idx, inner_bitmaps = get_OuterTile_storage(mask, BLOCK_M, BLOCK_N)

    #  STOF_attention ------------------------------------
    
    if(test_STOF):    
        attention_type="STOF_attention"

        for i in range(warmup_iters + running_iters):
            if i == warmup_iters:    
                t_start = time_stamp_cudasync()

            if model_selection == "vit_base":
                inference_model(vit_mask)
            else:
                inference_model(mask)
            STOF_output=transformer_output[-1]
          
        
        t_end = time_stamp_cudasync()
        STOF_time = (t_end - t_start) * 1000 / running_iters
        
        print("e2e {} | bs:{} | seq:{}  |  STOF MHA        : {:.3f} ms / iter".format(model_selection, batch_size, args.seq_len, STOF_time)) 
                 