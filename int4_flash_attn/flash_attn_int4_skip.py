# Copyright (c) 2023, Tri Dao.

from typing import Optional, Sequence, Tuple, Union
import math
import torch

# isort: off
# We need to import the CUDA kernels after importing torch
try:
    import flash_attn_2_cuda_int4_skip as flash_attn_cuda
except:
    import flash_attn_2_cuda_int4_skip as flash_attn_cuda
from .quant_triton import (
    int4_quant_sub_k_mean_pertoken_SmoothQ_allInOne,
    int4_quant_sub_k_mean_perwarp_SmoothQ_fullsize_allInOne
)



import spas_sage_attn._qattn as qattn
from spas_sage_attn import utils
from torch.nn.attention import SDPBackend, sdpa_kernel
import warnings
# isort: on

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

def round_multiple(x, m):
    return (x + m - 1) // m * m


# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


@_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qq: torch.Tensor, qk: torch.Tensor, sq: torch.Tensor, sk: torch.Tensor, qmk: torch.Tensor,
    compute_record: Optional[torch.Tensor], approx_score: Optional[torch.Tensor],
    threshold: torch.Tensor,
    block_q_size: int,
    block_k_size: int,
    local_window: int,
    block_seg: int,
    softmax_scale: float,
    causal: bool,
    softcap: float,
    impl_version: int,
    quant_mode: int,
    two_pass: bool,
    map_only: bool
) -> torch.Tensor:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out = flash_attn_cuda.fwd_sparse_int4(
        q,
        k,
        v,
        None,
        softmax_scale,
        causal,
        softcap,
        qq, qk, sq, sk, qmk, block_q_size, block_k_size, 
        compute_record, approx_score, threshold, local_window, block_seg, impl_version, quant_mode, two_pass, map_only
    )
    return out


@_torch_register_fake_wrapper("flash_attn::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qq: torch.Tensor, qk: torch.Tensor, sq: torch.Tensor, sk: torch.Tensor, qmk: torch.Tensor,
    compute_record: Optional[torch.Tensor], approx_score: Optional[torch.Tensor],
    threshold: torch.Tensor,
    block_q_size: int,
    block_k_size: int,
    local_window: int,
    block_seg: int,
    softmax_scale: float,
    causal: bool,
    softcap: float,
    impl_version: int,
    quant_mode: int,
    two_pass: bool,
    map_only
)  -> torch.Tensor:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out = torch.empty_like(q)

    return out

class FlashAttnFunc:
    @staticmethod
    def forward(
        q,
        k,
        v,
        qq, qk, sq, sk, qmk, compute_record, approx_score,
        threshold, block_q_size, block_k_size, local_window, block_seg,
        softmax_scale,
        causal,
        softcap,
        impl_version,
        quant_mode,
        two_pass,
        map_only
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        
        out_padded = _flash_attn_forward(
            q,
            k,
            v,
            qq, qk, sq, sk, qmk, compute_record, approx_score,
            threshold, block_q_size, block_k_size, local_window, block_seg, 
            softmax_scale,
            causal=causal,
            softcap=softcap,
            impl_version=impl_version,
            quant_mode = quant_mode,
            two_pass = two_pass,
            map_only=map_only
        )
        out = out_padded[..., :head_size_og]
        return out

def flash_attn_func_int4_skip_sage(
    q, # [bsz, n_head, q_len, dim]
    k, # [bsz, n_kv_head, q_len, dim]
    v, # [bsz, n_kv_head, q_len, dim]
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    block_q_size = 64, 
    block_k_size = 32,
    local_window = 128,
    block_seg_size = 256,
    debug = False,
    layer_idx = 0,
    impl_version = 0, # 0 for original impl. 1 for v2 impl
    quant_mode = 2, # 2 for per-token, 3 for per-warp
    thresholds = None
):
    bsz, n_head, q_len, dim = q.shape
    if q_len <= 1024:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale = 1 / math.sqrt(dim),
                is_causal=True,
                enable_gqa=True
            )
        return attn_output
    
    assert block_q_size == 64 and block_k_size >= 16
    # assert local_window in [64, 128, 256]
    assert local_window == 128
    assert block_seg_size % 128 == 0

    qq, qk, sq, sk, qmk, smooth_k, mean_k = int4_quant_sub_k_mean_perwarp_SmoothQ_fullsize_allInOne.quant(q, k, block_k_size, False, True)
    # qq, qk, _, _, _, _ = int4_quant_sub_k_mean_pertoken_SmoothQ_allInOne.quant(q, k, False)
    # _, sq, _, sk, _, smooth_k, qmk = int4_quant_sub_k_mean_pertoken_SmoothQ(q, k, 128, 128)
    assert qq.dtype == torch.uint8 and qk.dtype == torch.uint8, f"{qq.dtype},{qk.dtype}"
        

    n_Q_block = math.ceil(q.shape[2] / block_q_size)
    n_K_block = math.ceil(k.shape[2] / block_k_size)

    compute_record = torch.zeros([bsz, n_head, n_Q_block, n_K_block], dtype=torch.int32, device=q.device)
    approx_score = torch.zeros([bsz, n_head, 1, 1], dtype=torch.float32, device=q.device)
    # approx_score = torch.zeros([bsz, n_head, q_len + (block_q_size - q_len % block_q_size), q_len + (8 - q_len % 8)], dtype=torch.float32, device=q.device)

    two_pass = True
    impl_version = 1

    result = FlashAttnFunc.forward(
        q.transpose(1,2), smooth_k.transpose(1,2), v.transpose(1,2),
        qq, qk, sq, sk, qmk, compute_record, approx_score,
        thresholds, block_q_size, block_k_size, local_window, block_seg_size,
        softmax_scale,
        causal,
        softcap,
        impl_version,
        quant_mode,
        two_pass,
        map_only=True
    ).transpose(1,2)
    lut, valid_block_num, q_int8, k_int8, sq8, sk8 = utils.get_block_map_and_quant(
        q, k, mean_k, compute_record, is_causal=True
    )
    scale = 1.0 / (dim ** 0.5)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf(q_int8, k_int8, v, result, lut, valid_block_num, sq8, sk8, 1, 1, 1, scale, True)

    return result

def flash_attn_func_int4_skip(
    q, # [bsz, n_head, q_len, dim]
    k, # [bsz, n_kv_head, q_len, dim]
    v, # [bsz, n_kv_head, q_len, dim]
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    block_q_size = 64,
    block_k_size = 32,
    local_window = 128,
    block_seg_size = 256,
    debug = False,
    layer_idx = 0,
    impl_version = 0, # 0 for original impl. 1 for v2 impl
    quant_mode = 2, # 2 for per-token, 3 for per-warp
    thresholds = None
):
    bsz, n_head, q_len, dim = q.shape
    if q_len <= 1024:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                scale = 1 / math.sqrt(dim),
                is_causal=True,
                enable_gqa=True
            )
        return attn_output
    # assert threshold in [0.001, 0.002, 0.004, 0.008]
    # assert block_q_size in [16, 32, 64, 128, 256]
    # assert block_k_size in [16, 32, 64, 128, 256]
    assert block_q_size == 64 and block_k_size >= 16
    # assert local_window in [64, 128, 256]
    assert local_window == 128
    assert block_seg_size % 128 == 0
    if quant_mode == 2:
        qq, qk, sq, sk, qmk, smooth_k = int4_quant_sub_k_mean_pertoken_SmoothQ_allInOne.quant(q, k, False)
        # qq, qk, _, _, _, _ = int4_quant_sub_k_mean_pertoken_SmoothQ_allInOne.quant(q, k, False)
        # _, sq, _, sk, _, smooth_k, qmk = int4_quant_sub_k_mean_pertoken_SmoothQ(q, k, 128, 128)
        assert qq.dtype == torch.uint8 and qk.dtype == torch.uint8, f"{qq.dtype},{qk.dtype}"
    else:
        qq, qk, sq, sk, qmk, smooth_k = int4_quant_sub_k_mean_perwarp_SmoothQ_fullsize_allInOne.quant(q, k, block_k_size, False)
        assert qq.dtype == torch.uint8 and qk.dtype == torch.uint8, f"{qq.dtype},{qk.dtype}"
        

    n_Q_block = math.ceil(q.shape[2] / block_q_size)
    n_K_block = math.ceil(k.shape[2] / block_k_size)


    if debug:
        compute_record = torch.zeros([bsz, n_head, n_Q_block, n_K_block], dtype=torch.int32, device=q.device)
        # approx_score = torch.zeros([bsz, n_head, n_Q_block, n_K_block], dtype=q.dtype, device=q.device)
        approx_score = torch.zeros([bsz, n_head, q_len + (block_q_size - q_len % block_q_size), q_len + (8 - q_len % 8)], dtype=torch.float32, device=q.device)
    else:
        compute_record, approx_score = None, None

    if impl_version > 1:
        two_pass = True
        impl_version = 1
    else:
        two_pass = False

    result = FlashAttnFunc.forward(
        q.transpose(1,2), smooth_k.transpose(1,2), v.transpose(1,2),
        qq, qk, sq, sk, qmk, compute_record, approx_score,
        thresholds, block_q_size, block_k_size, local_window, block_seg_size,
        softmax_scale,
        causal,
        softcap,
        impl_version,
        quant_mode,
        two_pass,
        False
    ).transpose(1,2)

    return result