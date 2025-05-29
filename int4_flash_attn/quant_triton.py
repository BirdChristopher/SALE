import os
# os.environ["TRITON_INTERPRET"] = "1"
import torch
import math
import triton.language as tl
import triton

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

pertoken_autotune_configs =  [
    triton.Config(kwargs={"BLOCK_SEQ_LEN":16}, num_stages=8, num_warps=4) 
]

class int4_quant_sub_k_mean_pertoken_SmoothQ_allInOne:
    @staticmethod
    @triton.autotune(configs=pertoken_autotune_configs, key = ["H","N_CTX", "HEAD_GROUP_SIZE"])
    @triton.jit
    def triton_fwd(q, k, q_mean, k_mean, qq, qk, sq, sk, qmk, smooth_k,
                    stride_q_bsz, stride_q_head, stride_q_seqlen, stride_q_dim,
                    stride_k_bsz, stride_k_head, stride_k_seqlen, stride_k_dim,
                    stride_q_mean_bsz, stride_q_mean_head, stride_q_mean_dim,
                    stride_k_mean_bsz, stride_k_mean_head, stride_k_mean_dim,
                    stride_qq_bsz, stride_qq_head, stride_qq_seqlen, stride_qq_dim,
                    stride_qk_bsz, stride_qk_head, stride_qk_seqlen, stride_qk_dim,
                    stride_sq_bsz, stride_sq_head, stride_sq_seqlen, 
                    stride_sk_bsz, stride_sk_head, stride_sk_seqlen,
                    stride_qmk_bsz, stride_qmk_head, stride_qmk_seqlen,
                    stride_smk_bsz, stride_smk_head, stride_smk_seqlen, stride_smk_dim,
                    Z, N_CTX, N_CTX_PART,
                    H: tl.constexpr,
                    HeadDim: tl.constexpr,
                    HEAD_GROUP_SIZE: tl.constexpr,
                    BLOCK_SEQ_LEN: tl.constexpr
                    ):
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1) 
        off_z = off_hz // H
        off_h = off_hz % H
        
        q_offset = off_z * stride_q_bsz + off_h * stride_q_head * HEAD_GROUP_SIZE
        k_offset = off_z * stride_k_bsz + off_h * stride_k_head
        
        q_mean_offset = off_z * stride_q_mean_bsz + off_h * stride_q_mean_head * HEAD_GROUP_SIZE
        k_mean_offset = off_z * stride_k_mean_bsz + off_h * stride_k_mean_head

        qq_offset = off_z * stride_qq_bsz + off_h * stride_qq_head * HEAD_GROUP_SIZE
        qk_offset = off_z * stride_qk_bsz + off_h * stride_qk_head

        sq_offset = off_z * stride_sq_bsz + off_h * stride_sq_head * HEAD_GROUP_SIZE
        sk_offset = off_z * stride_sk_bsz + off_h * stride_sk_head

        qmk_offset = off_z * stride_qmk_bsz + off_h * stride_qmk_head * HEAD_GROUP_SIZE

        smooth_k_offset = off_z * stride_smk_bsz + off_h * stride_smk_head

        D_idx = tl.arange(0, HeadDim)
        HALF_D_idx = tl.arange(0, HeadDim // 2)
        SEQLEN_idx = tl.arange(0, BLOCK_SEQ_LEN) + start_n * N_CTX_PART
        HG_idx = tl.arange(0, HEAD_GROUP_SIZE)

        q_ptr = q + q_offset + HG_idx[:, None, None] * stride_q_head + SEQLEN_idx[None, :, None] * stride_q_seqlen + D_idx[None, None, :] * stride_q_dim
        k_ptr = k + k_offset + SEQLEN_idx[:, None] * stride_k_seqlen + D_idx[None, :] * stride_k_dim
        smooth_k_ptr = smooth_k + smooth_k_offset + SEQLEN_idx[:, None] * stride_k_seqlen + D_idx[None, :] * stride_k_dim

        qq_ptr = qq + qq_offset + HG_idx[:, None, None] * stride_qq_head + SEQLEN_idx[None, :, None] * stride_qq_seqlen + HALF_D_idx[None, None, :] * stride_qq_dim
        qk_ptr = qk + qk_offset + SEQLEN_idx[ :, None] * stride_qk_seqlen + HALF_D_idx[None, :] * stride_qk_dim

        sq_ptr = sq + sq_offset + HG_idx[:, None] * stride_sq_head + SEQLEN_idx[None, :] * stride_sq_seqlen
        sk_ptr = sk + sk_offset + SEQLEN_idx * stride_sk_seqlen

        qmk_ptr = qmk + qmk_offset + HG_idx[:, None] * stride_qmk_head + SEQLEN_idx[None, :] * stride_qmk_seqlen

        q_mean_ptr = q_mean + q_mean_offset + HG_idx[:, None] * stride_q_mean_head + D_idx[None,:] * stride_q_mean_dim
        k_mean_ptr = k_mean + k_mean_offset + D_idx * stride_k_mean_dim

        q_mean = tl.load(q_mean_ptr) # [H_G, dim]
        k_mean = tl.load(k_mean_ptr) # [dim]

        INBOUND_N_CTX_PART = tl.minimum(N_CTX_PART, N_CTX - start_n * N_CTX_PART)
        total_iter = tl.ceil(INBOUND_N_CTX_PART / BLOCK_SEQ_LEN).to(tl.int32)
        lo = start_n * N_CTX_PART
        hi = start_n * N_CTX_PART + (total_iter - 1) * BLOCK_SEQ_LEN

        seqlen_offset = lo
        # off_band
        for _ in tl.range(lo, hi, BLOCK_SEQ_LEN):
            q_smem = tl.load(q_ptr) # [H_G, BLOCK_SEQ_LEN, dim]
            k_smem = tl.load(k_ptr) # [BLOCK_SEQ_LEN, dim]
            
            q_smem = q_smem - q_mean[:, None, :]
            k_smem = k_smem - k_mean[None, :]

            tl.store(smooth_k_ptr, k_smem)

            qmk_value = tl.sum(q_mean[:, None, :] * k_smem[None, :, :], axis=2)
            
            tl.store(qmk_ptr, qmk_value)

            q_abs = tl.abs(q_smem)
            q_max = tl.max(q_abs, axis=2) # [H_G, BLOCK_SEQ_LEN]
            k_abs = tl.abs(k_smem)
            k_max = tl.max(k_abs, axis=1) # [BLOCK_SEQ_LEN]

            q_scale = 7 / q_max
            k_scale = 7 / k_max

            int32_q = tl.extra.cuda.libdevice.float2int_rn((q_smem * q_scale[:, :, None]).to(tl.float32)) 
            int32_k = tl.extra.cuda.libdevice.float2int_rn((k_smem * k_scale[:, None]).to(tl.float32))

            int4_q = int32_q.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
            int4_q_odd_channel, int4_q_even_channel = int4_q.reshape([HEAD_GROUP_SIZE, BLOCK_SEQ_LEN, HeadDim // 2, 2]).split()
            int4_pack_q = int4_q_odd_channel << 4 | int4_q_even_channel

            int4_k = int32_k.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
            int4_k_odd_channel, int4_k_even_channel = int4_k.reshape([BLOCK_SEQ_LEN, HeadDim // 2, 2]).split()
            int4_pack_k = int4_k_odd_channel << 4 | int4_k_even_channel

            tl.store(qq_ptr, int4_pack_q)
            tl.store(qk_ptr, int4_pack_k)

            tl.store(sq_ptr, 1 / q_scale)
            tl.store(sk_ptr, 1 / k_scale)

            q_ptr = q_ptr + BLOCK_SEQ_LEN * stride_q_seqlen
            k_ptr = k_ptr + BLOCK_SEQ_LEN * stride_k_seqlen
            smooth_k_ptr = smooth_k_ptr + BLOCK_SEQ_LEN * stride_k_seqlen
            qq_ptr = qq_ptr + BLOCK_SEQ_LEN * stride_qq_seqlen
            qk_ptr = qk_ptr + BLOCK_SEQ_LEN * stride_qk_seqlen
            qmk_ptr = qmk_ptr + BLOCK_SEQ_LEN  * stride_qmk_seqlen
            sq_ptr = sq_ptr + BLOCK_SEQ_LEN * stride_sq_seqlen
            sk_ptr = sk_ptr + BLOCK_SEQ_LEN * stride_sk_seqlen
            
            seqlen_offset += BLOCK_SEQ_LEN

        # on_band
        # n_ctx_mask = (seqlen_offset + SEQLEN_idx) < INBOUND_N_CTX_PART
        n_ctx_mask = (seqlen_offset + tl.arange(0, BLOCK_SEQ_LEN)) < N_CTX
        q_smem = tl.load(q_ptr, mask=n_ctx_mask[None, :, None], other=0.01) # [H_G, BLOCK_SEQ_LEN, dim]
        k_smem = tl.load(k_ptr, mask=n_ctx_mask[:, None], other = 0.01) # [BLOCK_SEQ_LEN, dim]
        
        q_smem = q_smem - q_mean[:, None, :]
        k_smem = k_smem - k_mean[None, :]

        tl.store(smooth_k_ptr, k_smem, mask = n_ctx_mask[:, None])

        qmk_value = q_mean[:, None, :] * k_smem[None, :, :]
        qmk_value = tl.sum(qmk_value, axis=2)
        tl.store(qmk_ptr, qmk_value, mask = n_ctx_mask[None, :])

        q_abs = tl.abs(q_smem)
        q_max = tl.max(q_abs, axis=2) # [H_G, BLOCK_SEQ_LEN]
        k_abs = tl.abs(k_smem)
        k_max = tl.max(k_abs, axis=1) # [BLOCK_SEQ_LEN]

        q_scale = 7 / q_max
        k_scale = 7 / k_max

        int32_q = tl.extra.cuda.libdevice.float2int_rn((q_smem * q_scale[:, :, None]).to(tl.float32)) 
        int32_k = tl.extra.cuda.libdevice.float2int_rn((k_smem * k_scale[:, None]).to(tl.float32))

        int4_q = int32_q.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
        int4_q_odd_channel, int4_q_even_channel = int4_q.reshape([HEAD_GROUP_SIZE, BLOCK_SEQ_LEN, HeadDim // 2, 2]).split()
        int4_pack_q = int4_q_odd_channel << 4 | int4_q_even_channel

        int4_k = int32_k.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
        int4_k_odd_channel, int4_k_even_channel = int4_k.reshape([BLOCK_SEQ_LEN, HeadDim // 2, 2]).split()
        int4_pack_k = int4_k_odd_channel << 4 | int4_k_even_channel

        tl.store(qq_ptr, int4_pack_q, mask = n_ctx_mask[None, :, None]) # [HG, BLOCK_SEQ_LEN, dim // 2]
        tl.store(qk_ptr, int4_pack_k, mask = n_ctx_mask[:, None]) # [BLOCK_SEQ_LEN, dim // 2]

        tl.store(sq_ptr, 1 / q_scale, mask = n_ctx_mask[None, :])
        tl.store(sk_ptr, 1 / k_scale, mask = n_ctx_mask)

    @staticmethod
    def quant(q:torch.Tensor, k:torch.Tensor, new_tensor = False):
        if new_tensor: # for latency profiling
            q = torch.rand_like(q)
            k = torch.rand_like(k)
        bsz, n_head, q_len, dim = q.shape
        _, kv_head, k_len, dim = k.shape
        assert q_len == k_len
        assert k_len > kv_head, "Seems the order of BNHD is not right."

        # NOTE: hopefully it won't cost too much time.
        
        q_mean = q.mean(dim = 2) # [bsz, n_head, dim] 
        k_mean = k.mean(dim = 2) # [bsz, kv_head, dim]
        
        # q_mean and k_mean' digits is not strictly consistent. 

        smooth_k = torch.empty_like(k) 

        qq = torch.zeros([bsz, n_head, q_len, dim // 2], device=q.device, dtype = torch.uint8)
        qk = torch.zeros([bsz, kv_head, k_len,dim // 2], device=k.device, dtype = torch.uint8)
        sq = q.new_zeros([bsz, n_head, q_len + (64 - q_len % 64)])
        sk = q.new_zeros([bsz, kv_head, k_len + (64 - k_len % 64)])
        qmk = q.new_zeros([bsz,  n_head, k_len + (64 - k_len % 64)])

        num_SM = torch.cuda.get_device_properties(q.device).multi_processor_count
        sm_per_kv_head = num_SM // (bsz * kv_head)
        assert sm_per_kv_head > 0, f"{num_SM}, {bsz}, {kv_head}"

        seqlen_per_SM = math.ceil(q_len / sm_per_kv_head)

        grid = (sm_per_kv_head, bsz * kv_head, 1)
        # seqlen_per_SM = q_len
        # grid = (1, bsz * kv_head, 1)

        with torch.cuda.device(q.device):
            int4_quant_sub_k_mean_pertoken_SmoothQ_allInOne.triton_fwd[grid](
                q, k, q_mean, k_mean, qq, qk, sq, sk, qmk, smooth_k,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                q_mean.stride(0), q_mean.stride(1), q_mean.stride(2),
                k_mean.stride(0), k_mean.stride(1), k_mean.stride(2),
                qq.stride(0), qq.stride(1), qq.stride(2), qq.stride(3),
                qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
                sq.stride(0), sq.stride(1), sq.stride(2),
                sk.stride(0), sk.stride(1), sk.stride(2),
                qmk.stride(0), qmk.stride(1), qmk.stride(2),
                smooth_k.stride(0), smooth_k.stride(1), smooth_k.stride(2), smooth_k.stride(3),
                bsz, q_len, seqlen_per_SM, kv_head, dim, n_head//kv_head
            )

        return qq, qk, sq, sk, qmk, smooth_k


perwarp_autotune_configs =  [
    triton.Config(kwargs={}, num_stages=4, num_warps=4) 
]


class int4_quant_sub_k_mean_perwarp_SmoothQ_allInOne:

    @staticmethod
    @triton.jit
    def select(target, index, n):
        indicator = tl.arange(0, n) == index
        return tl.sum(target * indicator)
    
    @staticmethod
    @triton.autotune(configs=perwarp_autotune_configs, key = ["H","N_CTX", "HEAD_GROUP_SIZE"])
    @triton.jit
    def triton_fwd(q, k, q_mean, k_mean, qq, qk, sq, sk, qmk, smooth_k,
                    stride_q_bsz, stride_q_head, stride_q_seqlen, stride_q_dim,
                    stride_k_bsz, stride_k_head, stride_k_seqlen, stride_k_dim,
                    stride_q_mean_bsz, stride_q_mean_head, stride_q_mean_dim,
                    stride_k_mean_bsz, stride_k_mean_head, stride_k_mean_dim,
                    stride_qq_bsz, stride_qq_head, stride_qq_seqlen, stride_qq_dim,
                    stride_qk_bsz, stride_qk_head, stride_qk_seqlen, stride_qk_dim,
                    stride_sq_bsz, stride_sq_head, stride_sq_seqlen, 
                    stride_sk_bsz, stride_sk_head, stride_sk_seqlen,
                    stride_qmk_bsz, stride_qmk_head, stride_qmk_seqlen,
                    stride_smk_bsz, stride_smk_head, stride_smk_seqlen, stride_smk_dim,
                    Z, N_CTX, N_CTX_PART, 
                    kBlockN: tl.constexpr,
                    H: tl.constexpr,
                    HeadDim: tl.constexpr,
                    HEAD_GROUP_SIZE: tl.constexpr
                    ):
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1) 
        off_z = off_hz // H
        off_h = off_hz % H

        INBOUND_N_CTX_PART = tl.minimum(N_CTX_PART, N_CTX - start_n * N_CTX_PART)
        total_iter = tl.ceil(INBOUND_N_CTX_PART / kBlockN).to(tl.int32)

        # Out of range, exit.
        if total_iter <= 0:
            return 
        
        q_offset = off_z * stride_q_bsz + off_h * stride_q_head * HEAD_GROUP_SIZE
        k_offset = off_z * stride_k_bsz + off_h * stride_k_head
        
        q_mean_offset = off_z * stride_q_mean_bsz + off_h * stride_q_mean_head * HEAD_GROUP_SIZE
        k_mean_offset = off_z * stride_k_mean_bsz + off_h * stride_k_mean_head

        qq_offset = off_z * stride_qq_bsz + off_h * stride_qq_head * HEAD_GROUP_SIZE
        qk_offset = off_z * stride_qk_bsz + off_h * stride_qk_head

        sq_offset = off_z * stride_sq_bsz + off_h * stride_sq_head * HEAD_GROUP_SIZE
        sk_offset = off_z * stride_sk_bsz + off_h * stride_sk_head

        qmk_offset = off_z * stride_qmk_bsz + off_h * stride_qmk_head * HEAD_GROUP_SIZE

        smooth_k_offset = off_z * stride_smk_bsz + off_h * stride_smk_head

        D_idx = tl.arange(0, HeadDim)
        HALF_D_idx = tl.arange(0, HeadDim // 2)
        SEQLEN_idx = tl.arange(0, kBlockN) + start_n * N_CTX_PART 
        SK_SEQ_idx = tl.arange(0, 4) + start_n * (N_CTX_PART // kBlockN * 4)
        SQ_SEQ_idx = tl.arange(0, kBlockN // 2) + start_n * N_CTX_PART // 2

        HG_idx = tl.arange(0, HEAD_GROUP_SIZE)

        q_ptr = q + q_offset + HG_idx[:, None, None] * stride_q_head + SEQLEN_idx[None, :, None] * stride_q_seqlen + D_idx[None, None, :] * stride_q_dim
        k_ptr = k + k_offset + SEQLEN_idx[:, None] * stride_k_seqlen + D_idx[None, :] * stride_k_dim
        smooth_k_ptr = smooth_k + smooth_k_offset + SEQLEN_idx[:, None] * stride_k_seqlen + D_idx[None, :] * stride_k_dim

        qq_ptr = qq + qq_offset + HG_idx[:, None, None] * stride_qq_head + SEQLEN_idx[None, :, None] * stride_qq_seqlen + HALF_D_idx[None, None, :] * stride_qq_dim
        qk_ptr = qk + qk_offset + SEQLEN_idx[ :, None] * stride_qk_seqlen + HALF_D_idx[None, :] * stride_qk_dim

        sq_ptr = sq + sq_offset + HG_idx[:, None] * stride_sq_head + SQ_SEQ_idx[None, :] * stride_sq_seqlen
        sk_ptr = sk + sk_offset + SK_SEQ_idx * stride_sk_seqlen

        qmk_ptr = qmk + qmk_offset + HG_idx[:, None] * stride_qmk_head + SEQLEN_idx[None, :] * stride_qmk_seqlen

        q_mean_ptr = q_mean + q_mean_offset + HG_idx[:, None] * stride_q_mean_head + D_idx[None,:] * stride_q_mean_dim
        k_mean_ptr = k_mean + k_mean_offset + D_idx * stride_k_mean_dim

        q_mean = tl.load(q_mean_ptr) # [H_G, dim]
        k_mean = tl.load(k_mean_ptr) # [dim]

        lo = start_n * N_CTX_PART
        hi = start_n * N_CTX_PART + (total_iter - 1) * kBlockN

        seqlen_offset = lo
        # off_band
        for i in tl.range(lo, hi, kBlockN):
            q_smem = tl.load(q_ptr) # [H_G, BLOCK_SEQ_LEN, dim]
            k_smem = tl.load(k_ptr) # [BLOCK_SEQ_LEN, dim]
            
            q_smem = q_smem - q_mean[:, None, :]
            k_smem = k_smem - k_mean[None, :]

            tl.store(smooth_k_ptr, k_smem)
            qmk_value = tl.sum(q_mean[:, None, :] * k_smem[None, :, :], axis=2) # [H_G, BLOCK_SEQ_LEM]

            q_smem_per_thr = q_smem.reshape([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, HeadDim])
            q_abs = tl.abs(q_smem_per_thr)
            q_max = tl.max(tl.max(q_abs, axis = 2, keep_dims=True), axis = 4, keep_dims=True)
            q_scale = 7 / q_max 
            to_store_sq = (q_max / 7).reshape([HEAD_GROUP_SIZE, kBlockN // 2])
            tl.store(sq_ptr, to_store_sq)
            q_scale = q_scale.broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, HeadDim]).reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim])

            k_smem_per_thr = k_smem.reshape([kBlockN // 8, 4, 2, HeadDim])
            k_abs = tl.abs(k_smem_per_thr)
            k_max = tl.max(tl.max(tl.max(k_abs,axis = 3, keep_dims=True), axis = 2, keep_dims = True), axis = 0, keep_dims = True) # [1, 4, 1, 1]
            k_scale = 7 / k_max   # [4]
            tl.store(sk_ptr, 1 / k_scale.reshape([4]))
            k_scale = k_scale.broadcast_to([kBlockN // 8, 4, 2, 1]).reshape([kBlockN, 1])

            qmk_sk = qmk_value * k_scale.reshape([kBlockN])[None, :]
            tl.store(qmk_ptr, qmk_sk)

            k_scale = k_scale.broadcast_to([kBlockN, HeadDim])

            int32_q = tl.extra.cuda.libdevice.float2int_rn((q_smem * q_scale).to(tl.float32)) 
            int32_k = tl.extra.cuda.libdevice.float2int_rn((k_smem * k_scale).to(tl.float32))

            int4_q = int32_q.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
            int4_q_odd_channel, int4_q_even_channel = int4_q.reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim // 2, 2]).split()
            int4_pack_q = int4_q_odd_channel << 4 | int4_q_even_channel

            int4_k = int32_k.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
            int4_k_odd_channel, int4_k_even_channel = int4_k.reshape([kBlockN, HeadDim // 2, 2]).split()
            int4_pack_k = int4_k_odd_channel << 4 | int4_k_even_channel

            tl.store(qq_ptr, int4_pack_q)
            tl.store(qk_ptr, int4_pack_k)

            q_ptr = q_ptr + kBlockN * stride_q_seqlen
            k_ptr = k_ptr + kBlockN * stride_k_seqlen
            smooth_k_ptr = smooth_k_ptr + kBlockN * stride_k_seqlen
            qq_ptr = qq_ptr + kBlockN * stride_qq_seqlen
            qk_ptr = qk_ptr + kBlockN * stride_qk_seqlen
            qmk_ptr = qmk_ptr + kBlockN  * stride_qmk_seqlen
            sq_ptr = sq_ptr + kBlockN // 2 * stride_sq_seqlen
            sk_ptr = sk_ptr + 4 * stride_sk_seqlen
            
            seqlen_offset += kBlockN

        # on_band
        # n_ctx_mask = (seqlen_offset + SEQLEN_idx) < INBOUND_N_CTX_PART
        n_ctx_mask = (seqlen_offset + tl.arange(0, kBlockN)) < N_CTX

        q_smem = tl.load(q_ptr, mask=n_ctx_mask[None, :, None], other = q_mean[:, None, :]) # [H_G, BLOCK_SEQ_LEN, dim]
        k_smem = tl.load(k_ptr, mask=n_ctx_mask[:, None], other = k_mean[None, :]) # [BLOCK_SEQ_LEN, dim]
        
        q_smem = q_smem - q_mean[:, None, :]
        k_smem = k_smem - k_mean[None, :]

        tl.store(smooth_k_ptr, k_smem, mask = n_ctx_mask[:, None])

        qmk_value = q_mean[:, None, :] * k_smem[None, :, :]
        qmk_value = tl.sum(qmk_value, axis=2) # [H_G, BLOCK_SEQ_LEM]

        q_smem_per_thr = q_smem.reshape([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, HeadDim])
        q_abs = tl.abs(q_smem_per_thr)
        q_max = tl.max(tl.max(q_abs, axis = 2, keep_dims=True), axis = 4, keep_dims=True)
        q_scale = 7 / q_max 
        to_store_sq = 1 / q_scale.reshape([HEAD_GROUP_SIZE, kBlockN // 2])
        tl.store(sq_ptr, to_store_sq)
        q_scale = q_scale.broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8,  HeadDim]).reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim])

        k_smem_per_thr = k_smem.reshape([kBlockN // 8, 4, 2, HeadDim])
        k_abs = tl.abs(k_smem_per_thr)
        k_max = tl.max(tl.max(tl.max(k_abs, axis = 3, keep_dims=True), axis = 2, keep_dims = True), axis = 0, keep_dims = True) # [1, 4, 1, 1]
        k_scale = 7 / k_max   # [4]
        tl.store(sk_ptr, 1 / k_scale.reshape([4]))
        k_scale = k_scale.broadcast_to([(kBlockN // 8), 4, 2, 1]).reshape([kBlockN, 1])

        qmk_sk = qmk_value * k_scale.reshape([kBlockN])[None, :]
        tl.store(qmk_ptr, qmk_sk)

        k_scale = k_scale.broadcast_to([kBlockN, HeadDim])

        int32_q = tl.extra.cuda.libdevice.float2int_rn((q_smem * q_scale).to(tl.float32)) 
        int32_k = tl.extra.cuda.libdevice.float2int_rn((k_smem * k_scale).to(tl.float32))

        int4_q = int32_q.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
        int4_q_odd_channel, int4_q_even_channel = int4_q.reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim // 2, 2]).split()
        int4_pack_q = int4_q_odd_channel << 4 | int4_q_even_channel

        int4_k = int32_k.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
        int4_k_odd_channel, int4_k_even_channel = int4_k.reshape([kBlockN, HeadDim // 2, 2]).split()
        int4_pack_k = int4_k_odd_channel << 4 | int4_k_even_channel

        tl.store(qq_ptr, int4_pack_q, mask = n_ctx_mask[None, :, None]) # [HG, BLOCK_SEQ_LEN, dim // 2]
        tl.store(qk_ptr, int4_pack_k, mask = n_ctx_mask[:, None]) # [BLOCK_SEQ_LEN, dim // 2]


    @staticmethod
    def quant(q:torch.Tensor, k:torch.Tensor, kBlockN: int, new_tensor = False):
        assert kBlockN % 16 == 0, "kBlockN should be multiple of 16."
        assert kBlockN <= 64, "kBlockN should be less than 64, or kernel may be much slower."

        if new_tensor: # for latency profiling
            q = torch.rand_like(q)
            k = torch.rand_like(k)
        bsz, n_head, q_len, dim = q.shape
        _, kv_head, k_len, dim = k.shape
        assert q_len == k_len
        assert k_len > kv_head, "Seems the order of BNHD is not right."

        # NOTE: hopefully it won't cost too much time.
        
        q_mean = q.mean(dim = 2) # [bsz, n_head, dim] 
        k_mean = k.mean(dim = 2) # [bsz, kv_head, dim]
        
        # q_mean and k_mean' digits is not strictly consistent. 
        # # Replace q/k_mean with zeros/ones make computation more stable.

        smooth_k = torch.empty_like(k) 

        n_blocks_k = math.ceil(k_len / kBlockN)
        n_scale_k = n_blocks_k * 4           
        n_scale_q = math.ceil(q_len / 2)

        qq = torch.zeros([bsz, n_head, q_len, dim // 2], device=q.device, dtype = torch.uint8)
        qk = torch.zeros([bsz, kv_head, k_len,dim // 2], device=k.device, dtype = torch.uint8)
        sq = q.new_zeros([bsz, n_head, n_scale_q + (64 - n_scale_q % 64)]) # per token
        sk = q.new_zeros([bsz, kv_head, n_scale_k + (64 - n_scale_k % 64)]) # per warp
        qmk = q.new_empty([bsz,  n_head, k_len + (64 - k_len % 64)])

        num_SM = torch.cuda.get_device_properties(q.device).multi_processor_count
        sm_per_kv_head = num_SM // (bsz * kv_head)
        assert sm_per_kv_head > 0, f"{num_SM}, {bsz}, {kv_head}"

        seqlen_per_SM = math.ceil(q_len / (kBlockN * sm_per_kv_head)) * kBlockN

        grid = (sm_per_kv_head, bsz * kv_head, 1)
        # seqlen_per_SM = q_len
        # grid = (1, bsz * kv_head, 1)

        with torch.cuda.device(q.device):
            int4_quant_sub_k_mean_perwarp_SmoothQ_allInOne.triton_fwd[grid](
                q, k, q_mean, k_mean, qq, qk, sq, sk, qmk, smooth_k,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                q_mean.stride(0), q_mean.stride(1), q_mean.stride(2),
                k_mean.stride(0), k_mean.stride(1), k_mean.stride(2),
                qq.stride(0), qq.stride(1), qq.stride(2), qq.stride(3),
                qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
                sq.stride(0), sq.stride(1), sq.stride(2),
                sk.stride(0), sk.stride(1), sk.stride(2),
                qmk.stride(0), qmk.stride(1), qmk.stride(2),
                smooth_k.stride(0), smooth_k.stride(1), smooth_k.stride(2), smooth_k.stride(3),
                bsz, q_len, seqlen_per_SM, kBlockN, kv_head, dim, n_head//kv_head
            )

        return qq, qk, sq, sk, qmk, smooth_k

class int4_quant_sub_k_mean_perwarp_SmoothQ_fullsize_allInOne_fuse_sage8:
    @staticmethod
    @triton.autotune(configs=perwarp_autotune_configs, key = ["H","N_CTX", "HEAD_GROUP_SIZE"])
    @triton.jit
    def triton_fwd(q, k, q_mean, k_mean, qq, qk, qq8, qk8, sq, sk, qmk, smooth_k,
                    stride_q_bsz, stride_q_head, stride_q_seqlen, stride_q_dim,
                    stride_k_bsz, stride_k_head, stride_k_seqlen, stride_k_dim,
                    stride_q_mean_bsz, stride_q_mean_head, stride_q_mean_dim,
                    stride_k_mean_bsz, stride_k_mean_head, stride_k_mean_dim,
                    stride_qq_bsz, stride_qq_head, stride_qq_seqlen, stride_qq_dim,
                    stride_qk_bsz, stride_qk_head, stride_qk_seqlen, stride_qk_dim,
                    stride_qq8_bsz, stride_qq8_head, stride_qq8_seqlen, stride_qq8_dim,
                    stride_qk8_bsz, stride_qk8_head, stride_qk8_seqlen, stride_qk8_dim,
                    stride_sq_bsz, stride_sq_head, stride_sq_seqlen, 
                    stride_sk_bsz, stride_sk_head, stride_sk_seqlen,
                    stride_sq8_bsz, stride_sq8_head, stride_sq8_seqlen, 
                    stride_sk8_bsz, stride_sk8_head, stride_sk8_seqlen,
                    stride_qmk_bsz, stride_qmk_head, stride_qmk_seqlen,
                    stride_smk_bsz, stride_smk_head, stride_smk_seqlen, stride_smk_dim,
                    Z, N_CTX, N_CTX_PART, 
                    kBlockN: tl.constexpr,
                    H: tl.constexpr,
                    HeadDim: tl.constexpr,
                    HEAD_GROUP_SIZE: tl.constexpr
                    ):
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1) 
        off_z = off_hz // H
        off_h = off_hz % H

        INBOUND_N_CTX_PART = tl.minimum(N_CTX_PART, N_CTX - start_n * N_CTX_PART)
        total_iter = tl.ceil(INBOUND_N_CTX_PART / kBlockN).to(tl.int32)

        # Out of range, exit.
        if total_iter <= 0:
            return 
        
        q_offset = off_z * stride_q_bsz + off_h * stride_q_head * HEAD_GROUP_SIZE
        k_offset = off_z * stride_k_bsz + off_h * stride_k_head
        
        q_mean_offset = off_z * stride_q_mean_bsz + off_h * stride_q_mean_head * HEAD_GROUP_SIZE
        k_mean_offset = off_z * stride_k_mean_bsz + off_h * stride_k_mean_head

        qq_offset = off_z * stride_qq_bsz + off_h * stride_qq_head * HEAD_GROUP_SIZE
        qk_offset = off_z * stride_qk_bsz + off_h * stride_qk_head

        qq8_offset = off_z * stride_qq8_bsz + off_h * stride_qq8_head * HEAD_GROUP_SIZE
        qk8_offset = off_z * stride_qk8_bsz + off_h * stride_qk8_head

        sq_offset = off_z * stride_sq_bsz + off_h * stride_sq_head * HEAD_GROUP_SIZE
        sk_offset = off_z * stride_sk_bsz + off_h * stride_sk_head

        sq8_offset = off_z * stride_sq8_bsz + off_h * stride_sq8_head * HEAD_GROUP_SIZE
        sk8_offset = off_z * stride_sk8_bsz + off_h * stride_sk8_head

        qmk_offset = off_z * stride_qmk_bsz + off_h * stride_qmk_head * HEAD_GROUP_SIZE

        smooth_k_offset = off_z * stride_smk_bsz + off_h * stride_smk_head

        D_idx = tl.arange(0, HeadDim)
        HALF_D_idx = tl.arange(0, HeadDim // 2)
        SEQLEN_idx = tl.arange(0, kBlockN) + start_n * N_CTX_PART 

        HG_idx = tl.arange(0, HEAD_GROUP_SIZE)

        q_ptr = q + q_offset + HG_idx[:, None, None] * stride_q_head + SEQLEN_idx[None, :, None] * stride_q_seqlen + D_idx[None, None, :] * stride_q_dim
        k_ptr = k + k_offset + SEQLEN_idx[:, None] * stride_k_seqlen + D_idx[None, :] * stride_k_dim
        smooth_k_ptr = smooth_k + smooth_k_offset + SEQLEN_idx[:, None] * stride_k_seqlen + D_idx[None, :] * stride_k_dim

        qq_ptr = qq + qq_offset + HG_idx[:, None, None] * stride_qq_head + SEQLEN_idx[None, :, None] * stride_qq_seqlen + HALF_D_idx[None, None, :] * stride_qq_dim
        qk_ptr = qk + qk_offset + SEQLEN_idx[ :, None] * stride_qk_seqlen + HALF_D_idx[None, :] * stride_qk_dim

        qq8_ptr = qq8 + qq8_offset + HG_idx[:, None, None] * stride_qq8_head + SEQLEN_idx[None, :, None] * stride_qq8_seqlen + HALF_D_idx[None, None, :] * stride_qq8_dim
        qk8_ptr = qk8 + qk8_offset + SEQLEN_idx[ :, None] * stride_qk8_seqlen + HALF_D_idx[None, :] * stride_qk8_dim

        sq_ptr = sq + sq_offset + HG_idx[:, None] * stride_sq_head + SEQLEN_idx[None, :] * stride_sq_seqlen
        sk_ptr = sk + sk_offset + SEQLEN_idx * stride_sk_seqlen

        sq8_ptr = sq + sq8_offset + HG_idx[:, None] * stride_sq8_head + SEQLEN_idx[None, :] * stride_sq8_seqlen
        sk8_ptr = sk + sk8_offset + SEQLEN_idx * stride_sk8_seqlen

        qmk_ptr = qmk + qmk_offset + HG_idx[:, None] * stride_qmk_head + SEQLEN_idx[None, :] * stride_qmk_seqlen

        q_mean_ptr = q_mean + q_mean_offset + HG_idx[:, None] * stride_q_mean_head + D_idx[None,:] * stride_q_mean_dim
        k_mean_ptr = k_mean + k_mean_offset + D_idx * stride_k_mean_dim

        q_mean = tl.load(q_mean_ptr) # [H_G, dim]
        k_mean = tl.load(k_mean_ptr) # [dim]

        lo = start_n * N_CTX_PART
        hi = start_n * N_CTX_PART + (total_iter - 1) * kBlockN

        seqlen_offset = lo
        # off_band
        for i in tl.range(lo, hi, kBlockN):
            q_smem = tl.load(q_ptr) # [H_G, BLOCK_SEQ_LEN, dim]
            k_smem = tl.load(k_ptr) # [BLOCK_SEQ_LEN, dim]
            
            q_no_smooth = q_smem
            q_smem = q_smem - q_mean[:, None, :]
            k_smem = k_smem - k_mean[None, :]

            tl.store(smooth_k_ptr, k_smem)
            
            # Block wise int8 quant
            q_ns_abs = tl.abs(q_no_smooth)
            q_ns_max = tl.max(tl.max(q_ns_abs, axis=2), axis=1)
            scale_q_ns_8 = q_ns_max.to(tl.float32) / 127.
            tl.store(sq8_ptr, scale_q_ns_8)

            scale_k_s_8 = tl.max(tl.abs(k_smem)) / 127.
            tl.store(sk8_ptr, scale_q_ns)

            q_int8 = tl.extra.cuda.libdevice.float2int(q_no_smooth / scale_q_ns_8[:,None,None]).to(tl.int8)
            k_int8 = tl.extra.cuda.libdevice.float2int(k_smem / scale_q_ns_8[:,None,None]).to(tl.int8)
            tl.store(qq8_ptr, q_int8)
            tl.store()


            qmk_value = tl.sum(q_mean[:, None, :] * k_smem[None, :, :], axis=2) # [H_G, BLOCK_SEQ_LEM]

            q_smem_per_thr = q_smem.reshape([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, HeadDim])
            q_abs = tl.abs(q_smem_per_thr)
            q_max = tl.max(tl.max(q_abs, axis = 2, keep_dims=True), axis = 4, keep_dims=True) # [HEAD_GROUP_SIZE, kBlockN // 16, 1, 8, 1]
            q_scale = 7 / q_max 
            to_store_sq = (q_max / 7).broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, 1]).reshape([HEAD_GROUP_SIZE, kBlockN])
            tl.store(sq_ptr, to_store_sq)
            q_scale = q_scale.broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, HeadDim]).reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim])

            k_smem_per_thr = k_smem.reshape([kBlockN // 8, 4, 2, HeadDim])
            k_abs = tl.abs(k_smem_per_thr)
            k_max = tl.max(tl.max(tl.max(k_abs,axis = 3, keep_dims=True), axis = 2, keep_dims = True), axis = 0, keep_dims = True) # [1, 4, 1, 1]
            k_scale = 7 / k_max   # [4]
            k_scale = k_scale.broadcast_to([kBlockN // 8, 4, 2, 1]).reshape([kBlockN, 1])
            tl.store(sk_ptr, 1 / k_scale.reshape([kBlockN]))

            qmk_sk = qmk_value * k_scale.reshape([kBlockN])[None, :]
            tl.store(qmk_ptr, qmk_sk)

            k_scale = k_scale.broadcast_to([kBlockN, HeadDim])
            int32_q = tl.extra.cuda.libdevice.float2int_rn((q_smem * q_scale).to(tl.float32)) 
            int32_k = tl.extra.cuda.libdevice.float2int_rn((k_smem * k_scale).to(tl.float32))

            int4_q = int32_q.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
            int4_q_odd_channel, int4_q_even_channel = int4_q.reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim // 2, 2]).split()
            int4_pack_q = int4_q_odd_channel << 4 | int4_q_even_channel

            int4_k = int32_k.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
            int4_k_odd_channel, int4_k_even_channel = int4_k.reshape([kBlockN, HeadDim // 2, 2]).split()
            int4_pack_k = int4_k_odd_channel << 4 | int4_k_even_channel

            tl.store(qq_ptr, int4_pack_q)
            tl.store(qk_ptr, int4_pack_k)

            q_ptr = q_ptr + kBlockN * stride_q_seqlen
            k_ptr = k_ptr + kBlockN * stride_k_seqlen
            smooth_k_ptr = smooth_k_ptr + kBlockN * stride_k_seqlen
            qq_ptr = qq_ptr + kBlockN * stride_qq_seqlen
            qk_ptr = qk_ptr + kBlockN * stride_qk_seqlen
            qmk_ptr = qmk_ptr + kBlockN  * stride_qmk_seqlen
            sq_ptr = sq_ptr + kBlockN * stride_sq_seqlen
            sk_ptr = sk_ptr + kBlockN * stride_sk_seqlen
            
            seqlen_offset += kBlockN

        # on_band
        # n_ctx_mask = (seqlen_offset + SEQLEN_idx) < INBOUND_N_CTX_PART
        n_ctx_mask = (seqlen_offset + tl.arange(0, kBlockN)) < N_CTX
        q_smem = tl.load(q_ptr, mask=n_ctx_mask[None, :, None], other = q_mean[:, None, :]) # [H_G, BLOCK_SEQ_LEN, dim]
        k_smem = tl.load(k_ptr, mask=n_ctx_mask[:, None], other = k_mean[None, :]) # [BLOCK_SEQ_LEN, dim]
        
        q_smem = q_smem - q_mean[:, None, :]
        k_smem = k_smem - k_mean[None, :]

        tl.store(smooth_k_ptr, k_smem, mask = n_ctx_mask[:, None])

        qmk_value = q_mean[:, None, :] * k_smem[None, :, :]
        qmk_value = tl.sum(qmk_value, axis=2) # [H_G, BLOCK_SEQ_LEM]

        q_smem_per_thr = q_smem.reshape([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, HeadDim])
        q_abs = tl.abs(q_smem_per_thr)
        q_max = tl.max(tl.max(q_abs, axis = 2, keep_dims=True), axis = 4, keep_dims=True)
        q_scale = 7 / q_max 
        to_store_sq = (q_max / 7).broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, 1]).reshape([HEAD_GROUP_SIZE, kBlockN])
        tl.store(sq_ptr, to_store_sq)
        q_scale = q_scale.broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8,  HeadDim]).reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim])

        k_smem_per_thr = k_smem.reshape([kBlockN // 8, 4, 2, HeadDim])
        k_abs = tl.abs(k_smem_per_thr)
        k_max = tl.max(tl.max(tl.max(k_abs, axis = 3, keep_dims=True), axis = 2, keep_dims = True), axis = 0, keep_dims = True) # [1, 4, 1, 1]
        k_scale = 7 / k_max   # [4]
        k_scale = k_scale.broadcast_to([(kBlockN // 8), 4, 2, 1]).reshape([kBlockN, 1])
        tl.store(sk_ptr, 1 / k_scale.reshape([kBlockN]))


        qmk_sk = qmk_value * k_scale.reshape([kBlockN])[None, :]
        tl.store(qmk_ptr, qmk_sk)

        k_scale = k_scale.broadcast_to([kBlockN, HeadDim])

        int32_q = tl.extra.cuda.libdevice.float2int_rn((q_smem * q_scale).to(tl.float32)) 
        int32_k = tl.extra.cuda.libdevice.float2int_rn((k_smem * k_scale).to(tl.float32))

        int4_q = int32_q.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
        int4_q_odd_channel, int4_q_even_channel = int4_q.reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim // 2, 2]).split()
        int4_pack_q = int4_q_odd_channel << 4 | int4_q_even_channel

        int4_k = int32_k.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
        int4_k_odd_channel, int4_k_even_channel = int4_k.reshape([kBlockN, HeadDim // 2, 2]).split()
        int4_pack_k = int4_k_odd_channel << 4 | int4_k_even_channel

        tl.store(qq_ptr, int4_pack_q, mask = n_ctx_mask[None, :, None]) # [HG, BLOCK_SEQ_LEN, dim // 2]
        tl.store(qk_ptr, int4_pack_k, mask = n_ctx_mask[:, None]) # [BLOCK_SEQ_LEN, dim // 2]


    @staticmethod
    def quant(q:torch.Tensor, k:torch.Tensor, kBlockN: int, new_tensor = False):
        assert kBlockN % 16 == 0, "kBlockN should be multiple of 16."
        assert kBlockN <= 64, "kBlockN should be less than 64, or kernel may be much slower."

        if new_tensor: # for latency profiling
            q = torch.rand_like(q)
            k = torch.rand_like(k)
        bsz, n_head, q_len, dim = q.shape
        _, kv_head, k_len, dim = k.shape
        assert q_len == k_len
        assert k_len > kv_head, "Seems the order of BNHD is not right."

        # NOTE: hopefully it won't cost too much time.
        
        q_mean = q.mean(dim = 2) # [bsz, n_head, dim] 
        k_mean = k.mean(dim = 2) # [bsz, kv_head, dim]
        
        # q_mean and k_mean' digits is not strictly consistent. 

        smooth_k = torch.empty_like(k) 

        qq = torch.zeros([bsz, n_head, q_len, dim // 2], device=q.device, dtype = torch.uint8)
        qk = torch.zeros([bsz, kv_head, k_len,dim // 2], device=k.device, dtype = torch.uint8)
        qq8 = torch.zeros([bsz, n_head, q_len, dim], device=q.device, dtype = torch.int8)
        qk8 = torch.zeros([bsz, n_head, q_len, dim], device=k.device, dtype = torch.int8)
        sq = q.new_zeros([bsz, n_head, q_len + (64 - q_len % 64)])
        sk = q.new_zeros([bsz, kv_head, k_len + (64 - k_len % 64)])
        sq8 = q.new_zeros([bsz, n_head, (q_len + 64 - 1) // 64], dtype=torch.float32)
        sk8 = q.new_zeros([bsz, kv_head, (k_len + 64 - 1) // 32], dtype=torch.float32)
        qmk = q.new_zeros([bsz,  n_head, k_len + (64 - k_len % 64)])

        num_SM = torch.cuda.get_device_properties(q.device).multi_processor_count
        sm_per_kv_head = num_SM // (bsz * kv_head) if num_SM > (bsz * kv_head) else 1
        assert sm_per_kv_head > 0, f"{num_SM}, {bsz}, {kv_head}"

        seqlen_per_SM = math.ceil(q_len / (kBlockN * sm_per_kv_head)) * kBlockN

        grid = (sm_per_kv_head, bsz * kv_head, 1)
        # seqlen_per_SM = q_len
        # grid = (1, bsz * kv_head, 1)

        with torch.cuda.device(q.device):
            int4_quant_sub_k_mean_perwarp_SmoothQ_fullsize_allInOne_fuse_sage8.triton_fwd[grid](
                q, k, q_mean, k_mean, qq, qk, qq8, qk8, sq, sk, sq8, sk8, qmk, smooth_k,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                q_mean.stride(0), q_mean.stride(1), q_mean.stride(2),
                k_mean.stride(0), k_mean.stride(1), k_mean.stride(2),
                qq.stride(0), qq.stride(1), qq.stride(2), qq.stride(3),
                qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
                qq8.stride(0), qq8.stride(1), qq8.stride(2), qq8.stride(3),
                qk8.stride(0), qk8.stride(1), qk8.stride(2), qk8.stride(3),
                sq.stride(0), sq.stride(1), sq.stride(2),
                sk.stride(0), sk.stride(1), sk.stride(2),
                sq8.stride(0), sq8.stride(1), sq8.stride(2),
                sk8.stride(0), sk8.stride(1), sk8.stride(2),
                qmk.stride(0), qmk.stride(1), qmk.stride(2),
                smooth_k.stride(0), smooth_k.stride(1), smooth_k.stride(2), smooth_k.stride(3),
                bsz, q_len, seqlen_per_SM, kBlockN, kv_head, dim, n_head//kv_head
            )

        return qq, qk, qq8, qk8, sq, sk, sq8, sk8, qmk, smooth_k

class int4_quant_sub_k_mean_perwarp_SmoothQ_fullsize_allInOne:
    @staticmethod
    @triton.autotune(configs=perwarp_autotune_configs, key = ["H","N_CTX", "HEAD_GROUP_SIZE"])
    @triton.jit
    def triton_fwd_q(q, k, q_mean, k_mean, qq, qk, sq, sk, qmk, smooth_k,
                    stride_q_bsz, stride_q_head, stride_q_seqlen, stride_q_dim,
                    stride_k_bsz, stride_k_head, stride_k_seqlen, stride_k_dim,
                    stride_q_mean_bsz, stride_q_mean_head, stride_q_mean_dim,
                    stride_k_mean_bsz, stride_k_mean_head, stride_k_mean_dim,
                    stride_qq_bsz, stride_qq_head, stride_qq_seqlen, stride_qq_dim,
                    stride_qk_bsz, stride_qk_head, stride_qk_seqlen, stride_qk_dim,
                    stride_sq_bsz, stride_sq_head, stride_sq_seqlen, 
                    stride_sk_bsz, stride_sk_head, stride_sk_seqlen,
                    stride_qmk_bsz, stride_qmk_head, stride_qmk_seqlen,
                    stride_smk_bsz, stride_smk_head, stride_smk_seqlen, stride_smk_dim,
                    Z, N_CTX, N_CTX_PART, 
                    kBlockN: tl.constexpr,
                    H: tl.constexpr,
                    HeadDim: tl.constexpr,
                    HEAD_GROUP_SIZE: tl.constexpr
                    ):
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1) 
        off_z = off_hz // H
        off_h = off_hz % H

        INBOUND_N_CTX_PART = tl.minimum(N_CTX_PART, N_CTX - start_n * N_CTX_PART)
        total_iter = tl.ceil(INBOUND_N_CTX_PART / kBlockN).to(tl.int32)

        # Out of range, exit.
        if total_iter <= 0:
            return 
        
        q_offset = off_z * stride_q_bsz + off_h * stride_q_head * HEAD_GROUP_SIZE
        q_mean_offset = off_z * stride_q_mean_bsz + off_h * stride_q_mean_head * HEAD_GROUP_SIZE
        qq_offset = off_z * stride_qq_bsz + off_h * stride_qq_head * HEAD_GROUP_SIZE
        sq_offset = off_z * stride_sq_bsz + off_h * stride_sq_head * HEAD_GROUP_SIZE

        D_idx = tl.arange(0, HeadDim)
        HALF_D_idx = tl.arange(0, HeadDim // 2)
        SEQLEN_idx = tl.arange(0, kBlockN) + start_n * N_CTX_PART 

        HG_idx = tl.arange(0, HEAD_GROUP_SIZE)

        q_ptr = q + q_offset + HG_idx[:, None, None] * stride_q_head + SEQLEN_idx[None, :, None] * stride_q_seqlen + D_idx[None, None, :] * stride_q_dim
        qq_ptr = qq + qq_offset + HG_idx[:, None, None] * stride_qq_head + SEQLEN_idx[None, :, None] * stride_qq_seqlen + HALF_D_idx[None, None, :] * stride_qq_dim
        sq_ptr = sq + sq_offset + HG_idx[:, None] * stride_sq_head + SEQLEN_idx[None, :] * stride_sq_seqlen
        q_mean_ptr = q_mean + q_mean_offset + HG_idx[:, None] * stride_q_mean_head + D_idx[None,:] * stride_q_mean_dim

        q_mean = tl.load(q_mean_ptr) # [H_G, dim]

        lo = start_n * N_CTX_PART
        hi = start_n * N_CTX_PART + (total_iter - 1) * kBlockN

        seqlen_offset = lo
        # off_band
        for i in tl.range(lo, hi, kBlockN):
            q_smem = tl.load(q_ptr) # [H_G, BLOCK_SEQ_LEN, dim]
            q_smem = q_smem - q_mean[:, None, :]
            q_smem_per_thr = q_smem.reshape([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, HeadDim])
            q_abs = tl.abs(q_smem_per_thr)
            q_max = tl.max(tl.max(q_abs, axis = 2, keep_dims=True), axis = 4, keep_dims=True) # [HEAD_GROUP_SIZE, kBlockN // 16, 1, 8, 1]
            q_scale = 7 / q_max 
            to_store_sq = (q_max / 7).broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, 1]).reshape([HEAD_GROUP_SIZE, kBlockN])
            tl.store(sq_ptr, to_store_sq)
            q_scale = q_scale.broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, HeadDim]).reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim])
            int32_q = tl.extra.cuda.libdevice.float2int_rn((q_smem * q_scale).to(tl.float32)) 
            int4_q = int32_q.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
            int4_q_odd_channel, int4_q_even_channel = int4_q.reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim // 2, 2]).split()
            int4_pack_q = int4_q_odd_channel << 4 | int4_q_even_channel
            tl.store(qq_ptr, int4_pack_q)

            q_ptr = q_ptr + kBlockN * stride_q_seqlen
            qq_ptr = qq_ptr + kBlockN * stride_qq_seqlen
            sq_ptr = sq_ptr + kBlockN * stride_sq_seqlen
            
            seqlen_offset += kBlockN

        # on_band
        # n_ctx_mask = (seqlen_offset + SEQLEN_idx) < INBOUND_N_CTX_PART
        n_ctx_mask = (seqlen_offset + tl.arange(0, kBlockN)) < N_CTX

        q_smem = tl.load(q_ptr, mask=n_ctx_mask[None, :, None], other = q_mean[:, None, :]) # [H_G, BLOCK_SEQ_LEN, dim]
        q_smem = q_smem - q_mean[:, None, :]
        q_smem_per_thr = q_smem.reshape([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, HeadDim])
        q_abs = tl.abs(q_smem_per_thr)
        q_max = tl.max(tl.max(q_abs, axis = 2, keep_dims=True), axis = 4, keep_dims=True)
        q_scale = 7 / q_max 
        to_store_sq = (q_max / 7).broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8, 1]).reshape([HEAD_GROUP_SIZE, kBlockN])
        tl.store(sq_ptr, to_store_sq)
        q_scale = q_scale.broadcast_to([HEAD_GROUP_SIZE, kBlockN // 16, 2, 8,  HeadDim]).reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim])
        int32_q = tl.extra.cuda.libdevice.float2int_rn((q_smem * q_scale).to(tl.float32)) 
        int4_q = int32_q.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
        int4_q_odd_channel, int4_q_even_channel = int4_q.reshape([HEAD_GROUP_SIZE, kBlockN, HeadDim // 2, 2]).split()
        int4_pack_q = int4_q_odd_channel << 4 | int4_q_even_channel
        tl.store(qq_ptr, int4_pack_q, mask = n_ctx_mask[None, :, None]) # [HG, BLOCK_SEQ_LEN, dim // 2]

    @staticmethod
    @triton.autotune(configs=perwarp_autotune_configs, key = ["H","N_CTX", "HEAD_GROUP_SIZE"])
    @triton.jit
    def triton_fwd_k(q, k, q_mean, k_mean, qq, qk, sq, sk, qmk, smooth_k,
                    stride_q_bsz, stride_q_head, stride_q_seqlen, stride_q_dim,
                    stride_k_bsz, stride_k_head, stride_k_seqlen, stride_k_dim,
                    stride_q_mean_bsz, stride_q_mean_head, stride_q_mean_dim,
                    stride_k_mean_bsz, stride_k_mean_head, stride_k_mean_dim,
                    stride_qq_bsz, stride_qq_head, stride_qq_seqlen, stride_qq_dim,
                    stride_qk_bsz, stride_qk_head, stride_qk_seqlen, stride_qk_dim,
                    stride_sq_bsz, stride_sq_head, stride_sq_seqlen, 
                    stride_sk_bsz, stride_sk_head, stride_sk_seqlen,
                    stride_qmk_bsz, stride_qmk_head, stride_qmk_seqlen,
                    stride_smk_bsz, stride_smk_head, stride_smk_seqlen, stride_smk_dim,
                    Z, N_CTX, N_CTX_PART, 
                    kBlockN: tl.constexpr,
                    H: tl.constexpr,
                    HeadDim: tl.constexpr,
                    HEAD_GROUP_SIZE: tl.constexpr
                    ):
        start_n = tl.program_id(0)
        off_hz = tl.program_id(1) 
        off_z = off_hz // H
        off_h = off_hz % H

        INBOUND_N_CTX_PART = tl.minimum(N_CTX_PART, N_CTX - start_n * N_CTX_PART)
        total_iter = tl.ceil(INBOUND_N_CTX_PART / kBlockN).to(tl.int32)

        # Out of range, exit.
        if total_iter <= 0:
            return 
        
        k_offset = off_z * stride_k_bsz + off_h * stride_k_head
        
        q_mean_offset = off_z * stride_q_mean_bsz + off_h * stride_q_mean_head * HEAD_GROUP_SIZE
        k_mean_offset = off_z * stride_k_mean_bsz + off_h * stride_k_mean_head

        qk_offset = off_z * stride_qk_bsz + off_h * stride_qk_head
        sk_offset = off_z * stride_sk_bsz + off_h * stride_sk_head
        qmk_offset = off_z * stride_qmk_bsz + off_h * stride_qmk_head * HEAD_GROUP_SIZE
        smooth_k_offset = off_z * stride_smk_bsz + off_h * stride_smk_head

        D_idx = tl.arange(0, HeadDim)
        HALF_D_idx = tl.arange(0, HeadDim // 2)
        SEQLEN_idx = tl.arange(0, kBlockN) + start_n * N_CTX_PART 

        HG_idx = tl.arange(0, HEAD_GROUP_SIZE)

        k_ptr = k + k_offset + SEQLEN_idx[:, None] * stride_k_seqlen + D_idx[None, :] * stride_k_dim
        smooth_k_ptr = smooth_k + smooth_k_offset + SEQLEN_idx[:, None] * stride_k_seqlen + D_idx[None, :] * stride_k_dim
        qk_ptr = qk + qk_offset + SEQLEN_idx[ :, None] * stride_qk_seqlen + HALF_D_idx[None, :] * stride_qk_dim
        sk_ptr = sk + sk_offset + SEQLEN_idx * stride_sk_seqlen
        qmk_ptr = qmk + qmk_offset + HG_idx[:, None] * stride_qmk_head + SEQLEN_idx[None, :] * stride_qmk_seqlen

        q_mean_ptr = q_mean + q_mean_offset + HG_idx[:, None] * stride_q_mean_head + D_idx[None,:] * stride_q_mean_dim
        k_mean_ptr = k_mean + k_mean_offset + D_idx * stride_k_mean_dim

        q_mean = tl.load(q_mean_ptr) # [H_G, dim]
        k_mean = tl.load(k_mean_ptr) # [dim]

        lo = start_n * N_CTX_PART
        hi = start_n * N_CTX_PART + (total_iter - 1) * kBlockN

        seqlen_offset = lo
        # off_band
        for i in tl.range(lo, hi, kBlockN):
            k_smem = tl.load(k_ptr) # [BLOCK_SEQ_LEN, dim]
            k_smem = k_smem - k_mean[None, :]

            tl.store(smooth_k_ptr, k_smem)

            qmk_value = tl.sum(q_mean[:, None, :] * k_smem[None, :, :], axis=2) # [H_G, BLOCK_SEQ_LEM]
            
            k_smem_per_thr = k_smem.reshape([kBlockN // 8, 4, 2, HeadDim])
            k_abs = tl.abs(k_smem_per_thr)
            k_max = tl.max(tl.max(tl.max(k_abs,axis = 3, keep_dims=True), axis = 2, keep_dims = True), axis = 0, keep_dims = True) # [1, 4, 1, 1]
            k_scale = 7 / k_max   # [4]
            k_scale = k_scale.broadcast_to([kBlockN // 8, 4, 2, 1]).reshape([kBlockN, 1])
            tl.store(sk_ptr, 1 / k_scale.reshape([kBlockN]))

            qmk_sk = qmk_value * k_scale.reshape([kBlockN])[None, :]
            tl.store(qmk_ptr, qmk_sk)

            k_scale = k_scale.broadcast_to([kBlockN, HeadDim])
            int32_k = tl.extra.cuda.libdevice.float2int_rn((k_smem * k_scale).to(tl.float32))
            int4_k = int32_k.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
            int4_k_odd_channel, int4_k_even_channel = int4_k.reshape([kBlockN, HeadDim // 2, 2]).split()
            int4_pack_k = int4_k_odd_channel << 4 | int4_k_even_channel
            tl.store(qk_ptr, int4_pack_k)

            k_ptr = k_ptr + kBlockN * stride_k_seqlen
            smooth_k_ptr = smooth_k_ptr + kBlockN * stride_k_seqlen
            qk_ptr = qk_ptr + kBlockN * stride_qk_seqlen
            qmk_ptr = qmk_ptr + kBlockN  * stride_qmk_seqlen
            sk_ptr = sk_ptr + kBlockN * stride_sk_seqlen
            
            seqlen_offset += kBlockN

        # on_band
        # n_ctx_mask = (seqlen_offset + SEQLEN_idx) < INBOUND_N_CTX_PART
        n_ctx_mask = (seqlen_offset + tl.arange(0, kBlockN)) < N_CTX
        
        k_smem = tl.load(k_ptr, mask=n_ctx_mask[:, None], other = k_mean[None, :]) # [BLOCK_SEQ_LEN, dim]
        k_smem = k_smem - k_mean[None, :]

        tl.store(smooth_k_ptr, k_smem, mask = n_ctx_mask[:, None])

        qmk_value = q_mean[:, None, :] * k_smem[None, :, :]
        qmk_value = tl.sum(qmk_value, axis=2) # [H_G, BLOCK_SEQ_LEM]

        k_smem_per_thr = k_smem.reshape([kBlockN // 8, 4, 2, HeadDim])
        k_abs = tl.abs(k_smem_per_thr)
        k_max = tl.max(tl.max(tl.max(k_abs, axis = 3, keep_dims=True), axis = 2, keep_dims = True), axis = 0, keep_dims = True) # [1, 4, 1, 1]
        k_scale = 7 / k_max   # [4]
        k_scale = k_scale.broadcast_to([(kBlockN // 8), 4, 2, 1]).reshape([kBlockN, 1])
        tl.store(sk_ptr, 1 / k_scale.reshape([kBlockN]))

        qmk_sk = qmk_value * k_scale.reshape([kBlockN])[None, :]
        tl.store(qmk_ptr, qmk_sk)

        k_scale = k_scale.broadcast_to([kBlockN, HeadDim])
        int32_k = tl.extra.cuda.libdevice.float2int_rn((k_smem * k_scale).to(tl.float32))
        int4_k = int32_k.to(tl.int8).cast(tl.uint8, bitcast=True) & 0x0F
        int4_k_odd_channel, int4_k_even_channel = int4_k.reshape([kBlockN, HeadDim // 2, 2]).split()
        int4_pack_k = int4_k_odd_channel << 4 | int4_k_even_channel
        tl.store(qk_ptr, int4_pack_k, mask = n_ctx_mask[:, None]) # [BLOCK_SEQ_LEN, dim // 2]


    @staticmethod
    def quant(q:torch.Tensor, k:torch.Tensor, kBlockN: int, new_tensor = False, return_mean_k=False):
        assert kBlockN % 16 == 0, "kBlockN should be multiple of 16."
        assert kBlockN <= 64, "kBlockN should be less than 64, or kernel may be much slower."

        if new_tensor: # for latency profiling
            q = torch.rand_like(q)
            k = torch.rand_like(k)
        bsz, n_head, q_len, dim = q.shape
        _, kv_head, k_len, dim = k.shape
        assert q_len == k_len
        assert k_len > kv_head, "Seems the order of BNHD is not right."

        # NOTE: hopefully it won't cost too much time.
        
        q_mean = q.mean(dim = 2) # [bsz, n_head, dim] 
        k_mean = k.mean(dim = 2) # [bsz, kv_head, dim]
        
        # q_mean and k_mean' digits is not strictly consistent. 

        smooth_k = torch.empty_like(k) 

        qq = torch.zeros([bsz, n_head, q_len, dim // 2], device=q.device, dtype = torch.uint8)
        qk = torch.zeros([bsz, kv_head, k_len,dim // 2], device=k.device, dtype = torch.uint8)
        sq = q.new_zeros([bsz, n_head, q_len + (64 - q_len % 64)])
        sk = q.new_zeros([bsz, kv_head, k_len + (64 - k_len % 64)])
        qmk = q.new_zeros([bsz,  n_head, k_len + (64 - k_len % 64)])

        num_SM = torch.cuda.get_device_properties(q.device).multi_processor_count
        sm_per_head = num_SM // (bsz * n_head) if num_SM > (bsz * n_head) else 1
        assert sm_per_head > 0, f"{num_SM}, {bsz}, {n_head}"
        seqlen_per_SM = math.ceil(q_len / (kBlockN * sm_per_head)) * kBlockN

        grid = (sm_per_head, bsz * n_head, 1)
        # grid = (sm_per_kv_head, bsz * kv_head, 1)
        # seqlen_per_SM = q_len
        # grid = (1, bsz * kv_head, 1)

        with torch.cuda.device(q.device):
            int4_quant_sub_k_mean_perwarp_SmoothQ_fullsize_allInOne.triton_fwd_q[grid](
                q, k, q_mean, k_mean, qq, qk, sq, sk, qmk, smooth_k,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                q_mean.stride(0), q_mean.stride(1), q_mean.stride(2),
                k_mean.stride(0), k_mean.stride(1), k_mean.stride(2),
                qq.stride(0), qq.stride(1), qq.stride(2), qq.stride(3),
                qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
                sq.stride(0), sq.stride(1), sq.stride(2),
                sk.stride(0), sk.stride(1), sk.stride(2),
                qmk.stride(0), qmk.stride(1), qmk.stride(2),
                smooth_k.stride(0), smooth_k.stride(1), smooth_k.stride(2), smooth_k.stride(3),
                bsz, q_len, seqlen_per_SM, kBlockN, kv_head, dim, 1
            )
        
        sm_per_kv_head = num_SM // (bsz * kv_head) if num_SM > (bsz * kv_head) else 1
        assert sm_per_kv_head > 0, f"{num_SM}, {bsz}, {kv_head}"

        seqlen_per_SM = math.ceil(q_len / (kBlockN * sm_per_kv_head)) * kBlockN

        grid = (sm_per_kv_head, bsz * kv_head, 1)
        with torch.cuda.device(q.device):
            int4_quant_sub_k_mean_perwarp_SmoothQ_fullsize_allInOne.triton_fwd_k[grid](
                q, k, q_mean, k_mean, qq, qk, sq, sk, qmk, smooth_k,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                q_mean.stride(0), q_mean.stride(1), q_mean.stride(2),
                k_mean.stride(0), k_mean.stride(1), k_mean.stride(2),
                qq.stride(0), qq.stride(1), qq.stride(2), qq.stride(3),
                qk.stride(0), qk.stride(1), qk.stride(2), qk.stride(3),
                sq.stride(0), sq.stride(1), sq.stride(2),
                sk.stride(0), sk.stride(1), sk.stride(2),
                qmk.stride(0), qmk.stride(1), qmk.stride(2),
                smooth_k.stride(0), smooth_k.stride(1), smooth_k.stride(2), smooth_k.stride(3),
                bsz, q_len, seqlen_per_SM, kBlockN, kv_head, dim, n_head // kv_head
            )
        if return_mean_k:
            return qq, qk, sq, sk, qmk, smooth_k, k_mean
        return qq, qk, sq, sk, qmk, smooth_k
