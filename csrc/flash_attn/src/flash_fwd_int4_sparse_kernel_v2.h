/******************************************************************************
* Copyright (c) 2024, Tri Dao.
******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <type_traits>

#include "block_info.h"
#include "cute/algorithm/copy.hpp"
#include "cute/arch/copy.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/config.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/tensor_impl.hpp"
#include "cute/util/debug.hpp"
#include "cutlass/half.h"
#include "cutlass/integer_subbyte.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

#include "cub/cub.cuh"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb); 
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return; 

    // n_block_min = 0
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN); 
    
    if (Is_causal || Is_local) { 
        n_block_max = std::min(n_block_max,
                            cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                            + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // P = softmax(QK^T) Useless value in our use case.
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;


    // The whole Q tensor of this batch shape: [1, q_len, n_head, dim]
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) 
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    
    // Q sub-tensor that belong to this CTA
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    
    // The whole K tensor of this batch.
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                        + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));

    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)

    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                        + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), 
                            typename Kernel_traits::SmemLayoutQ{}); // Swizzled Layout here.
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{}); 

    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx); // Type: ThrCopy<TiledCopy, ThrIdx>

    // tQgQ: Partitioning pattern tQ apply to gQ.
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ); 
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ); 
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN) 

    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA,MMA_K,MMA_N) 

    Tensor tSgS  = thr_mma.partition_C(gP); // useless

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K


    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma); // SM75_U32x4_LDSM_N
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma); // SM75_U16x8_LDSM_T
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK  
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ))); 
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                    binfo.actual_seqlen_q - m_block * kBlockM 
                                    );
    if (Kernel_traits::Is_Q_in_regs) { 
        cute::cp_async_fence();
    }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1; 
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                    binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);


    flash::Softmax<2 * size<1>(acc_o), Kernel_traits> softmax;

    const float alibi_slope = 0.0f;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1. 
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1 // Not possible in my use case.
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1); 

    // On-band, need causal masking.
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            // tVgV (CPY, CPY_N, CPY_K, n_block)
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        } else {
            // Clear the smem tiles to account for predicated off loads
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        cute::cp_async_fence(); // Sync kernel。

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        // acc_s Layout: (MMA, MMA_M, MMA_N)
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence(); 
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout())); 
        // if (cute::thread0()) { print(tOrP); }
        
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue 
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                        + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

template<int max_idx = 16>
struct offsets_iterator {
    int pt;

    __forceinline__ __device__ void clear() {
        pt = -1;
    }

    __forceinline__ __device__ int next(unsigned record){
        pt += 1;
        while(pt < max_idx) {
            if ((record & (1 << pt)) > 0) 
                return pt;
            else
                pt += 1;
        }
        return -1;
    }
};

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Extra_reserve_blocks, bool Record_approx, int Quant_mode, bool Two_pass, typename Params>
inline __device__ void compute_attn_1rowblock_skipping_v2(const Params &params, const int bidb, const int bidh, const int m_block, const float threshold_per_head) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Int4_Element = typename Kernel_traits::Int4_Element;
    using UInt8_Type = typename Kernel_traits::UInt8_Type;

    // Shared memory. Its size is dynamicly declared but actually can be determined in compile-time.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const bool tid_is_leader = tidx % 32 == 0;
    const int tid_warp = tidx % 32;
    const int warp_idx = tidx / 32;
    const int warp_row_tile_offset = warp_idx % Kernel_traits::nWarpM;
    const int row_offset_within_warp = (tidx % 32) / 4;
    

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kPipeSize = Kernel_traits::kPipeSize;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb); 
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return; 

    const int n_block_min = 0; 
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN); 
    if (Is_causal) { 
        n_block_max = std::min(n_block_max,
                            cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || !Is_even_MN) && n_block_max <= n_block_min) {
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    // The whole Q tensor of this batch shape: [1, q_len, n_head, dim]
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    
    // Q sub-tensor that belong to this CTA
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    
    // The whole K tensor of this batch.
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                        + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));

    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)

    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                        + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{}); // Swizzled Layout here.
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{}); 
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{}); 

    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx); // Type: ThrCopy<TiledCopy, ThrIdx>

    // tQgQ: Partitioning pattern tQ apply to gQ.
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ); 
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ); 
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN) 
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVt);                          // (MMA,MMA_K,MMA_N) 
    static_assert(std::is_same_v<
        decltype(thr_mma.partition_fragment_B(sVt)), 
        decltype(thr_mma.partition_fragment_B(sVtNoSwizzle))
    >, "sVtNoSwizzle can't be replaced by sVt.");

    Tensor tSgS  = thr_mma.partition_C(gP); // useless

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma); // SM75_U32x4_LDSM_N
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma); // SM75_U16x8_LDSM_T
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);


    Tensor mQQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<UInt8_Type *>(params.qq_ptr) 
            + binfo.q_offset(params.qq_batch_stride, params.qq_row_stride, bidb)),
        make_shape(params.h, binfo.actual_seqlen_q, kHeadDim / 2),
        make_stride(params.qq_head_stride, params.qq_row_stride, _1{})
    );
    
    Tensor gQQ = local_tile(mQQ(bidh, _, _), Shape<Int<kBlockM>, Int<kHeadDim / 2>>{}, make_coord(m_block, _0{}));

    Tensor mQK = make_tensor(
        make_gmem_ptr(reinterpret_cast<UInt8_Type *>(params.qk_ptr)
            + binfo.q_offset(params.qk_batch_stride, params.qk_row_stride, bidb)),
        make_shape(params.h_k, binfo.actual_seqlen_k, kHeadDim / 2),
        make_stride(params.qk_head_stride, params.qk_row_stride, _1{})
    );

    Tensor gQK = local_tile(mQK(bidh / params.h_h_k_ratio, _, _), Shape<Int<kBlockN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));
    // (kBlockN, kHeadDim, n_blocks_N)

    constexpr index_t one = 1; // make compiler happy

    Tensor mSQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.sq_ptr) 
        + binfo.q_offset(params.sq_batch_stride, one, bidb)),
        make_shape(params.h, binfo.actual_seqlen_q),
        make_stride(params.sq_head_stride, _1{})
    );

    Tensor gSQ = local_tile(mSQ(bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));

    Tensor mSK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.sk_ptr)
            + binfo.q_offset(params.sk_batch_stride, one, bidb)),
        make_shape(params.h_k, binfo.actual_seqlen_k),
        make_stride(params.sk_head_stride, _1{})
    );

    Tensor gSK = local_tile(mSK(bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>>{}, make_coord(_));

    Tensor mQMK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.qmk_ptr)
            + binfo.q_offset(params.qmk_batch_stride, one, bidb)),
        make_shape(params.h, binfo.actual_seqlen_k),
        make_stride(params.qmk_head_stride, _1{})
    );

    Tensor gQMK = local_tile(mQMK(bidh, _), Shape<Int<kBlockN>>{}, make_coord(_));

    Tensor mCPRC = make_tensor(
        make_gmem_ptr(reinterpret_cast<int32_t *>(params.cprc_ptr)
            + binfo.q_offset(params.cprc_batch_stride, one, bidb)),
        make_shape(params.h, params.n_Q_block, params.n_K_block),   
        make_stride(params.cprc_head_stride, params.cprc_qblk_stride, _1{})
    );

    Tensor gCPRC = mCPRC(bidh, m_block, _);

    Tensor mApprox = make_tensor(
        make_gmem_ptr(reinterpret_cast<float *>(params.approx_ptr)
            + binfo.q_offset(params.approx_batch_stride, params.approx_row_stride, bidb)),
        make_shape(params.h, binfo.actual_seqlen_q, binfo.actual_seqlen_k + (8 - binfo.actual_seqlen_k % 8)),   
        make_stride(params.approx_head_stride, params.approx_row_stride, _1{})
    );

    Tensor gApprox = local_tile(mApprox(bidh, _, _), Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));

    UInt8_Type * sQQ_ptr = reinterpret_cast<UInt8_Type *>((sV.data() + size(sV)).get());
    Tensor sQQ = make_tensor(make_smem_ptr(sQQ_ptr), typename Kernel_traits::SmemLayoutUInt8Q{});
    Tensor sQK = make_tensor(sQQ.data() + (Kernel_traits::Share_QQ_QK_smem ? 0 : size(sQQ)), typename Kernel_traits::SmemLayoutUInt8K{});

    Tensor sQQ_Int4 = make_tensor(make_smem_ptr(reinterpret_cast<Int4_Element *>(sQQ_ptr)), typename Kernel_traits::SmemLayoutInt4Q{});
    Tensor sQK_Int4 = make_tensor(make_smem_ptr(reinterpret_cast<Int4_Element *>(sQK.data().get())), typename Kernel_traits::SmemLayoutInt4K{});

    Element * sSK_ptr = reinterpret_cast<Element *>((sQK.data() + size(sQK)).get());
    Tensor sSK = make_tensor(make_smem_ptr(sSK_ptr), typename Kernel_traits::SmemLayoutSK{}); 

    Element * sQMK_ptr = reinterpret_cast<Element *>((sSK.data() + size(sSK)).get());
    Tensor sQMK = make_tensor(make_smem_ptr(sQMK_ptr), typename Kernel_traits::SmemLayoutSK{});

    unsigned * sbuf_ptr = reinterpret_cast<unsigned *>((sQMK.data() + size(sQMK)).get());
    Tensor temp_storage = make_tensor(make_smem_ptr(sbuf_ptr), Shape<Int<kNWarps>>{});

    typename Kernel_traits::GmemInt4QKTiledCopy gmem_tiled_copy_Quant;

    auto gmem_thr_copy_Quant = gmem_tiled_copy_Quant.get_thread_slice(tidx % Kernel_traits::GmemInt4QKCopyMaxThreadCnt);
    typename Kernel_traits::GmemKScaleTiledCopy gmem_tiled_copy_K_Scale;
    auto gmem_thr_copy_K_Scale = gmem_tiled_copy_K_Scale.get_thread_slice(tidx % Kernel_traits::kGmemThreadsPerRow_SK_QMK);
    typename Kernel_traits::GmemQMKTiledCopy gmem_tiled_copy_QMK;
    auto gmem_thr_copy_QMK = gmem_tiled_copy_QMK.get_thread_slice(tidx % Kernel_traits::kGmemThreadsPerRow_SK_QMK);

    typename Kernel_traits::GmemTiledCopyApprox gmem_tiled_copy_Approx;
    auto gmem_thr_copy_approx = gmem_tiled_copy_Approx.get_thread_slice(tidx);

    Tensor tQgQQ = gmem_thr_copy_Quant.partition_S(gQQ); // (QCPY, QCPY_M, QCPY_K) 
    Tensor tQsQQ = gmem_thr_copy_Quant.partition_D(sQQ);

    Tensor tKgQK = gmem_thr_copy_Quant.partition_S(gQK); // (KCPY, KCPY_M, KCPY_K, n_blocks_N)
    Tensor tKsQK = gmem_thr_copy_Quant.partition_D(sQK); // (KCPY, KCPY_M, KCPY_K, kPipesize)
    
    Tensor tKgSK = gmem_thr_copy_K_Scale.partition_S(gSK); 
    Tensor tKsSK = gmem_thr_copy_K_Scale.partition_D(sSK); // (CPY, kCPY_N, kPipesize)

    Tensor tQgQMK = gmem_thr_copy_QMK.partition_S(gQMK); 
    Tensor tQsQMK = gmem_thr_copy_QMK.partition_D(sQMK); // (CPY, kCPY_N, kPipesize)

    Tensor tAgApprox = gmem_thr_copy_approx.partition_D(gApprox);

    typename Kernel_traits::Int4TiledMMA int4_tiled_mma;
    auto int4_thr_mma = int4_tiled_mma.get_thread_slice(tidx);

    Tensor t4rQQ_int4 = int4_thr_mma.partition_fragment_A(sQQ_Int4); // (MMA, MMA_M, MMA_K) , type=int4
    Tensor t4rQK_int4 = int4_thr_mma.partition_fragment_B(sQK_Int4(_, _, _0{})); // (MMA, MMA_N, MMA_k, kPipesize)
    static_assert(size<0>(t4rQK_int4) * 2 == size<0>(t4rQQ_int4));
    static_assert(size<2>(t4rQQ_int4) == 2 && size<1>(t4rQQ_int4) == 1 && size<0>(t4rQQ_int4) == 32);

    Tensor vSK = make_tensor(make_smem_ptr(sSK.data()), typename Kernel_traits::SmemLayoutSK_repeat{});
    Tensor vQMK = make_tensor(make_smem_ptr(sQMK.data()), typename Kernel_traits::SmemLayoutSK_repeat{});
    Tensor t4rSk = make_tensor<Element>(Shape<Shape<_2, Int<Kernel_traits::kCoreMatrixN>>, Int<1>, Int<1>>{}); 
    Tensor t4rQMK = make_tensor<Element>(Shape<Shape<_2, Int<Kernel_traits::kCoreMatrixN>>, Int<1>, Int<1>>{});
    // Tensor t4rSk = convert_type<Element>(int4_thr_mma.partition_fragment_C(vSK(_, _, _0{}))); // (V, V_M, V_N, kPipeSize)
    // Tensor t4rQMK = convert_type<Element>(int4_thr_mma.partition_fragment_C(vQMK(_, _, _0{}))); // (V, V_M, V_N, kPipeSize)

    typename Kernel_traits::DummyUInt8TiledMMA dummy_uint8_tiled_mma;

    auto smem_tiled_copy_QQ_int8 = make_tiled_copy_A(typename Kernel_traits::INT8_SmemCopyAtom_x4{}, dummy_uint8_tiled_mma);
    auto smem_tiled_copy_QK_int8 = make_tiled_copy_B(typename Kernel_traits::INT8_SmemCopyAtom_x4{}, dummy_uint8_tiled_mma);
    
    typename Kernel_traits::SmemKScaleTiledCopy smem_tiled_copy_SQ_SK_QMK;

    auto smem_thr_copy_QQ_int8 = smem_tiled_copy_QQ_int8.get_thread_slice(tidx);
    auto smem_thr_copy_QK_int8 = smem_tiled_copy_QK_int8.get_thread_slice(tidx);

    auto smem_thr_copy_SQ_SK_QMK = smem_tiled_copy_SQ_SK_QMK.get_thread_slice(tidx % 32);

    Tensor t4sQQ_int8 = smem_thr_copy_QQ_int8.partition_S(sQQ);
    Tensor t4sQK_int8 = smem_thr_copy_QK_int8.partition_S(sQK); 

    Tensor t4sSK = smem_thr_copy_SQ_SK_QMK.partition_S(vSK); // (CPY, CPY_M, CPY_N, kPipeSize)
    Tensor t4sQMK = smem_thr_copy_SQ_SK_QMK.partition_S(vQMK); // (CPY, CPY_M, CPY_N, kPipeSize)

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ))); 
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                    binfo.actual_seqlen_q - m_block * kBlockM 
                                    );
    if (Kernel_traits::Is_Q_in_regs) { 
        cute::cp_async_fence();
    }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                    binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o), Kernel_traits> softmax;

    const float alibi_slope =  0.0f;
    flash::Mask<Is_causal, false, false> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);


    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1. 
    constexpr int n_masking_steps = (Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1; 
    // constexpr int n_masking_steps = cute::ceil_div(kBlockM, kBlockN) ;

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            // tVgV (CPY, CPY_N, CPY_K, n_block)
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
        } else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        cute::cp_async_fence(); 

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap) flash::apply_softcap(acc_s, params.softcap);

        // acc_s Layout: (MMA, MMA_M, MMA_N)
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence(); 
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout())); 

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        if (Record_approx && tid_is_leader) {gCPRC(n_block) = 1;}

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    constexpr int blocks_per_seg = Kernel_traits::kBlockSeg / kBlockN;
    int n_blocks_wo_sink = n_block_max -1;
    int n_segs = cute::ceil_div(n_blocks_wo_sink, blocks_per_seg);
    int remain_blocks = n_blocks_wo_sink - n_segs * blocks_per_seg;
    int n_local_blocks;
    if (threshold_per_head < 0.00001 ){
        n_segs = 0;
        remain_blocks = n_blocks_wo_sink;
        n_local_blocks = remain_blocks;
    }
    else if(remain_blocks >= Kernel_traits::kNLocalBlocks)
        n_local_blocks = remain_blocks;
    else{
        int n_supplement_seg = cute::ceil_div(Kernel_traits::kNLocalBlocks - remain_blocks, blocks_per_seg);
        n_local_blocks = blocks_per_seg * n_supplement_seg + remain_blocks;
        n_segs -= n_supplement_seg;
    }

    
    int n_off_band_local_steps = n_local_blocks - n_masking_steps; 
    bool sink_token_covered = (n_local_blocks >= n_block_max); /* sink token block has not been covered. */
    // local and sink. 
    if constexpr (Extra_reserve_blocks){
        for (int rest_local = n_off_band_local_steps; n_block >= n_block_min && rest_local > 0; --n_block, --rest_local) {
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
            cute::cp_async_fence();

            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            if constexpr (Is_softcap){ flash::apply_softcap(acc_s, params.softcap); }

            flash::cp_async_wait<0>(); 
            __syncthreads();
            if (n_block > n_block_min && rest_local > 1) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            }
            else if (rest_local == 1 /* last iter */ && !sink_token_covered) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, 0), tKsK, tKVcKV, tKVpKV);
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            }

            mask.template apply_mask<false>(
                acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
            );

            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false, /* row_sum_sync */true>(acc_s, acc_o, params.scale_softmax_log2);
            Tensor rP = flash::convert_type<Element>(acc_s);

            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
            flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

            if(Record_approx && tid_is_leader) { gCPRC(n_block) = 1; }
        }
        // sink area.
        if (!sink_token_covered){
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, 0), tVsV, tKVcKV, tKVpKV);
            cute::cp_async_fence();

            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            if constexpr (Is_softcap){ flash::apply_softcap(acc_s, params.softcap); }

            flash::cp_async_wait<0>(); 
            __syncthreads();

            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false, /* row_sum_sync */true>(acc_s, acc_o, params.scale_softmax_log2);

            Tensor rP = flash::convert_type<Element>(acc_s);

            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
            flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            
            if(Record_approx && tid_is_leader) {gCPRC(0) = 1; }
        }
    }

    /*
        prologue for int4 skipping phase.
    */   
    Tensor cQQ = make_identity_tensor(make_shape(size<0>(sQQ), size<1>(sQQ)));
    Tensor tQQcQQ = gmem_thr_copy_Quant.partition_S(cQQ); // (CPY, CPY_M, CPY_K)
    Tensor tQQpQQ = make_tensor<bool>(make_layout(make_shape(size<1>(tQQcQQ), size<2>(tQQcQQ)), make_stride(_1{}, _0{})));

    /* Retiling */
    Tensor t4rQQ_int8_copy_view = smem_thr_copy_QQ_int8.retile_D(recast<UInt8_Type>(t4rQQ_int4));
    Tensor t4rQK_int8_copy_view = smem_thr_copy_QK_int8.retile_D(recast<UInt8_Type>(t4rQK_int4));
    CUTE_STATIC_ASSERT_V(size<1>(t4sQQ_int8) == size<1>(t4rQQ_int8_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(t4sQK_int8) == size<1>(t4rQK_int8_copy_view));        

    #pragma unroll
    for (int k = 0; k < size<0>(tQQpQQ); ++k)
        tQQpQQ(k, 0) = get<0>(tQQcQQ(0, k, 0)) < (binfo.actual_seqlen_q - m_block * kBlockM);

    cute::copy_if(gmem_tiled_copy_Quant, tQQpQQ, tQgQQ, tQsQQ);
    cute::cp_async_fence();
    
    if (Kernel_traits::Is_QQ_in_regs) {
        flash::cp_async_wait<0>();
        __syncthreads(); // Ensure that all cp instruction has completed.
        cute::copy(smem_tiled_copy_QQ_int8, t4sQQ_int8, t4rQQ_int8_copy_view); // src layout cannot be vectorize. 
    }

    Tensor rSQ = make_tensor<Element>(Shape<_2, Int<size<1>(t4rQQ_int4)>>{}); 
    { 
        const int base_offset = warp_row_tile_offset * 16 + row_offset_within_warp;
        const int max_M_idx = binfo.actual_seqlen_q - m_block * kBlockM;
        // load sQ directly from global memory.
        #pragma unroll
        for(int mi = 0; mi < size<1>(t4rQQ_int4); mi ++){ 
            int offset = base_offset + tile_size<0>(int4_tiled_mma) * mi;
            rSQ(make_coord(_0{}, mi)) = offset < max_M_idx ? gSQ(offset) : Element(0.0);
            rSQ(make_coord(_1{}, mi)) = (offset + 8) < max_M_idx ? gSQ(offset + 8) : Element(0.0);
        }
    }
    
    int seg_offset = 1, big_cnt = 0;
    Element sq_reciprocal;
    sq_reciprocal = static_cast<Element>(1 / static_cast<float>(rSQ(_0{}))); // make compiler happy :)
    Tensor dummy_weight_int32 = partition_fragment_C(int4_tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(dummy_weight_int32);
    CUTE_STATIC_ASSERT(size<0>(dummy_weight_int32) == 4);
    using m_extent = decltype(size<1>(dummy_weight_int32));
    using n_extent = decltype(size<2>(dummy_weight_int32));
    Tensor dummy_weight_fp32 = recast<float>(dummy_weight_int32);
    Tensor dummy_weight_int32_ = make_tensor(
        dummy_weight_int32.data(), 
        flash::convert_layout_acc_rowcol(dummy_weight_int32.layout())
    );

    Tensor acc_s = recast<float>(dummy_weight_int32); // recast == reinterpret_cast

    // Tensor dummy_weight_fp32 = make_tensor_like<float>(dummy_weight_int32);
    Tensor rowwise_max = make_tensor<int>(Shape<Int<size(rSQ)>>{}); // 2 * M
    Tensor rowwise_max_dequant = make_tensor_like<Element>(rowwise_max); // Do not declare a new array.

    Tensor t4rQMK_div_sq_sk = make_tensor_like<int>(t4rQMK); 

    // Tensor big_block_offsets = make_tensor<unsigned>(Shape<Int<blocks_per_seg>>{});
    Tensor cta_records = make_tensor<unsigned>(Shape<Int<Kernel_traits::kNWarps>>{});

    offsets_iterator<blocks_per_seg> off_iter;

    if constexpr (Two_pass)
        softmax.template reduce_row_sum<Element>(threshold_per_head, params.scale_softmax_log2);

    for (; n_segs > 0; n_segs--) { 
        if constexpr (!Two_pass)
            softmax.template reduce_row_sum<Element>(threshold_per_head, params.scale_softmax_log2);

        unsigned record = 0;
        static_assert(blocks_per_seg <= 32, "segment is too large!");

        int istage_prefetch = kPipeSize - 1;
        int in_prefetch = kPipeSize - 1;
        int istage_now = 0; 

        #pragma unroll
        for (int istage = 0; istage < (Kernel_traits::kPipeSize - 1); istage++){
            flash::copy<true, Is_even_K>(gmem_tiled_copy_Quant, tKgQK(_, _, _, istage + seg_offset), tKsQK(_, _, _, istage), tKVcKV, tKVpKV);
            flash::copy_vector_off_band(gmem_tiled_copy_K_Scale, tKgSK(_, _, istage + seg_offset), tKsSK(_, _, istage));
            flash::copy_vector_off_band(gmem_tiled_copy_QMK, tQgQMK(_, _, istage + seg_offset), tQsQMK(_, _, istage));
            cute::cp_async_fence(); 
        }

        #pragma unroll
        for (int b_i = 0; b_i < blocks_per_seg; b_i++){
            flash::cp_async_wait<kPipeSize - 2>();
            __syncthreads(); 
    
            // Pre-issue s2r to start the MMA pipeline.
            if (!Kernel_traits::Is_QQ_in_regs) { cute::copy(smem_tiled_copy_QQ_int8, t4sQQ_int8(_, _, _0{}), t4rQQ_int8_copy_view(_, _, _0{})); } 
            cute::copy(smem_tiled_copy_QK_int8, t4sQK_int8(_, _, _0{}, istage_now), t4rQK_int8_copy_view(_, _, _0{}));
    
            if (b_i < blocks_per_seg - (Kernel_traits::kPipeSize - 1)) {
                flash::copy<true, Is_even_K>(gmem_tiled_copy_Quant, tKgQK(_, _, _, in_prefetch + seg_offset), tKsQK(_, _, _, istage_prefetch), tKVcKV, tKVpKV);
                flash::copy_vector_off_band(gmem_tiled_copy_K_Scale, tKgSK(_, _, in_prefetch + seg_offset), tKsSK(_, _, istage_prefetch));
                flash::copy_vector_off_band(gmem_tiled_copy_QMK, tQgQMK(_, _, in_prefetch + seg_offset), tQsQMK(_, _, istage_prefetch));
                cute::cp_async_fence(); 
                istage_prefetch = (istage_prefetch + 1) & (Kernel_traits::kPipemask);
                in_prefetch = in_prefetch + 1;
            } else {
                cute::cp_async_fence();
            }
    
            bool is_big_block;

            CUTE_STATIC_ASSERT(size<0>(get<0>(dummy_weight_int32.layout())) == _2{});
            CUTE_STATIC_ASSERT(size<1>(get<0>(dummy_weight_int32.layout())) == _2{});
            // dequant.
            if constexpr (Quant_mode == 2) /* Per-token Quant */{
                clear(dummy_weight_int32);
                #pragma unroll
                for (int i = 0; i < size<2>(t4rQQ_int4); ++i) {
                    if (i < size<2>(t4rQQ_int4) - 1) {
                        if (!Kernel_traits::Is_QQ_in_regs) { cute::copy(smem_tiled_copy_QQ_int8, t4sQQ_int8(_, _, i+1), t4rQQ_int8_copy_view(_, _, i+1)); }
                        cute::copy(smem_tiled_copy_QK_int8, t4sQK_int8(_, _, i+1, istage_now), t4rQK_int8_copy_view(_, _, i+1));
                    }

                    if (i == size<2>(t4rQQ_int4) - 1) { 
                        cute::copy(smem_tiled_copy_SQ_SK_QMK, t4sSK(_, _, _, istage_now), t4rSk);
                        cute::copy(smem_tiled_copy_SQ_SK_QMK, t4sQMK(_, _, _, istage_now), t4rQMK);
                    }
                    cute::gemm(int4_thr_mma, t4rQQ_int4(_, _, i), t4rQK_int4(_, _, i), dummy_weight_int32); 
                }

                Tensor dummy_weight_fp32 = flash::convert_type<float>(dummy_weight_int32); // size 16
                #pragma unroll
                for (int mi = 0; mi < size<1>(dummy_weight_fp32); mi++){
                    #pragma unroll
                    for (int ni = 0; ni < size<2>(dummy_weight_fp32); ni++){
                        #pragma unroll
                        for (int vi = 0; vi < size<0>(dummy_weight_fp32); vi++){ 
                            const int frag_row = vi / 2;
                            const int frag_col = vi % size<1>(get<0>(dummy_weight_fp32.layout()));
    
                            // const auto target_reg = make_coord(make_coord(frag_col, frag_row), mi, ni);
                            const auto target_reg = make_coord(frag_col + frag_row * 2, mi, ni);
                            const auto target_SK = make_coord(make_coord(frag_col, ni), _0{}, _0{});
                            const int rSQ_idx = frag_row + 2 * mi;
                            dummy_weight_fp32(target_reg) = t4rSk(target_SK) * rSQ(rSQ_idx) * dummy_weight_fp32(target_reg) + t4rQMK(target_SK);
                        }
                    }
                }   
                if (Record_approx) {
                    auto tAgApprox_view = gmem_thr_copy_approx.retile_D(tAgApprox);
                    cute::copy(gmem_tiled_copy_Approx, dummy_weight_fp32, tAgApprox_view(_, _, _, seg_offset + b_i));
                }
                    
                is_big_block = softmax.template is_big_block</* debug */true,true>(dummy_weight_fp32, params.scale_softmax_log2, threshold_per_head);    
            } else if constexpr (Quant_mode == 3) /* Per-warp Quant */{


                CUTE_STATIC_ASSERT(size<1>(dummy_weight_int32_) == size(t4rQMK_div_sq_sk));

                clear(dummy_weight_int32);

                #pragma unroll
                for (int i = 0; i < size<2>(t4rQQ_int4); ++i) {
                    if (i < size<2>(t4rQQ_int4) - 1) {
                        if (!Kernel_traits::Is_QQ_in_regs) { cute::copy(smem_tiled_copy_QQ_int8, t4sQQ_int8(_, _, i+1), t4rQQ_int8_copy_view(_, _, i+1)); }
                        cute::copy(smem_tiled_copy_QK_int8, t4sQK_int8(_, _, i+1, istage_now), t4rQK_int8_copy_view(_, _, i+1));
                    }

                    cute::gemm(int4_thr_mma, t4rQQ_int4(_, _, i), t4rQK_int4(_, _, i), dummy_weight_int32);
                }

                cute::copy(smem_tiled_copy_SQ_SK_QMK, t4sQMK(_, _, _, istage_now), t4rQMK);

                #pragma unroll
                for (int mi = 0; mi < size(t4rQMK); mi++)
                    t4rQMK(mi) = t4rQMK(mi) * sq_reciprocal;
                
                #pragma unroll
                for (int mi = 0; mi < size(t4rQMK); mi++){
                    int qmk_v;
                    if constexpr (std::is_same_v<Element, cutlass::half_t>)
                        qmk_v = __half2int_rn(t4rQMK(mi).to_half());
                    else
                        qmk_v = __bfloat162int_rn(t4rQMK(mi).to_nv_bfloat16());      

                    t4rQMK_div_sq_sk(mi)  = qmk_v;              
                }
                
                cute::copy(smem_tiled_copy_SQ_SK_QMK, t4sSK(_, _, _, istage_now), t4rSk);

                // from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
                CUTE_STATIC_ASSERT(size(t4rQMK_div_sq_sk) == size<1>(dummy_weight_int32_));
                #pragma unroll
                for (int mi = 0; mi < size<0>(dummy_weight_int32_); mi++){
                    rowwise_max(mi) = dummy_weight_int32_(mi, _0{}) + t4rQMK_div_sq_sk(_0{}, _0{}, _0{});
                    // rowwise_max(mi) = dummy_weight_int32_(mi, _0{});
                    #pragma unroll
                    for (int ni = 1; ni < size<1>(dummy_weight_int32_); ni++){
                        int temp = dummy_weight_int32_(mi, ni) + t4rQMK_div_sq_sk(ni);
                        rowwise_max(mi) = temp > rowwise_max(mi) ? temp : rowwise_max(mi);
                    }
                }   

                if constexpr (Record_approx) {
                    auto tAgApprox_view = gmem_thr_copy_approx.retile_D(tAgApprox);
                    cute::copy(gmem_tiled_copy_Approx, convert_type<float>(dummy_weight_int32), tAgApprox_view(_, _, _, seg_offset + b_i));
                }

                Element scale = rSQ[0] * t4rSk[0];
                #pragma unroll
                for (int mi = 0; mi < size(rowwise_max_dequant); mi++){
                    rowwise_max_dequant[mi] = static_cast<Element>(rowwise_max[mi]);
                }
                rowwise_max_dequant[0] = rowwise_max_dequant[0] * scale;
                rowwise_max_dequant[1] = rowwise_max_dequant[1] * scale;

                is_big_block = rowwise_max_dequant[0] > softmax.threshold_temp[0] || rowwise_max_dequant[1] > softmax.threshold_temp[1];
            }

            record = record | (static_cast<unsigned>(is_big_block) << b_i); 

            istage_now = (istage_now + 1) & (Kernel_traits::kPipemask); // kPipeSize
        }


        /* CTA allreduce */
        #pragma unroll 
        for (int t = 32; t > 1; t >>= 1){
            unsigned temp = __shfl_xor_sync(0xffffffff, record, t - 1); // mask: 31, 15, 7, 3, 1
            record |= temp;
        }

        if (tid_is_leader) temp_storage(warp_idx) = record;
        static_assert(Kernel_traits::kNWarps * 4 <= kBlockN * sizeof(Element), "Smem space for all reduce is too small!!");    
        __syncthreads(); 

        cute::copy(Copy_Atom<DefaultCopy, unsigned>{}, temp_storage, cta_records);
        record = cta_records[0];
        #pragma unroll
        for(int i=1;i<Kernel_traits::kNWarps;i++) record |= cta_records[i];
        
        off_iter.clear();
        int pos_now = -1, pos_next = off_iter.next(record);

        // Pre-issue K tensor g2s to start fp16 pipeline
        if (/*big_block_cnt > 0*/ pos_next >= 0){
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, seg_offset + pos_next), tKsK, tKVcKV, tKVpKV);
            Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
            CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
            cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);    
            cute::cp_async_fence();
        }
        
        // FP16 pipeline.
        // for (int b_i = 0; b_i < big_block_cnt; b_i++){
        while(pos_next >= 0) {
            pos_now = pos_next;
            pos_next = off_iter.next(record);
            // Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads(); 

            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, seg_offset + pos_now /*big_block_offsets[b_i]*/), tVsV, tKVcKV, tKVpKV);
            cute::cp_async_fence();

            // flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            flash::gemm</*A_in_regs=*/true>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            if constexpr (Is_softcap){
                flash::apply_softcap(acc_s, params.softcap);
            }

            flash::cp_async_wait<0>();
            __syncthreads();
            if (pos_next >= 0){
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, seg_offset + pos_next/*big_block_offsets[b_i + 1]*/), tKsK, tKVcKV, tKVpKV);
                cute::cp_async_fence();
            }

            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false, /* row_sum_sync */true>(acc_s, acc_o, params.scale_softmax_log2);

            Tensor rP = flash::convert_type<Element>(acc_s);

            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

            flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            if(Record_approx && tid_is_leader) {gCPRC(seg_offset + pos_now /* big_block_offsets[b_i] */) = 1;}
        }
        seg_offset += blocks_per_seg;
    }

    // Epilogue 
    cute::cp_async_wait<0>(); // In case we meet race condition when kBlockN is small (e.g. 16)
    Tensor lse = softmax.template normalize_softmax_lse<false>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                        + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Extra_reserve_blocks, int Quant_mode, typename Params>
inline __device__ void compute_map_1rowblock_skipping_v2(const Params &params, const int bidb, const int bidh, const int m_block, const float threshold_per_head) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Int4_Element = typename Kernel_traits::Int4_Element;
    using UInt8_Type = typename Kernel_traits::UInt8_Type;

    // Shared memory. Its size is dynamicly declared but actually can be determined in compile-time.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const bool tid_is_leader = tidx % 32 == 0;
    const int tid_warp = tidx % 32;
    const int warp_idx = tidx / 32;
    const int warp_row_tile_offset = warp_idx % Kernel_traits::nWarpM;
    const int row_offset_within_warp = (tidx % 32) / 4;
    

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kPipeSize = Kernel_traits::kPipeSize;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb); 
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = 0; 
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN); 

    if (Is_causal) { 
        n_block_max = std::min(n_block_max,
                            cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || !Is_even_MN) && n_block_max <= n_block_min) {
        return; 
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    // The whole Q tensor of this batch shape: [1, q_len, n_head, dim]
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    
    // Q sub-tensor that belong to this CTA
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    
    // The whole K tensor of this batch.
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                        + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));

    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{}); // Swizzled Layout here.
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx); // Type: ThrCopy<TiledCopy, ThrIdx>

    // tQgQ: Partitioning pattern tQ apply to gQ.
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ); 
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ); 
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN) 
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma); // SM75_U32x4_LDSM_N
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    Tensor mQQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<UInt8_Type *>(params.qq_ptr) 
            + binfo.q_offset(params.qq_batch_stride, params.qq_row_stride, bidb)),
        make_shape(params.h, binfo.actual_seqlen_q, kHeadDim / 2),
        make_stride(params.qq_head_stride, params.qq_row_stride, _1{})
    );
    
    Tensor gQQ = local_tile(mQQ(bidh, _, _), Shape<Int<kBlockM>, Int<kHeadDim / 2>>{}, make_coord(m_block, _0{}));

    Tensor mQK = make_tensor(
        make_gmem_ptr(reinterpret_cast<UInt8_Type *>(params.qk_ptr)
            + binfo.q_offset(params.qk_batch_stride, params.qk_row_stride, bidb)),
        make_shape(params.h_k, binfo.actual_seqlen_k, kHeadDim / 2),
        make_stride(params.qk_head_stride, params.qk_row_stride, _1{})
    );

    Tensor gQK = local_tile(mQK(bidh / params.h_h_k_ratio, _, _), Shape<Int<kBlockN>, Int<kHeadDim / 2>>{}, make_coord(_, _0{}));
    // (kBlockN, kHeadDim, n_blocks_N)

    constexpr index_t one = 1; // make compiler happy

    Tensor mSQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.sq_ptr) 
        + binfo.q_offset(params.sq_batch_stride, one, bidb)),
        make_shape(params.h, binfo.actual_seqlen_q),
        make_stride(params.sq_head_stride, _1{})
    );

    Tensor gSQ = local_tile(mSQ(bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));

    Tensor mSK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.sk_ptr)
            + binfo.q_offset(params.sk_batch_stride, one, bidb)),
        make_shape(params.h_k, binfo.actual_seqlen_k),
        make_stride(params.sk_head_stride, _1{})
    );

    Tensor gSK = local_tile(mSK(bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>>{}, make_coord(_));

    Tensor mQMK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element *>(params.qmk_ptr)
            + binfo.q_offset(params.qmk_batch_stride, one, bidb)),
        make_shape(params.h, binfo.actual_seqlen_k),
        make_stride(params.qmk_head_stride, _1{})
    );

    Tensor gQMK = local_tile(mQMK(bidh, _), Shape<Int<kBlockN>>{}, make_coord(_));

    Tensor mCPRC = make_tensor(
        make_gmem_ptr(reinterpret_cast<int32_t *>(params.cprc_ptr)
            + binfo.q_offset(params.cprc_batch_stride, one, bidb)),
        make_shape(params.h, params.n_Q_block, params.n_K_block),   
        make_stride(params.cprc_head_stride, params.cprc_qblk_stride, _1{})
    );

    Tensor gCPRC = mCPRC(bidh, m_block, _);

    UInt8_Type * sQQ_ptr = reinterpret_cast<UInt8_Type *>(smem_);
    Tensor sQQ = make_tensor(make_smem_ptr(sQQ_ptr), typename Kernel_traits::SmemLayoutUInt8Q{});
    Tensor sQK = make_tensor(sQQ.data() + (Kernel_traits::Share_QQ_QK_smem ? 0 : size(sQQ)), typename Kernel_traits::SmemLayoutUInt8K{});

    Tensor sQQ_Int4 = make_tensor(make_smem_ptr(reinterpret_cast<Int4_Element *>(sQQ_ptr)), typename Kernel_traits::SmemLayoutInt4Q{});
    Tensor sQK_Int4 = make_tensor(make_smem_ptr(reinterpret_cast<Int4_Element *>(sQK.data().get())), typename Kernel_traits::SmemLayoutInt4K{});

    Element * sSK_ptr = reinterpret_cast<Element *>((sQK.data() + size(sQK)).get());
    Tensor sSK = make_tensor(make_smem_ptr(sSK_ptr), typename Kernel_traits::SmemLayoutSK{});

    Element * sQMK_ptr = reinterpret_cast<Element *>((sSK.data() + size(sSK)).get());
    Tensor sQMK = make_tensor(make_smem_ptr(sQMK_ptr), typename Kernel_traits::SmemLayoutSK{});

    unsigned * sbuf_ptr = reinterpret_cast<unsigned *>((sQMK.data() + size(sQMK)).get());
    Tensor temp_storage = make_tensor(make_smem_ptr(sbuf_ptr), Shape<Int<kNWarps>>{});

    typename Kernel_traits::GmemInt4QKTiledCopy gmem_tiled_copy_Quant;

    auto gmem_thr_copy_Quant = gmem_tiled_copy_Quant.get_thread_slice(tidx % Kernel_traits::GmemInt4QKCopyMaxThreadCnt);
    typename Kernel_traits::GmemKScaleTiledCopy gmem_tiled_copy_K_Scale;
    auto gmem_thr_copy_K_Scale = gmem_tiled_copy_K_Scale.get_thread_slice(tidx % Kernel_traits::kGmemThreadsPerRow_SK_QMK);
    typename Kernel_traits::GmemQMKTiledCopy gmem_tiled_copy_QMK;
    auto gmem_thr_copy_QMK = gmem_tiled_copy_QMK.get_thread_slice(tidx % Kernel_traits::kGmemThreadsPerRow_SK_QMK);

    typename Kernel_traits::GmemTiledCopyApprox gmem_tiled_copy_Approx;
    auto gmem_thr_copy_approx = gmem_tiled_copy_Approx.get_thread_slice(tidx);

    Tensor tQgQQ = gmem_thr_copy_Quant.partition_S(gQQ); // (QCPY, QCPY_M, QCPY_K) 
    Tensor tQsQQ = gmem_thr_copy_Quant.partition_D(sQQ);

    Tensor tKgQK = gmem_thr_copy_Quant.partition_S(gQK); // (KCPY, KCPY_M, KCPY_K, n_blocks_N)
    Tensor tKsQK = gmem_thr_copy_Quant.partition_D(sQK); // (KCPY, KCPY_M, KCPY_K, kPipesize)
    
    Tensor tKgSK = gmem_thr_copy_K_Scale.partition_S(gSK); 
    Tensor tKsSK = gmem_thr_copy_K_Scale.partition_D(sSK); // (CPY, kCPY_N, kPipesize)

    Tensor tQgQMK = gmem_thr_copy_QMK.partition_S(gQMK); 
    Tensor tQsQMK = gmem_thr_copy_QMK.partition_D(sQMK); // (CPY, kCPY_N, kPipesize)

    typename Kernel_traits::Int4TiledMMA int4_tiled_mma;
    auto int4_thr_mma = int4_tiled_mma.get_thread_slice(tidx);

    Tensor t4rQQ_int4 = int4_thr_mma.partition_fragment_A(sQQ_Int4); // (MMA, MMA_M, MMA_K) , type=int4
    Tensor t4rQK_int4 = int4_thr_mma.partition_fragment_B(sQK_Int4(_, _, _0{})); // (MMA, MMA_N, MMA_k, kPipesize)
    static_assert(size<0>(t4rQK_int4) * 2 == size<0>(t4rQQ_int4));
    static_assert(size<2>(t4rQQ_int4) == 2 && size<1>(t4rQQ_int4) == 1 && size<0>(t4rQQ_int4) == 32);

    Tensor vSK = make_tensor(make_smem_ptr(sSK.data()), typename Kernel_traits::SmemLayoutSK_repeat{});
    Tensor vQMK = make_tensor(make_smem_ptr(sQMK.data()), typename Kernel_traits::SmemLayoutSK_repeat{});
    Tensor t4rSk = make_tensor<Element>(Shape<Shape<_2, Int<Kernel_traits::kCoreMatrixN>>, Int<1>, Int<1>>{}); 
    Tensor t4rQMK = make_tensor<Element>(Shape<Shape<_2, Int<Kernel_traits::kCoreMatrixN>>, Int<1>, Int<1>>{});
    // Tensor t4rSk = convert_type<Element>(int4_thr_mma.partition_fragment_C(vSK(_, _, _0{}))); // (V, V_M, V_N, kPipeSize)
    // Tensor t4rQMK = convert_type<Element>(int4_thr_mma.partition_fragment_C(vQMK(_, _, _0{}))); // (V, V_M, V_N, kPipeSize)

    typename Kernel_traits::DummyUInt8TiledMMA dummy_uint8_tiled_mma;

    auto smem_tiled_copy_QQ_int8 = make_tiled_copy_A(typename Kernel_traits::INT8_SmemCopyAtom_x4{}, dummy_uint8_tiled_mma);
    auto smem_tiled_copy_QK_int8 = make_tiled_copy_B(typename Kernel_traits::INT8_SmemCopyAtom_x4{}, dummy_uint8_tiled_mma);
    
    typename Kernel_traits::SmemKScaleTiledCopy smem_tiled_copy_SQ_SK_QMK;

    auto smem_thr_copy_QQ_int8 = smem_tiled_copy_QQ_int8.get_thread_slice(tidx);
    auto smem_thr_copy_QK_int8 = smem_tiled_copy_QK_int8.get_thread_slice(tidx);

    auto smem_thr_copy_SQ_SK_QMK = smem_tiled_copy_SQ_SK_QMK.get_thread_slice(tidx % 32);

    Tensor t4sQQ_int8 = smem_thr_copy_QQ_int8.partition_S(sQQ);
    Tensor t4sQK_int8 = smem_thr_copy_QK_int8.partition_S(sQK); 

    Tensor t4sSK = smem_thr_copy_SQ_SK_QMK.partition_S(vSK); // (CPY, CPY_M, CPY_N, kPipeSize)
    Tensor t4sQMK = smem_thr_copy_SQ_SK_QMK.partition_S(vQMK); // (CPY, CPY_M, CPY_N, kPipeSize)

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ))); 
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                    binfo.actual_seqlen_q - m_block * kBlockM 
                                    );
    if (Kernel_traits::Is_Q_in_regs) { 
        cute::cp_async_fence();
    }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                    binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o), Kernel_traits> softmax;

    const float alibi_slope =  0.0f;
    flash::Mask<Is_causal, false, false> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);


    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1. 
    constexpr int n_masking_steps = (Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1; 
    // constexpr int n_masking_steps = cute::ceil_div(kBlockM, kBlockN) ;

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap) flash::apply_softcap(acc_s, params.softcap);

        // acc_s Layout: (MMA, MMA_M, MMA_N)
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence(); 
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o_norescale_out</*Is_first=*/true,  /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o_norescale_out</*Is_first=*/false, /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2);


        if (tid_is_leader) {gCPRC(n_block) = 1;}

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    constexpr int blocks_per_seg = Kernel_traits::kBlockSeg / kBlockN;
    int n_blocks_wo_sink = n_block_max -1;
    int n_segs = cute::ceil_div(n_blocks_wo_sink, blocks_per_seg);
    int remain_blocks = n_blocks_wo_sink - n_segs * blocks_per_seg;
    int n_local_blocks;
    if (threshold_per_head < 0.00001){
        n_segs = 0;
        remain_blocks = n_blocks_wo_sink;
        n_local_blocks = remain_blocks;
    }
    else if(remain_blocks >= Kernel_traits::kNLocalBlocks)
        n_local_blocks = remain_blocks;
    else{
        int n_supplement_seg = cute::ceil_div(Kernel_traits::kNLocalBlocks - remain_blocks, blocks_per_seg);
        n_local_blocks = blocks_per_seg * n_supplement_seg + remain_blocks;
        n_segs -= n_supplement_seg;
    }

    
    int n_off_band_local_steps = n_local_blocks - n_masking_steps;
    bool sink_token_covered = (n_local_blocks >= n_block_max); /* sink token block has not been covered. */
    // local and sink. 
    if constexpr (Extra_reserve_blocks){
        for (int rest_local = n_off_band_local_steps; n_block >= n_block_min && rest_local > 0; --n_block, --rest_local) {
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();

            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            if constexpr (Is_softcap){ flash::apply_softcap(acc_s, params.softcap); }

            __syncthreads();
            if (n_block > n_block_min && rest_local > 1) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            }
            else if (rest_local == 1 /* last iter */ && !sink_token_covered) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, 0), tKsK, tKVcKV, tKVpKV);
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            }

            mask.template apply_mask<false>(
                acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
            );

            softmax.template softmax_rescale_o_norescale_out</*Is_first=*/false, /*Check_inf=*/false, /* row_sum_sync */true>(acc_s, acc_o, params.scale_softmax_log2);

            if(tid_is_leader) { gCPRC(n_block) = 1; }
        }
        // sink area.
        if (!sink_token_covered){
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();

            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            if constexpr (Is_softcap){ flash::apply_softcap(acc_s, params.softcap); }


            softmax.template softmax_rescale_o_norescale_out</*Is_first=*/false, /*Check_inf=*/false, /* row_sum_sync */true>(acc_s, acc_o, params.scale_softmax_log2);

            if(tid_is_leader) {gCPRC(0) = 1; }
        }
    }

    __syncthreads();
    /*
        prologue for int4 skipping phase.
    */   
    Tensor cQQ = make_identity_tensor(make_shape(size<0>(sQQ), size<1>(sQQ)));
    Tensor tQQcQQ = gmem_thr_copy_Quant.partition_S(cQQ); // (CPY, CPY_M, CPY_K)
    Tensor tQQpQQ = make_tensor<bool>(make_layout(make_shape(size<1>(tQQcQQ), size<2>(tQQcQQ)), make_stride(_1{}, _0{})));

    /* Retiling */
    Tensor t4rQQ_int8_copy_view = smem_thr_copy_QQ_int8.retile_D(recast<UInt8_Type>(t4rQQ_int4));
    Tensor t4rQK_int8_copy_view = smem_thr_copy_QK_int8.retile_D(recast<UInt8_Type>(t4rQK_int4));
    CUTE_STATIC_ASSERT_V(size<1>(t4sQQ_int8) == size<1>(t4rQQ_int8_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(t4sQK_int8) == size<1>(t4rQK_int8_copy_view));        

    #pragma unroll
    for (int k = 0; k < size<0>(tQQpQQ); ++k)
        tQQpQQ(k, 0) = get<0>(tQQcQQ(0, k, 0)) < (binfo.actual_seqlen_q - m_block * kBlockM);

    cute::copy_if(gmem_tiled_copy_Quant, tQQpQQ, tQgQQ, tQsQQ);
    cute::cp_async_fence();

    
    if (Kernel_traits::Is_QQ_in_regs) {
        flash::cp_async_wait<0>();
        __syncthreads(); // Ensure that all cp instruction has completed.
        cute::copy(smem_tiled_copy_QQ_int8, t4sQQ_int8, t4rQQ_int8_copy_view); 
    }
    Tensor rSQ = make_tensor<Element>(Shape<_2, Int<size<1>(t4rQQ_int4)>>{}); 
    { 
        const int base_offset = warp_row_tile_offset * 16 + row_offset_within_warp;
        const int max_M_idx = binfo.actual_seqlen_q - m_block * kBlockM;
        // load sQ directly from global memory.
        #pragma unroll
        for(int mi = 0; mi < size<1>(t4rQQ_int4); mi ++){ 
            int offset = base_offset + tile_size<0>(int4_tiled_mma) * mi;
            rSQ(make_coord(_0{}, mi)) = offset < max_M_idx ? gSQ(offset) : Element(0.0);
            rSQ(make_coord(_1{}, mi)) = (offset + 8) < max_M_idx ? gSQ(offset + 8) : Element(0.0);
        }
    }
    

    Element sq_reciprocal;
    sq_reciprocal = static_cast<Element>(1 / static_cast<float>(rSQ(_0{}))); // make compiler happy :)
    Tensor dummy_weight_int32 = partition_fragment_C(int4_tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(dummy_weight_int32);
    CUTE_STATIC_ASSERT(size<0>(dummy_weight_int32) == 4);
    using m_extent = decltype(size<1>(dummy_weight_int32));
    using n_extent = decltype(size<2>(dummy_weight_int32));
    Tensor dummy_weight_fp32 = recast<float>(dummy_weight_int32);
    Tensor dummy_weight_int32_ = make_tensor(
        dummy_weight_int32.data(), 
        flash::convert_layout_acc_rowcol(dummy_weight_int32.layout())
    );

    // Tensor dummy_weight_fp32 = make_tensor_like<float>(dummy_weight_int32);
    Tensor rowwise_max = make_tensor<int>(Shape<Int<size(rSQ)>>{}); // 2 * M
    Tensor rowwise_max_dequant = make_tensor_like<Element>(rowwise_max); // Do not declare a new array.

    Tensor t4rQMK_div_sq_sk = make_tensor_like<int>(t4rQMK); 
    Tensor cta_records = make_tensor<unsigned>(Shape<Int<Kernel_traits::kNWarps>>{});

    offsets_iterator<blocks_per_seg> off_iter;
    offsets_iterator<32> big_off_iter;
    int seg_offset = 1, big_cnt = 0;

    softmax.template reduce_row_sum<Element>(threshold_per_head, params.scale_softmax_log2);

    constexpr int big_seg = 32 * kBlockN;
    constexpr int seg_per_big_seg = big_seg / Kernel_traits::kBlockSeg;
    int n_big_segs = n_segs / seg_per_big_seg;
    n_segs = n_segs - n_big_segs * seg_per_big_seg;

    
    for (; n_big_segs > 0; n_big_segs--) {
        int istage_prefetch = kPipeSize - 1;
        int in_prefetch = kPipeSize - 1;
        int istage_now = 0; 
        unsigned record = 0;
        #pragma unroll
        for (int istage = 0; istage < (Kernel_traits::kPipeSize - 1); istage++){
            flash::copy<true, Is_even_K>(gmem_tiled_copy_Quant, tKgQK(_, _, _, istage + seg_offset), tKsQK(_, _, _, istage), tKVcKV, tKVpKV);
            flash::copy_vector_off_band(gmem_tiled_copy_K_Scale, tKgSK(_, _, istage + seg_offset), tKsSK(_, _, istage));
            flash::copy_vector_off_band(gmem_tiled_copy_QMK, tQgQMK(_, _, istage + seg_offset), tQsQMK(_, _, istage));
            cute::cp_async_fence(); 
        }

        #pragma unroll
        for (int b_i = 0; b_i < 32; b_i++){
            flash::cp_async_wait<kPipeSize - 2>();
            __syncthreads();
    
            // Pre-issue s2r to start the MMA pipeline.
            if (!Kernel_traits::Is_QQ_in_regs) { cute::copy(smem_tiled_copy_QQ_int8, t4sQQ_int8(_, _, _0{}), t4rQQ_int8_copy_view(_, _, _0{})); } 
            cute::copy(smem_tiled_copy_QK_int8, t4sQK_int8(_, _, _0{}, istage_now), t4rQK_int8_copy_view(_, _, _0{}));
    
            if (b_i < 32 - (Kernel_traits::kPipeSize - 1)) {
                flash::copy<true, Is_even_K>(gmem_tiled_copy_Quant, tKgQK(_, _, _, in_prefetch + seg_offset), tKsQK(_, _, _, istage_prefetch), tKVcKV, tKVpKV);
                flash::copy_vector_off_band(gmem_tiled_copy_K_Scale, tKgSK(_, _, in_prefetch + seg_offset), tKsSK(_, _, istage_prefetch));
                flash::copy_vector_off_band(gmem_tiled_copy_QMK, tQgQMK(_, _, in_prefetch + seg_offset), tQsQMK(_, _, istage_prefetch));
                cute::cp_async_fence(); 
                istage_prefetch = (istage_prefetch + 1) & (Kernel_traits::kPipemask);
                in_prefetch = in_prefetch + 1;
            } else {
                cute::cp_async_fence();
            }
    
            bool is_big_block;

            CUTE_STATIC_ASSERT(size<0>(get<0>(dummy_weight_int32.layout())) == _2{});
            CUTE_STATIC_ASSERT(size<1>(get<0>(dummy_weight_int32.layout())) == _2{});
            // dequant.
            if constexpr (Quant_mode == 3) /* Per-warp Quant */{
                CUTE_STATIC_ASSERT(size<1>(dummy_weight_int32_) == size(t4rQMK_div_sq_sk));

                clear(dummy_weight_int32);
                #pragma unroll
                for (int i = 0; i < size<2>(t4rQQ_int4); ++i) {
                    if (i < size<2>(t4rQQ_int4) - 1) {
                        if (!Kernel_traits::Is_QQ_in_regs) { cute::copy(smem_tiled_copy_QQ_int8, t4sQQ_int8(_, _, i+1), t4rQQ_int8_copy_view(_, _, i+1)); }
                        cute::copy(smem_tiled_copy_QK_int8, t4sQK_int8(_, _, i+1, istage_now), t4rQK_int8_copy_view(_, _, i+1));
                    }
                    cute::gemm(int4_thr_mma, t4rQQ_int4(_, _, i), t4rQK_int4(_, _, i), dummy_weight_int32);
                }

                cute::copy(smem_tiled_copy_SQ_SK_QMK, t4sQMK(_, _, _, istage_now), t4rQMK);

                #pragma unroll
                for (int mi = 0; mi < size(t4rQMK); mi++)
                    t4rQMK(mi) = t4rQMK(mi) * sq_reciprocal;

                #pragma unroll
                for (int mi = 0; mi < size(t4rQMK); mi++){
                    int qmk_v;
                    if constexpr (std::is_same_v<Element, cutlass::half_t>)
                        qmk_v = __half2int_rn(t4rQMK(mi).to_half());
                    else
                        qmk_v = __bfloat162int_rn(t4rQMK(mi).to_nv_bfloat16());      

                    t4rQMK_div_sq_sk(mi)  = qmk_v;              
                }
                
                cute::copy(smem_tiled_copy_SQ_SK_QMK, t4sSK(_, _, _, istage_now), t4rSk);

                CUTE_STATIC_ASSERT(size(t4rQMK_div_sq_sk) == size<1>(dummy_weight_int32_));
                #pragma unroll
                for (int mi = 0; mi < size<0>(dummy_weight_int32_); mi++){
                    rowwise_max(mi) = dummy_weight_int32_(mi, _0{}) + t4rQMK_div_sq_sk(_0{}, _0{}, _0{});
                    #pragma unroll
                    for (int ni = 1; ni < size<1>(dummy_weight_int32_); ni++){
                        int temp = dummy_weight_int32_(mi, ni) + t4rQMK_div_sq_sk(ni);
                        rowwise_max(mi) = temp > rowwise_max(mi) ? temp : rowwise_max(mi);
                    }
                }   
                Element scale = rSQ[0] * t4rSk[0];
                #pragma unroll
                for (int mi = 0; mi < size(rowwise_max_dequant); mi++){
                    rowwise_max_dequant[mi] = static_cast<Element>(rowwise_max[mi]);
                }
                rowwise_max_dequant[0] = rowwise_max_dequant[0] * scale;
                rowwise_max_dequant[1] = rowwise_max_dequant[1] * scale;

                is_big_block = rowwise_max_dequant[0] > softmax.threshold_temp[0] || rowwise_max_dequant[1] > softmax.threshold_temp[1];
            }

            record = record | (static_cast<unsigned>(is_big_block) << b_i); 

            istage_now = (istage_now + 1) & (Kernel_traits::kPipemask); // kPipeSize
        }


        /* CTA allreduce */
        #pragma unroll
        for (int t = 32; t > 1; t >>= 1){
            unsigned temp = __shfl_xor_sync(0xffffffff, record, t - 1); // mask: 31, 15, 7, 3, 1
            record |= temp;
        }

        if (tid_is_leader) temp_storage(warp_idx) = record;
        static_assert(Kernel_traits::kNWarps * 4 <= kBlockN * sizeof(Element), "Smem space for all reduce is too small!!");    
        __syncthreads(); 

        cute::copy(Copy_Atom<DefaultCopy, unsigned>{}, temp_storage, cta_records);
        record = cta_records[0];
        #pragma unroll
        for(int i=1;i<Kernel_traits::kNWarps;i++) record |= cta_records[i];
        
        big_off_iter.clear();
        int pos_now = -1, pos_next = big_off_iter.next(record);
        
        // FP16 pipeline.
        // for (int b_i = 0; b_i < big_block_cnt; b_i++){
        while(pos_next >= 0) {
            pos_now = pos_next;
            pos_next = big_off_iter.next(record);

            if(tid_is_leader) {gCPRC(seg_offset + pos_now /* big_block_offsets[b_i] */) = 1;}
        }
        seg_offset += 32;
    }
    __syncthreads();

    for (; n_segs > 0; n_segs--) {
        unsigned record = 0;
        static_assert(blocks_per_seg <= 32, "segment is too large!");

        int istage_prefetch = kPipeSize - 1;
        int in_prefetch = kPipeSize - 1;
        int istage_now = 0; 
        #pragma unroll
        for (int istage = 0; istage < (Kernel_traits::kPipeSize - 1); istage++){
            flash::copy<true, Is_even_K>(gmem_tiled_copy_Quant, tKgQK(_, _, _, istage + seg_offset), tKsQK(_, _, _, istage), tKVcKV, tKVpKV);
            flash::copy_vector_off_band(gmem_tiled_copy_K_Scale, tKgSK(_, _, istage + seg_offset), tKsSK(_, _, istage));
            flash::copy_vector_off_band(gmem_tiled_copy_QMK, tQgQMK(_, _, istage + seg_offset), tQsQMK(_, _, istage));
            cute::cp_async_fence(); 
        }

        #pragma unroll
        for (int b_i = 0; b_i < blocks_per_seg; b_i++){
            flash::cp_async_wait<kPipeSize - 2>();
            __syncthreads();
    
            // Pre-issue s2r to start the MMA pipeline.
            if (!Kernel_traits::Is_QQ_in_regs) { cute::copy(smem_tiled_copy_QQ_int8, t4sQQ_int8(_, _, _0{}), t4rQQ_int8_copy_view(_, _, _0{})); } 
            cute::copy(smem_tiled_copy_QK_int8, t4sQK_int8(_, _, _0{}, istage_now), t4rQK_int8_copy_view(_, _, _0{}));
    
            if (b_i < blocks_per_seg - (Kernel_traits::kPipeSize - 1)) {
                flash::copy<true, Is_even_K>(gmem_tiled_copy_Quant, tKgQK(_, _, _, in_prefetch + seg_offset), tKsQK(_, _, _, istage_prefetch), tKVcKV, tKVpKV);
                flash::copy_vector_off_band(gmem_tiled_copy_K_Scale, tKgSK(_, _, in_prefetch + seg_offset), tKsSK(_, _, istage_prefetch));
                flash::copy_vector_off_band(gmem_tiled_copy_QMK, tQgQMK(_, _, in_prefetch + seg_offset), tQsQMK(_, _, istage_prefetch));
                cute::cp_async_fence(); 
                istage_prefetch = (istage_prefetch + 1) & (Kernel_traits::kPipemask);
                in_prefetch = in_prefetch + 1;
            } else {
                cute::cp_async_fence(); 
            }
    
            bool is_big_block;

            CUTE_STATIC_ASSERT(size<0>(get<0>(dummy_weight_int32.layout())) == _2{});
            CUTE_STATIC_ASSERT(size<1>(get<0>(dummy_weight_int32.layout())) == _2{});
            // dequant.
            if constexpr (Quant_mode == 3) /* Per-thread Quant */{
                CUTE_STATIC_ASSERT(size<1>(dummy_weight_int32_) == size(t4rQMK_div_sq_sk));

                clear(dummy_weight_int32);
                #pragma unroll
                for (int i = 0; i < size<2>(t4rQQ_int4); ++i) {
                    if (i < size<2>(t4rQQ_int4) - 1) {
                        if (!Kernel_traits::Is_QQ_in_regs) { cute::copy(smem_tiled_copy_QQ_int8, t4sQQ_int8(_, _, i+1), t4rQQ_int8_copy_view(_, _, i+1)); }
                        cute::copy(smem_tiled_copy_QK_int8, t4sQK_int8(_, _, i+1, istage_now), t4rQK_int8_copy_view(_, _, i+1));
                    }
                    cute::gemm(int4_thr_mma, t4rQQ_int4(_, _, i), t4rQK_int4(_, _, i), dummy_weight_int32);
                }

                cute::copy(smem_tiled_copy_SQ_SK_QMK, t4sQMK(_, _, _, istage_now), t4rQMK);

                #pragma unroll
                for (int mi = 0; mi < size(t4rQMK); mi++)
                    t4rQMK(mi) = t4rQMK(mi) * sq_reciprocal;

                #pragma unroll
                for (int mi = 0; mi < size(t4rQMK); mi++){
                    int qmk_v;
                    if constexpr (std::is_same_v<Element, cutlass::half_t>)
                        qmk_v = __half2int_rn(t4rQMK(mi).to_half());
                    else
                        qmk_v = __bfloat162int_rn(t4rQMK(mi).to_nv_bfloat16());      

                    t4rQMK_div_sq_sk(mi)  = qmk_v;              
                }
                
                cute::copy(smem_tiled_copy_SQ_SK_QMK, t4sSK(_, _, _, istage_now), t4rSk);

                CUTE_STATIC_ASSERT(size(t4rQMK_div_sq_sk) == size<1>(dummy_weight_int32_));
                #pragma unroll
                for (int mi = 0; mi < size<0>(dummy_weight_int32_); mi++){
                    rowwise_max(mi) = dummy_weight_int32_(mi, _0{}) + t4rQMK_div_sq_sk(_0{}, _0{}, _0{});
                    #pragma unroll
                    for (int ni = 1; ni < size<1>(dummy_weight_int32_); ni++){
                        int temp = dummy_weight_int32_(mi, ni) + t4rQMK_div_sq_sk(ni);
                        rowwise_max(mi) = temp > rowwise_max(mi) ? temp : rowwise_max(mi);
                    }
                }   
                Element scale = rSQ[0] * t4rSk[0];
                #pragma unroll
                for (int mi = 0; mi < size(rowwise_max_dequant); mi++){
                    rowwise_max_dequant[mi] = static_cast<Element>(rowwise_max[mi]);
                }
                rowwise_max_dequant[0] = rowwise_max_dequant[0] * scale;
                rowwise_max_dequant[1] = rowwise_max_dequant[1] * scale;

                is_big_block = rowwise_max_dequant[0] > softmax.threshold_temp[0] || rowwise_max_dequant[1] > softmax.threshold_temp[1];
            }

            record = record | (static_cast<unsigned>(is_big_block) << b_i); 

            istage_now = (istage_now + 1) & (Kernel_traits::kPipemask); // kPipeSize
        }


        /* CTA allreduce */
        #pragma unroll 
        for (int t = 32; t > 1; t >>= 1){
            unsigned temp = __shfl_xor_sync(0xffffffff, record, t - 1); // mask: 31, 15, 7, 3, 1
            record |= temp;
        }

        if (tid_is_leader) temp_storage(warp_idx) = record;
        static_assert(Kernel_traits::kNWarps * 4 <= kBlockN * sizeof(Element), "Smem space for all reduce is too small!!");    
        __syncthreads(); 

        cute::copy(Copy_Atom<DefaultCopy, unsigned>{}, temp_storage, cta_records);
        record = cta_records[0];
        #pragma unroll
        for(int i=1;i<Kernel_traits::kNWarps;i++) record |= cta_records[i];
        
        off_iter.clear();
        int pos_now = -1, pos_next = off_iter.next(record);
        
        // FP16 pipeline.
        // for (int b_i = 0; b_i < big_block_cnt; b_i++){
        while(pos_next >= 0) {
            pos_now = pos_next;
            pos_next = off_iter.next(record);

            if(tid_is_leader) {gCPRC(seg_offset + pos_now /* big_block_offsets[b_i] */) = 1;}
        }
        seg_offset += blocks_per_seg;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// NOTE: Main Enter.
template<typename Sparse_Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Record_approx, int Quant_mode, bool Two_pass, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x; 
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;


    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    Tensor gThresholds = make_tensor(
        make_gmem_ptr(reinterpret_cast<float*>(params.threshold_ptr)),
        make_shape(params.h), 
        make_stride(_1{})
    );
    // const float* thresh_ptr = reinterpret_cast<float*>(params.threshold_ptr) + params.h * bidb + bidh;
    const float threshold = gThresholds(bidh);
    // const float threshold = 0.004;
    flash::compute_attn_1rowblock_skipping_v2<
        Sparse_Kernel_traits, Is_causal, Is_even_MN, Is_even_K, false, true, Record_approx, Quant_mode, Two_pass
    >(params, bidb, bidh, m_block, threshold);
}

template<typename Sparse_Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Record_approx, int Quant_mode, bool Two_pass, typename Params>
inline __device__ void compute_map(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;


    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    Tensor gThresholds = make_tensor(
        make_gmem_ptr(reinterpret_cast<float*>(params.threshold_ptr)),
        make_shape(params.h), 
        make_stride(_1{})
    );
    // const float* thresh_ptr = reinterpret_cast<float*>(params.threshold_ptr) + params.h * bidb + bidh;
    const float threshold = gThresholds(bidh);
    // const float threshold = 0.004;
    flash::compute_map_1rowblock_skipping_v2<
        Sparse_Kernel_traits, Is_causal, Is_even_MN, Is_even_K, false, true, Quant_mode
    >(params, bidb, bidh, m_block, threshold);
}


} // namespace flash
