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

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Extra_reserve_blocks, int Quant_mode, typename Params>
inline __device__ void compute_map_1rowblock_skipping_fp16(const Params &params, const int bidb, const int bidh, const int m_block, const float threshold_per_head) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

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
                            typename Kernel_traits::SmemLayoutK{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx); // Type: ThrCopy<TiledCopy, ThrIdx>

    // tQgQ: Partitioning pattern tQ apply to gQ.
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ); 
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN) 
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);  // (kCPY, KCPY_N, KCPY_K, 2)

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK(_, _, _0{}));                           // (MMA,MMA_N,MMA_K)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma); // SM75_U32x4_LDSM_N
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    constexpr index_t one = 1; // make compiler happy

    Tensor mCPRC = make_tensor(
        make_gmem_ptr(reinterpret_cast<int32_t *>(params.cprc_ptr)
            + binfo.q_offset(params.cprc_batch_stride, one, bidb)),
        make_shape(params.h, params.n_Q_block, params.n_K_block),   
        make_stride(params.cprc_head_stride, params.cprc_qblk_stride, _1{})
    );

    Tensor gCPRC = mCPRC(bidh, m_block, _);

    unsigned * sbuf_ptr = reinterpret_cast<unsigned *>((sK.data() + size(sK)).get());
    Tensor temp_storage = make_tensor(make_smem_ptr(sbuf_ptr), Shape<Int<kNWarps>>{});
    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ))); 
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK(_,_,_,_0{}))));

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
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK(_, _, _, _0{}), tKVcKV, tKVpKV,
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
    constexpr int n_masking_steps = (Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1; 

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, _0{}), tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap) flash::apply_softcap(acc_s, params.softcap);

        // acc_s Layout: (MMA, MMA_M, MMA_N)
        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK(_, _, _, _0{}), tKVcKV, tKVpKV);
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
                acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, _0{}), tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            if constexpr (Is_softcap){ flash::apply_softcap(acc_s, params.softcap); }

            __syncthreads();
            if (n_block > n_block_min && rest_local > 1) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK(_, _, _, _0{}), tKVcKV, tKVpKV);
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            }
            else if (rest_local == 1 /* last iter */ && !sink_token_covered) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, 0), tKsK(_, _, _, _0{}), tKVcKV, tKVpKV);
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
                acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, _0{}), tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            if constexpr (Is_softcap){ flash::apply_softcap(acc_s, params.softcap); }

            softmax.template softmax_rescale_o_norescale_out</*Is_first=*/false, /*Check_inf=*/false, /* row_sum_sync */true>(acc_s, acc_o, params.scale_softmax_log2);

            if(tid_is_leader) {gCPRC(0) = 1; }
        }
    }

    __syncthreads();
    // Tensor dummy_weight_fp32 = make_tensor_like<float>(dummy_weight_int32);
    Tensor rowwise_max = make_tensor<float>(Shape<Int<size(softmax.row_max)>>{}); // 2 * M

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
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, istage + seg_offset), tKsK(_, _, _, istage), tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        #pragma unroll
        for (int b_i = 0; b_i < 32; b_i++){
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            Tensor acc_s_ = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            clear(acc_s);
            flash::cp_async_wait<kPipeSize - 2>();
            __syncthreads();

            if (b_i < 32 - (Kernel_traits::kPipeSize - 1)) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, in_prefetch + seg_offset), tKsK(_, _, _,  istage_prefetch), tKVcKV, tKVpKV);
                cute::cp_async_fence();
                istage_prefetch = (istage_prefetch + 1) & (Kernel_traits::kPipemask);
                in_prefetch = in_prefetch + 1;
            } else {
                cute::cp_async_fence();
            }
    
            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, istage_now), tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );

            bool is_big_block;
            
            #pragma unroll
            for (int mi = 0; mi < size<0>(acc_s_); mi++){
                rowwise_max(mi) = acc_s_(mi, _0{});
                #pragma unroll
                for (int ni = 1; ni < size<1>(acc_s_); ni++){
                    rowwise_max(mi) = acc_s_(mi, ni) > rowwise_max(mi) ? acc_s_(mi, ni) : rowwise_max(mi);
                }
            }   

            
            is_big_block = rowwise_max[0] > softmax.threshold_temp[0] || rowwise_max[1] > softmax.threshold_temp[1];

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
        static_assert(blocks_per_seg <= 32, "segment is too big!");

        int istage_prefetch = kPipeSize - 1;
        int in_prefetch = kPipeSize - 1;
        int istage_now = 0; 
        #pragma unroll
        for (int istage = 0; istage < (Kernel_traits::kPipeSize - 1); istage++){
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, istage + seg_offset), tKsK(_, _, _, istage), tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        #pragma unroll
        for (int b_i = 0; b_i < blocks_per_seg; b_i++){
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            Tensor acc_s_ = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            clear(acc_s);
            flash::cp_async_wait<kPipeSize - 2>();
            __syncthreads(); 

            if (b_i < blocks_per_seg - (Kernel_traits::kPipeSize - 1)) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, in_prefetch + seg_offset), tKsK(_, _, _,  istage_prefetch), tKVcKV, tKVpKV);
                cute::cp_async_fence();
                istage_prefetch = (istage_prefetch + 1) & (Kernel_traits::kPipemask);
                in_prefetch = in_prefetch + 1;
            } else {
                cute::cp_async_fence();
            }
    
            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, istage_now), tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );

            bool is_big_block;
            
            #pragma unroll
            for (int mi = 0; mi < size<0>(acc_s_); mi++){
                rowwise_max(mi) = acc_s_(mi, _0{});
                #pragma unroll
                for (int ni = 1; ni < size<1>(acc_s_); ni++){
                    rowwise_max(mi) = acc_s_(mi, ni) > rowwise_max(mi) ? acc_s_(mi, ni) : rowwise_max(mi);
                }
            }   

            is_big_block = rowwise_max[0] > softmax.threshold_temp[0] || rowwise_max[1] > softmax.threshold_temp[1];

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
    flash::compute_map_1rowblock_skipping_fp16<
        Sparse_Kernel_traits, Is_causal, Is_even_MN, Is_even_K, false, true, Quant_mode
    >(params, bidb, bidh, m_block, threshold);
}


} // namespace flash
