/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once
#include <ATen/cuda/CUDAContext.h>

#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_int4_sparse_kernel_v2.h"
#include <cstdlib>

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Sparse_Kernel_traits,  __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_int4_v2, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Record_approx, int Quant_mode, bool Two_pass) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn<Sparse_Kernel_traits,  Is_causal, Is_even_MN, Is_even_K, Is_softcap, Record_approx, Quant_mode, Two_pass>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_int4_v2_map, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Record_approx, int Quant_mode, bool Two_pass) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_map<Sparse_Kernel_traits,  Is_causal, Is_even_MN, Is_even_K, Is_softcap, Record_approx, Quant_mode, Two_pass>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Sparse_Kernel_traits, bool Is_dropout, bool Is_causal, bool map_only>
void run_flash_fwd_int4_v2(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Sparse_Kernel_traits::kSmemSize;

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Sparse_Kernel_traits::kBlockM - 1) / Sparse_Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h); 
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Sparse_Kernel_traits::kBlockN == 0 && params.seqlen_q % Sparse_Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Sparse_Kernel_traits::kHeadDim;
    const bool recording = params.recording;
    if constexpr (map_only){
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                BOOL_SWITCH(recording, RecordApprox, [&] {
                    QUANT_MODE_SWITCH(params.quant_mode, Quant_mode, [&] {
                        BOOL_SWITCH(params.two_pass, Two_pass, [&] {
                            auto kernel = &flash_fwd_kernel_int4_v2_map<Sparse_Kernel_traits, true, IsEvenMNConst && IsEvenKConst && Sparse_Kernel_traits::kHeadDim <= 128, true, false, RecordApprox, Quant_mode, Two_pass>;
                            // Will only return softmax if dropout, to reduce compilation time.
                            // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                            // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                            // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                            // If Is_local, set Is_causal to false
                            // auto kernel = &flash_fwd_kernel_int4_v2<Sparse_Kernel_traits, Full_Kernel_traits, Is_causal, IsEvenMNConst && IsEvenKConst && Sparse_Kernel_traits::kHeadDim <= 128, IsEvenKConst, false, RecordApprox, Quant_mode, Two_pass>;

                            // Set shared memory size to maximum since we doesn't utilize L1 cache.
                            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                            if (smem_size >= 48 * 1024) {
                                C10_CUDA_CHECK(cudaFuncSetAttribute(
                                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,smem_size));
                            }
                            // int ctas_per_sm;
                            // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Sparse_Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    } else {
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                BOOL_SWITCH(recording, RecordApprox, [&] {
                    QUANT_MODE_SWITCH(params.quant_mode, Quant_mode, [&] {
                        BOOL_SWITCH(params.two_pass, Two_pass, [&] {
                            auto kernel = &flash_fwd_kernel_int4_v2<Sparse_Kernel_traits, true, IsEvenMNConst && IsEvenKConst && Sparse_Kernel_traits::kHeadDim <= 128, true, false, RecordApprox, Quant_mode, Two_pass>;
                            // Will only return softmax if dropout, to reduce compilation time.
                            // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                            // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                            // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                            // If Is_local, set Is_causal to false
                            // auto kernel = &flash_fwd_kernel_int4_v2<Sparse_Kernel_traits, Full_Kernel_traits, Is_causal, IsEvenMNConst && IsEvenKConst && Sparse_Kernel_traits::kHeadDim <= 128, IsEvenKConst, false, RecordApprox, Quant_mode, Two_pass>;

                            // Set shared memory size to maximum since we doesn't utilize L1 cache.
                            // cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                            if (smem_size >= 48 * 1024) {
                                C10_CUDA_CHECK(cudaFuncSetAttribute(
                                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,smem_size));
                            }
                            // int ctas_per_sm;
                            // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Sparse_Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    }
    
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128_int4_v2(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;

    // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
    // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM. 
    if (is_sm8x) {
        BLOCK_N_SWITCH(params.block_k_size, kBlockN, [&]{
            BLOCK_SEG_SWITCH(params.block_seg_size, kBlockSeg, [&]{
                run_flash_fwd_int4_v2<
                    Int4_Flash_fwd_kernel_traits_v1<Headdim, 64, kBlockN, 4, 256, kBlockSeg, false, false, true, true,false, T>, false, Is_causal, false
                >(params, stream);
            });
        });
    } else {
        // // Ignore this branch for now.
        // run_flash_fwd_int4_v2<
        //     Int4_Flash_fwd_kernel_traits_v1<Headdim, 64, 16, 4, 128, 128, false, false, true, true, T>, 
        //     Int4_Flash_fwd_kernel_traits_v1<Headdim, 64, 16, 4, 128, 128, false, false, true, true, T>,
        //     false, Is_causal
        // >(params, stream);
    }
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, false, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, false, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, false, Is_causal>(params, stream);
    // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, false, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, false, Is_causal>(params, stream);
    // 1st ones are good for H100, A100
    // 2nd one is good for A6000 bc we get slightly better occupancy
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128_int4_v2_map(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;

    if (is_sm8x) {
        BLOCK_N_SWITCH(params.block_k_size, kBlockN, [&]{
            BLOCK_SEG_SWITCH(params.block_seg_size, kBlockSeg, [&]{
                run_flash_fwd_int4_v2<
                    Int4_Flash_fwd_kernel_traits_v1<Headdim, 64, kBlockN, 4, 128, kBlockSeg, false, false, true, true, true, T>, false, Is_causal, true
                >(params, stream);
            });
        });
    } else {
    }
}