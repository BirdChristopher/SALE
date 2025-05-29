// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "cutlass/bfloat16.h"
#include "flash_fwd_fp16_sparse_launch_template.h"

template<>
void run_mha_fwd_fp16_sparse_v2_map_<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_fp16_v2_map<cutlass::bfloat16_t, true>(params, stream);
}

