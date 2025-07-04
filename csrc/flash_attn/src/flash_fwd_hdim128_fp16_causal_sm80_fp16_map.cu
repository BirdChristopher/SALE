// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_fp16_sparse_launch_template.h"

template<>
void run_mha_fwd_fp16_sparse_v2_map_<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_fp16_v2_map<cutlass::half_t, true>(params, stream);
}