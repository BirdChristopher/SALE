// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "cutlass/bfloat16.h"
#include "flash_fwd_int4_sparse_launch_template.h"

template<>
void run_mha_fwd_int4_sparse_v1_<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_int4_v1<cutlass::bfloat16_t, true>(params, stream);
}

