#!/bin/bash 
set -x
SEED="4321"
COMPRESSOR="int4_skip_cuda" 
EXP_NAME=sale_default
DEVICE=0,1,2,3,4,5,6,7
TIMER_UP=1

CONF_NAME=dynamic_0.4
RATIO=0.1
THRESHOLD=0.002
BLOCK_SIZE_Q=64
BLOCK_SIZE_K=32
START_LAYER=0
QUANT_MODE=3
IMPL_VERSION=2
BLOCK_SEG=128
SPARSITY_THRESHOLD=0.5
PRINT_RECALL=0
SHOW_Q_ERROR=0

# CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TIMER=${TIMER_UP} \
CUDA_VISIBLE_DEVICES=${DEVICE} \
RPINT_RECALL=${PRINT_RECALLP} SHOW_Q_ERROR=${SHOW_Q_ERROR} \
torchrun --nproc-per-node 8 pred.py \
--model Qwen-2.5-32B-Instruct \
--pp-size 1 \
--compressor ${COMPRESSOR} \
--exp_name ${EXP_NAME} \
--ratio ${RATIO} \
--threshold ${THRESHOLD} \
--block_size_q ${BLOCK_SIZE_Q} \
--block_size_k ${BLOCK_SIZE_K} \
--start_layer ${START_LAYER} \
--quant-mode ${QUANT_MODE} \
--sparsity-threshold ${SPARSITY_THRESHOLD} \
--impl-version ${IMPL_VERSION} \
--block-seg ${BLOCK_SEG} \
--conf-name ${CONF_NAME}
# --test_mode 
# --model llama-7b \