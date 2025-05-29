set -x

export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR=$(dirname "$0")/..
MODEL_PATH="Qwen/Qwen2.5-32B-Instruct"
MAX_SEQ_LEN=129500
COMPRESSOR="int4_skip_cuda" 
EXP_NAME=sale_default
DEVICE=0,1,2,3

THRESHOLD=0.001
CONF_NAME=dynamic_0.4
SPARSITY_THRESHOLD=0.5
QUANT_MODE=3
IMPL_VERSION=2
BLOCK_SEG=128
BLOCK_SIZE_Q=64
BLOCK_SIZE_K=32

TIMER_UP=1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TIMER=${TIMER_UP} \
CUDA_VISIBLE_DEVICES=${DEVICE} \
torchrun --nproc-per-node 4 "$SCRIPT_DIR/run_infinitebench.py" \
    --model_name_or_path ${MODEL_PATH} \
    --data_dir ./infinite_bench_data \
    --output_dir ./results_neq \
    --max_seq_length ${MAX_SEQ_LEN} \
    --num_eval_examples -1 \
    --fp16 \
    --exp_name ${EXP_NAME} \
    --compressor ${COMPRESSOR} \
    --threshold ${THRESHOLD} \
    --block_size_q ${BLOCK_SIZE_Q} \
    --block_size_k ${BLOCK_SIZE_K} \
    --quant-mode ${QUANT_MODE} \
    --sparsity-threshold ${SPARSITY_THRESHOLD} \
    --impl-version ${IMPL_VERSION} \
    --block-seg ${BLOCK_SEG} \
    --conf-name ${CONF_NAME}
