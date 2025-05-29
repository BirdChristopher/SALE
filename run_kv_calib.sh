set -x

model_path=${1}
theta=${2}

# calib_accum
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CALIBRATION="calib_error" \
CUDA_VISIBLE_DEVICES=0,1 \
python kv_retrieve_calib.py \
--model ${model_path} \
--error ${theta}