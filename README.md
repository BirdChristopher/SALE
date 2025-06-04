# SALE
Code repository for paper: **SALE : Low-bit Estimation for Efficient Sparse Attention in Long-context LLM Prefilling**

## Environment requirements
- `python == 3.10`
- `gcc >= 9.4.0`
- `CUDA >= 12.4`

## Available API
```
import torch
from int4_flash_attn.flash_attn_int4_skip import flash_attn_func_int4_skip_sage, flash_attn_func_int4_skip

Q = torch.randn([1, 8, 32000, 128], dtype=torch.half).to("cuda")
K = torch.randn([1, 8, 32000, 128], dtype=torch.half).to("cuda")
V = torch.randn([1, 8, 32000, 128], dtype=torch.half).to("cuda")
thresholds = Q.new_zeros([Q.shape[1]], dtype=torch.float32).fill_(0.004)

# Full precision Computation Pass.
attn_output = flash_attn_func_int4_skip(
    Q,K,V, 
    softmax_scale = None,
    causal = True, 
    softcap=0, 
    thresholds=thresholds, 
    block_q_size=64, 
    block_k_size=32, 
    local_window=128, 
    block_seg_size=128, 
    debug=False, 
    impl_version=2, 
    quant_mode=3
) 

# Using SageAttention to accelerate Computation Pass.
attn_output = flash_attn_func_int4_skip_sage(
    Q,K,V, 
    softmax_scale = None,
    causal = True, 
    softcap=0, 
    thresholds=thresholds, 
    block_q_size=64, 
    block_k_size=32, 
    local_window=128, 
    block_seg_size=128, 
    debug=False, 
    impl_version=2, 
    quant_mode=3
) 

```
## Supported models
Currently, our monkeypatch implementation supports the following models:
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)

## Installation
```
cd SALE
conda env create -f env.yml
pip install -e . 
cd sparge_utils
pip install -e .
# Installation takes several minutes.
```



## Run Evaluation
All of the following experiments can be conducted on a server equipped with four NVIDIA RTX 4090 GPUs. 

### Model ckpt download
```
huggingface-cli download --resume-download meta-llama/Llama-3.1-8B-Instruct --token [YOUR_TOKEN]
```


### Per-head threshold calibration
```
bash run_kv_calib.sh meta-llama/Llama-3.1-8B-Instruct 0.4
```
- Optimal hyperparameters will be saved at `int4_flash_attn/models_hyperparam/Llama-3.1-8B-Instruct/dynamic_0.4.pt`.
- You can tune the sparsity level by change "0.4" to other positive value. 

### Single input latency
```
cd experiments
python 128k_niah_latency.py
```

### LongBench evaluation
- First, download the samples of [LongBench](https://huggingface.co/datasets/THUDM/LongBench) and place the original JSON-formatted data to `experiments/longbench/data`
- Second, run evaluation:
```
cd experiments/longbench

# Generation. Takes approximately 3 hours.
bash scripts/run_llama_cuda.sh

# Evaluation
python eval.py \
--model Llama-3.1-8B-Instruct \
--dataset narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum trec triviaqa samsum lsht passage_count passage_retrieval_en \
 --exp_name sale_default

# Summarize result, the evaluation result will saved in longbench/result/sale_default.json .
python parse_result.py \
--model Llama-3.1-8B-Instruct \
--result_path pred \
--exp_name sale_default \
--output_path result/sale_default.json
```
### InfiniteBench evaluation
- First, download the samples of [InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench) and place the original JSON-formatted data to `experiments/infinite_bench/infinite_bench_data`
```
cd experiments/infinite_bench

# Generation. Takes more than 10 hours to complete.
bash scripts/run_cuda.sh

# Evaluation result will be saved at infinite_bench/results_neq/Llama-3.1-8B-Instruct_sale_default_int4_skip_cuda_quant_3_local_128_seg_128_impl_3_conf_dynamic_0.4/
```

## Acknowledgement
During the development of this work, we learned design ideas from the following works and utilized their code implementations.
- [MInference](https://github.com/microsoft/MInference)
- [FlexPrefill](https://github.com/ByteDance-Seed/FlexPrefill)
- [SpargeAttn](https://github.com/thu-ml/SpargeAttn)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [LongBench](https://github.com/THUDM/LongBench)
- [InfiniteBench](https://github.com/OpenBMB/InfiniteBench)
- [Needle-In-A-Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)

## Citation
If you find our work useful, please cite [our paper](https://arxiv.org/abs/2505.24179).
