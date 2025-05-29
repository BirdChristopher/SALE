import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import argparse

from loguru import logger

prompt_format = "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None) # Model path
    parser.add_argument('--threshold', type=float, default=0.008)
    parser.add_argument('--coverage', type=float, default=0.99)
    parser.add_argument("--error", type=float, default=1)
    parser.add_argument("--calib-dataset", type=str, default="kv_retrieval_calibration_llama.jsonl")
    return parser.parse_args(args)

def get_pred(model, tokenizer, data, prompt_format):
    device = model.device    
    for i, json_obj in tqdm(enumerate(data)): 
        prompt = prompt_format.format(**json_obj) # Ignoring chat template is just fine.

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            
        context_length = input.input_ids.shape[-1]
        
        # NOTE: Code for calibration (Determine sparsity of different heads). Comment it out when doing evaluation.
        output = model.generate(
            input_ids=input.input_ids,
            attention_mask=input.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=1,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            top_p = 1
        )[0]

def load_model_and_tokenizer(args, path, model_name, device, pp_size = 1):
    config = AutoConfig.from_pretrained(path)
    config.pp_size = pp_size
    config.device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, config=config)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, config=config)
    config.threshold = args.threshold
    config.coverage = args.coverage
    config.error = args.error

    if "Llama" in model_name or "llama" in model_name:
        from int4_flash_attn.patch.llama31_sparse_patch_4_51 import LlamaForCausalLMPatch
        model = LlamaForCausalLMPatch(model, config)
    elif "Qwen" in model_name or "qwen" in model_name:
        from int4_flash_attn.patch.qwen25_sparse_patch_4_51 import Qwen2ForCausalLMPatch
        model = Qwen2ForCausalLMPatch(model, config)
    
    return model, tokenizer

if __name__ == '__main__':   
    args = parse_args()     
    device = torch.device("cuda:0")
    model_name = args.model.split("/")[-1]
    model, tokenizer = load_model_and_tokenizer(args, args.model, args.model, device, len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))

    data_all = []
    prompt_path = args.calib_dataset
    logger.warning(f"Calibrating on {prompt_path}")
    with open(f"calib_samples/{prompt_path}", "r") as f:
        while True:
            line = f.readline()
            if len(line) > 0:
                data_all.append(json.loads(line))
            else:
                break
    
    get_pred(model, tokenizer, data_all, prompt_format)
    dynamic_threshes, _ = model.analyse_sparsity(args.coverage)
    os.makedirs(f"int4_flash_attn/models_hyperparam/{model_name}/",exist_ok=True)
    torch.save(dynamic_threshes, f"int4_flash_attn/models_hyperparam/{model_name}/dynamic_{args.error}.pt")
