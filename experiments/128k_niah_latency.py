import os
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["DEBUG"] = "0"
PP_SIZE=4
os.environ["TIMER"] = "1"
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
from loguru import logger

model_path = "meta-llama/Llama-3.1-8B-Instruct" 
model_name = model_path.split("/")[-1]

def main():
    config = AutoConfig.from_pretrained(model_path)
    config.pp_size = PP_SIZE
    config.device = torch.device("cuda:0")
    os.environ["SAGE_UP"] = "1" # Integrated with Sage1
    config.compressor = "int4_skip_cuda" # int4_skip_cuda  or  original

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, config=config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config)

    config.start_layer = 0
    config.sparsity_threshold = 0.00001
    config.block_size_q = 64
    config.block_size_k = 32
    if config.compressor == "int4_skip_cuda":
        config.ratio = 0.1
        config.threshold = 0.001 #
        config.start_layer = 0
        config.quant_mode = 3 # 1: perblock 2. pertoken 3. perwarp
        config.sparsity_threshold = 0.000001 
        config.conf_name = "dynamic_0.4"

        config.local_window = 128
        config.block_seg = 128
        config.impl_version = 3
        try:
            path = f"../int4_flash_attn/models_hyperparam/{model_name}"

            sparsity = torch.load(path+f"/{config.conf_name}.pt", weights_only=True)
            logger.info(f"We are loading config from {path}/{config.conf_name}.pt")
            
            config.sparsity = sparsity
        except:
            raise Exception("Cannot load config file!")

    # from int4_flash_attn.patch.llama31_sparse_patch import LlamaForCausalLMPatch
    from int4_flash_attn.patch.llama31_sparse_patch_4_51 import LlamaForCausalLMPatch
    model = LlamaForCausalLMPatch(model, config)

    with open("./nah_input_128k.txt","r") as f:
        input_string = f.read()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_string}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    input_ = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda:0")
    context_length = input_.input_ids.shape[-1]
    print(f"input seq: {context_length}, {prompt[-500:]}")

    # warmup
    begin = time.perf_counter()
    output = model.generate(
                input_ids=input_.input_ids[:,:],
                attention_mask=None,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=1, 
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                use_cache=False
            )[0]
    end = time.perf_counter()
    pred = tokenizer.decode(output[context_length - 100:], skip_special_tokens=True) 
    logger.info(f"Output:'{pred}', time elasped:{end - begin}")   

    for _ in range(3):
        for ratio in [1, 1/2, 1/4, 1/8, 1/16]:
            begin = time.perf_counter()
            output = model.generate(
                    input_ids=input_.input_ids[:,:int(context_length * ratio)],
                    attention_mask=None,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=1, 
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    use_cache=False
                )[0]
            end = time.perf_counter()
            pred = tokenizer.decode(output[int(context_length * ratio):], skip_special_tokens=True)     
            logger.info(f"Output:'{pred}', context_len is:{int(context_length * ratio)}, time elasped:{end - begin}")   

            attn_time = 0
            for i in range(config.num_hidden_layers):
                start_e = model.model.layers[i].self_attn.attn_timer_start
                end_e = model.model.layers[i].self_attn.attn_timer_end
                attn_time += start_e.elapsed_time(end_e)
            print("Attn time part is:", attn_time)

if __name__ == "__main__":
    main()

