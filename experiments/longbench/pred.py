import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse

import time
from loguru import logger

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    compressor_choices = [
        "sparge",
        "original",
        "oracle_topk",
        "oracle_offline_skip",
        "fp16_online_skip",
        "int4_skip",
        "int4_skip_cuda",
        "cuda_triton_ref_v1",
        "cuda_triton_ref_v2",
        "cuda_triton_ref_v0",
        "cuda_triton_ref_v4", # dynamic threshold
        "minfer",
        "flex"
    ]
    parser.add_argument('--model', type=str, default=None, choices=[ "Qwen2.5-32B-Instruct", "Llama-3.1-8B-Instruct"])
    parser.add_argument("--exp_name", type=str, default="dafault_exp")
    parser.add_argument("--compressor", type=str, default="off", choices=compressor_choices)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument('--pp-size', type=int, choices=[1,2,4,8])
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--start_layer', type = int, default=0)

    # For oracle topk
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument('--block_size_q', type=int, default=64)
    parser.add_argument('--block_size_k', type=int, default=64)

    # For skip kernel
    parser.add_argument("--threshold", type=float, default=0.004)
    parser.add_argument("--quant-mode", type=int, default=2)
    parser.add_argument("--sparsity-threshold", type=float, default=0)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--block-seg", type=int, default=256)
    parser.add_argument("--impl-version", type=int, default=0)
    parser.add_argument("--conf-name", type=str, default="sparsity")

    # For sparge kernel
    parser.add_argument("--l1", type=str, default="0.08")
    parser.add_argument("--pv-l1", type=str, default="0.09")

    # For FlexPrefill
    parser.add_argument("--flex-block-size", type=int, default=128)
    parser.add_argument("--flex-gamma", type=float, default=0.9)
    parser.add_argument("--flex-tau", type=float, default=0.1)
    parser.add_argument("--flex-min-budget", type=int, default=512)
    parser.add_argument("--flex-max-budget", type=int, default=0)

    # For Minfer
    parser.add_argument("--minfer-config-path", type=str, default="./null")

    return parser.parse_args(args)

# This is the customized building prompt for chat models

def build_chat(tokenizer, prompt, model_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(args, model, tokenizer, rank, world_size, data, max_length, \
                max_gen, prompt_format, dataset, model_name, model2path, out_path, attn_time_record):
    data_idx = None
    # data_idx = 55
    device = model.device
    
    min_ctx_length = 100000
    line_num = 0
    if not os.path.exists(out_path):
        line_num = 0
    else:
        with open(out_path, "r", encoding="utf-8") as f:
            while True:
                l = f.readline()
                if l == "":
                    break
                line_num += 1
    
    accumulated_records = 0
    for i, json_obj in tqdm(enumerate(data)):
        if i < line_num:
            continue

        a = time.perf_counter()
        if data_idx is not None and i != (data_idx - 1):
            continue
        
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt").input_ids[0]
        original_token_cnt = len(tokenized_prompt)
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True)
            original_token_cnt = max_length

        # chat models are better off without build prompts on these tasks
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False,
                            return_tensors="pt").to(device)
        
        context_length = input.input_ids.shape[-1]
        min_ctx_length = min(min_ctx_length, context_length)

        print("context_length", context_length)
        # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode(
                    "\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                input_ids=input.input_ids,
                attention_mask=input.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                top_p = 1
            )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        
        if data_idx is not None and i == (data_idx - 1):
            print("output:",pred)
            exit()

        if os.environ.get("RANK","0") == "0": 
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], 
                            "all_classes": json_obj["all_classes"], "length": json_obj["length"], 
                            "request_time": {"batch_time": 0, "batch_size": 1}, 
                            "input_tokens":int(original_token_cnt)}, f, ensure_ascii=False)
                f.write('\n')        
            if os.environ["TIMER"] == "1":
                try:
                    attn_time = 0
                    for layer in  model.model.layers:
                        start_e = layer.self_attn.attn_timer_start
                        end_e = layer.self_attn.attn_timer_end
                        attn_time += start_e.elapsed_time(end_e)
                    # print("Attn time part is:", attn_time)
                    attn_time_record[dataset] += attn_time
                    attn_time_record["Total_TTFT"] += attn_time
                    print("context_length", context_length, "Attn time sumup", attn_time_record["Total_TTFT"])
                except:
                    logger.warning("CUDA Event are not recorded!")

    print("minimum length is ", min_ctx_length)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args, path, model_name, device, pp_size = 1):
    config = AutoConfig.from_pretrained(path)
    config.pp_size = pp_size
    config.device = torch.device("cuda:0")
    config.compressor = args.compressor
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, config=config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if os.environ.get("WORLD_SIZE","1") == "1":
        model = AutoModelForCausalLM.from_pretrained(path, config=config) # .....
    else:
        model = AutoModelForCausalLM.from_pretrained(path, config=config, tp_plan="auto") # .....

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    config.start_layer = 0
    start_head_idx = config.num_attention_heads // world_size * rank
    end_head_idx = config.num_attention_heads // world_size * (rank + 1)
    if args.compressor in ["int4_skip_cuda", "original"]:
        config.block_size_q = args.block_size_q
        config.block_size_k = args.block_size_k
        config.ratio = args.ratio
        config.threshold = args.threshold
        config.quant_mode = args.quant_mode
        config.sparsity_threshold = args.sparsity_threshold
        config.local_window = args.local_window
        config.block_seg = args.block_seg
        config.impl_version = args.impl_version
        config.conf_name = args.conf_name

        if args.compressor == "int4_skip_cuda":
            try:
                path = f"../../int4_flash_attn/models_hyperparam/{model_name}"

                sparsity = torch.load(path+f"/{config.conf_name}.pt", weights_only=True)
                logger.info(f"We are loading config from {path}/{config.conf_name}.pt")
                
                config.sparsity = sparsity
                assert config.sparsity.shape == (config.num_hidden_layers, config.num_attention_heads)

                config.sparsity = torch.narrow(config.sparsity, 1, start_head_idx, end_head_idx-start_head_idx)
            except:
                raise Exception("Cannot load config file!")

        if "Llama" in model_name or "llama" in model_name:
            from int4_flash_attn.patch.llama31_sparse_patch_4_51 import LlamaForCausalLMPatch
            model = LlamaForCausalLMPatch(model, config)
        elif "Qwen" in model_name or "qwen" in model_name:
            from int4_flash_attn.patch.qwen25_sparse_patch_4_51 import Qwen2ForCausalLMPatch
            model = Qwen2ForCausalLMPatch(model, config)
    model = model.half().eval()
    
    return model, tokenizer


def get_config_str_list(args):
    if args.compressor in ["int4_skip_cuda"]:
        return [
                args.compressor,
                f"thresh_{args.threshold}",
                f"bq_{args.block_size_q}",
                f"bk_{args.block_size_k}",
                f"quant_{args.quant_mode}",
                f"min_sparse_{args.sparsity_threshold}"
                f"local_{args.local_window}",
                f"seg_{args.block_seg}",
                f"impl_{args.impl_version}",
                f"conf_{args.conf_name}"
            ]
    return [f"{args.compressor}_default"]

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en"]

    # narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum trec triviaqa samsum lsht passage_count passage_retrieval_en

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
        
    device = torch.device("cuda:0")
    model, tokenizer = load_model_and_tokenizer(args, model2path[model_name], model_name, device, len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))

    attn_latency = {"Total_TTFT": 0}

    for dataset in datasets:
        attn_latency[dataset] = 0

        logger.info(f"Yes we are evaluating {dataset}")
        # data = load_dataset('./data', dataset, split='test')
        data = load_dataset('json', data_files='./data/' + dataset+'.jsonl', split='train')
        exp_name = args.exp_name
        os.makedirs(f"pred/{model_name}/{dataset}/{exp_name}", exist_ok=True)
        config_str_list = get_config_str_list(args)
        out_path = f"pred/{model_name}/{dataset}/{exp_name}/{'_'.join(config_str_list)}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        
        get_pred(args, model, tokenizer, 0, world_size, data_all, max_length, max_gen,
                    prompt_format, dataset, model_name, model2path, out_path, attn_latency)
    
    print(attn_latency)
    print("All evaluation done.")
