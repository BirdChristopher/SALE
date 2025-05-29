# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from args import parse_args
from compute_scores import compute_scores
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    check_benchmark_availability,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM
)

from loguru import logger

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None

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

def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    args,
    tok: AutoTokenizer,
    input_text: str,
    attn_latency: dict,
    max_input_length: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_tokens = truncate_by_tokens(input_text, tok, max_input_length)

    input_tensors = {
        "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)
    }
    outputs = model.generate(**input_tensors, generation_config=generation_config)

    output = outputs[0, len(input_tokens) :]
    output = tok.decode(output, skip_special_tokens=True)
    output = output.strip()
    if os.environ.get("RANK","0") == "0": 
        if os.environ["TIMER"] == "1":
            try:
                attn_time = 0
                for layer in  model.model.layers:
                    start_e = layer.self_attn.attn_timer_start
                    end_e = layer.self_attn.attn_timer_end
                    attn_time += start_e.elapsed_time(end_e)
                # print("Attn time part is:", attn_time)
                attn_latency["TTFT"] += attn_time
                print("attn sum up", attn_latency["TTFT"])
            except:
                logger.warning("CUDA Event are not recorded!")

    return output

def load_model(
    args,
    path,
    model_name: str,
    starting_layer: int = -1,
    topk_dims_file_path: str = "",
    max_seq_length: int = None
):
    tok = AutoTokenizer.from_pretrained(
        path, resume_download=None, trust_remote_code=True, use_fast = False
    )
    tok.pad_token = tok.eos_token

    config = AutoConfig.from_pretrained(
        path, resume_download=None, trust_remote_code=True
    )
    config.pp_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    config.device = torch.device("cuda:0")
    config.compressor = args.compressor
    if os.environ.get("WORLD_SIZE","1") == "1":
        model = AutoModelForCausalLM.from_pretrained(path, config=config) # .....
    else:
        model = AutoModelForCausalLM.from_pretrained(path, config=config, tp_plan="auto") # .....


    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config.start_layer = 0
    start_head_idx = config.num_attention_heads // world_size * rank
    end_head_idx = config.num_attention_heads // world_size * (rank + 1)

    if args.compressor in [ "int4_skip_cuda", "original"]:
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

                config.sparsity = torch.narrow(config.sparsity, 1, start_head_idx, end_head_idx-start_head_idx).contiguous()
            except:
                raise Exception("Cannot load config file!")
        if "Llama" in model_name or "llama" in model_name:
            from int4_flash_attn.patch.llama31_sparse_patch_4_51 import LlamaForCausalLMPatch
            model = LlamaForCausalLMPatch(model, config)
        elif "Qwen" in model_name or "qwen" in model_name:
            from int4_flash_attn.patch.qwen25_sparse_patch_4_51 import Qwen2ForCausalLMPatch
            model = Qwen2ForCausalLMPatch(model, config)

    model = model.half().eval()
    print("Model and tokenizer loaded.")
    return model, tok

def get_config_str_list(args):
    if args.compressor in ["int4_skip_cuda",]:
        return [
                args.compressor,
                f"quant_{args.quant_mode}",
                f"local_{args.local_window}",
                f"seg_{args.block_seg}",
                f"impl_{args.impl_version}",
                f"conf_{args.conf_name}"
            ]
    return [f"{args.compressor}_default"]

@record
def main():
    args = parse_args()

    check_benchmark_availability(args.data_dir)
    print("Benchmark data has been found in local disk.")
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]

    tasks = ("kv_retrieval","longbook_choice_eng","math_find","longbook_qa_eng","longdialogue_qa_eng")

    # Model
    model, tok = load_model(
        args,
        args.model_name_or_path,
        real_model_name,
        args.starting_layer,
        args.topk_dims_file_path,
        max_seq_length=max_seq_length
        # Other args are useless and have been pruned.
    )
    
    config_strs = get_config_str_list(args)
    attn_latency = {"TTFT":0}
    for data_name in tasks:
        print("Going to evaluate", data_name)

        if "," in data_name:
            data_names = data_name.split(",")
        else:
            data_names = [data_name]
        results = {}

        for data_name in data_names:
            max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
            if max_new_tokens >= max_seq_length:
                max_new_tokens = 500

            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )

            # Data
            result_dir = Path(args.output_dir, f"{real_model_name}_{args.exp_name}_{'_'.join(config_strs)}")
            result_dir.mkdir(exist_ok=True, parents=True)
            output_path = result_dir / f"prediction_{data_name}.jsonl"
            examples = load_data(data_name, data_dir=args.data_dir)

            if args.num_eval_examples != -1:
                num_eval_examples = min(args.num_eval_examples, len(examples))
                examples = examples[:num_eval_examples]

            preds = []
            print("==== Evaluation ====")
            print(f"# examples: {len(examples)}")
            print(f"Num eval examples: {args.num_eval_examples}")
            print(f"Verbose: {args.verbose}")
            print(f"Max new tokens: {max_new_tokens}")

            if os.path.exists(output_path) and not args.rewrite:
                print(f"Output file {output_path} exists. Loading from file.")
                compute_scores(output_path, data_name, real_model_name, max_seq_length)
                with open(output_path) as f:
                    preds = [json.loads(ii) for ii in f.readlines()]

            for i, eg in tqdm(enumerate(examples)):
                if i < len(preds):
                    continue
                input_text = create_prompt(eg, data_name, real_model_name, args.data_dir)
                ground_truth = get_answer(eg, data_name)
                pred = get_pred(
                    model,
                    args,
                    tok,
                    input_text,
                    attn_latency,
                    max_input_length=max_seq_length - max_new_tokens,
                    verbose=args.verbose,
                    generation_config=generation_config,
                )
                preds.append({
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                })
                if os.environ.get("RANK","0") == "0": 
                    dump_jsonl(preds, output_path)
                torch.cuda.empty_cache()

            result_file_path = f"{args.exp_name}_{'_'.join(config_strs)}"
            score = compute_scores(output_path, data_name, result_file_path)
            results[data_name] = score

        print("==== Results ====")
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()