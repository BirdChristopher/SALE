# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from argparse import ArgumentParser, Namespace

from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS


def parse_args() -> Namespace:
    p = ArgumentParser()

    # MInference args.
    p.add_argument(
        "--data_dir", type=str, default="./infinite_bench_data", help="The directory of data."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Where to dump the prediction results.",
    )  # noqa
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-350m",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument(
        "--num_eval_examples",
        type=int,
        default=-1,
        help="The number of test examples to use, use all examples in default.",
    )  # noqa
    p.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.",
    )  # noqa
    p.add_argument(
        "--stop_idx",
        type=int,
        help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.",
    )  # noqa
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--max_seq_length", type=int, default=100000)
    p.add_argument("--rewrite", action="store_true") 
    p.add_argument("--starting_layer", type=int, default=-1)
    p.add_argument("--topk_dims_file_path", type=str, default=None)
    p.add_argument("--is_search", action="store_true")

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
    p.add_argument("--exp_name", type=str, default="dafault_exp")
    p.add_argument("--compressor", type=str, default="off", choices=compressor_choices)
    p.add_argument(
        "--fp16",
        action="store_true",
    )
    # For oracle topk
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument('--block_size_q', type=int, default=64)
    p.add_argument('--block_size_k', type=int, default=64)

    # For skip kernel
    p.add_argument("--threshold", type=float, default=0.004)
    p.add_argument("--quant-mode", type=int, default=2)
    p.add_argument("--sparsity-threshold", type=float, default=0)
    p.add_argument("--local-window", type=int, default=128)
    p.add_argument("--block-seg", type=int, default=256)
    p.add_argument("--impl-version", type=int, default=0)
    p.add_argument("--conf-name", type=str, default="0.001_0.7")


    # For sparge kernel
    p.add_argument("--l1", type=str, default="0.08")
    p.add_argument("--pv-l1", type=str, default="0.09")

    # For FlexPrefill
    p.add_argument("--flex-block-size", type=int, default=128)
    p.add_argument("--flex-gamma", type=float, default=0.9)
    p.add_argument("--flex-tau", type=float, default=0.1)
    p.add_argument("--flex-min-budget", type=int, default=512)
    p.add_argument("--flex-max-budget", type=int, default=0)

    # For Minfer
    p.add_argument("--minfer-config-path", type=str, default="./null")
    return p.parse_args()
