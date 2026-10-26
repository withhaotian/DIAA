import argparse
import os
from fastchat.utils import str_to_torch_dtype

from evaluation.eval_chat import run_eval, reorg_answer_file

from transformers import AutoTokenizer

from model.diaa.diaa_model import DIAA
from model.diaa.modeling_patch.modeling_llama_kv import LlamaForCausalLM
from model.diaa.utils import *

import torch
import gc

@torch.inference_mode()
def diaa_forward(
    input_ids,
    model,
    tokenizer,
    max_new_tokens,
    temperature=0.0,
    do_sample=False,
    output_id_topk=8,
    tree_version="dynamic",
    tree_budget=None,
    min_thres=None,
    max_depth=None,
    dy_interval=None,
):
    return model.diaa_forward(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        output_id_topk=output_id_topk,
        tree_version=tree_version,
        tree_budget=tree_budget,
        min_thres=min_thres,
        max_depth=max_depth,
        dy_interval=dy_interval,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=2048,
        help="The maximum number of context tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--output-id-topk",
        type=int,
        default=8,
        help="how many tokens from output_ids to be used to renew the tree.",
    )
    parser.add_argument(
        "--tree_version",
        type=str,
        default="dynamic",
        help="the version of tree.",
    )
    parser.add_argument(
        "--tree_budget",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=7,
        help="the maximum depth of the tree.",
    )
    parser.add_argument(
        "--min_thres",
        type=float,
        default=0.0005,
        help="the minimum threshold of the tree.",
    )
    parser.add_argument(
        "--dy_interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-num",
        type=int,
        default=10,
        help="The number of samples.",
    )
    parser.add_argument(
        "--cache-lut",
        action="store_true",
        default=False,
        help="Whether to use cached LUT.",
    )
    parser.add_argument(
        "--cache-lut-path",
        type=str,
        default=None,
        help="The path to the cached LUT.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The sampling seed.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="The timestamp of the run.",
    )
    
    args = parser.parse_args()

    args.model_name = (args.model_id + "-diaa-" + str(args.cache_lut) + '-tree_version-' + str(args.tree_version) + '-' + str(args.dtype)+ "-temp-" + str(args.temperature)
                       + "-seed-" + str(args.seed) + "-max_new_tokens-" + str(args.max_new_tokens) + '-tree_budget-' + str(args.tree_budget) + '-max_depth-' + str(args.max_depth) + '-min_thres-' + str(args.min_thres) + '-dy_interval-' + str(args.dy_interval))
    
    answer_file = f"outputs/{args.bench_name}/{args.task_name}_{args.data_num}/model_answer/{args.model_id}/{args.timestamp}/{args.model_name}.jsonl"

    question_file = f"data/{args.bench_name}/{args.task_name}/question.jsonl"

    gc.collect()

    if "llama" in args.model_id.lower() or "vicuna" in args.model_id.lower():
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="cuda:0",
            attn_implementation="sdpa",    # by default
        )
    else:
        raise ValueError("[Error] Unsupported model: " + args.model_id)

    if args.cache_lut:
        cache_dir = "model/diaa/offline/cache"
        os.makedirs(cache_dir, exist_ok=True)
        lut_cache_path = os.path.join(cache_dir, f'lut_cache_{args.cache_lut_path}')
        print(f"Using cached LUT: {lut_cache_path}")

        try:
            token_LUT, prob_LUT = load_LUT(lut_cache_path, model.device)
        except FileNotFoundError:
            print("LUT cache not found. Initializing new LUT...")
            token_LUT, prob_LUT = initialize_LUT(model.vocab_size, args.output_id_topk, model.device)
    else:
        token_LUT, prob_LUT = initialize_LUT(model.vocab_size, args.output_id_topk, model.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # diaa model
    model = DIAA(
        base_model=model,
        tokenizer=tokenizer,
        token_LUT=token_LUT,
        prob_LUT=prob_LUT,
        max_output_length=args.max_output_length,
        output_id_topk=args.output_id_topk,
        tree_version=args.tree_version,
        tree_budget=args.tree_budget,
        min_thres=args.min_thres,
        max_depth=args.max_depth,
        dy_interval=args.dy_interval,
    )

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=diaa_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        temperature=args.temperature,
        do_sample=do_sample,
        output_id_topk=args.output_id_topk,
        tree_version=args.tree_version,
        seed=args.seed,
        task_name=args.task_name,
        data_num=args.data_num,
        max_output_length=args.max_output_length,
        tree_budget=args.tree_budget,
        max_depth=args.max_depth,
        min_thres=args.min_thres,
        dy_interval=args.dy_interval,
    )

    # reuse the LUT
    if args.cache_lut:
        save_LUT(lut_cache_path, token_LUT, prob_LUT)