"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import logging
import os
import time
import torch
import random
import numpy as np
import shortuuid

from tqdm import tqdm
from datasets import load_dataset
from human_eval.data import read_problems
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from random import sample
from safetensors.torch import save_file, load_file

import argparse
import os
from fastchat.utils import str_to_torch_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from model.diaa.kv_cache import initialize_past_key_values
from model.diaa.modeling_patch.modeling_llama_kv import LlamaForCausalLM
from model.diaa.diaa_tree import build_diaa_tree
from model.diaa.utils import *

import torch

def seed_everything(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clip_input(input_ids, max_new_tokens=512, tree_length=128, max_output_length=4096):
    total_len = input_ids.shape[1]
    budget = max_output_length - max_new_tokens - tree_length - 1
    if total_len > budget:
        input_ids = input_ids[:, total_len - budget:]
    return input_ids

def load_data(task_name, seed=42, data_num=10):
    data = []
    prompt_shots = ''
    
    if task_name == 'sharegpt':
        import re
        URL_PATTERN = re.compile(
            r'^(https?://|ftp://|www\.)|(?:dropbox\.com|drive\.google\.com|mega\.nz|onedrive\.live\.com|we.tl|pan\.baidu\.com|bit\.ly|t\.co)'
        )

        def contains_url(text):
            return bool(URL_PATTERN.search(text.lower()))
        
        dataset = load_dataset('RyokoAI/ShareGPT52K', split='train', data_files='old/sg_52k.json')
        dataset = dataset.map(
            lambda x: {"human_texts": [turn["value"] for turn in x["conversations"] if turn["from"] == "human"]},
            remove_columns=dataset.column_names
        ).filter(
            lambda x: len(x["human_texts"]) > 0 and len(x["human_texts"]) <= 3
            and all(isinstance(t, str) and len(t.strip()) > 0 for t in x["human_texts"])
            and not any(contains_url(t) for t in x["human_texts"])
        )
        indices = sample(range(len(dataset)), data_num)
        data = dataset.select(indices)
    
    else:
        raise ValueError("Error: Unsupported task.")

    return data, prompt_shots

def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_gpus_per_model,
        num_gpus_total,
        task_name,
        data_num,
        seed,
        tree_length,
        **kwargs,
):
    assert num_gpus_total % num_gpus_per_model == 0

    seed_everything(seed)

    data, prompt_shots = load_data(task_name, seed, data_num=data_num)
    get_answers_func = get_model_answers

    get_answers_func(
        model,
        tokenizer,
        forward_func,
        model_id,
        data,
        prompt_shots,
        answer_file,
        max_new_tokens,
        task_name,
        tree_length,
        **kwargs,
    )

@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        data,
        prompt_shots,
        answer_file,
        max_new_tokens,
        task_name,
        tree_length,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    sys_p = "You are a helpful, respectful and honest assistant."   
    
    for question in tqdm(data):
        if "vicuna" in model_id.lower():
            conv = get_conversation_template("vicuna")
        elif "llama-2" in model_id.lower():
            conv = get_conversation_template("llama-2")
            conv.system_message = sys_p
        elif "llama-3" in model_id.lower():
            conv = get_conversation_template("llama-3")
        
        if task_name == 'sharegpt':
            all_qs = question["human_texts"]
        for qs in all_qs:
            conv.messages = []
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            if "vicuna" in model_id.lower():
                conv.stop_str = "</s>"
            prompt = conv.get_prompt()
            if "llama-2" in model_id.lower():
                prompt += " "

            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids

            input_ids = clip_input(input_ids, 
                                max_new_tokens=max_new_tokens,
                                tree_length=tree_length, 
                                max_output_length=args.max_output_length)

            cur_accept_lengths_tree = []
            cur_draft_num = 0

            output_ids, new_token_num, step, accept_length_tree, draft_token_num = forward_func(
                input_ids,
                model,
                tokenizer,
                max_new_tokens,
                **kwargs,
            )

            cur_accept_lengths_tree.extend(accept_length_tree)
            cur_draft_num += draft_token_num
            output_ids = output_ids[0][len(input_ids[0]):]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

@torch.inference_mode()
def diaa_forward(input_ids, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False, output_id_topk=8, tree_version="2.2.2"):
    accept_length_list = []
    step = 0
    start_len = input_ids.size(1)
    cur_seq_len = start_len - 1
    verify_input_ids = input_ids[:, :-1]
    input_ids = input_ids[:, -1:]
    
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model, args.max_output_length)

    model(input_ids=verify_input_ids, past_key_values=past_key_values)
    
    tree_template = build_diaa_tree(tree_version)
    
    pad_len = len(tree_template[-1]) + 1

    def prepare_data(template, input_ids):
        candi_attention_ids = torch.zeros([input_ids.size(0), len(template)+1, len(template)+1], dtype=torch.long)
        candi_positon_ids = torch.zeros([input_ids.size(0), len(template)+1], dtype=torch.long)

        candi_attention_ids[:, :, 0] = 1
        candi_positon_ids[:, 0] = 0
        search_path = []
        father_index = [-1]
        candi_index = [-1]
        deep_split = []
        
        for candi_i in range(len(template)):
            tree_deep = len(template[candi_i])
            tree_index = template[candi_i][-1]

            if tree_deep == 1:
                father_index.append(0)
            else:
                father_index.append(template.index(template[candi_i][:-1])+1)
            candi_index.append(tree_index)

            candi_positon_ids[:, candi_i+1] = tree_deep

            if tree_deep != candi_positon_ids[:, candi_i]:
                deep_split.append(candi_i+1)

            cur_path = [0]
            candi_attention_ids[:, candi_i+1, candi_i+1] = 1

            for candi_j in range(candi_i):
                if template[candi_j] == template[candi_i][:len(template[candi_j])]:
                    candi_attention_ids[:, candi_i+1, candi_j+1] = 1
                    cur_path.append(candi_j+1)
            cur_path.append(candi_i+1)

            for sub_i in range(1, len(cur_path)):
                if cur_path[:sub_i] in search_path:
                    search_path.remove(cur_path[:sub_i])
            search_path.append(cur_path)

        deep_split.append(len(template)+1)

        search_path = torch.tensor([path + [-1] * (pad_len-len(path)) for path in search_path], dtype=torch.long, device=torch.device('cuda:0'))
        father_index = torch.tensor(father_index, dtype=torch.long, device=torch.device('cuda:0'))
        candi_index = torch.tensor(candi_index, dtype=torch.long, device=torch.device('cuda:0'))
        
        return candi_attention_ids.cuda(), candi_positon_ids.cuda(), search_path, father_index, candi_index, deep_split

    def prepare_data_input_ids(cur_input_ids, father_index, candi_index, deep_split):
        input_ids = torch.zeros_like(father_index, dtype=torch.long, device=cur_input_ids.device)
        input_probs = torch.zeros_like(father_index, dtype=torch.float32, device=cur_input_ids.device)
        input_ids[0] = cur_input_ids
        input_probs[0] = 1.0
        for layer in range(len(deep_split)-1):
            cur_father = input_ids[father_index[deep_split[layer]:deep_split[layer+1]]]
            cur_father_index = cur_father * output_id_topk + candi_index[deep_split[layer]:deep_split[layer+1]]
            input_ids[deep_split[layer]:deep_split[layer+1]] = token_LUT.view(-1)[cur_father_index]
            input_probs[deep_split[layer]:deep_split[layer+1]] = prob_LUT.view(-1)[cur_father_index]
        
        return input_ids.unsqueeze(0), input_probs

    eos_token_id = torch.tensor(tokenizer.eos_token_id, dtype=torch.long, device=torch.device('cuda:0'))
    
    attention_mask, position_ids, search_path, father_index, candi_index, deep_split= prepare_data(tree_template, input_ids)
    
    for i in range(max_new_tokens):
        input_ids, input_probs = prepare_data_input_ids(input_ids, father_index, candi_index, deep_split)
        merge_attention_mask = torch.cat([torch.ones([input_ids.size(0), len(tree_template)+1, verify_input_ids.size(1)], dtype=torch.long, device=torch.device('cuda:0')), attention_mask], dim=-1)
        merge_positon_ids = position_ids + verify_input_ids.size(1)
        outputs = model(input_ids=input_ids, attention_mask=merge_attention_mask.unsqueeze(1), position_ids=merge_positon_ids, past_key_values=past_key_values)
        
        model_res = torch.argmax(outputs.logits, dim=-1)
        
        all_input_path = input_ids[0][search_path]
        all_input_path[search_path==-1] = -100
        all_output_path = model_res[0][search_path]
        reward = torch.cumprod(all_input_path[:, 1:].eq(all_output_path[:, :-1]), dim=-1).sum(dim=-1)
        best_reward = reward.max()
        
        accept_len = 1 + best_reward
        accept_length_list.append(accept_len.item())
        best_path_index = torch.argmax(reward, dim=-1).to(torch.long)
        index_path = search_path[best_path_index][:accept_len]
        best_path_input = torch.index_select(input_ids, index=index_path, dim=1)

        tgt = past_key_values_data[..., verify_input_ids.size(1)+index_path, :]
        dst = past_key_values_data[..., verify_input_ids.size(1) : verify_input_ids.size(1) + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
        
        current_length_data.fill_(verify_input_ids.size(1) + tgt.shape[-2])

        verify_input_ids = torch.cat([verify_input_ids, best_path_input], dim=-1)
        if (best_path_input == eos_token_id).any() or verify_input_ids.size(1) - start_len > max_new_tokens:
            break
        
        topk = torch.topk(outputs.logits, k=output_id_topk, dim=-1)
        probs = outputs.logits.softmax(-1)
        
        topk_probs = torch.gather(probs, dim=-1, index=topk.indices)
        
        token_update = torch.topk(outputs.logits, k=output_id_topk, dim=-1)[1][0]
        prob_update = topk_probs
        token_LUT[input_ids.squeeze(0)] = token_update
        prob_LUT[input_ids.squeeze(0)] = prob_update
        
        input_ids = model_res[:, search_path[best_path_index][accept_len-1]].unsqueeze(-1).cuda()
        if (input_ids == eos_token_id).any():
            start_len -= 1
            break

        if verify_input_ids.size(1) - start_len > max_new_tokens:
            break
        
        step += 1
    
    non_ptg = cal_percentages(token_LUT)

    if non_ptg > args.max_ptg:
        exit(1)
    
    if args.cache_lut:
        save_file({'token_LUT': token_LUT.cpu(), 'prob_LUT': prob_LUT.cpu()}, lut_cache_path + '.safetensors')
    
    return verify_input_ids, verify_input_ids.size(1) - start_len, step, accept_length_list, verify_input_ids.size(1) - start_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
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
        default=1960,
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
        default="full",
        help="the version of tree template.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="The top-p for sampling.",
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
        "--max-ptg",
        type=int,
        default=50,
        help="Maximum percentage of non-zero tokens in the LUT.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The sampling seed.",
    )
    
    args = parser.parse_args()

    answer_file = ""
    question_file = ""

    if "llama" in args.model_id.lower() or "vicuna" in args.model_id.lower():
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=str_to_torch_dtype(args.dtype),
            low_cpu_mem_usage=True,
            device_map="cuda:0",
            offload_buffers=True,
            offload_state_dict=True,
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tree_length = len(build_diaa_tree(version='full'))

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
        tree_length=tree_length,
    )
