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
import inspect
import warnings

from tqdm import tqdm
# from datasets import load_dataset
# from human_eval.data import read_problems
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from fastchat.conversation import Conversation
import gc

# Silence noisy warnings during evaluation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def seed_everything(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clip_input(input_ids, max_new_tokens=512, tree_length=82, max_output_length=4096):
    total_len = input_ids.shape[1]
    budget = max_output_length - max_new_tokens - tree_length
    if total_len > budget:
        input_ids = input_ids[:, total_len - budget:]
    return input_ids


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
        max_output_length,
        **kwargs,
):
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    seed_everything(seed)

    # data, prompt_shots = load_data(task_name, seed, data_num=data_num)
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                task_name,
                data_num,
                max_output_length,
                num_choices=1,
                **kwargs,
            )
        )

    if use_ray:
        ray.get(ans_handles)


def _call_forward_func(forward_func, input_ids, model, tokenizer, max_new_tokens, **kwargs):
    sig = inspect.signature(forward_func)
    param_names = list(sig.parameters.keys())
    expects_model = "model" in param_names or "tokenizer" in param_names
    if expects_model or len(param_names) >= 4:
        return forward_func(
            input_ids,
            model,
            tokenizer,
            max_new_tokens,
            **kwargs,
        )
    return forward_func(
        input_ids,
        max_new_tokens,
        **kwargs,
    )


@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        task_name,
        data_num,
        max_output_length,
        num_choices,
        **kwargs,
):
    gc.collect()
    
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # questions = questions[:data_num]
    questions = random.sample(questions, data_num)

    # question = questions[0]
    # # warmup
    # for i in range(3):
    #     choices = []

    #     sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

    #     for i in range(num_choices):
    #         torch.manual_seed(i)

    #         # vicuna, llama-2-chat-hf
    #         if "vicuna" in model_id.lower():
    #             conv = get_conversation_template("vicuna")
    #         elif "-chat" in model_id.lower():
    #             conv = get_conversation_template("llama-2-chat")
    #             conv.system_message = sys_p
    #         elif "llama-2" in model_id.lower():
    #             conv = get_conversation_template("llama-2")
    #             conv.system_message = sys_p
    #         elif "llama-3" in model_id.lower():
    #             conv = get_conversation_template("llama-3")
            
    #         cur_accept_lengths_tree = []
    #         cur_draft_num = 0
    #         steps = []
    #         new_tokens = []
    #         wall_time = []
    #         for j in range(len(question["turns"])):
    #             qs = question["turns"][j]
    #             conv.append_message(conv.roles[0], qs)
    #             conv.append_message(conv.roles[1], None)
    #             if "vicuna" in model_id.lower():
    #                 conv.stop_str = "</s>"
    #             prompt = conv.get_prompt()
    #             if "llama-2" in model_id.lower():
    #                 prompt += " "

    #             inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    #             input_ids = inputs.input_ids

    #             input_ids = clip_input(input_ids,
    #                     max_new_tokens=max_new_tokens,
    #                     max_output_length=model.config.max_position_embeddings)

    #             torch.cuda.synchronize()
    #             start_time = time.perf_counter()
    #             output_ids, new_token_num, step, accept_length_tree, draft_token_num = forward_func(
    #                 input_ids,
    #                 model,
    #                 tokenizer,
    #                 max_new_tokens,
    #                 **kwargs,
    #             )
    #             torch.cuda.synchronize()
    #             total_time = time.perf_counter() - start_time

    #             cur_accept_lengths_tree.extend(accept_length_tree)
    #             cur_draft_num += draft_token_num
    #             output_ids = output_ids[0][len(input_ids[0]):]

    #             if conv.stop_token_ids:
    #                 stop_token_ids_index = [
    #                     i
    #                     for i, id in enumerate(output_ids)
    #                     if id in conv.stop_token_ids
    #                 ]
    #                 if len(stop_token_ids_index) > 0:
    #                     output_ids = output_ids[: stop_token_ids_index[0]]

    #             output = tokenizer.decode(
    #                 output_ids,
    #                 spaces_between_special_tokens=False,
    #             )
    #             if conv.stop_str and output.find(conv.stop_str) > 0:
    #                 output = output[: output.find(conv.stop_str)]
    #             for special_token in tokenizer.special_tokens_map.values():
    #                 if isinstance(special_token, list):
    #                     for special_tok in special_token:
    #                         output = output.replace(special_tok, "")
    #                 else:
    #                     output = output.replace(special_token, "")
    #             output = output.strip()

    #             if conv.name == "xgen" and output.startswith("Assistant:"):
    #                 output = output.replace("Assistant:", "", 1).strip()

    #             steps.append(int(step))
    #             new_tokens.append(int(new_token_num))
    #             wall_time.append(total_time)

    print('Warmup done')

    accept_lengths_tree = []
    total_draft_num = 0

    speeds = []
    time_per_token = []
    wall_times = []

    for question in tqdm(questions):
        choices = []
        
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

        for i in range(num_choices):
            torch.manual_seed(i)
            # vicuna
            if "vicuna" in model_id.lower():
                conv = get_conversation_template("vicuna")
            elif "llama-2" in model_id.lower():
                conv = get_conversation_template("llama-2")
                conv.system_message = sys_p
            elif "llama-3" in model_id.lower():
                conv = get_conversation_template("llama-3")
            elif "tinyllama" in model_id.lower():
                conv = get_conversation_template("TinyLlama")
            elif "mistral" in model_id.lower():
                conv = get_conversation_template("mistral")
            elif "qwen" in model_id.lower():
                conv = get_conversation_template("qwen2")
            else:
                raise ValueError("[Error] Unsupported model: {}".format(model_id))
            
            cur_accept_lengths_tree = []
            cur_draft_num = 0
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                if "llama-3" in model_id.lower() and "instruct" not in model_id.lower():
                    # manully set prompt for llama-3-base
                    prompt = qs
                else:
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    if "vicuna" in model_id.lower():
                        conv.stop_str = "</s>"
                    prompt = conv.get_prompt()
                    if "llama-2" in model_id.lower() or "qwen" in model_id.lower():
                        prompt += " "
                

                inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                input_ids = inputs.input_ids

                input_ids = clip_input(input_ids,
                        max_new_tokens=max_new_tokens,
                        max_output_length=max_output_length)

                torch.cuda.synchronize()
                start_time = time.perf_counter()
                output_ids, new_token_num, step, accept_length_tree, draft_token_num = _call_forward_func(
                    forward_func,
                    input_ids,
                    model,
                    tokenizer,
                    max_new_tokens,
                    **kwargs,
                )
                torch.cuda.synchronize()
                total_time = time.perf_counter() - start_time

                cur_accept_lengths_tree.extend(accept_length_tree)
                cur_draft_num += draft_token_num
                output_ids = output_ids[0][len(input_ids[0]):]

                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()

                turns.append(output)
                steps.append(int(step))
                new_tokens.append(int(new_token_num))
                wall_time.append(total_time)

                accept_lengths_tree.extend(cur_accept_lengths_tree)
                total_draft_num += cur_draft_num

                conv.messages[-1][-1] = output
            
            choices.append({"index": i, "turns": turns, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                            "accept_lengths": cur_accept_lengths_tree,
                            "acceptance_rate": (sum(cur_accept_lengths_tree) - len(
                                cur_accept_lengths_tree)) / cur_draft_num})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
        
        tokens = sum(choices[0]["new_tokens"])
        times = sum(choices[0]["wall_time"])
        speeds.append(tokens / times)

        time_per_token.append(times / tokens)
        wall_times.append(times)

        # break
    mean_accepted_tokens = np.mean(accept_lengths_tree)
    mean_speeds = np.array(speeds).mean()
    mean_time_per_token = np.array(time_per_token).mean()
    mean_wall_time = np.array(wall_times).mean()

    with open(os.path.expanduser(answer_file), "a") as fout:
        ans_json = {
            "Mean accepted tokens": mean_accepted_tokens,
            "Token acceptance rate": (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num,
            "Tokens per second": mean_speeds,
            "Time per token": mean_time_per_token,
            "Mean wall time": mean_wall_time,
        }
        fout.write(json.dumps(ans_json) + "\n")
        print("#Mean accepted tokens:", mean_accepted_tokens)
        print("Token acceptance rate:", (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num)
        print('Tokens per second:', mean_speeds)
        print('Time per token:', mean_time_per_token*1000, 'ms')
        print('Mean wall time:', mean_wall_time, 's')


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
