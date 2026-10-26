import torch
from torch import nn

from model.diaa.kv_cache import initialize_past_key_values
from model.diaa.diaa_tree import build_diaa_tree


class DIAA(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        tokenizer,
        token_LUT: torch.Tensor,
        prob_LUT: torch.Tensor,
        max_output_length: int,
        output_id_topk: int = 8,
        tree_version: str = "dynamic",
        tree_budget: int | None = None,
        min_thres: float | None = None,
        max_depth: int | None = None,
        dy_interval: int | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.token_LUT = token_LUT
        self.prob_LUT = prob_LUT
        self.max_output_length = max_output_length
        self.output_id_topk = output_id_topk
        self.tree_version = tree_version
        self.tree_budget = tree_budget
        self.min_thres = min_thres
        self.max_depth = max_depth
        self.dy_interval = dy_interval

        self.s_background = torch.cuda.Stream()

        self.config = getattr(base_model, "config", None)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def train(self, mode: bool = True):
        self.base_model.train(mode)
        return super().train(mode)

    def eval(self):
        self.base_model.eval()
        return super().eval()

    @torch.inference_mode()
    def diaa_forward(
        self,
        input_ids,
        max_new_tokens,
        temperature: float = 0.0,
        do_sample: bool = False,
        output_id_topk: int | None = None,
        tree_version: str | None = None,
        tree_budget: int | None = None,
        min_thres: float | None = None,
        max_depth: int | None = None,
        dy_interval: int | None = None,
    ):
        if output_id_topk is None:
            output_id_topk = self.output_id_topk
        if tree_version is None:
            tree_version = self.tree_version
        if tree_budget is None:
            tree_budget = self.tree_budget
        if min_thres is None:
            min_thres = self.min_thres
        if max_depth is None:
            max_depth = self.max_depth
        if dy_interval is None:
            dy_interval = self.dy_interval

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
        ) = initialize_past_key_values(self.base_model, self.max_output_length)

        self.base_model(input_ids=verify_input_ids, past_key_values=past_key_values)

        tree_template = build_diaa_tree(
            version=tree_version,
            token_LUT=self.token_LUT.cpu().numpy(),
            prob_LUT=self.prob_LUT.cpu().numpy(),
            input_ids=input_ids.squeeze(0).item(),
            topk=output_id_topk,
            tree_budget=tree_budget,
            min_thres=min_thres,
            max_depth=max_depth,
            print_tree=False,
        )

        def prepare_data(template, input_ids_len, pad_len, device):
            candi_attention_ids = torch.zeros(
                [input_ids_len, len(template) + 1, len(template) + 1],
                dtype=torch.long,
                device=device,
            )
            candi_positon_ids = torch.zeros(
                [input_ids_len, len(template) + 1],
                dtype=torch.long,
                device=device,
            )

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
                    father_index.append(template.index(template[candi_i][:-1]) + 1)
                candi_index.append(tree_index)

                candi_positon_ids[:, candi_i + 1] = tree_deep

                if tree_deep != candi_positon_ids[:, candi_i]:
                    deep_split.append(candi_i + 1)

                cur_path = [0]
                candi_attention_ids[:, candi_i + 1, candi_i + 1] = 1

                for candi_j in range(candi_i):
                    if template[candi_j] == template[candi_i][: len(template[candi_j])]:
                        candi_attention_ids[:, candi_i + 1, candi_j + 1] = 1
                        cur_path.append(candi_j + 1)
                cur_path.append(candi_i + 1)

                for sub_i in range(1, len(cur_path)):
                    if cur_path[:sub_i] in search_path:
                        search_path.remove(cur_path[:sub_i])
                search_path.append(cur_path)

            deep_split.append(len(template) + 1)

            processed_search_path = []
            for path in search_path:
                if len(path) > pad_len:
                    print(template)
                    path = path[:pad_len]
                padded = path + [-1] * (pad_len - len(path))
                processed_search_path.append(padded)

            search_path = torch.tensor(
                processed_search_path, dtype=torch.long, device=device
            )

            father_index = torch.tensor(
                father_index, dtype=torch.long, device=device
            )
            candi_index = torch.tensor(
                candi_index, dtype=torch.long, device=device
            )

            return (
                candi_attention_ids,
                candi_positon_ids,
                search_path,
                father_index,
                candi_index,
                deep_split,
            )

        def prepare_data_input_ids(cur_input_ids, father_index, candi_index, deep_split):
            input_ids_local = torch.zeros_like(
                father_index, dtype=torch.long, device=cur_input_ids.device
            )
            input_ids_local[0] = cur_input_ids
            for layer in range(len(deep_split) - 1):
                cur_father = input_ids_local[
                    father_index[deep_split[layer] : deep_split[layer + 1]]
                ]
                cur_father_index = (
                    cur_father * output_id_topk
                    + candi_index[deep_split[layer] : deep_split[layer + 1]]
                )
                input_ids_local[deep_split[layer] : deep_split[layer + 1]] = (
                    self.token_LUT.view(-1)[cur_father_index]
                )

            return input_ids_local.unsqueeze(0)

        device = input_ids.device
        eos_token_ids = set()
        if self.tokenizer.eos_token_id is not None:
            eos_token_ids.add(int(self.tokenizer.eos_token_id))
        gen_cfg = getattr(self.base_model, "generation_config", None)
        if gen_cfg is not None and gen_cfg.eos_token_id is not None:
            if isinstance(gen_cfg.eos_token_id, (list, tuple, set)):
                eos_token_ids.update(int(x) for x in gen_cfg.eos_token_id)
            else:
                eos_token_ids.add(int(gen_cfg.eos_token_id))
        eos_token_ids_tensor = (
            torch.tensor(sorted(eos_token_ids), dtype=torch.long, device=device)
            if eos_token_ids
            else None
        )

        (
            attention_mask,
            position_ids,
            search_path,
            father_index,
            candi_index,
            deep_split,
        ) = prepare_data(tree_template, input_ids.size(0), len(tree_template[-1]) + 1, device)

        for i in range(max_new_tokens):
            torch.cuda.current_stream().wait_stream(self.s_background)

            input_ids = prepare_data_input_ids(
                input_ids, father_index, candi_index, deep_split
            )

            seq_len = verify_input_ids.size(1)

            merge_attention_mask = torch.cat(
                [
                    torch.ones(
                        [input_ids.size(0), len(tree_template) + 1, seq_len],
                        dtype=torch.long,
                        device=device,
                    ),
                    attention_mask,
                ],
                dim=-1,
            )
            merge_positon_ids = position_ids + seq_len

            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=merge_attention_mask.unsqueeze(1),
                position_ids=merge_positon_ids,
                past_key_values=past_key_values,
            )

            model_res = torch.argmax(outputs.logits, dim=-1)

            all_input_path = input_ids[0][search_path]
            all_input_path[search_path == -1] = -100
            all_output_path = model_res[0][search_path]
            reward = (
                torch.cumprod(all_input_path[:, 1:].eq(all_output_path[:, :-1]), dim=-1)
                .sum(dim=-1)
            )

            best_reward = reward.max()
            accept_len = 1 + best_reward

            best_path_index = torch.argmax(reward, dim=-1).to(torch.long)
            index_path = search_path[best_path_index][:accept_len]
            best_path_input = torch.index_select(input_ids, index=index_path, dim=1)
            verify_input_ids = torch.cat([verify_input_ids, best_path_input], dim=-1)

            if (
                (eos_token_ids_tensor is not None)
                and (best_path_input.unsqueeze(-1) == eos_token_ids_tensor).any()
            ) or (
                verify_input_ids.size(1) - start_len > max_new_tokens
            ):
                break

            with torch.cuda.stream(self.s_background):
                tgt = past_key_values_data[..., seq_len + index_path, :]
                dst = past_key_values_data[
                    ..., seq_len : seq_len + tgt.shape[-2], :
                ]
                dst.copy_(tgt, non_blocking=True)

                current_length_data.fill_(seq_len + tgt.shape[-2])

                topk = torch.topk(outputs.logits, k=output_id_topk, dim=-1)
                token_update = topk.indices[0]
                prob_update = topk.values[0].softmax(-1).to(self.prob_LUT.dtype)

                self.token_LUT[input_ids.squeeze(0)] = token_update
                self.prob_LUT[input_ids.squeeze(0)] = prob_update

            input_ids = (
                model_res[:, search_path[best_path_index][accept_len - 1]]
                .unsqueeze(-1)
                .to(device)
            )

            if (
                eos_token_ids_tensor is not None
                and (input_ids.unsqueeze(-1) == eos_token_ids_tensor).any()
            ):
                start_len -= 1
                break

            if tree_version == "dynamic" and (i + 1) % dy_interval == 0:
                tree_template = build_diaa_tree(
                    version=tree_version,
                    token_LUT=self.token_LUT.cpu().numpy(),
                    prob_LUT=self.prob_LUT.cpu().numpy(),
                    input_ids=input_ids.squeeze(0).item(),
                    topk=output_id_topk,
                    tree_budget=tree_budget,
                    min_thres=min_thres,
                    max_depth=max_depth,
                    print_tree=False,
                )

                (
                    attention_mask,
                    position_ids,
                    search_path,
                    father_index,
                    candi_index,
                    deep_split,
                ) = prepare_data(
                    tree_template,
                    input_ids.size(0),
                    len(tree_template[-1]) + 1,
                    device,
                )

            accept_length_list.append(accept_len.item())

        return (
            verify_input_ids,
            verify_input_ids.size(1) - start_len,
            step,
            accept_length_list,
            verify_input_ids.size(1) - start_len,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        do_sample: bool = False,
        output_id_topk: int | None = None,
        tree_version: str | None = None,
        tree_budget: int | None = None,
        min_thres: float | None = None,
        max_depth: int | None = None,
        dy_interval: int | None = None,
        return_diaa_details: bool = False,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids is required for DIAA.generate")
        if max_new_tokens is None:
            max_new_tokens = kwargs.get("max_length", 0)
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens or max_length must be provided")

        outputs = self.diaa_forward(
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

        if return_diaa_details:
            return outputs
        return outputs[0]