from collections import defaultdict
from typing import Optional

import deepspeed
import torch
from accelerate import PartialState
from accelerate.utils import TORCH_DISTRIBUTED_OPERATION_TYPES, gather_object, recursively_apply
from torch import Tensor, nn
from torch.distributed.nn import functional
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from cheems.train.tokenizer import get_tokenizer


def gather(tensor):
    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]

        # Can only gather contiguous tensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
            # a backend of `None` is always CPU
            # also gloo does not support `all_gather_into_tensor`,
            # which will result in a larger memory overhead for the op
            gathered_tensor = functional.all_gather(tensor)
            return torch.cat(gathered_tensor, dim=0)  # type: ignore
        else:
            return tensor

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


def build_pairs(rewards, input_indices, output_indices, potential_labels):
    inout_map = {
        (in_id.item(), out_id.item()): i
        for i, (in_id, out_id) in enumerate(zip(input_indices, output_indices))
    }
    input_reward_idx_map = defaultdict(list)
    chosen_rewards, rejected_rewards = [], []
    for label in potential_labels:
        chosen_input_id, chosen_output_id, rejected_input_id, rejected_output_id = label
        if (chosen_input_id, chosen_output_id) in inout_map and (rejected_input_id, rejected_output_id) in inout_map:
            chosen_idx = inout_map[chosen_input_id, chosen_output_id]
            rejected_idx = inout_map[rejected_input_id, rejected_output_id]
            chosen_rewards.append(rewards[chosen_idx])
            rejected_rewards.append(rewards[rejected_idx])
            input_reward_idx_map[chosen_input_id].append(len(chosen_rewards) - 1)
    assert len(chosen_rewards) > 0 and len(rejected_rewards) > 0, \
        f"[ERROR] No valid pairs found in the batch.\n{inout_map=}\n{potential_labels=}"
    return chosen_rewards, rejected_rewards


def get_reward_model_class(model_path):
    config = AutoConfig.from_pretrained(model_path)
    base_pretrained_class = AutoModelForCausalLM._model_mapping[type(config)]

    class RewardModel(base_pretrained_class):  # type: ignore
        def __init__(self, config, seed: int = 0, reg_coef: float = 0.):
            super().__init__(config)
            self.reg_coef = reg_coef
            self.v_head = nn.Linear(self.config.hidden_size, 1, bias=False)
            with deepspeed.zero.GatheredParameters([self.v_head.weight], modifier_rank=0):
                tensor = self.v_head.weight.data.clone()
                torch.manual_seed(seed)
                tensor = tensor.normal_(mean=0.0, std=1 / (self.config.hidden_size + 1))
                with torch.no_grad():
                    self.v_head.weight = nn.Parameter(tensor)

        @property
        def tokenizer(self):
            return get_tokenizer(self.config._name_or_path)

        def truncate_input_ids(self, input_ids: Tensor):
            max_token_id = self.model.embed_tokens.weight.shape[0]
            if input_ids.max().item() >= max_token_id:
                print(
                    f"Encounter input_ids contain token_id larger than embedding size {max_token_id}.\n"
                    f"The unexpected id includes {input_ids.max().item()}.\n"
                    f"The tokenizer and model may not be consistent, pls check.\n"
                    f"Unexpected ids are replaced by {self.tokenizer.pad_token_id=}"
                )
                input_ids = input_ids.masked_fill(input_ids.ge(max_token_id), self.tokenizer.pad_token_id)
            if input_ids.min().item() < 0:
                print(
                    f"Encounter input_ids contain token_id less than 0.\n"
                    f"The unexpected id includes {input_ids.min().item()}.\n"
                    f"The tokenizer and model may not be consistent, pls check.\n"
                    f"Unexpected ids are replaced by {self.tokenizer.pad_token_id=}"
                )
                input_ids = input_ids.masked_fill(input_ids.lt(0), self.tokenizer.pad_token_id)
            return input_ids

        def inference(self, input_ids: Tensor, attention_mask: Tensor):
            input_ids = self.truncate_input_ids(input_ids)
            hidden_states = self.model(input_ids, attention_mask=attention_mask)[0]
            end_ids = [mask.ne(0).nonzero()[-1].item() for mask in attention_mask]
            hidden_states = hidden_states[list(range(hidden_states.shape[0])), end_ids]
            return self.v_head(hidden_states).squeeze(-1).float()  # [micro_batch_size]

        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor = None,  # type: ignore
            input_indices: Optional[Tensor] = None,
            output_indices: Optional[Tensor] = None,
            category_ids: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            ranks: Optional[Tensor] = None,
        ):
            """

            :param input_ids:
            :param attention_mask:
            :param input_indices:
            :param output_indices:
            :param labels:[batch_size, num_pairs, 2]
                num_pairs是pad过之后的pair对个数，[正样本索引，负样本索引，margin]
            :return:
            """
            rewards = self.inference(input_ids, attention_mask)  # [micro_batch_size]

            loss = torch.as_tensor(0., device=rewards.device)
            gathered_rewards = gather(rewards)  # [global_batch_size]
            gathered_input_indices = gather_object(input_indices)  # (global_batch_size,):[1,1,1,2,2,2,3,3]
            gathered_output_indices = gather_object(output_indices)  # (global_batch_size,):[1,2,3,1,2,3,1,2]

            # labels [micro_batch_size, num_pairs, 2(chosen_ans_id, rejected_ans_id)]
            # input_indices [micro_batch_size]
            expand_input_indices = input_indices.unsqueeze(-1).expand(*labels.shape[:-1])  # type: ignore
            labels = torch.stack([
                expand_input_indices, labels[..., 0],  # type: ignore
                expand_input_indices, labels[..., 1],  # type: ignore
            ], dim=-1)  # [micro_batch_size, num_pairs, 4]
            gathered_labels = labels.tolist()
            gathered_labels = gather_object(gathered_labels)

            potential_labels = sum(gathered_labels, [])  # [num_potential_pairs, 4]
            potential_labels = {tuple(labels) for labels in potential_labels}
            chosen_rewards, rejected_rewards = build_pairs(  # type: ignore
                rewards=gathered_rewards,
                input_indices=gathered_input_indices,
                output_indices=gathered_output_indices,
                potential_labels=potential_labels,
            )
            chosen_rewards, rejected_rewards = torch.stack(chosen_rewards), torch.stack(rejected_rewards)
            loss += torch.mean(-F.logsigmoid(chosen_rewards - rejected_rewards))
            if self.reg_coef > 0:
                loss = loss + self.reg_coef * torch.mean(rewards ** 2)

            returned = {
                "rewards": rewards,
                "input_indices": input_indices,
                "output_indices": output_indices,
                "loss": loss,
            }
            returned["ranks"] = ranks if ranks is not None else torch.as_tensor([-1] * len(rewards), device=rewards.device)
            if category_ids is not None:
                returned["category_ids"] = category_ids
            return returned

    return RewardModel
