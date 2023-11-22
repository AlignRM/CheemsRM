import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from cheems.train.dataset import SampleDataset
from cheems.train.model import build_pairs


def metric_fn(rewards, input_indices, output_indices, ranks=None, labels=None, category_ids=None):
    metrics = {
        "reward_mean": torch.mean(rewards).item(),
        "reward_std": torch.std(rewards).item(),
    }
    if category_ids is None:
        if ranks is not None:
            rank_metrics = get_rank_metrics(rewards, input_indices, output_indices, ranks)
            metrics.update(rank_metrics)  # type: ignore
        if labels is not None:
            expand_input_indices = input_indices.unsqueeze(-1).expand(*labels.shape[:-1])
            potential_labels = torch.stack([  # [micro_batch_size, num_pairs, 4]
                expand_input_indices,
                labels[..., 0],
                expand_input_indices,
                labels[..., 1],
            ], dim=-1).view(-1, 4)  # [num_potential_pairs, 4]
            potential_labels = {tuple(label) for label in potential_labels.tolist()}
            chosen_rewards, rejected_rewards = build_pairs(
                rewards=rewards,
                input_indices=input_indices,
                output_indices=output_indices,
                potential_labels=potential_labels,
            )
            chosen_rewards, rejected_rewards = torch.stack(chosen_rewards), torch.stack(rejected_rewards)
            metrics.update(get_pair_metrics(chosen_rewards, rejected_rewards))
    else:
        all_metrics = defaultdict(list)
        for c in set(category_ids.tolist()):
            category = SampleDataset.ID_CATEGORY_MAP[c]
            masks = category_ids.eq(c)
            rank_c = None if ranks is None else ranks[masks]
            labels_c = None if labels is None else labels[masks]
            category_metrics = metric_fn(
                rewards=rewards[masks],
                input_indices=input_indices[masks],
                output_indices=output_indices[masks],
                ranks=rank_c,
                labels=labels_c,
            )
            for metric_name, value in category_metrics.items():
                all_metrics[metric_name].append(value)
            metrics.update({f"{metric_name}-{category}": value for metric_name, value in category_metrics.items()})  # type: ignore
        metrics.update({metric_name: np.mean(value) for metric_name, value in all_metrics.items()})
    return metrics


def get_pair_metrics(chosen_rewards: Tensor, rejected_rewards: Tensor):
    reward_diverge = chosen_rewards - rejected_rewards
    return {
        "accuracy-labels": torch.mean(chosen_rewards.gt(rejected_rewards).float()).item(),
        "chosen_reward_mean-labels": torch.mean(chosen_rewards).item(),
        "chosen_reward_std-labels": torch.std(chosen_rewards).item(),
        "rejected_reward_mean-labels": torch.mean(rejected_rewards).item(),
        "rejected_reward_std-labels": torch.std(rejected_rewards).item(),
        "reward_diverge_mean-labels": torch.mean(reward_diverge).item(),
        "reward_diverge_std-labels": torch.std(reward_diverge).item(),
        "expectation_calibration_error-labels": compute_ece(chosen_rewards, rejected_rewards),
    }


def get_rank_metrics(rewards: Tensor, input_indices: Tensor, output_indices: Tensor, ranks: Tensor):
    all_data = defaultdict(dict)
    for reward, rank, input_id, output_id in zip(rewards, ranks, input_indices, output_indices):
        all_data[input_id.item()][output_id.item()] = (reward.item(), rank.item())

    metrics = defaultdict(list)
    chosen_rewards, rejected_rewards = [], []
    for data in all_data.values():
        rewards, ranks = zip(*data.values())
        exact_match = 1
        # lower the rank, the better the response
        for l_idx, r_idx in itertools.combinations(list(range(len(ranks))), 2):
            if ranks[l_idx] < ranks[r_idx]:
                chosen_rewards.append(rewards[l_idx])
                rejected_rewards.append(rewards[r_idx])
                if rewards[l_idx] < rewards[r_idx]:
                    exact_match = 0
            elif ranks[l_idx] > ranks[r_idx]:
                chosen_rewards.append(rewards[r_idx])
                rejected_rewards.append(rewards[l_idx])
                if rewards[l_idx] > rewards[r_idx]:
                    exact_match = 0
        metrics["exact_match"].append(exact_match)

    metrics = {f"{name}-rank": np.mean(metrics[name]) for name in metrics}
    metrics["accuracy-rank"] = torch.as_tensor(chosen_rewards).gt(torch.as_tensor(rejected_rewards)).float().mean().item()  # type: ignore
    return metrics


def calibration_curve(chosen_rewards, rejected_rewards, n_bins: int = 5, max_diff: Optional[float] = None):
    if not isinstance(chosen_rewards, Tensor):
        chosen_rewards = torch.stack(chosen_rewards)
    if not isinstance(rejected_rewards, Tensor):
        rejected_rewards = torch.stack(rejected_rewards)

    rewards = chosen_rewards - rejected_rewards
    diffs, label = torch.abs(rewards), rewards.ge(0).float()

    if max_diff is None:
        max_diff = diffs.max().item()

    acc, score_diffs, nums = [], [], []
    step = max_diff / n_bins
    for i in range(n_bins):
        mask = diffs.ge(i * step) & diffs.lt((i + 1) * step)
        if mask.sum().item() != 0:
            nums.append(torch.sum(mask).item())
            acc.append(torch.mean(label.masked_select(mask)).item())
            score_diffs.append(torch.mean(diffs.masked_select(mask)).item())
    return acc, score_diffs, nums, max_diff


def compute_ece(chosen_rewards, rejected_rewards, n_bins: int = 5, max_diff: float = None):
    acc, score_diffs, nums, max_diff = calibration_curve(chosen_rewards, rejected_rewards, n_bins, max_diff)
    acc, score_diffs, nums = list(map(lambda i: torch.as_tensor(i), [acc, score_diffs, nums]))
    return torch.sum(torch.abs(acc - torch.sigmoid(score_diffs)) * nums / torch.sum(nums)).item()
