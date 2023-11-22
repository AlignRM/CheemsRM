import itertools
import json
import random
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Union

import ray
import torch
from jsonargparse import CLI
from ray._private.utils import hex_to_binary
from ray._raylet import PlacementGroupID
from ray.data import DataContext
from ray.util.placement_group import PlacementGroup, placement_group_table, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from cheems.eval.gen_predictor import PREDICTOR_MAP as GENERATIVE_PREDICTOR_MAP
from cheems.eval.gen_predictor import GenerativePredictor
from cheems.eval.metrics import metric_fn
from cheems.eval.rm_predictor import PREDICTOR_MAP as RM_PREDICTOR_MAP
from cheems.eval.rm_predictor import DefaultRMPredictor, RMPredictor
from cheems.train.dataset import SampleDataset

PREDICTOR_MAP = {**RM_PREDICTOR_MAP, **GENERATIVE_PREDICTOR_MAP}

random.seed(0)


def create_rm_predictor(predictor_class, model_name):
    class Predictor(predictor_class):  # Create a class to do batch inference.
        def __init__(self):
            super().__init__(model_name)

        def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
            row['reward'] = self.reward_fn([
                {"role": "user", "content": row['inputs']},
                {"role": "assistant", "content": row['outputs']}
            ])
            return row
    return Predictor


def rm_evaluate(
    data_path: Union[str, List[str]],
    predictor_class,
    model_name: str,
    num_gpus: int,
    num_gpus_per_predictor: int
):
    all_dataset = []
    if isinstance(data_path, str):
        data_path = [data_path]  # type: ignore
    for path in data_path:
        dataset = SampleDataset(path).to_list()
        for item in dataset:
            item['origin_path'] = path
        all_dataset.extend(dataset)

    predictor_class = create_rm_predictor(predictor_class, model_name)
    num_instances = (num_gpus // num_gpus_per_predictor, 512)
    ds = ray.data.from_items(all_dataset)
    ds = ds.map(
        predictor_class,
        concurrency=num_instances,
        num_gpus=num_gpus_per_predictor
    )
    scored_dataset = list(ds.iter_rows())

    all_metrics = {}
    for path in data_path:
        dataset = [item for item in scored_dataset if item['origin_path'] == path]
        all_metrics[path] = metric_fn(
            rewards=torch.as_tensor([item['reward'] for item in dataset]),
            input_indices=torch.as_tensor([item['input_indices'] for item in dataset]),
            output_indices=torch.as_tensor([item['output_indices'] for item in dataset]),
            ranks=torch.as_tensor([item['ranks'] for item in dataset]),
            category_ids=torch.as_tensor([item['category_ids'] for item in dataset]),
        )
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    overall = sum(
        metrics['accuracy-rank'] + metrics['exact_match-rank']
        for metrics in all_metrics.values()
    ) / len(all_metrics) / 2
    print(f"Overall: {overall}")
    return all_metrics


def create_generative_predictor(predictor_class, model_name, tensor_parallel_size: int = 1):
    class Predictor(predictor_class):  # Create a class to do batch inference.
        def __init__(self):
            super().__init__(model_name, tensor_parallel_size)

        def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
            batch['judgment'] = self.compare_fn(
                question=batch['question'],
                answer_a=batch['answer_a'],
                answer_b=batch['answer_b'],
            )
            return batch
    return Predictor


def generative_evaluate(
    data_path: Union[str, List[str]],
    predictor_class,
    model_name: str,
    num_gpus: int = 8,
    tensor_parallel_size: int = 1,
    batch_size: int = 64,
):
    all_dataset = []
    if isinstance(data_path, str):
        data_path = [data_path]
    for path in data_path:
        dataset = SampleDataset.load_dataset(path)
        for item in dataset:
            responses = {res['id']: res['content'] for res in item['responses']}
            ranks = item['rank']
            for l_idx, r_idx in itertools.combinations(list(range(len(ranks))), 2):
                (id_a, rank_a), (id_b, rank_b) = ranks[l_idx], ranks[r_idx]
                if int(rank_a) != int(rank_b):
                    if int(rank_a) < int(rank_b):
                        answer_a, answer_b, preference = responses[id_a], responses[id_b], 0  # A
                    elif int(rank_a) > int(rank_b):
                        answer_a, answer_b, preference = responses[id_a], responses[id_b], 1  # B
                    if random.random() < 0.5:
                        answer_a, answer_b, preference = answer_b, answer_a, 1 - preference
                    all_dataset.append({
                        'question': item['input'],
                        'answer_a': answer_a,
                        'answer_b': answer_b,
                        'preference': "A" if preference == 0 else "B",
                        'category': item['category'][0],
                        'origin_path': path,
                    })

    if tensor_parallel_size == 0:
        # Do not use GPT, and use APIPredictor
        ray_map_kwarg = {
            "num_gpus": 0,
            "num_cpus": min(int(ray.cluster_resources()['CPU']) / batch_size, 1),
            "concurrency": batch_size,
        }
        override_num_blocks = len(all_dataset) // 2
    else:
        override_num_blocks = None
        num_instances = (num_gpus // tensor_parallel_size, 512)
        ctx = DataContext.get_current()
        ctx.wait_for_min_actors_s = 60 * 15 * tensor_parallel_size
        if tensor_parallel_size == 1:
            # For tensor_parallel_size == 1, we simply set num_gpus=1.
            ray_map_kwarg = {
                "num_gpus": 1,
                "concurrency": num_instances,
                "batch_size": batch_size,
            }
        elif tensor_parallel_size > 1:
            # Otherwise, we have to set num_gpus=0 and provide
            # a function that will create a placement group for
            # each instance.
            def scheduling_strategy_fn(tensor_parallel_size):
                # One bundle per tensor parallel worker
                pg = ray.util.placement_group(
                    [{"GPU": 1, "CPU": 1}] * tensor_parallel_size,
                    strategy="STRICT_PACK",
                )
                return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
                    pg, placement_group_capture_child_tasks=True
                ))

            ray_map_kwarg = {
                "num_gpus": 0,
                "concurrency": num_instances,
                "ray_remote_args_fn": partial(
                    scheduling_strategy_fn,
                    tensor_parallel_size=tensor_parallel_size
                ),
                "batch_size": batch_size,
            }

    predictor_class = create_generative_predictor(predictor_class, model_name, tensor_parallel_size)
    ds = ray.data.from_items(all_dataset, override_num_blocks=override_num_blocks)
    ds = ds.map_batches(predictor_class, **ray_map_kwarg)
    judged_dataset = list(ds.iter_rows())

    for placement_group_info in placement_group_table().values():
        # https://github.com/ray-project/ray/blob/ray-2.7.0/python/ray/util/placement_group.py#L291
        pg = PlacementGroup(
            PlacementGroupID(hex_to_binary(placement_group_info["placement_group_id"]))
        )
        remove_placement_group(pg)

    all_metrics = dict()
    for path in data_path:
        results = defaultdict(lambda: defaultdict(list))
        for item in judged_dataset:
            if item['origin_path'] == path:
                results[item['category']][item['question']].append(1 if item['judgment'] == item['preference'] else 0)
        metrics = dict()
        for category, questions in results.items():
            exact_match = []
            accuracy = []
            for judgments in questions.values():
                accuracy.extend(judgments)
                exact_match.append(sum(judgments) == len(judgments))
            metrics[category] = {
                'accuracy-rank': sum(accuracy) / len(accuracy),
                'exact_match-rank': sum(exact_match) / len(exact_match),
            }
        all_metrics[path] = {
            "accuracy-rank": sum([m['accuracy-rank'] for m in metrics.values()]) / len(metrics),
            "exact_match-rank": sum([m['exact_match-rank'] for m in metrics.values()]) / len(metrics),
        }
        for category, metric in metrics.items():
            all_metrics[path][f"accuracy-rank-{category}"] = metric['accuracy-rank']
            all_metrics[path][f"exact_match-rank-{category}"] = metric['exact_match-rank']
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    overall = sum(
        metrics['accuracy-rank'] + metrics['exact_match-rank']
        for metrics in all_metrics.values()
    ) / len(all_metrics) / 2
    print(f"Overall: {overall}")
    return all_metrics


def evaluate(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_gpus_per_predictor: int = 1,
    data_path: Union[str, List[str]] = [
        "data/cheems_bench/human.jsonl",
        "data/cheems_bench/open.jsonl",
    ],
):
    # Initialize a Ray cluster
    ray.init(ignore_reinit_error=True)
    num_gpus = int(ray.cluster_resources()['GPU'])
    predictor_class = PREDICTOR_MAP.get(model_name, DefaultRMPredictor)

    if issubclass(predictor_class, RMPredictor):
        return rm_evaluate(
            data_path=data_path,
            predictor_class=predictor_class,
            model_name=model_name,
            num_gpus=num_gpus,
            num_gpus_per_predictor=num_gpus_per_predictor
        )
    elif issubclass(predictor_class, GenerativePredictor):
        return generative_evaluate(
            data_path=data_path,
            predictor_class=predictor_class,
            model_name=model_name,
            num_gpus=num_gpus,
            tensor_parallel_size=num_gpus_per_predictor
        )
    else:
        raise NotImplementedError(f"predictor_class={predictor_class}")


if __name__ == "__main__":
    CLI(evaluate)
