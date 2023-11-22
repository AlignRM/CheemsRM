import hashlib
import os
import random
from collections import defaultdict
from typing import Iterator, List, Optional, Union

import jsonlines
import torch
from datasets import Dataset, tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

IGNORE_ID = -100


class SampleDataset(data.Dataset):
    CATEGORY_ID_MAP = {}
    ID_CATEGORY_MAP = {}

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer=None,
        num_proc: int = 8,
        add_generation_prompt: bool = False,
        do_strip: bool = True,
        max_length: int = 2048,
        padding: Union[str, bool] = False,
        truncation: bool = False,
    ):
        self.add_generation_prompt = add_generation_prompt
        self.do_strip = do_strip
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.num_proc = num_proc

        self.dataset = self.load_dataset(data_path)
        self.dataset = self.convert(self.dataset)
        if self.tokenizer is not None:
            self.dataset = self.tokenize(self.dataset)

    @staticmethod
    def load_dataset(data_path: Union[str, List[str]]):
        data_path = [data_path] if isinstance(data_path, str) else data_path
        dataset = defaultdict(lambda: {"responses": {}, "annotations": []})
        for path in data_path:
            assert path.endswith('jsonl'), f"Unexpected {path=}"
            with jsonlines.open(path) as reader:
                for item in tqdm(reader, desc=f"Loading {os.path.basename(path)}"):
                    input = item.get('input', item.get("prompt"))
                    if len(set(item.keys()) & {"input", "answer1", "answer2", "preference"}) == 4:
                        item['annotations'] = [{
                            "answer1": item.pop("answer1"),
                            "answer2": item.pop("answer2"),
                            "preference": item.pop("preference")
                        }]

                    for key, value in item.items():
                        if key == "annotations":
                            assert isinstance(value, list), f"Unexpected {value=}"
                            for anno in value:
                                if "answer1" in anno and "answer2" in anno and "preference" in anno:
                                    res1 = {
                                        "content": anno["answer1"],
                                        "id": hashlib.md5(anno["answer1"].encode('utf-8')).hexdigest()
                                    }
                                    res2 = {
                                        "content": anno["answer2"],
                                        "id": hashlib.md5(anno["answer2"].encode('utf-8')).hexdigest()
                                    }
                                    dataset[input]["responses"][res1['id']] = res1  # type: ignore
                                    dataset[input]["responses"][res2['id']] = res2  # type: ignore
                                    if str(anno["preference"]) == "1":
                                        dataset[input]["annotations"].append((res1["id"], res2["id"], '1'))
                                    elif str(anno["preference"]) == "2":
                                        dataset[input]["annotations"].append((res2["id"], res1["id"], '1'))
                                    else:
                                        raise ValueError(f"Unexpected {item['preference']=}")
                                else:
                                    dataset[input][key].append(tuple(anno))
                        elif key == "responses":
                            for res in value:
                                dataset[input][key][res['id']] = res
                        elif key == "rank":
                            assert key not in dataset[input], "Unable to fuse \"rank\" field"
                            dataset[input][key] = list(tuple(rank) for rank in value.items())
                        else:
                            dataset[input][key] = value

        common_keys = set.intersection(*(set(item.keys()) for item in dataset.values()))
        for input, item in dataset.items():
            item['responses'] = list(item['responses'].values())  # type: ignore
            dataset[input] = {key: value for key, value in item.items() if key in common_keys}
        return list(dataset.values())

    def convert(self, dataset):
        for item in dataset:
            if 'category' in item:
                if item['category'][0] not in self.CATEGORY_ID_MAP:
                    id, category = len(self.CATEGORY_ID_MAP), item['category'][0]
                    self.CATEGORY_ID_MAP[category] = id
                    self.ID_CATEGORY_MAP[id] = category
                item['category'] = self.CATEGORY_ID_MAP[item['category'][0]]

        def flatten(items, item_ids):
            data = defaultdict(list)
            for item_id, annotations in enumerate(items['annotations']):
                responses_map = {
                    res['id']: {"output_id": output_id, **res}
                    for output_id, res in enumerate(items['responses'][item_id])
                }
                labels = []  # 必需有一个"labels"，主要是为了hf trainer能够进去compute metrics流程
                for (res1_id, res2_id, preference) in annotations:
                    # [[],[chosen_output_id (itself),rejected_output_id]]
                    if preference == '1':
                        chosen_output_id = responses_map[res1_id]['output_id']
                        rejected_output_id = responses_map[res2_id]['output_id']
                        labels.append((chosen_output_id, rejected_output_id))
                responses = ["" for _ in range(len(responses_map))]
                for res in responses_map.values():
                    responses[res['output_id']] = res['content']

                ranks = None
                if "rank" in items:
                    ranks = [0 for _ in range(len(responses_map))]
                    for rank in items["rank"][item_id]:
                        ranks[responses_map[rank[0]]['output_id']] = int(rank[1])
                # lower the rank, the better the response

                input = items['input'][item_id]
                for output_id, output in enumerate(responses):
                    data["input_indices"].append(item_ids[item_id])
                    data["output_indices"].append(output_id)
                    data["labels"].append(labels)
                    data['inputs'].append(input)
                    data['outputs'].append(output)
                    if 'category' in items:
                        data['category_ids'].append(items['category'][item_id])

                    if ranks is not None:
                        data['ranks'].append(ranks[output_id])
                    if 'sys_prompt' in items:
                        data['sys_prompts'].append(items['sys_prompt'][item_id])
            return data

        dataset = Dataset.from_list(self.dataset)
        dataset = dataset.map(
            flatten,
            batched=True,
            with_indices=True,
            remove_columns=dataset.column_names,
            desc="Convert"
        )
        return dataset

    def tokenize(self, dataset):
        dataset = dataset.map(
            lambda item: build_sample(
                tokenizer=self.tokenizer,
                input_or_conv=item['inputs'],
                output=item['outputs'],
                sys_prompt=item.get('sys_prompt', None),
                add_generation_prompt=self.add_generation_prompt,
                do_strip=self.do_strip,
                tokenize=True,
                truncation=self.truncation,
                max_length=self.max_length,
                padding=self.padding,
            ),  # type: ignore
            desc="Tokenize",
            num_proc=self.num_proc,
        ).to_list()
        if not self.truncation and self.max_length is not None:
            input_map = defaultdict(dict)
            for item in tqdm(dataset, desc="Filter & Index"):
                if len(item['input_ids']) <= self.max_length:
                    input_map[item['input_indices']][item['output_indices']] = item
            for output_map in tqdm(input_map.values(), desc="Label Filter", total=len(input_map)):
                for output_id, item in output_map.items():
                    new_labels = [
                        label for label in item['labels']
                        if len(set(label[:2]) & set(output_map.keys())) == 2
                    ]  # 因为可能会做长度filter导致部分output_id无效，所以先只保留在labels中output还在的
                    item['labels'] = new_labels
                    if new_labels == 0:  # 如果没有有效的labels，就删除这个sample
                        output_map.pop(output_id)
            dataset = []
            for output_map in tqdm(input_map.values(), desc="Output Filter", total=len(input_map)):
                labels = next(iter(output_map.values()))['labels']
                cur_output_ids = set(sum(labels, []))  # 过滤完labels之后，可能存在某些output已经无法组成pair了
                for output_id in cur_output_ids:
                    dataset.append(output_map[output_id])

        return dataset

    def to_list(self):
        if not isinstance(self.dataset, list):
            return self.dataset.to_list()
        else:
            return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return self.dataset.__iter__()


class BatchSampler:
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: SampleDataset, batch_size: int, do_shuffle: bool = True, seed: int = 0) -> None:
        # batch_size = micro_batch * world_size means the collected batch in one time of forward
        self.data_source = data_source
        # 建立input_id到sample_id的映射，实际上是把同一个input的数据点合并到一起便于遍历
        self.input_map = defaultdict(dict)
        self.labels_map = {}
        for sample_id, item in enumerate(tqdm(self.data_source, desc="Index")):
            input_id = item['input_indices']  # type: ignore
            self.input_map[input_id][item['output_indices']] = sample_id  # type: ignore
            self.labels_map[input_id] = list(map(tuple, item['labels']))  # type: ignore

        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        out_sample_maps = list(self.input_map.items())
        if self.do_shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(out_sample_maps)
            # print(f"{[input_id for input_id,_ in out_sample_maps]}\n")
            # [1, 4, 3, 2, 0]

        # 贪心求解组batch逻辑
        batched_samples = set()
        for input_id, out_sample_map in out_sample_maps:
            output_left, labels = set(out_sample_map.keys()), set(self.labels_map[input_id])
            while len(labels) > 0 or len(output_left) > 0:
                output_selected = set()
                if len(labels) > 0:
                    for label in labels:  # 遍历所有的pair
                        tmp_select = output_selected | set(label[:2])
                        # 检查如果添加这个pair后的batch大小
                        if len(tmp_select) + len(batched_samples) <= self.batch_size:
                            # 如果batch大小没超，就可以合并它对应的sample
                            # 可能添加0个sample，意味着完全包含在现有的batch里
                            output_selected = tmp_select

                    # labels -= labels_selected  # 删掉所有已经选择的pair
                    # 如果pair中的sample已经被组成batch则删除
                    labels = {label for label in labels if len(set(label[:2]) & output_selected) == 0}
                    output_left -= output_selected
                    # 保证下个batch一定不包含这个batch里已经有的sample

                else:  # 如果labels为空，但是output_left不为空，这些output就是上一轮batch中没有被选中的
                    tmp_output_left = list(output_left)
                    output_selected = set(tmp_output_left[:self.batch_size])
                    output_left = set(tmp_output_left[self.batch_size:])

                batched_samples |= set(out_sample_map[output_id] for output_id in output_selected)
                if len(batched_samples) == self.batch_size - 1 and len(labels) > 0:
                    output_id_a, _ = labels.pop()
                    batched_samples.add(out_sample_map[output_id_a])
                    labels = {label for label in labels if output_id_a not in label[:2]}
                    output_left -= {output_id_a}

                if len(batched_samples) == self.batch_size:
                    yield from batched_samples
                    batched_samples = set()

        yield from batched_samples
        self.set_epoch(self.epoch + 1)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        # return len(list(self.__iter__()))
        return sum(len(out_sample_map) for out_sample_map in self.input_map.values())


def build_sample(
    tokenizer,
    input_or_conv: Union[str, List],
    output: Optional[str] = None,
    sys_prompt: Optional[str] = None,
    add_generation_prompt: bool = False,
    do_strip: bool = True,
    tokenize: bool = False,
    max_length: int = 1024,
    padding: Union[str, bool] = False,
    truncation: bool = False,
    tokenizer_kwargs: Optional[dict] = None,
):
    if isinstance(input_or_conv, str):
        chatml = [{"role": "user", "content": input_or_conv}]
    else:
        chatml = input_or_conv
    if output is not None:
        chatml.append({"role": "assistant", "content": output},)

    if sys_prompt is not None:
        chatml.insert(0, {"role": "system", "content": sys_prompt})
    if do_strip and tokenize:
        # NOTE: In some model's chat template, there are extra \n at the END
        # while for the vllm of OpenRLHF generation, it will not be included
        # Thus result in the gap between the RM the generation
        sample = tokenizer.apply_chat_template(
            chatml,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        tokenizer_kwargs = tokenizer_kwargs or {}
        return tokenizer(
            sample.strip(),
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **tokenizer_kwargs,
        )
    else:
        return tokenizer.apply_chat_template(
            chatml,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            tokenizer_kwargs=tokenizer_kwargs,
            return_dict=tokenize,
        )


def collate_fn(batch):
    batch_data = {}
    for key in batch[0].keys():
        tensors = [
            torch.as_tensor(i[key]) if i[key] != []
            else torch.as_tensor([[IGNORE_ID, IGNORE_ID]])  # empty labels
            for i in batch
        ]
        try:
            batch_data[key] = torch.stack(tensors, dim=0)
        except RuntimeError:
            padding_value = 0 if key in ['input_ids', 'attention_mask'] else IGNORE_ID
            batch_data[key] = pad_sequence(
                tensors,
                batch_first=True,
                padding_side="left",
                padding_value=padding_value
            )
    return batch_data
