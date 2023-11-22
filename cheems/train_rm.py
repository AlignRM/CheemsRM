import logging
import os
import pathlib
from datetime import timedelta

import deepspeed as ds
import torch
from accelerate import PartialState
from colorama import Fore, Style
from transformers import HfArgumentParser

from cheems.train.dataset import SampleDataset, collate_fn
from cheems.train.model import get_reward_model_class
from cheems.train.tokenizer import get_tokenizer
from cheems.train.trainer import Trainer, TrainingArguments, compute_metrics


def main():
    # logging.basicConfig(level=logging.DEBUG)
    parser = HfArgumentParser((TrainingArguments))  # type: ignore
    training_args, = parser.parse_args_into_dataclasses()
    ds.init_distributed(timeout=timedelta(minutes=int(os.getenv("DEEPSPEED_TIMEOUT", default=training_args.timeout))))

    state = PartialState()
    if state.is_main_process:
        logging.info(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Run with from {training_args=}")

    checkpoint_path = training_args.output_dir
    os.makedirs(checkpoint_path, exist_ok=True)

    if state.is_main_process:
        logging.info(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Initializing from {training_args.model_path=}\n")

    tokenizer = get_tokenizer(training_args.model_path)
    dataset_kwargs = dict(
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        truncation=training_args.truncation,
        do_strip=training_args.do_strip,
    )

    if training_args.do_train:
        # Make pairwise datasets for training
        if state.is_main_process:
            logging.info(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Prepare train_dataset from {training_args.train_data_path=}")
        train_dataset = SampleDataset(training_args.train_data_path, **dataset_kwargs)
    else:
        train_dataset = None

    if state.is_main_process:
        logging.info(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Prepare val_dataset from {training_args.test_data_path=}")

    if training_args.test_data_path == training_args.train_data_path and train_dataset is not None:
        eval_dataset = train_dataset
    elif isinstance(training_args.test_data_path, dict):
        eval_dataset = {name: SampleDataset(path, **dataset_kwargs) for name, path in training_args.test_data_path.items()}
    else:
        eval_dataset = SampleDataset(training_args.test_data_path, **dataset_kwargs)
    logging.info(f"{Fore.BLUE}[LOCAL RANK {state.local_process_index}]{Style.RESET_ALL} Datasets loaded.")

    RewardModelClass = get_reward_model_class(training_args.model_path)
    model = RewardModelClass.from_pretrained(
        training_args.model_path,
        use_flash_attention_2="gemma" not in training_args.model_path,
        torch_dtype=torch.bfloat16,  # 使用flash_attention必须要bf16
        reg_coef=training_args.reg_coef,
        seed=training_args.seed,
    )
    logging.info(f"{Fore.BLUE}[LOCAL RANK {state.local_process_index}]{Style.RESET_ALL} Model loaded.")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    resume_from_checkpoint = training_args.resume_if_possible and \
        len(list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))) > 0
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-last"))


if __name__ == "__main__":
    main()
