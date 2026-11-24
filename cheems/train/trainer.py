import ast
import json
import math
import os
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import List, Optional, Union

import torch
import transformers
from torch.utils.data import Dataset, SequentialSampler
from transformers import IntervalStrategy, SchedulerType, TrainerCallback
from transformers.trainer import TRAINER_STATE_NAME, logger
from transformers.trainer_callback import ExportableState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy, has_length, rotate_checkpoints

from cheems.eval.metrics import metric_fn
from cheems.train.dataset import BatchSampler

STORE_ACC_NAME = "eval_acc.json"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    reg_coef: float = -1
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    model_path: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": (
                "模型路径"
            )
        },
    )
    train_data_path: str = field(
        default="data/cheems_preference.jsonl",
        metadata={
            "help": (
                "训练数据路径，可以是一个list"
            )
        },
    )
    test_data_path: str = field(
        default="data/cheems_bench/open.jsonl",
        metadata={
            "help": (
                "测试数据路径，可以是一个list"
            )
        },
    )
    truncation: bool = field(
        default=False,
        metadata={
            "help": (
                "如果为True，则使用左truncate，如果为False，超长样本会被丢弃"
            )
        },
    )
    max_length: int = field(
        default=2048,
        metadata={
            "help": (
                "输入样本tokenize后的最大长度，指prompt+response的总长度，对于超长样本会被左truncate"
            )
        },
    )
    do_strip: bool = field(
        default=True,
        metadata={"help": "build_sample的时候是否要do_strip"},
    )
    do_shuffle: bool = field(default=True)
    timeout: int = field(default=60)
    logging_first_step: bool = field(default=True, metadata={"help": "Log the first global_step"})
    output_dir: str = field(
        default="models/rm_checkpoint",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    learning_rate: float = field(default=5e-6, metadata={"help": "The initial learning rate for AdamW."})
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default='constant',
        metadata={"help": "The scheduler type to use."},
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    eval_steps: Optional[float] = field(
        default=100,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    num_train_epochs: float = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    logging_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_strategy: Union[SaveStrategy, str] = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={
            "aliases": ["--evaluation_strategy", "--evaluation-strategy"],
            "help": "The evaluation strategy to use.",
        },
    )
    warmup_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Fraction of training steps used for warmup when warmup_steps is zero."},
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["tensorboard"],
        metadata={"help": "The list of integrations to report the results and logs to."}
    )
    resume_if_possible: bool = field(
        default=True,
        metadata={
            "help": "If True, resume checkpoints if possible."
        },
    )

    @classmethod
    def parse(cls, string):
        try:
            return json.loads(string)
        except JSONDecodeError:
            try:
                return ast.literal_eval(string)
            except BaseException:
                if string.startswith("[") and string.endswith("]"):
                    string = string[1:-1]
                    return [cls.parse(s.strip()) for s in string.split(',')]
                else:
                    return string

    def __post_init__(self):
        super(TrainingArguments, self).__post_init__()
        if self.warmup_ratio is not None and not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        self.train_data_path = self.parse(self.train_data_path)
        assert type(self.train_data_path) in [list, str], "train_data_path should be list or str"

        self.test_data_path = self.parse(self.test_data_path)
        if isinstance(self.test_data_path, dict):
            assert all(type(path) in [str, list] for path in self.test_data_path.values())
        elif isinstance(self.test_data_path, list):
            assert all(isinstance(path, str) for path in self.test_data_path)
        else:
            assert isinstance(self.test_data_path, str)

    def get_warmup_steps(self, num_training_steps: int):
        if self.warmup_ratio is not None and self.warmup_steps == 0:
            return math.ceil(num_training_steps * self.warmup_ratio)
        return super().get_warmup_steps(num_training_steps)


class EvaluateAtEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        control.should_save = True


class Trainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(EvaluateAtEpochCallback())
        self._checkpoint_metrics = None

    def _get_train_sampler(self, train_dataset=None) -> Optional[BatchSampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None
        return BatchSampler(
            data_source=train_dataset,
            batch_size=self.args.world_size * self.args.per_device_train_batch_size,
            do_shuffle=self.args.do_shuffle,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(eval_dataset)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        self._checkpoint_metrics = None
        super()._maybe_log_save_evaluate(*args, **kwargs)

    def _determine_best_metric(self, metrics, trial):
        self._checkpoint_metrics = metrics
        if not self.control.should_save and self.args.save_strategy != SaveStrategy.BEST:
            return False
        is_best = super()._determine_best_metric(metrics, trial)
        if is_best:
            self.state.best_global_step = self.state.global_step
        return is_best

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.warning(
                f"Checkpoint destination directory {output_dir} already exists and is non-empty. "
                "Saving will proceed but saved results may be invalid."
            )
            staging_output_dir = output_dir
        else:
            staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
        self.save_model(staging_output_dir, _internal_call=True)

        if self.state.best_global_step:
            best_dir = os.path.join(run_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}")
            if best_dir == output_dir or os.path.isdir(best_dir):
                self.state.best_model_checkpoint = best_dir

        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(staging_output_dir)
            self._save_scaler(staging_output_dir)
            self._save_rng_state(staging_output_dir)

        # Save the Trainer state
        if self.args.should_save:
            for callback in self.callback_handler.callbacks + [self.control]:
                if isinstance(callback, ExportableState):
                    name = type(callback).__name__
                    state = self.state.stateful_callbacks.get(name)
                    if isinstance(state, list):
                        state.append(callback.state())
                    else:
                        self.state.stateful_callbacks[name] = callback.state()
            self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))
            if self.is_world_process_zero():
                output_file_path = os.path.join(staging_output_dir, STORE_ACC_NAME)
                with open(output_file_path, 'w') as f:
                    json.dump(self._checkpoint_metrics, f, indent=4, ensure_ascii=False)

        if self.args.push_to_hub:
            self._push_from_checkpoint(staging_output_dir)

        # Place checkpoint in final location after all saving is finished.
        # First wait for everyone to finish writing
        self.args.distributed_state.wait_for_everyone()

        # Then go through the rewriting process, only renaming and rotating from main process(es)
        if self.is_local_process_zero() if self.args.save_on_each_node else self.is_world_process_zero():
            if staging_output_dir != output_dir:
                if os.path.exists(staging_output_dir):
                    os.rename(staging_output_dir, output_dir)

                    # Ensure rename completed in cases where os.rename is not atomic
                    # And can only happen on non-windows based systems
                    if os.name != "nt":
                        fd = os.open(output_dir, os.O_RDONLY)
                        os.fsync(fd)
                        os.close(fd)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                rotate_checkpoints(
                    output_dir=run_dir,
                    save_total_limit=self.args.save_total_limit,
                    best_model_checkpoint=self.state.best_model_checkpoint,
                )

        self.args.distributed_state.wait_for_everyone()


def compute_metrics(eval_preds):
    kwargs = {
        "rewards": torch.as_tensor(eval_preds.predictions[0]),
        "input_indices": torch.as_tensor(eval_preds.predictions[1]),
        "output_indices": torch.as_tensor(eval_preds.predictions[2]),
        "labels": torch.as_tensor(eval_preds.label_ids),
    }
    if -1 not in eval_preds.predictions[3]:
        kwargs["ranks"] = torch.as_tensor(eval_preds.predictions[3])
    if len(eval_preds.predictions) > 4:
        kwargs["category_ids"] = torch.as_tensor(eval_preds.predictions[4])
    return metric_fn(**kwargs)
