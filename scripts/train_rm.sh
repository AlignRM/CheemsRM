#!/bin/bash
set -x
export PYTHONPATH="${PYTHONPATH}:./"
cd ${WORKSPACE_PATH:-"./"} || exit

N_GPUS=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}
torchrun --nproc_per_node="${N_GPUS:-"8"}" \
    --nnodes="${WORLD_SIZE:-"1"}" \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT:-"2333"}" \
    cheems/train_rm.py \
    --deepspeed="${CONFIG:-"configs/zero3.json"}" \
    --gradient_checkpointing="${GRADIENT_CHECKPOINTING:-"True"}" \
    --max_length="${MAX_LENGTH:-"2048"}" \
    --truncation="${TRUNCATION:-"False"}" \
    --do_shuffle="${SHUFFLE:-"true"}" \
    --model_path"=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}" \
    --reg_coef="${REG_COEF:-"0.1"}" \
    --gradient_accumulation_steps="${ACCUMULATION_STEPS:-"1"}" \
    --output_dir="${OUTPUT_DIR:-"models"}" \
    --train_data_path="${TRAIN_DATA_PATH:-"data/cheems_preference.jsonl"}" \
    --test_data_path="${TEST_DATA_PATH:-"[data/cheems_bench/open.jsonl,data/cheems_bench/human.jsonl]"}" \
    --num_train_epochs="${EPOCHS:-"1"}" \
    --evaluation_strategy="${EVAL_STRATEGY:-"steps"}" \
    --eval_steps="${EVAL_STEPS:-"100"}" \
    --save_strategy="${SAVE_STRATEGY:-"steps"}" \
    --save_steps="${SAVE_STEPS:-"100"}" \
    --save_total_limit="${SAVE_TOTAL_LIMIT:-"3"}" \
    --per_device_train_batch_size="${MICRO_BATCH_SIZE:-"1"}" \
    --per_device_eval_batch_size="${EVAL_MICRO_BATCH_SIZE:-"1"}" \
    --learning_rate="${LR:-"5e-6"}" \
    --warmup_ratio="${WARMUP_RATIO:-"0.1"}" \
    --lr_scheduler_type="${LR_SCHEDULER:-"cosine"}" \
    --seed="${SEED:-"0"}" \
    --resume_if_possible="${USE_RESUME:-"false"}"
