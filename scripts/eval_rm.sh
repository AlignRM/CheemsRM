#!/bin/bash
set -x
export PYTHONPATH="${PYTHONPATH}:./"
cd ${WORKSPACE_PATH:-"./"} || exit

python cheems/eval_rm.py \
    --model_name="${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}" \
    --num_gpus_per_predictor="${NUM_GPUS_PER_PREDICTOR:-"1"}"
