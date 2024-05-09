#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# export WANDB_DISABLED=true
export WANDB_MODE=offline

# Avoid timeout error 
export WANDB__SERVICE_WAIT=300

# TODO: Change it to your local path
export MAIN_DIR="/home/ubuntu/CrossLingualAlignment"

cd ${MAIN_DIR}

TASK_NAME="afp_en-zh"

# TRAIN_FILE="${MAIN_DIR}/data/AFP.en_zh.X0.5.R3.0.json"
TRAIN_FILE="${MAIN_DIR}/data/AFP.en_zh.train.json"

# TODO: Modify the config josn file (path of dataset, number of demos, languages)
EVAL_CONFIG_PATH="${MAIN_DIR}/data/config/pawsx_en-zh_seed0.json"


MODEL_NAME="xglm-564m"
# MODEL_NAME="xglm-7.5b"
# MODEL_NAME="bloom-560m"

# MODEL_PATH="/LocalPath2PretrainModels/${MODEL_NAME}"
MODEL_PATH="facebook/${MODEL_NAME}"

SEED=0

# PoolingType-LayerIndex (The index of embedding layer is layer 0)
POOLER="last-1"
# POOLER="avg-1"

CL_TEMP=0.05

# MAX_TRAIN="--num_train_epochs 1 "
# MAX_TRAIN="--num_train_epochs 3 "
MAX_TRAIN="--max_steps 5000 "
# MAX_TRAIN="--max_steps 10000 "

DEVICE_BATCH_SIZE=2
# DEVICE_BATCH_SIZE=8

GRADIENT_ACC=1
# GRADIENT_ACC=2
# GRADIENT_ACC=4

# EVAL_STEP=125
# EVAL_STEP=250
EVAL_STEP=500
# EVAL_STEP=1000

# SAVE_STEP=100
# SAVE_STEP=500
# SAVE_STEP=1000
SAVE_STEP=2500
# SAVE_STEP=5000

EVAL_BATCH_SIZE=1
# EVAL_BATCH_SIZE=2
# EVAL_BATCH_SIZE=4

# Learning rate
LR=1e-5

# MAX_SEQ_LENGTH=128
# MAX_SEQ_LENGTH=256
MAX_SEQ_LENGTH=512
# MAX_SEQ_LENGTH=1024

# CLM="--do_clm --clm_weight 1"
CLM="--do_clm --clm_weight 1.5"
# CLM="--do_clm --clm_weight 2"

# METRIC_BEST=avg_all
METRIC_BEST=avg_zero

OUT_DIR="${MAIN_DIR}/log/${MODEL_NAME}/${TASK_NAME}/seed${SEED}"
OUT_FILE="${MAIN_DIR}/log/${MODEL_NAME}/${TASK_NAME}/seed${SEED}.log"

MAX_SAVE_NUM=1
# MAX_SAVE_NUM=20

cd ${MAIN_DIR}

CACHE_DIR="${MAIN_DIR}/data/cache"

mkdir -p ${OUT_DIR}
mkdir -p ${CACHE_DIR}

python src/train.py \
    --model_name_or_path ${MODEL_PATH} \
    --pooler_type ${POOLER} \
    --train_file ${TRAIN_FILE} \
    --output_dir ${OUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED} \
    --temp ${CL_TEMP} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --per_device_train_batch_size ${DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --learning_rate ${LR} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --clm_block_size ${MAX_SEQ_LENGTH} \
    --evaluation_strategy steps \
    --eval_steps ${EVAL_STEP} \
    --save_steps ${SAVE_STEP} \
    --save_total_limit ${MAX_SAVE_NUM} \
    --overwrite_output_dir \
    --eval_config_file ${EVAL_CONFIG_PATH} \
    --metric_for_best_model ${METRIC_BEST} \
    --load_best_model_at_end \
    --do_train \
    ${MAX_TRAIN} \
    ${CLM} \
    ${DP} \
    --tf32 True \
    --bf16 \
    --do_eval >>${OUT_FILE} 2>&1
