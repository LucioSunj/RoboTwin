#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}" # ensure GPU < 24G
export ROBOTWIN_PI05_HOME="/root/autodl-tmp"
export XDG_CACHE_HOME="/root/autodl-tmp/cache"
export UV_CACHE_DIR="/root/autodl-tmp/uv-cache"
export OPENPI_DATA_HOME="/root/autodl-tmp/cache/openpi"
export JAX_COMPILATION_CACHE_DIR="/root/autodl-tmp/cache/jax"
# Keep PyAV on FFmpeg 7.1 libs via LD_LIBRARY_PATH, but let video logging
# use /usr/bin/ffmpeg because the local FFmpeg 7.1 build lacks libx264.
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/root/autodl-tmp/ffmpeg-7.1-build/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
export ROBOTWIN_RT_DENOISER="${ROBOTWIN_RT_DENOISER:-none}"

policy_name=pi05
task_name=${1:?Usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [checkpoint_id]}
task_config=${2:?Usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [checkpoint_id]}
train_config_name=${3:?Usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [checkpoint_id]}
model_name=${4:?Usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [checkpoint_id]}
seed=${5:?Usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [checkpoint_id]}
gpu_id=${6:?Usage: bash eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id> [checkpoint_id]}
checkpoint_id=${7:-30000}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# source .venv/bin/activate
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --clear_cache_freq 1 \
    --seed ${seed} \
    --policy_name ${policy_name} 

# ROBOTWIN=/home/lsk/long-horizon-manipulation/Long-horizon-manipulation-decision-vla/submodules/RoboTwin
# CKPT=/home/lsk/long-horizon-manipulation/checkpoints/pi05/robotwin/stack_block_three/19000

# cd "$ROBOTWIN"

# mkdir -p \
# policy/pi05/checkpoints/pi05_aloha_stack_three_blocks_full/stack_three_blocks_full \
# policy/pi05/assets/pi05_aloha_stack_three_blocks_full \
# /home/lsk/long-horizon-manipulation/.cache/openpi \
# /home/lsk/long-horizon-manipulation/.cache/jax

# ln -sfnT "$CKPT" \
# policy/pi05/checkpoints/pi05_aloha_stack_three_blocks_full/stack_three_blocks_full/19000

# ln -sfnT "$CKPT/assets/stack_three_blocks_clean_repo" \
# policy/pi05/assets/pi05_aloha_stack_three_blocks_full/stack_three_blocks_clean_repo

# policy/pi05/.venv/bin/python script/update_embodiment_config_path.py

# CUDA_VISIBLE_DEVICES=0 \
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
# XDG_CACHE_HOME=/home/lsk/long-horizon-manipulation/.cache \
# OPENPI_DATA_HOME=/home/lsk/long-horizon-manipulation/.cache/openpi \
# JAX_COMPILATION_CACHE_DIR=/home/lsk/long-horizon-manipulation/.cache/jax \
# ROBOTWIN_PI05_ASSETS_BASE_DIR="$ROBOTWIN/policy/pi05/assets" \
# ROBOTWIN_PI05_CHECKPOINT_BASE_DIR="$ROBOTWIN/policy/pi05/checkpoints" \
# XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
# PYTHONWARNINGS=ignore::UserWarning \
# policy/pi05/.venv/bin/python script/eval_policy.py \
# --config policy/pi05/deploy_policy.yml \
# --overrides \
# --task_name stack_blocks_three \
# --task_config demo_clean \
# --train_config_name pi05_aloha_stack_three_blocks_full \
# --model_name stack_three_blocks_full \
# --ckpt_setting stack_three_blocks_full \
# --checkpoint_id 19000 \
# --clear_cache_freq 1 \
# --seed 0 \
# --policy_name pi05