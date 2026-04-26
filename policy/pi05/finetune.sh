#!/usr/bin/env bash
set -euo pipefail

train_config_name=${1:?Usage: bash finetune.sh <train_config_name> <model_name> <gpu_use> [resume|overwrite]}
model_name=${2:?Usage: bash finetune.sh <train_config_name> <model_name> <gpu_use> [resume|overwrite]}
gpu_use=${3:?Usage: bash finetune.sh <train_config_name> <model_name> <gpu_use> [resume|overwrite]}
run_mode=${4:-resume}

case "$run_mode" in
    resume)
        run_flag="--resume"
        ;;
    overwrite)
        run_flag="--overwrite"
        ;;
    *)
        echo "Unknown run mode: $run_mode"
        echo "Usage: bash finetune.sh <train_config_name> <model_name> <gpu_use> [resume|overwrite]"
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES=$gpu_use
export ROBOTWIN_PI05_HOME="/root/autodl-tmp"
export XDG_CACHE_HOME="/root/autodl-tmp/cache"
export UV_CACHE_DIR="/root/autodl-tmp/uv-cache"
export OPENPI_DATA_HOME="/root/autodl-tmp/cache/openpi"
export JAX_COMPILATION_CACHE_DIR="/root/autodl-tmp/cache/jax"
export WANDB_MODE="offline"
export WANDB_DIR="/root/autodl-tmp/wandb"
export WANDB_CACHE_DIR="/root/autodl-tmp/wandb/cache"
export WANDB_CONFIG_DIR="/root/autodl-tmp/wandb/config"
export PATH="/usr/local/cuda-12.8/bin:/root/autodl-tmp/ffmpeg-7.1-build/bin:$PATH"
export LD_LIBRARY_PATH="/root/autodl-tmp/ffmpeg-7.1-build/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "run_mode=$run_mode"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}" \
    uv run --frozen scripts/train.py "$train_config_name" --exp-name="$model_name" "$run_flag"
