#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-RoboTwin5090}"
WORKDIR="${WORKDIR:-/tmp/robotwin5090}"
WHEEL_DIR="$WORKDIR/wheels"
CUROBO_DIR="$ROOT_DIR/envs/curobo"
CONDA_BASE="$(conda info --base)"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
ASSET_ROOT="${ASSET_ROOT:-/root/autodl-tmp/RoboTwin-assets}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}"

TORCH_WHEEL="$WHEEL_DIR/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TORCHVISION_WHEEL="$WHEEL_DIR/torchvision-0.22.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TORCHAUDIO_WHEEL="$WHEEL_DIR/torchaudio-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
FLASH_WHEEL="$WORKDIR/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

TORCH_URL="https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TORCHVISION_URL="https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TORCHAUDIO_URL="https://download-r2.pytorch.org/whl/cu128/torchaudio-2.7.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
FLASH_URL="https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
BACKGROUND_TEXTURE_URL="https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/background_texture.zip"
OBJECTS_URL="https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/objects.zip"
EMBODIMENTS_URL="https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/embodiments.zip"

NO_PROXY_ENV=(
  env
  -u HTTP_PROXY
  -u HTTPS_PROXY
  -u ALL_PROXY
  -u http_proxy
  -u https_proxy
  -u all_proxy
)

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

run_in_env() {
  conda run --no-capture-output -n "$ENV_NAME" "$@"
}

run_in_env_no_proxy() {
  "${NO_PROXY_ENV[@]}" conda run --no-capture-output -n "$ENV_NAME" "$@"
}

download_no_proxy() {
  local url="$1"
  local out="$2"
  "${NO_PROXY_ENV[@]}" wget -c -O "$out" "$url"
}

download_default() {
  local url="$1"
  local out="$2"
  wget -c -O "$out" "$url"
}

extract_asset_zip() {
  local zip_path="$1"
  local asset_name="$2"
  if [ ! -d "$ASSET_ROOT/$asset_name" ]; then
    unzip -q -o "$zip_path" -d "$ASSET_ROOT"
    rm -rf "$ASSET_ROOT/__MACOSX"
  fi
  rm -f "$zip_path"
}

link_asset_dir() {
  local asset_name="$1"
  local src="$ASSET_ROOT/$asset_name"
  local dst="$ROOT_DIR/assets/$asset_name"

  if [ ! -d "$src" ]; then
    echo "Missing asset directory: $src" >&2
    exit 1
  fi

  if [ -L "$dst" ]; then
    rm -f "$dst"
  elif [ -e "$dst" ]; then
    mv "$dst" "${dst}.bak.$(date +%s)"
  fi

  ln -s "$src" "$dst"
}

for cmd in apt-get conda git python3 wget; do
  need_cmd "$cmd"
done

mkdir -p "$WORKDIR" "$WHEEL_DIR"

apt-get update
apt-get install -y ffmpeg libegl1-mesa-dev libgles2-mesa-dev

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10.14
fi

run_in_env python -m pip install --upgrade pip wheel setuptools==80.9.0

download_no_proxy "$TORCH_URL" "$TORCH_WHEEL"
download_no_proxy "$TORCHVISION_URL" "$TORCHVISION_WHEEL"
download_no_proxy "$TORCHAUDIO_URL" "$TORCHAUDIO_WHEEL"

run_in_env_no_proxy python -m pip install \
  filelock \
  'typing-extensions>=4.10.0' \
  'sympy>=1.13.3' \
  networkx \
  jinja2 \
  fsspec \
  triton==3.3.1 \
  nvidia-cuda-nvrtc-cu12==12.8.61 \
  nvidia-cuda-runtime-cu12==12.8.57 \
  nvidia-cuda-cupti-cu12==12.8.57 \
  nvidia-cudnn-cu12==9.7.1.26 \
  nvidia-cublas-cu12==12.8.3.14 \
  nvidia-cufft-cu12==11.3.3.41 \
  nvidia-curand-cu12==10.3.9.55 \
  nvidia-cusolver-cu12==11.7.2.55 \
  nvidia-cusparse-cu12==12.5.7.53 \
  nvidia-cusparselt-cu12==0.6.3 \
  nvidia-nccl-cu12==2.26.2 \
  nvidia-nvtx-cu12==12.8.55 \
  nvidia-nvjitlink-cu12==12.8.61 \
  nvidia-cufile-cu12==1.13.0.11

run_in_env_no_proxy python -m pip install --no-deps \
  "$TORCH_WHEEL" \
  "$TORCHVISION_WHEEL" \
  "$TORCHAUDIO_WHEEL"

run_in_env_no_proxy python -m pip install \
  numpy==1.26.4 \
  transforms3d==0.4.2 \
  sapien==3.0.0b1 \
  scipy==1.10.1 \
  mplib==0.2.1 \
  gymnasium==0.29.1 \
  trimesh==4.4.3 \
  open3d==0.18.0 \
  imageio==2.34.2 \
  pydantic \
  zarr \
  openai \
  huggingface_hub==0.25.0 \
  h5py \
  'pyglet<2' \
  wandb \
  moviepy \
  termcolor \
  av \
  matplotlib

run_in_env python -m pip install \
  ninja \
  fvcore \
  iopath

run_in_env python -m pip install \
  'git+https://github.com/facebookresearch/pytorch3d.git@stable' \
  --no-build-isolation

mkdir -p "$ENV_PREFIX/etc/conda/activate.d" \
         "$ENV_PREFIX/etc/conda/deactivate.d"

cat >"$ENV_PREFIX/etc/conda/activate.d/robotwin_env.sh" <<'EOF'
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
EOF

cat >"$ENV_PREFIX/etc/conda/deactivate.d/robotwin_env.sh" <<'EOF'
unset VK_DRIVER_FILES
unset VK_ICD_FILENAMES
EOF

export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

run_in_env_no_proxy python - <<'PY'
from pathlib import Path
import inspect
import mplib
import sapien

sapien_file = Path(inspect.getfile(sapien)).resolve().parent / "wrapper" / "urdf_loader.py"
mplib_file = Path(inspect.getfile(mplib)).resolve().parent / "planner.py"

sapien_text = sapien_file.read_text()
sapien_text = sapien_text.replace('with open(urdf_file, "r") as f:', 'with open(urdf_file, "r", encoding="utf-8") as f:')
sapien_text = sapien_text.replace('srdf_file = urdf_file[:-4] + "srdf"', 'srdf_file = urdf_file[:-4] + ".srdf"')
sapien_text = sapien_text.replace('with open(srdf_file, "r") as f:', 'with open(srdf_file, "r", encoding="utf-8") as f:')
sapien_file.write_text(sapien_text)

mplib_text = mplib_file.read_text()
mplib_text = mplib_text.replace(
    'if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:',
    'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:',
)
mplib_file.write_text(mplib_text)
PY

if [ ! -d "$CUROBO_DIR/.git" ]; then
  git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi

run_in_env_no_proxy python -m pip install -e "$CUROBO_DIR" --no-build-isolation

download_default "$FLASH_URL" "$FLASH_WHEEL"
run_in_env_no_proxy python -m pip install --no-deps "$FLASH_WHEEL"

run_in_env_no_proxy python -m pip install \
  einops==0.8.1 \
  transformers==4.47.0 \
  timm==1.0.16 \
  diffusers==0.34.0 \
  qwen-vl-utils==0.0.11 \
  accelerate==0.26.0 \
  sentencepiece

run_in_env_no_proxy python - <<'PY'
import os
import flash_attn
import mplib
import open3d as o3d
import pytorch3d
import sapien
import torch
from curobo.wrap.reacher.motion_gen import MotionGen

print("NVIDIA_DRIVER_CAPABILITIES", os.environ.get("NVIDIA_DRIVER_CAPABILITIES"))
print("VK_ICD_FILENAMES", os.environ.get("VK_ICD_FILENAMES"))
print("torch", torch.__version__)
print("flash_attn", flash_attn.__version__)
print("pytorch3d", pytorch3d.__version__)
print("sapien", sapien.__version__)
print("mplib", mplib.__version__)
print("open3d", o3d.__version__)
print("MotionGen", MotionGen)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(2, 3, device="cuda", dtype=torch.float16)
    y = torch.randn(2, 3, device="cuda", dtype=torch.float16)
    print("cuda_sum_ok", float((x + y).sum().item()))
scene = sapien.Scene()
print("sapien_scene_ok", scene is not None)
PY

mkdir -p "$ASSET_ROOT"

if [ "$DOWNLOAD_ASSETS" = "1" ]; then
  download_default "$EMBODIMENTS_URL" "$ASSET_ROOT/embodiments.zip"
  download_default "$OBJECTS_URL" "$ASSET_ROOT/objects.zip"
  download_default "$BACKGROUND_TEXTURE_URL" "$ASSET_ROOT/background_texture.zip"
fi

if [ -f "$ASSET_ROOT/embodiments.zip" ]; then
  extract_asset_zip "$ASSET_ROOT/embodiments.zip" "embodiments"
fi

if [ -f "$ASSET_ROOT/objects.zip" ]; then
  extract_asset_zip "$ASSET_ROOT/objects.zip" "objects"
fi

if [ -f "$ASSET_ROOT/background_texture.zip" ]; then
  extract_asset_zip "$ASSET_ROOT/background_texture.zip" "background_texture"
fi

if [ -d "$ASSET_ROOT/embodiments" ]; then
  link_asset_dir "embodiments"
fi

if [ -d "$ASSET_ROOT/objects" ]; then
  link_asset_dir "objects"
fi

if [ -d "$ASSET_ROOT/background_texture" ]; then
  link_asset_dir "background_texture"
fi

if [ -d "$ROOT_DIR/assets/embodiments" ]; then
  run_in_env python "$ROOT_DIR/script/update_embodiment_config_path.py"
fi

echo
echo "Environment ready."
echo "Activate with: conda activate $ENV_NAME"
echo
echo "Asset root: $ASSET_ROOT"
echo "Current zipped asset sizes:"
echo "  background_texture.zip : 10.97 GB"
echo "  objects.zip            : 3.74 GB"
echo "  embodiments.zip        : 0.22 GB"
echo
echo "To download and link assets automatically, run:"
echo "  DOWNLOAD_ASSETS=1 ASSET_ROOT=$ASSET_ROOT bash $ROOT_DIR/script/setup_rtx5090.sh"
