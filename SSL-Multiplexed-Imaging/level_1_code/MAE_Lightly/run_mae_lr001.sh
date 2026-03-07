#!/bin/bash
#SBATCH -A berzelius-2025-315
#SBATCH -N 1
#SBATCH -C thin
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=2
#SBATCH --gpus=2
#SBATCH --time=36:00:00
#SBATCH --job-name=mae_local_lr0.001_epochs500
#SBATCH --output=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/%x_%j.out
#SBATCH --error=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/%x_%j.err

set -euo pipefail

# --- directories ---
REPO=/proj/berzelius-2025-315/users/x_dilwi/AI-supported-Survival-Prediction-from-Multichannel-Microscopy-Images-of-Cancer-Tissue/SSL-Multiplexed-Imaging/level_1_code/MAE_Lightly
DATA=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/patches_542_h5_lzf/patches_542_lzf
OUT=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/results/mae_local_run1

mkdir -p "$(dirname /proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/dummy)"
mkdir -p "$OUT"

# --- environment ---
module purge
module load Miniforge3/25.3.1-0
conda activate /proj/berzelius-2025-315/users/x_dilwi/conda_envs/h5env_clean_py310

# Disable wandb unless you explicitly want it
export WANDB_API_KEY="$(cat ~/.wandb_api_key)"
export WANDB_MODE=online
export WANDB_DIR="$OUT"
export WANDB_CACHE_DIR="$OUT/wandb_cache"

cd "$REPO"

# --- run ---
# IMPORTANT: use srun so SLURM handles GPU binding correctly
srun python -u main.py \
  --n_nodes 1 \
  --n_devices 2 \
  --num_workers 8 \
  --seed 1 \
  --input_size 224 \
  --batch_size 128 \
  --max_epochs 500 \
  --lr 0.001 \
  --mask_ratio 0.75 \
  --data_path "$DATA" \
  --output_dir "$OUT" \
  --wandb_project_name "SSL_mIF" \
  --wandb_name "MAE_patches542_lr0.001_epochs500"