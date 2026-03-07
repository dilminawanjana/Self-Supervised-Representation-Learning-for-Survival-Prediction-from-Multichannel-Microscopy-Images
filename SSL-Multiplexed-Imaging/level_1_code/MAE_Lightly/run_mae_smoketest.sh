#!/bin/bash
#SBATCH -A berzelius-2025-315
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=mae_smoke
#SBATCH --output=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/%x_%j.out
#SBATCH --error=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/%x_%j.err

set -euo pipefail

REPO=/proj/berzelius-2025-315/users/x_dilwi/AI-supported-Survival-Prediction-from-Multichannel-Microscopy-Images-of-Cancer-Tissue/SSL-Multiplexed-Imaging/level_1_code/MAE_Lightly
DATA=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/patches_542_h5_lzf/patches_542_lzf
OUT=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/results/mae_smoke_run1

mkdir -p /proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs
mkdir -p "$OUT"

module purge
module load Miniforge3/25.3.1-0
conda activate /proj/berzelius-2025-315/users/x_dilwi/conda_envs/h5env_clean_py310

# Keep it simple: disable W&B for the smoke test
export WANDB_MODE=disabled

cd "$REPO"

echo "Node: $(hostname)"
echo "GPUs visible:"
nvidia-smi -L || true

# Optional auto-resume (if a previous smoke test exists)
CKPT="$OUT/last.ckpt"
RESUME_ARGS=""
if [ -f "$CKPT" ]; then
  echo "Resuming from: $CKPT"
  RESUME_ARGS="--resume_ckpt $CKPT"
fi

srun python -u main.py \
  --n_nodes 1 \
  --n_devices 1 \
  --num_workers 2 \
  --seed 1 \
  --input_size 224 \
  --batch_size 128 \
  --max_epochs 2 \
  --lr 0.03 \
  --mask_ratio 0.75 \
  --data_path "$DATA" \
  --output_dir "$OUT" \
  --wandb_project_name "SSL_mIF" \
  --wandb_name "MAE_smoketest" \
  $RESUME_ARGS

echo "Done. Checkpoints in: $OUT"
ls -lah "$OUT" | head