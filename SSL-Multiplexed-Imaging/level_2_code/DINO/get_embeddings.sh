#!/bin/bash
#SBATCH -A berzelius-2025-315
#SBATCH -N 1
#SBATCH -C thin
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --job-name=dino_extract_epoch99_all
#SBATCH --output=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/DINO/%x_%A_%a.out
#SBATCH --error=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/DINO/%x_%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

# -------------------------
# Repo + env
# -------------------------
REPO=/proj/berzelius-2025-315/users/x_dilwi/AI-supported-Survival-Prediction-from-Multichannel-Microscopy-Images-of-Cancer-Tissue/SSL-Multiplexed-Imaging/level_2_code/DINO

module purge
module load Miniforge3/25.3.1-0
conda activate /proj/berzelius-2025-315/users/x_dilwi/conda_envs/h5env_clean_py310

export WANDB_MODE=disabled

cd "$REPO"

# -------------------------
# MAE inputs used to build the dataset for extraction
# -------------------------
MAE_EMB_DIR=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/results/mae_local_run1_lr0.001_epochs500/selected_embeddings
STEMS=("epochepoch=0199-v1" "epochepoch=0299-v1" "epochepoch=0399" "epochepoch=0499")
HUMAN_EPOCHS=("200" "300" "400" "500")

STEM=${STEMS[$SLURM_ARRAY_TASK_ID]}
EPOCH_LABEL=${HUMAN_EPOCHS[$SLURM_ARRAY_TASK_ID]}

PATH_EMB="${MAE_EMB_DIR}/embeddings_${STEM}.npy"
PATH_NAMES="${MAE_EMB_DIR}/names_${STEM}.npy"

# -------------------------
# DINO checkpoint paths (your 4 trained runs)
# -------------------------
RESULTS_ROOT=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/results
DINO_RUN_DIR="${RESULTS_ROOT}/DINO_from_MAE_epoch${EPOCH_LABEL}"
CKPT="${DINO_RUN_DIR}/epoch=99.ckpt"

# Save extracted embeddings here
SAVE_DIR="${DINO_RUN_DIR}/extracted_embeddings_epoch100"
mkdir -p "$SAVE_DIR"

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "DINO run dir: $DINO_RUN_DIR"
echo "Checkpoint: $CKPT"
echo "MAE embeddings: $PATH_EMB"
echo "MAE names:      $PATH_NAMES"
echo "Save dir:       $SAVE_DIR"

# -------------------------
# Checks
# -------------------------
if [ ! -f "$CKPT" ]; then
  echo "ERROR: Missing checkpoint: $CKPT"
  echo "Available ckpts in $DINO_RUN_DIR:"
  ls -1 "$DINO_RUN_DIR" | grep ckpt || true
  exit 1
fi

if [ ! -f "$PATH_EMB" ]; then
  echo "ERROR: Missing MAE embeddings: $PATH_EMB"
  exit 1
fi

if [ ! -f "$PATH_NAMES" ]; then
  echo "ERROR: Missing MAE names: $PATH_NAMES"
  exit 1
fi

# -------------------------
# Run extraction (DINO/get_embeddings.py)
# -------------------------
srun python -u get_embeddings.py \
  --num_workers 8 \
  --input_size 14 \
  --batch_size 256 \
  --path_embeddings "$PATH_EMB" \
  --path_names "$PATH_NAMES" \
  --checkpoint_path "$CKPT" \
  --save_dir "$SAVE_DIR"

echo "DONE: extracted embeddings for DINO_from_MAE_epoch${EPOCH_LABEL} (epoch=99)"