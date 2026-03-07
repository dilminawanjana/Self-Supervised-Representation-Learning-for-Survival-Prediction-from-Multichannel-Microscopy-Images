#!/bin/bash
#SBATCH -A berzelius-2025-315
#SBATCH -N 1
#SBATCH -C thin
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=dino_from_mae_200_300_400_500
#SBATCH --output=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/DINO/%x_%A_%a.out
#SBATCH --error=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/DINO/%x_%A_%a.err
#SBATCH --array=0-3

set -euo pipefail

# -------------------------
# Fixed paths
# -------------------------
REPO=/proj/berzelius-2025-315/users/x_dilwi/AI-supported-Survival-Prediction-from-Multichannel-Microscopy-Images-of-Cancer-Tissue/SSL-Multiplexed-Imaging/level_2_code/DINO

MAE_EMB_DIR=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/results/mae_local_run1_lr0.001_epochs500/selected_embeddings

RESULTS_ROOT=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/results
LOG_DIR=/proj/berzelius-2025-315/users/x_dilwi/SSL_method/logs/DINO

mkdir -p "$LOG_DIR"

# -------------------------
# Environment
# -------------------------
module purge
module load Miniforge3/25.3.1-0
conda activate /proj/berzelius-2025-315/users/x_dilwi/conda_envs/h5env_clean_py310

# Disable wandb unless you explicitly want it
export WANDB_API_KEY="$(cat ~/.wandb_api_key)"
export WANDB_MODE=online
export WANDB_DIR="$RESULTS_ROOT"
export WANDB_CACHE_DIR="$RESULTS_ROOT/wandb_cache"

cd "$REPO"

# -------------------------
# Choose which MAE epoch for this task
# -------------------------
# These correspond to MAE checkpoints:
# 200 -> epochepoch=0199-v1
# 300 -> epochepoch=0299-v1
# 400 -> epochepoch=0399
# 500 -> epochepoch=0499
STEMS=("epochepoch=0199-v1" "epochepoch=0299-v1" "epochepoch=0399" "epochepoch=0499")
HUMAN_EPOCHS=("200" "300" "400" "500")

STEM=${STEMS[$SLURM_ARRAY_TASK_ID]}
EPOCH_LABEL=${HUMAN_EPOCHS[$SLURM_ARRAY_TASK_ID]}

PATH_EMB="${MAE_EMB_DIR}/embeddings_${STEM}.npy"
PATH_NAMES="${MAE_EMB_DIR}/names_${STEM}.npy"

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Using MAE embeddings stem: $STEM (Epoch ${EPOCH_LABEL})"
echo "PATH_EMB: $PATH_EMB"
echo "PATH_NAMES: $PATH_NAMES"

if [ ! -f "$PATH_EMB" ]; then
  echo "ERROR: Missing $PATH_EMB"
  exit 1
fi
if [ ! -f "$PATH_NAMES" ]; then
  echo "ERROR: Missing $PATH_NAMES"
  exit 1
fi

# -------------------------
# Output directory for this DINO run
# -------------------------
OUTDIR="${RESULTS_ROOT}/DINO_from_MAE_epoch${EPOCH_LABEL}"
mkdir -p "$OUTDIR"

echo "DINO output_dir: $OUTDIR"

# -------------------------
# Run DINO training
# -------------------------
# NOTE: Your DINO/main.py uses:
#   global_crop_size=14, local_crop_size=6
# so keep input_size=14 consistent when extracting.
#
# Also: your main.py sets args.wandb_name = args.wandb_name + '_gradacc'
#
srun python -u main.py \
  --n_nodes 1 \
  --n_devices 1 \
  --num_workers 8 \
  --seed 1 \
  --batch_size 64 \
  --max_epochs 100 \
  --lr 6e-4 \
  --momentum 0.9 \
  --accumulate_grad_batches 4 \
  --output_dir "$OUTDIR" \
  --path_embeddings "$PATH_EMB" \
  --path_names "$PATH_NAMES" \
  --wandb_project_name "SSL_mIF" \
  --wandb_name "DINO_from_${STEM}_epoch100"