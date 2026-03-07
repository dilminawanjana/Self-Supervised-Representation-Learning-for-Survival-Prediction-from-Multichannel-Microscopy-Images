#!/bin/bash
#SBATCH -A berzelius-2025-315
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -C thin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:20:00
#SBATCH -J h5_io_bench_gpu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# --- environment ---
module purge
module load Miniforge3/25.3.1-0
conda activate /proj/berzelius-2025-315/users/x_dilwi/conda_envs/h5env_clean_py310



export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DATA="/proj/berzelius-2025-315/users/x_dilwi/SSL_method/patches_542_h5_lzf/patches_542_lzf"

python -u benchmark_dataloader.py \
  --data_path "$DATA" \
  --batch_size 128 \
  --num_workers 4 \
  --n_batches 50 \
  --warmup_batches 2 \
  --move_to_gpu