#!/bin/bash
#SBATCH -A berzelius-2025-315
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -J mae_embed
#SBATCH -o mae_embed_%j.out
#SBATCH -e mae_embed_%j.err

# --- environment ---
module purge
module load Miniforge3/25.3.1-0
conda activate /proj/berzelius-2025-315/users/x_dilwi/conda_envs/h5env_clean_py310

cd /proj/berzelius-2025-315/users/x_dilwi/AI-supported-Survival-Prediction-from-Multichannel-Microscopy-Images-of-Cancer-Tissue/SSL-Multiplexed-Imaging/level_1_code/MAE_Lightly

python get_embeddings.py \
  --data_path /proj/berzelius-2025-315/users/x_dilwi/SSL_method/patches_542_h5_lzf/patches_542_lzf \
  --checkpoint_path /proj/berzelius-2025-315/users/x_dilwi/SSL_method/results/mae_local_run1_lr0.001_epochs500 \
  --save_dir /proj/berzelius-2025-315/users/x_dilwi/SSL_method/results/mae_local_run1_lr0.001_epochs500/selected_embeddings \
  --batch_size 128 \
  --num_workers 7 \
  --lr 0.001