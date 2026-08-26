#!/bin/bash
#SBATCH -A berzelius-2026-112
#SBATCH -N 1
#SBATCH -C thin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -J kronos_embed
#SBATCH -o logs/kronos_embed_%j.out
#SBATCH -e logs/kronos_embed_%j.err

module --force purge
module load Miniforge3/25.3.1-0

conda activate kronos310

export HF_HOME=/proj/berzelius-2025-315/users/x_dilwi/.cache/huggingface

cd /proj/berzelius-2025-315/users/x_dilwi/AI-supported-Survival-Prediction-from-Multichannel-Microscopy-Images-of-Cancer-Tissue/KRONOS/KRONOS

mkdir -p logs

python extract_kronos_embeddings_orimeanstd.py

