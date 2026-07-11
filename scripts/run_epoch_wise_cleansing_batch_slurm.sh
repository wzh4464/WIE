#!/bin/bash
#SBATCH --job-name=epoch_wise_cleansing
#SBATCH --output=%x_%j_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=debug
#SBATCH --array=1-5

# Configuration
WORK_DIR="$PWD"
cd "$WORK_DIR" || exit 1

# keep_ratio list
KEEP_RATIOS=(98 96 94 92 90)

# Slurm array index starts from 1
IDX=$((SLURM_ARRAY_TASK_ID - 1))
KEEP_RATIO=${KEEP_RATIOS[$IDX]}
SAVE_DIR=$(printf "epoch_wise_decay_True_keep_ratio_%03d" $KEEP_RATIO)

echo "Running epoch-wise cleansing with keep_ratio=$KEEP_RATIO, save_dir=$SAVE_DIR"

pixi run bash scripts/epoch_wise_keep_ratio.sh mnist dnn "$SAVE_DIR" 0 0 True $KEEP_RATIO

echo "Job completed at: $(date)" 