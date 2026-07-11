#!/bin/bash
#SBATCH --job-name=epochwise_cleansing_grid
#SBATCH --output=%x_%j_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=debug
#SBATCH --array=1-48

WORK_DIR="$PWD"
cd "$WORK_DIR" || exit 1

# Parameter lists
RELABELS=(0 5 10 15 20 30)
KEEP_RATIOS=(100 98 95 90 85 80 75 70)

# Generate all combinations
COMBINATIONS=()
for relabel in "${RELABELS[@]}"; do
  for keep_ratio in "${KEEP_RATIOS[@]}"; do
    COMBINATIONS+=("$relabel $keep_ratio")
  done
done

# Slurm array index starts from 1
IDX=$((SLURM_ARRAY_TASK_ID - 1))
PARAMS=(${COMBINATIONS[$IDX]})
RELABEL=${PARAMS[0]}
KEEP_RATIO=${PARAMS[1]}

SAVE_DIR=$(printf "epochwise_relabel_%03d_keep_ratio_%03d" $RELABEL $KEEP_RATIO)

echo "Running: relabel=$RELABEL, keep_ratio=$KEEP_RATIO, save_dir=$SAVE_DIR"

/backup/codes/sgd-influence/.pixi/envs/default/bin/python -m scripts.epoch_wise_keep_ratio \
  --target mnist \
  --model dnn \
  --save_dir "result/decay_True/$SAVE_DIR" \
  --seed 0 \
  --decay True \
  --relabel $RELABEL \
  --keep_ratio $KEEP_RATIO

echo "Job completed at: $(date)"