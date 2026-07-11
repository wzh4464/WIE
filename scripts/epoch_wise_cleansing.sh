#!/bin/bash
set -e

# This script is used for epoch-wise cleansing experiments. The main process includes:
# 1. Train the model (seed can be specified)
# 2. Calculate the influence score for each epoch
# 3. Perform data cleansing experiments based on influence scores

# If seed is not specified, default to 0
if [ -z "$5" ]; then
    seed=0
else
    seed=$5
fi

# Step 1: Train the model without leave-one-out, save the model to the specified directory
python -m experiment.train --target "$1" --model "$2" --no-loo --save_dir "$3" --relabel "$4" --seed "$seed" 

# Step 2: Calculate the influence score for all epochs (tim_all_epochs), results are saved in the specified directory
python -m experiment.infl --target "$1" --model "$2" --type tim_all_epochs --save_dir "$3" --relabel "$4" --seed "$seed"

# Step 3: Perform cleansing experiments based on influence scores
# The specific method is: for each epoch, read the influence score of that epoch,
# keep only the training samples with the highest proportion of scores (adjustable, default 90%) (remove the 10% with the lowest scores),
# retrain the model with the filtered data, evaluate performance on the validation/test set,
# and finally save the experimental results for each epoch.
if [ -z "$6" ]; then
    keep_ratio=0.9
else
    keep_ratio=$6
fi
python -m experiment.exp_influence_cleansing --target "$1" --model "$2" --save_dir "$3" --relabel "$4" --seed "$seed" --keep_ratio "$keep_ratio" 

