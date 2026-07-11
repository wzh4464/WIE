#! /bin/bash

for keep_ratio in 2 5 10 15 20 25 30
do
    save_dir=$(printf "epoch_wise_cleansing_keep_ratio_%03d" $keep_ratio)
    bash scripts/epoch_wise_cleansing.sh mnist dnn "$save_dir" 0 0 $keep_ratio
done 
