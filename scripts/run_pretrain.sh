#!/bin/bash
# set GPU
gpuid=7

config="metaworld_vision"
config_mode="pretrain"

run_mode="pretraining"

time=$(date +"%Y-%m-%d_%H:%M:%S")
folder_name=./logdir/nohup_out/pretraining/$time
mkdir -p $folder_name

export CUDA_VISIBLE_DEVICES=$gpuid
nohup python3 rcwm/dreamer.py --logtime $time --configs $config $config_mode --run-mode $run_mode >$folder_name/pretrain.log 2>&1 &
echo "PID: $!" >$folder_name/pretrain.pid
