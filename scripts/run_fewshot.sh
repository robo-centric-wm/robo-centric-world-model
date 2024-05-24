#!/bin/bash

reset_mode=2
seed=0

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reset-mode) reset_mode=$2; shift ;;
        --seed) seed=$2; shift ;;
        *) echo "unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

fewshot_tasks=(
  'handle-pull'
  'drawer-open'
  'lever-pull'
  'peg-unplug-side'
)

# set tasks
tasks=("${fewshot_tasks[@]}")

GPUs=(0 1 2 3 4 5 6 7)
gpus_num=${#GPUs[@]}

config="metaworld_vision"
config_mode="fewshot"
domain="metaworld_"

time=$(date +"%Y-%m-%d_%H:%M:%S")
folder_name=./logdir/nohup_out/fewshot/$time
mkdir -p $folder_name

for index in "${!tasks[@]}"; do
  task="${tasks[index]}"
  task=$domain$task
  gpu_idx=$((index % gpus_num))
  gpuid=${GPUs[gpu_idx]}
  export CUDA_VISIBLE_DEVICES=$gpuid
  nohup python3 fewshot/dreamer.py --reset_mode $reset_mode --task $task --configs $config $config_mode --logtime $time --seed $seed >$folder_name/$task.log 2>&1 &
  echo "PID: $!" >$folder_name/$task.pid
done