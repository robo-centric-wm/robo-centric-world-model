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

pretrain_tasks=(
  'window-open'
  'reach'
  'door-open'
  'button-press'
)

finetune_tasks=(
  'peg-insert-side'
  'coffee-push'
  'coffee-pull'
  'box-close'
  'stick-push'
  'hammer'
  'peg-unplug-side'
  'sweep-into'
)

fewshot_tasks=(
  'handle-pull'
  'drawer-open'
  'lever-pull'
  'peg-unplug-side'
)

# set tasks
tasks=("${finetune_tasks[@]}")

# set GPU
GPUs=(0 1 2 3 4 5 6 7)
gpus_num=${#GPUs[@]}

config="metaworld_vision"
config_mode="finetune"
domain="metaworld_"

run_mode="finetuning"

time=$(date +"%Y-%m-%d_%H:%M:%S")
folder_name=./logdir/nohup_out/finetuning/$time
mkdir -p $folder_name

for index in "${!tasks[@]}"; do
  task="${tasks[index]}"
  task=$domain$task
  gpu_idx=$((index % gpus_num))
  gpuid=${GPUs[gpu_idx]}
  export CUDA_VISIBLE_DEVICES=$gpuid
  nohup python3 rcwm/dreamer.py --reset_mode $reset_mode --task $task --configs $config $config_mode --logtime $time --seed $seed --run-mode $run_mode>$folder_name/$task.log 2>&1 &
  echo "PID: $!" >$folder_name/$task.pid
done
