#!/bin/bash

seed=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
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
tasks=("${pretrain_tasks[@]}")

# set GPUs
GPUs=(0 1 2 3 4 5 6 7)
gpus_num=${#GPUs[@]}

config="metaworld_vision"
domain="metaworld_"

run_mode="training"

time=$(date +"%Y-%m-%d_%H:%M:%S")
folder_name=./logdir/nohup_out/training/$time
mkdir -p $folder_name

for index in "${!tasks[@]}"; do
  task="${tasks[index]}"
  task=$domain$task
  gpu_idx=$((index % gpus_num))
  gpuid=${GPUs[gpu_idx]}
  export CUDA_VISIBLE_DEVICES=$gpuid
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qianlong/.mujoco/mujoco200/bin
  nohup python3 rcwm/dreamer.py --task $task --configs $config --logtime $time --seed $seed --run-mode $run_mode>$folder_name/$task.log 2>&1 &
  echo "PID: $!" >$folder_name/$task.pid
done
