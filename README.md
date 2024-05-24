# Pre-Training Robo-Centric World Models For Efficient Visual Control
This is the official repository for the paper titled “Pre-Training Robo-Centric World Models For Efficient Visual Control". Please refer to the [project website](https://robo-centric-wm.github.io/) for details of the algorithm.


## Table of Contents
- [Prerequisites](#installation)
- [Installation](#installation)
- [Basic Usage](#basic-usage)

## Prerequisites
Please ensure that the [modified mujoco-py](https://github.com/robo-centric-wm/mujoco-py-mask) is installed. Then download and install [Meta-world](https://github.com/Farama-Foundation/Metaworld/releases/tag/v2.0.0) manually.
```bash
cd Metaworld
pip install -e .
```

## Installation
```bash
pip install -r requirements.txt
```

## Basic Usage
The _rcwm_ folder contains code for training and fine-tuning, and the _fewshot_ folder provides code about few-shot expert-guided policy learning.

We provide startup scripts in the _scripts_ folder. Before running the script, specify the tasks and available GPUs.


### Training from scratch

```bash
cd robo-centric-world-model
bash scripts/run_train.sh --seed 0
```

### Pre-training
Before running the script, specify _offline_traindir_ under the _pretrain_ parameter group in configs.yaml
```bash
bash scripts/run_pretrain.sh --reset-mode 2 --seed 0
```

### Fine-tuning
Before running the script, specify _pretrain_model_dir_ under the _finetune_ parameter group in configs.yaml

```bash
bash scripts/run_finetune.sh --reset-mode 2 --seed 0
```

### Few-shot policy learning
Before running the script, specify _pretrain_model_dir_ and _base_data_dir_ under the _fewshot_ parameter group in configs.yaml

```bash
bash scripts/run_fewshot.sh --reset-mode 2 --seed 0
```

## Acknowledgments
This code is based on the following works:
- Naoki's Dreamer-v3 PyTorch implementation: https://github.com/NM512/dreamerv3-torch
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
