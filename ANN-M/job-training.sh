#!/bin/bash

#SBATCH --job-name=training

#SBATCH --nodes=1

#SBATCH --ntasks=10   

#SBATCH --cpus-per-task=1

#SBATCH --partition=gpu1

#SBATCH --gres=gpu:1

#SBATCH --time=168:00:00


source activate py39
### 加载环境
python compare-Y6-light-dark.py --epochs 200 --batch_sizes 16 32 64
