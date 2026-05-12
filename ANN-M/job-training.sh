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
bash run_pipeline.sh
### 指定输入、输出文件，并运行uni-mol任务
