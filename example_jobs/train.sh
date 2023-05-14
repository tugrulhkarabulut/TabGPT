#!/bin/sh
#SBATCH -p akya-cuda
#SBATCH --time=1-00:00:00
#SBATCH -A tbag130
#SBATCH -J tabgpt-train
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH -o /truba/home/tkarabulut/htgaa_project/TabGPT/logs/out-train-%j.out
#SBATCH -e /truba/home/tkarabulut/htgaa_project/TabGPT/logs/out-train-%j.err

python train.py --config configs/gpt2_large_train.yml