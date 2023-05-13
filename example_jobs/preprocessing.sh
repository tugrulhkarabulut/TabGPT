#!/bin/sh
#SBATCH -p long
#SBATCH --time=1-00:00:00
#SBATCH -A tbag130
#SBATCH -J tabgpt-preprocess
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -o /truba/home/tkarabulut/htgaa_project/TabGPT/logs/out-%j.out
#SBATCH -e /truba/home/tkarabulut/htgaa_project/TabGPT/logs/err-%j.err

conda activate htgaa
python preprocessing.py --input-path ../DadaGP-v1.1 \
                        --output-path ../DadaGP-processed-classic_rock \
                        --genre classic_rock


