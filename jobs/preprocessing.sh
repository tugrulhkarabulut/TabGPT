#!/bin/bash
#SBATCH -p long
#SBATCH --time=1-00:00:00
#SBATCH -A tbag130
#SBATCH -J tabgpt-preprocess
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH -o /truba/home/tkarabulut/htgaa_project/out-%j.out
#SBATCH -e /truba/home/tkarabulut/htgaa_project/err-%j.err

conda init bash
conda activate htgaa
python preprocessing.py --input-path ../dadaGP-v1.1 \
                        --output-path ../DadaGP-processed-classic_rock \
                        --genre classic_rock