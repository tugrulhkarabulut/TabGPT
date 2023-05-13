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
python inference.py --config configs/gpt2_train.yml \
                    --input-path /truba/home/tkarabulut/htgaa_project/DadaGP-example-prompts/here_comes_the_sun.txt \
                    --output-path ./output/ \
                    --output-file here_comes_the_sun.gp5 \
                    --encdec-path ./dadaGP/ \
                    --model-path /truba/home/tkarabulut/htgaa_project/output_test/checkpoint-2439 \
                    --n-warm-up 128 \
                    --max-length 1024 \
                    --overlap 128 \
                    --instruments clean0

python inference.py --config configs/gpt2_train.yml \
                    --input-path /truba/home/tkarabulut/htgaa_project/DadaGP-example-prompts/something.txt \
                    --output-path ./output/ \
                    --output-file something.gp5 \
                    --encdec-path ./dadaGP/ \
                    --model-path /truba/home/tkarabulut/htgaa_project/output_test/checkpoint-2439 \
                    --n-warm-up 128 \
                    --max-length 1024 \
                    --overlap 128 \
                    --instruments clean0


