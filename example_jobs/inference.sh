#!/bin/sh
#SBATCH -p akya-cuda
#SBATCH --time=1-00:00:00
#SBATCH -A tbag130
#SBATCH -J tabgpt-inference
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH -o /truba/home/tkarabulut/htgaa_project/TabGPT/logs/out-inf-%j.out
#SBATCH -e /truba/home/tkarabulut/htgaa_project/TabGPT/logs/err-inf-%j.err

python inference.py --config configs/gpt2_train.yml \
                    --input-path /truba/home/tkarabulut/htgaa_project/DadaGP-example-prompts/here_comes_the_sun.txt \
                    --output-path /truba/home/tkarabulut/htgaa_project/DadaGP-example-outputs/ \
                    --output-file here_comes_the_sun.gp5 \
                    --encdec-path ./dadaGP/ \
                    --n-warm-up 128 \
                    --max-length 1024 \
                    --overlap 128 \
                    --instruments clean0
cd ./dadaGP/
python dadagp.py decode /truba/home/tkarabulut/htgaa_project/DadaGP-example-outputs/input.txt \
                       /truba/home/tkarabulut/htgaa_project/DadaGP-example-outputs/here_comes_the_sun.gp5
cd ..

python inference.py --config configs/gpt2_train.yml \
                    --input-path /truba/home/tkarabulut/htgaa_project/DadaGP-example-prompts/something.txt \
                    --output-path /truba/home/tkarabulut/htgaa_project/DadaGP-example-outputs/ \
                    --output-file something.gp5 \
                    --encdec-path ./dadaGP/ \
                    --n-warm-up 128 \
                    --max-length 1024 \
                    --overlap 128 \
                    --instruments clean0
cd ./dadaGP/
python dadagp.py decode /truba/home/tkarabulut/htgaa_project/DadaGP-example-outputs/input.txt \
                       /truba/home/tkarabulut/htgaa_project/DadaGP-example-outputs/something.gp5
cd ..

