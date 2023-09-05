# TabGPT 

A project about generating guitar tabs using large language models. Mostly experimented on fine-tuning GPT-2.

[DadaGP](https://arxiv.org/abs/2107.14653) dataset is used for both training and generation. [Their encoder/decoder code](https://github.com/dada-bots/dadaGP) is used to convert guitar pro files to text and vice versa.

## How to Run

Python: 3.9.13

Prepare a config file in yaml format just like in the example_configs folder. Clone the dadaGP repo in the same folder as the project. Install the packages in the requirements file.

### Preprocessing

```bash
    python preprocessing.py --input-path ../DadaGP-v1.1 \
                            --output-path ../DadaGP-processed-classic_rock \
                            --genre classic_rock
```

### Training

```bash
    python train.py --config configs/gpt2_large_train.yml
```

### Inference

Prepare an input txt file that contains initial tokens. Or copy the initial tokens of your favorite song from the dataset. The rest is running the following commands:

```bash
    python inference.py --config configs/gpt2_train.yml \
                    --input-path /path/to/here_comes_the_sun.txt \
                    --output-path /path/to/output/folder/ \
                    --output-file here_comes_the_sun.gp5 \
                    --n-warm-up 128 \
                    --max-length 1024 \
                    --overlap 128 \
                    --instruments clean0
    cd ./dadaGP/
    python dadagp.py decode \
                        /path/to/output/folder/input.txt \
                        /path/to/output/folder/here_comes_the_sun.gp5
```

### Viewing the Tabs

Open the .gp5 file in Guitar Pro or TuxGuitar as a free alternative.


## Ideas

 - Using LLMs with larger context sizes and training custom tokenizers seem promising to improve generation results.
 - Token type embeddings

