import argparse
import ast
import json
import os
from math import ceil

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

import utils
from config import get_cfg_defaults


def get_files(path):
    return pd.read_json(os.path.join(path, "_DadaGP_all_filenames.json"))


def get_metadata(path):
    data = json.load(open(os.path.join(path, "_DadaGP_all_metadata.json")))
    return data


def filter(data, value, by="genre", col="genre_tokens"):
    return {k: v for k, v in data.items() if f"{by}:{value}" in v[col]}


def read_tokens(path):
    try:
        with open(path) as f:
            text = "".join(f.readlines())
            text = text.replace("\n", " ")
    except:
        print(path)

    return text


def add_tokens(data, path):
    data_new = data.copy()
    for key in tqdm(data):
        data_new[key]["text"] = read_tokens(os.path.join(path, key))
    return data_new


def prepare_train_val(data):
    train_data = []
    val_data = []

    for value in data.values():
        if value["validation_set"]:
            val_data.append(value)
        else:
            train_data.append(value)

    return train_data, val_data


def chunk_text(text, max_chunk_size=1000, split_by="new_measure"):
    sub_chunks = text.split(split_by)
    for i in range(1, len(sub_chunks)):
        sub_chunks[i] = split_by + sub_chunks[i]
    sub_chunks = [s.strip() for s in sub_chunks]

    chunks = [sub_chunks[0]]
    chunk_i = 0
    for i in range(1, len(sub_chunks)):
        merged_chunk = chunks[chunk_i] + " " + sub_chunks[i]
        if len(merged_chunk) <= max_chunk_size:
            chunks[chunk_i] = merged_chunk
        else:
            chunk_len = len(sub_chunks[i])
            chunk_num = ceil(chunk_len / max_chunk_size)
            if chunk_num > 1:
                chunk_num = 0
                chunk_index_start = 0
                chunk_index_end = 0
                for x in sub_chunks[i].split():
                    if chunk_index_end + len(x) - chunk_index_start > max_chunk_size:
                        chunks.append(sub_chunks[i][chunk_index_start:chunk_index_end])
                        chunk_num += 1
                        chunk_index_start = chunk_index_end
                    chunk_index_end += len(x)

                chunks.append(sub_chunks[i][chunk_index_end:])

                chunk_i += chunk_num
            else:
                chunks.append(sub_chunks[i])
                chunk_i += 1

    return chunks


def chunk_map(examples):
    chunked_texts = []
    for text in examples["text"]:
        chunked_texts += chunk_text(text)
    examples["text"] = chunked_texts
    return examples


# https://github.com/huggingface/notebooks/blob/main/examples/language_modeling_from_scratch.ipynb
def group_texts(examples, block_size=384):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def prepare_dataset(path):
    train_data = json.load(open(os.path.join(path, "train_data.json")))
    val_data = json.load(open(os.path.join(path, "val_data.json")))
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(val_data)
    return train_dataset, test_dataset


def tokenize_function(tokenizer, examples):
    examples = tokenizer(examples["text"])
    examples["labels"] = examples["input_ids"].copy()
    return examples


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help=".yml config path",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    cfg = get_cfg_defaults()

    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)

    train_dataset = Dataset.load_from_disk(cfg.DATA.TRAIN_DATASET)
    test_dataset = Dataset.load_from_disk(cfg.DATA.TEST_DATASET)
    print('Loaded train and test datasets')

    all_tokens = json.load(open(os.path.join(cfg.INPUT, "_DadaGP_all_tokens.json")))

    if cfg.DATA.EXTEND_TOKENIZER:
        tokenizer = utils.get_tokenizer(extend=all_tokens)
    else:
        tokenizer = utils.get_tokenizer()

    model = GPT2LMHeadModel.from_pretrained(cfg.MODEL, use_cache=False)

    training_args = TrainingArguments(
        output_dir=cfg.OUTPUT,
        learning_rate=cfg.SOLVER.LR,
        per_device_train_batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
        gradient_accumulation_steps=cfg.SOLVER.GRAD_ACC_STEPS,
        gradient_checkpointing=cfg.SOLVER.GRAD_CKPT,
        num_train_epochs=cfg.SOLVER.EPOCHS,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        fp16=cfg.SOLVER.FP16,
        evaluation_strategy="epoch",
        save_total_limit=2,
        save_strategy="epoch",
        use_cache=not cfg.SOLVER.GRAD_CKPT,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    if cfg.RESUME_FROM_CKPT:
        ckpt_path = cfg.CKPT_PATH
    else:
        ckpt_path = False

    trainer.train(resume_from_checkpoint=ckpt_path)


if __name__ == "__main__":
    main()
