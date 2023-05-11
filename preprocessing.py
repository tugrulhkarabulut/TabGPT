import os
import ast
import json
import argparse
from math import ceil

from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import GPT2TokenizerFast

import utils


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
    parser.add_argument("--input-path", type=str, default="/mnt/e/Data/DadaGP-v1.1")
    parser.add_argument(
        "--output-path", type=str, default="/mnt/e/Data/DadaGP-processed"
    )
    parser.add_argument("--genre", type=str, default="classic_rock")
    parser.add_argument("--extend-tokenizer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    df_metadata = get_metadata(args.input_path)
    if args.genre is not None:
        df_metadata = filter(df_metadata, args.genre)
    df_metadata = add_tokens(df_metadata, args.input_path)
    train_data, val_data = prepare_train_val(df_metadata)
    all_tokens = json.load(
        open(os.path.join(args.input_path, "_DadaGP_all_tokens.json"))
    )

    os.makedirs(args.output_path, exist_ok=True)
    json.dump(train_data, open(os.path.join(args.output_path, "train_data.json"), "w"))
    json.dump(val_data, open(os.path.join(args.output_path, "val_data.json"), "w"))

    train_dataset, test_dataset = prepare_dataset(args.output_path)
    if args.extend_tokenizer:
        tokenizer = utils.get_tokenizer(extend=all_tokens)
    else:
        tokenizer = utils.get_tokenizer()

    train_dataset = train_dataset.map(
        chunk_map,
        batched=True,
        num_proc=4,
        remove_columns=["validation_set", "tokens.txt", "artist_token", "genre_tokens"],
    )
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    train_dataset = train_dataset.map(group_texts, batched=True, num_proc=4)

    test_dataset = test_dataset.map(
        chunk_map,
        batched=True,
        num_proc=4,
        remove_columns=["validation_set", "tokens.txt", "artist_token", "genre_tokens"],
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    test_dataset = test_dataset.map(group_texts, batched=True, num_proc=4)

    train_dataset.save_to_disk(os.path.join(args.output_path, "train_dataset"))
    test_dataset.save_to_disk(os.path.join(args.output_path, "test_dataset"))


if __name__ == "__main__":
    main()
