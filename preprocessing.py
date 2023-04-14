import os
import ast
import json
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd


def get_files(path):
    return pd.read_json(os.path.join(path, "_DadaGP_all_filenames.json"))


def get_metadata(path):
    data = json.load(open(os.path.join(path, "_DadaGP_all_metadata.json")))
    return data


def filter(data, value, by="genre", col="genre_tokens"):
    return {k: v for k, v in data.items() if f"{by}:{value}" in v[col]}


def read_tokens(path):
    with open(path) as f:
        text = "".join(f.readlines())
        text = text.replace("\n", " ")

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/mnt/e/Data/DadaGP-v1.1')
    parser.add_argument('--output-path', type=str, default='/mnt/e/Data/DadaGP-processed')
    parser.add_argument('--genre', type=str, default='classic_rock')
    return parser.parse_args()

def main():
    args = parse_arguments()
    df_metadata = get_metadata(args.input_path)
    df_metadata = add_tokens(df_metadata, args.input_path)
    if args.genre is not None:
        df_metadata = filter(df_metadata, args.genre)
    train_data, val_data = prepare_train_val(df_metadata)

    print(len(train_data), len(val_data))

    os.makedirs(args.output_path, exist_ok=True)
    json.dump(train_data, open(os.path.join(args.output_path, 'train_data.json'), 'w'))
    json.dump(val_data, open(os.path.join(args.output_path, 'val_data.json'), 'w'))

if __name__ == "__main__":
    main()
