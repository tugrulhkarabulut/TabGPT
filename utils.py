import os
import json

from datasets import load_dataset, Dataset

def prepare_dataset(path):
    train_data = json.load(open(os.path.join(path, "train_data.json")))
    val_data = json.load(open(os.path.join(path, "val_data.json")))
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(val_data)
    return train_dataset, test_dataset