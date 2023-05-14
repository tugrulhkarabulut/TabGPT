import argparse
import json
import os
import re
import subprocess
import sys
from math import ceil

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Pipeline,
    TextGenerationPipeline,
    pipeline,
)
from peft import PeftModel, PeftConfig

import utils
from config import get_cfg_defaults

INSTRUMENTS = [
    "distorted0",
    "distorted1",
    "distorted2",
    "clean0",
    "clean1",
    "bass",
    "leads",
    "pads",
    "drums",
]


def is_in_instruments(note, instruments):
    for instr in instruments:
        if instr in note:
            return True
    return False


def get_bad_words(tokenizer, vocab, instruments=INSTRUMENTS):
    tok_vocab = tokenizer.vocab.keys()
    bad_words = list(set(tok_vocab).difference(vocab))
    unwanted_instr = set(INSTRUMENTS).difference(instruments)
    bad_notes = [s for s in vocab if is_in_instruments(s, unwanted_instr)]
    bad_words += bad_notes
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    return bad_words_ids


def postprocess(text, all_tokens, append_end=True, sep="/n"):
    text = re.sub("new_measure", " new_measure ", text)
    text = text.strip()
    tokens = text.split()
    filtered_tokens = [t for t in tokens if t in all_tokens]
    if append_end and filtered_tokens[-1] != "end":
        filtered_tokens.append("end")
    processed_text = f"{sep}".join(filtered_tokens)
    return processed_text


def generate_piece(generator, warm_up_tabs, max_length, overlap, all_tokens):
    current_input = warm_up_tabs
    generated_tabs = generator(current_input)[0]["generated_text"]
    generated_tabs = postprocess(generated_tabs, all_tokens, append_end=False, sep=" ")
    while len(generated_tabs.split()) < max_length:
        print(len(generated_tabs.split()))
        current_input = " ".join(generated_tabs.split()[-overlap:])
        current_gen = generator(current_input)[0]["generated_text"]
        current_gen = " ".join(current_gen.split()[overlap:])
        current_gen = postprocess(current_gen, all_tokens, append_end=False, sep=" ")
        generated_tabs += current_gen

    generated_tabs = postprocess(generated_tabs, all_tokens)
    return generated_tabs


def generate_gp(
    generated_text,
    encdec_path="./dadaGP/",
    output_path=".output/",
    output_file="output.gp5",
):
    txt_path = os.path.join(output_path, "input.txt")
    gp_path = os.path.join(output_path, output_file)
    with open(txt_path, "w") as f:
        f.write(generated_text)
    subprocess.run(
        [f"cd {encdec_path} && {sys.executable} dadagp.py decode {txt_path} {gp_path}"]
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help=".yml config path",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/mnt/e/Data/DadaGP-v1.1/B/Beatles (The)/Beatles (The) - Here Comes The Sun (3).gp4.tokens.txt",
    )
    parser.add_argument("--output-path", type=str, default="./output/")
    parser.add_argument("--output-file", type=str, default="output.gp5")
    parser.add_argument("--encdec-path", type=str, default="./dadaGP/")
    parser.add_argument("--n-warm-up", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--instruments", nargs="+", default=INSTRUMENTS)
    return parser.parse_args()


def main():
    args = parse_arguments()
    cfg = get_cfg_defaults()

    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)

    all_tokens = json.load(open(os.path.join(cfg.INPUT, "_DadaGP_all_tokens.json")))

    input_piece = utils.read_tokens(args.input_path)
    warm_up = " ".join(input_piece.split()[: args.n_warm_up])

    if cfg.DATA.EXTEND_TOKENIZER:
        tokenizer = utils.get_tokenizer(extend=all_tokens)
    else:
        tokenizer = utils.get_tokenizer()


    if cfg.USE_PEFT:
        config = PeftConfig.from_pretrained(os.path.join(cfg.CKPT_PATH, "adapter_model"))
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, os.path.join(cfg.CKPT_PATH, "adapter_model"))
    else:
        model = GPT2LMHeadModel.from_pretrained(cfg.CKPT_PATH)

    tab_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        device=torch.device(0),
        bad_words_ids=get_bad_words(
            tokenizer,
            all_tokens,
            instruments=args.instruments,
        ),
    )

    generated_text = generate_piece(
        tab_generator,
        warm_up,
        max_length=args.max_length,
        overlap=args.overlap,
        all_tokens=all_tokens,
    )

    generate_gp(generated_text, args.encdec_path, args.output_path, args.output_file)


if __name__ == "__main__":
    main()
