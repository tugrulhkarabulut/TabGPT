from transformers import GPT2TokenizerFast


def get_tokenizer(extend=None):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if extend:
        tokenizer.add_tokens(extend)
    return tokenizer


def read_tokens(path):
    with open(path) as f:
        text = "".join(f.readlines())
        text = text.replace("\n", " ")

    return text
