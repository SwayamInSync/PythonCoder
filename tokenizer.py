"""
Training GPT2 tokenizer on Python data for efficient tokenization
"""

import torch
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from datasets import load_dataset
from tqdm.auto import tqdm

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)
iter_dataset = iter(dataset)

byte_to_unicode_map = bytes_to_unicode()
unicode_to_byte_map = dict((v, k) for k, v in byte_to_unicode_map.items())
base_vocab = list(unicode_to_byte_map.keys())

length = 200000


def batch_iterator(batch_size=32):
    for _ in tqdm(range(0, length, batch_size)):
        yield [next(iter_dataset)['content'] for _ in range(batch_size)]


code_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(),
                                                   vocab_size=32768,
                                                   initial_alphabet=base_vocab)
code_tokenizer.add_tokens("<pad>")
code_tokenizer.pad_token = "<pad>"

code_tokenizer.push_to_hub("rootacess/FlashCoder")
