import os
from tqdm import tqdm
from config import HF_TOKEN

import random
import math
from datasets import load_dataset
from tokenizers import CharBPETokenizer

from huggingface_hub import create_repo
from huggingface_hub import upload_file
from datasets import DatasetDict

def evaluate_tokenizer_dataset(tokenizer, dataset, batch_size=10000):
    total_tokens = 0
    total_chars = 0
    total_sentences = 0

    num_batches = math.ceil(len(dataset) / batch_size)

    for batch in tqdm(
        dataset.iter(batch_size=batch_size),
        total=num_batches,
        desc="Tokenizing"
    ):
        texts = batch["text"]

        encodings = tokenizer.encode_batch(texts)

        total_tokens += sum(len(e.ids) for e in encodings)
        total_chars += sum(len(t) for t in texts)
        total_sentences += len(texts)

    avg_tokens = total_tokens / total_sentences
    avg_chars = total_chars / total_sentences
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0

    return {
        "avg_tokens_per_sentence": avg_tokens,
        "avg_chars_per_sentence": avg_chars,
        "compression_ratio": compression_ratio
    }


# Load the dataset
dataset = load_dataset("karpathy/tinystories-gpt4-clean",token=HF_TOKEN)
print(f"Tiny Stories dataset has {len(dataset['train'])} records")

train_test_dataset = dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=123)
train_val_dataset = train_test_dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=123)

train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']
test_dataset = train_test_dataset['test']

print(len(train_dataset), len(val_dataset), len(test_dataset))

print(f"Example Story :\n\n{train_dataset[random.randint(0, len(train_dataset))]['text']}")

def filter_short_text(example):
    return len(example["text"]) < 1500

train_dataset = train_dataset.filter(filter_short_text, num_proc=8)
val_dataset = val_dataset.filter(filter_short_text, num_proc=8)

train_dataset = train_dataset.select(range(100000))
val_dataset = val_dataset.select(range(20000))

print(train_dataset, val_dataset)

full_text = "\n".join(train_dataset['text'])
with open('full_text.txt', 'w') as f:
    f.write(full_text)

# Initialize a tokenizer
tokenizer = CharBPETokenizer()

tokenizer.train(
    files=["full_text.txt"],
    min_frequency=2,
    special_tokens=[
        "<|pad|>",
        "<|start_of_text|>",     # start of sequence
        "<|end_of_text|>",    # end of sequence
        "<unk>",   # unknown
    ]
)

tokenizer.save("char_bpe_tokenizer.json")

# Test Tokenizer
input_text = "Once upon a time there was a girl named Lily. <|end_of_text|> <|pad|> <|pad|> <|pad|>"
encoded_input = tokenizer.encode(input_text)
decoded_output = tokenizer.decode(encoded_input.ids)

print("Input Text:", input_text)
print("Encoded Input:", encoded_input.ids)

vocab = tokenizer.get_vocab()
vocab_size = tokenizer.get_vocab_size()

print(f"{vocab['<|end_of_text|>']=}, {vocab['<|pad|>']=}")

print(f"{evaluate_tokenizer_dataset(tokenizer, train_dataset)=}")
print(f"{evaluate_tokenizer_dataset(tokenizer, val_dataset)=}")

version = f"vb-{vocab_size}-cbpe-v1"

create_repo(
    f"aditya-6122/tinystories-tokenizer-{version}",
    repo_type="model",
    exist_ok=True,
    token=HF_TOKEN
)

upload_file(
    path_or_fileobj="char_bpe_tokenizer.json",   # your local file
    path_in_repo="bpe_tokenizer.json",      # name on HF
    repo_id=f"aditya-6122/tinystories-tokenizer-{version}",
    repo_type="model",                      # "dataset" or "model"
    token=HF_TOKEN
)

tokenized_train_dataset = train_dataset.map(lambda x: {"raw_ids": tokenizer.encode(x["text"]).ids}, num_proc=8)
tokenized_val_dataset = val_dataset.map(lambda x: {"raw_ids": tokenizer.encode(x["text"]).ids}, num_proc=8)

padded_train_dataset = tokenized_train_dataset.map(lambda x: {"ids": x['raw_ids'] + [vocab['<|end_of_text|>']]}, num_proc=8)
padded_validation_dataset = tokenized_val_dataset.map(lambda x: {"ids": x['raw_ids'] + [vocab['<|end_of_text|>']]}, num_proc=8)

final_train_dataset = padded_train_dataset.map(lambda x: {"input_ids": x['ids'][:-1], "output_ids": x['ids'][1:]}, num_proc=8)
final_validation_dataset = padded_validation_dataset.map(lambda x: {"input_ids": x['ids'][:-1], "output_ids": x['ids'][1:]}, num_proc=8)

token_lengths = list(map(len, final_train_dataset['ids']))
print(f"Max Tokens : {max(token_lengths)}, Min Tokens : {min(token_lengths)}")

dataset_dict = DatasetDict({"train": final_train_dataset, "validation": final_validation_dataset})
dataset_dict.push_to_hub(f"aditya-6122/tinystories-custom-dataset-{version}", token=HF_TOKEN)
