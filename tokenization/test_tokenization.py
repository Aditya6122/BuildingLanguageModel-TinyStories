"""
Test script for tokenizer.

This script downloads a tokenizer from Hugging Face and allows testing encoding and decoding interactively.
"""

import argparse
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from config import HF_TOKEN

def main(tokenizer_repo):
    """Load tokenizer and test encoding/decoding interactively."""
    print(f"Downloading tokenizer from {tokenizer_repo}...")
    tokenizer_path = hf_hub_download(
        repo_id=tokenizer_repo,
        filename="tokenizer.json",
        repo_type="model",
        token=HF_TOKEN
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
    vocab = tokenizer.get_vocab()
    print("Special tokens example:", {k: v for k, v in vocab.items() if k.startswith('<')})

    while True:
        text = input("\nEnter text to encode/decode (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break

        print(f"\nOriginal text: {text}")

        # Encode
        encoded = tokenizer.encode(text)
        print(f"Encoded: {encoded.ids}")

        # Decode
        decoded = tokenizer.decode(encoded.ids).strip()
        print(f"Decoded: {decoded}")

        # Check if round-trip works
        if text == decoded:
            print("✓ Round-trip successful")
        else:
            print("✗ Round-trip failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test tokenizer encoding/decoding')
    parser.add_argument('--tokenizer-repo', type=str, default="aditya-6122/tinystories-tokenizer-vb-None-char_bpe-v1", help='Tokenizer repo on Hugging Face')

    args = parser.parse_args()
    main(args.tokenizer_repo)