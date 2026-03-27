"""
Tokenizer inference script.

Downloads a tokenizer from Hugging Face and provides testing functionality.
"""

import logging
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from config import HF_TOKEN

logger = logging.getLogger(__name__)


def load_tokenizer_from_hub(tokenizer_repo, file_name='bpe_tokenizer.json', repo_type='model'):
    """
    Load a tokenizer from Hugging Face Hub.

    Args:
        tokenizer_repo (str): Repository name (e.g., 'user/repo-name').
        file_name (str): Filename of the tokenizer file.
        repo_type (str): Type of repository.

    Returns:
        Tokenizer: Loaded tokenizer.
    """
    logger.info(f"Downloading tokenizer from {tokenizer_repo}")
    file_path = hf_hub_download(
        repo_id=tokenizer_repo,
        filename=file_name,
        repo_type=repo_type,
        token=HF_TOKEN
    )
    tokenizer = Tokenizer.from_file(file_path)
    logger.info("Tokenizer loaded successfully")
    return tokenizer


def test_tokenizer(tokenizer, test_text=None):
    """
    Test the tokenizer with sample text.

    Args:
        tokenizer: The tokenizer to test.
        test_text (str, optional): Text to tokenize. If None, uses default.
    """
    if test_text is None:
        test_text = "Once upon a time there was a little girl named Lily."

    logger.info(f"Testing tokenizer with text: {test_text}")

    # Encode
    encoded = tokenizer.encode(test_text)
    logger.info(f"Encoded IDs: {encoded.ids}")
    logger.info(f"Tokens: {encoded.tokens}")

    # Decode
    decoded = tokenizer.decode(encoded.ids)
    logger.info(f"Decoded text: {decoded}")

    # Vocabulary info
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")

    # Special tokens
    vocab = tokenizer.get_vocab()
    special_tokens = {k: v for k, v in vocab.items() if k.startswith('<') and k.endswith('>')}
    logger.info(f"Special tokens: {special_tokens}")


def interactive_tokenizer_test(tokenizer):
    """
    Interactive testing of the tokenizer.

    Args:
        tokenizer: The tokenizer to test.
    """
    print("Interactive Tokenizer Testing")
    print("Enter text to tokenize (or 'quit' to exit):")

    while True:
        text = input("> ")
        if text.lower() == 'quit':
            break

        encoded = tokenizer.encode(text)
        print(f"IDs: {encoded.ids}")
        print(f"Tokens: {encoded.tokens}")
        decoded = tokenizer.decode(encoded.ids)
        print(f"Decoded: {decoded}")
        print()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Test tokenizer from Hugging Face')
    parser.add_argument('--repo', type=str, required=True, help='Tokenizer repository (e.g., user/repo-name)')
    parser.add_argument('--file', type=str, default='bpe_tokenizer.json', help='Tokenizer filename')
    parser.add_argument('--interactive', action='store_true', help='Run interactive testing')

    args = parser.parse_args()

    tokenizer = load_tokenizer_from_hub(args.repo, args.file)

    if args.interactive:
        interactive_tokenizer_test(tokenizer)
    else:
        test_tokenizer(tokenizer)