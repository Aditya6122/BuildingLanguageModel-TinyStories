"""
Tokenizer training module.

This module handles the training and evaluation of BPE tokenizers.
"""

import logging
import math
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer

logger = logging.getLogger(__name__)


def train_tokenizer(text_file, vocab_size=None, min_frequency=2, special_tokens=None, tokenizer_type='byte_level_bpe'):
    """
    Train a tokenizer.

    Args:
        text_file (str): Path to the text file for training.
        vocab_size (int, optional): Target vocabulary size.
        min_frequency (int): Minimum frequency for tokens.
        special_tokens (list): List of special tokens.
        tokenizer_type (str): Type of tokenizer ('byte_level_bpe', 'char_bpe', 'bpe', 'word_level').

    Returns:
        Tokenizer: Trained tokenizer.
    """
    if special_tokens is None:
        special_tokens = [
            "<|pad|>",
            "<|start_of_text|>",
            "<|end_of_text|>",
            "<unk>",
        ]

    logger.info(f"Training {tokenizer_type} tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}")
    logger.info(f"Special tokens: {special_tokens}")

    # Import here to avoid issues
    from tokenizers import Tokenizer, CharBPETokenizer, ByteLevelBPETokenizer
    from tokenizers.models import BPE, WordLevel
    from tokenizers.pre_tokenizers import Whitespace, ByteLevel
    from tokenizers.trainers import BpeTrainer, WordLevelTrainer

    if tokenizer_type == 'byte_level_bpe':
        tokenizer = Tokenizer(ByteLevelBPETokenizer())
        tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True
        )
    elif tokenizer_type == 'char_bpe':
        tokenizer = CharBPETokenizer()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True
        )
    elif tokenizer_type == 'bpe':
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True
        )
    elif tokenizer_type == 'word_level':
        tokenizer = Tokenizer(WordLevel())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    # Train
    tokenizer.train([text_file], trainer)

    logger.info("Tokenizer training completed")
    return tokenizer


def save_tokenizer(tokenizer, output_file="bpe_tokenizer.json"):
    """
    Save tokenizer to file.

    Args:
        tokenizer: Tokenizer to save.
        output_file (str): Output file path.
    """
    tokenizer.save(output_file)
    logger.info(f"Tokenizer saved to {output_file}")


def test_tokenizer(tokenizer):
    """
    Test the tokenizer with a sample text.

    Args:
        tokenizer: Trained tokenizer.
    """
    input_text = "Once upon a time there was a girl named Lily. <|end_of_text|><|pad|><|pad|><|pad|>"
    encoded_input = tokenizer.encode(input_text)
    decoded_output = tokenizer.decode(encoded_input.ids)

    logger.info(f"Test Input: {input_text}")
    logger.info(f"Encoded: {encoded_input.ids}")
    logger.info(f"Decoded: {decoded_output}")


def evaluate_tokenizer_dataset(tokenizer, dataset, batch_size=10000):
    """
    Evaluate tokenizer performance on a dataset.

    Args:
        tokenizer: Tokenizer to evaluate.
        dataset: Dataset to evaluate on.
        batch_size (int): Batch size for processing.

    Returns:
        dict: Evaluation metrics.
    """
    logger.info("Evaluating tokenizer on dataset")
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

    metrics = {
        "avg_tokens_per_sentence": avg_tokens,
        "avg_chars_per_sentence": avg_chars,
        "compression_ratio": compression_ratio
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics