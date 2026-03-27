"""
Main script for training and deploying a BPE tokenizer.

This script orchestrates the complete pipeline:
- Load and prepare data
- Train BPE tokenizer
- Evaluate tokenizer performance
- Upload tokenizer and processed dataset to Hugging Face Hub
"""

import logging
import os
from tokenization.data_preparation import (
    load_and_split_dataset,
    filter_and_sample_datasets,
    prepare_text_file,
    show_example
)
from tokenization.tokenizer_trainer import (
    train_tokenizer,
    save_tokenizer,
    test_tokenizer,
    evaluate_tokenizer_dataset
)
from tokenization.dataset_uploader import (
    upload_tokenizer_to_hub,
    prepare_and_upload_dataset
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATASET_NAME = "karpathy/tinystories-gpt4-clean"
VOCAB_SIZE = 4000  # Set to None for unlimited vocab
MIN_FREQUENCY = 2
MAX_TEXT_LENGTH = 1500
TRAIN_SAMPLES = 100000
VAL_SAMPLES = 20000
TOKENIZER_TYPE = "byte_level_bpe"  # Options: 'byte_level_bpe', 'char_bpe', 'bpe', 'word_level'
TOKENIZER_REPO = "aditya-6122/tinystories-tokenizer"
DATASET_REPO = "aditya-6122/tinystories-custom-dataset"
VERSION = f"vb-{VOCAB_SIZE}-{TOKENIZER_TYPE}-v1"

def main(tokenizer_type=None):
    """Main function to run the tokenizer training pipeline."""
    global TOKENIZER_TYPE, VERSION
    if tokenizer_type:
        TOKENIZER_TYPE = tokenizer_type
        VERSION = f"vb-{VOCAB_SIZE}-{TOKENIZER_TYPE}-v1"

    logger.info("Starting tokenizer training pipeline")

    # Load and prepare data
    logger.info("Step 1: Loading and preparing data")
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(DATASET_NAME)
    show_example(train_dataset)
    train_dataset, val_dataset = filter_and_sample_datasets(
        train_dataset, val_dataset,
        max_length=MAX_TEXT_LENGTH,
        train_samples=TRAIN_SAMPLES,
        val_samples=VAL_SAMPLES
    )
    text_file = prepare_text_file(train_dataset)

    # Train tokenizer
    logger.info("Step 2: Training tokenizer")
    tokenizer = train_tokenizer(
        text_file,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        tokenizer_type=TOKENIZER_TYPE
    )
    save_tokenizer(tokenizer)
    test_tokenizer(tokenizer)

    # Clean up temporary text file
    if os.path.exists(text_file):
        os.remove(text_file)
        logger.info(f"Cleaned up temporary file: {text_file}")

    # Evaluate tokenizer
    logger.info("Step 3: Evaluating tokenizer")
    vocab = tokenizer.get_vocab()
    vocab_size_actual = tokenizer.get_vocab_size()
    logger.info(f"Actual vocab size: {vocab_size_actual}")
    logger.info(f"Special tokens: end_of_text={vocab.get('<|end_of_text|>')}, pad={vocab.get('</s>')}")

    train_metrics = evaluate_tokenizer_dataset(tokenizer, train_dataset)
    val_metrics = evaluate_tokenizer_dataset(tokenizer, val_dataset)

    # Upload to Hugging Face
    logger.info("Step 4: Uploading to Hugging Face Hub")
    upload_tokenizer_to_hub("bpe_tokenizer.json", TOKENIZER_REPO, VERSION)
    prepare_and_upload_dataset(
        train_dataset, val_dataset, tokenizer, vocab,
        DATASET_REPO, VERSION
    )

    logger.info("Tokenizer training pipeline completed successfully")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train tokenizer')
    parser.add_argument('--tokenizer-type', type=str, default='byte_level_bpe',
                       choices=['byte_level_bpe', 'char_bpe', 'bpe', 'word_level'],
                       help='Type of tokenizer to train')

    args = parser.parse_args()
    main(tokenizer_type=args.tokenizer_type)