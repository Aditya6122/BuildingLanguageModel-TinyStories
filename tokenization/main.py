"""
Main script for training and deploying a BPE tokenizer.

This script orchestrates the complete pipeline:
- Load and prepare data
- Train BPE tokenizer
- Evaluate tokenizer performance
- Upload tokenizer to Hugging Face Hub
"""

import logging
import os
import uuid
import shutil
from data_processor.dataset import (
    load_and_split_dataset,
    filter_and_sample_datasets,
)
from tokenization.trainer import (
    train_tokenizer,
    save_tokenizer,
    test_tokenizer,
    evaluate_tokenizer_dataset,
    prepare_text_file,
    show_example
)

from tokenization.upload import upload_tokenizer_to_hub

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(dataset_name, vocab_size, min_frequency, max_text_length, train_samples, val_samples, tokenizer_type, tokenizer_repo, version_suffix):
    """Main function to run the tokenizer training pipeline."""

    # Create temporary directory
    training_id = str(uuid.uuid4())
    temp_dir = f"temp_{training_id}"
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Created temporary directory: {temp_dir}")

    try:
        logger.info("Starting tokenizer training pipeline")

        # Load and prepare data
        logger.info("Step 1: Loading and preparing data")
        train_val_dataset = load_and_split_dataset(dataset_name)
        show_example(train_val_dataset['train'])
        train_val_dataset = filter_and_sample_datasets(
            train_val_dataset,
            max_length=max_text_length,
            train_samples=train_samples,
            val_samples=val_samples
        )

        text_file = prepare_text_file(train_val_dataset['train'], temp_dir)

        # Train tokenizer
        logger.info("Step 2: Training tokenizer")
        tokenizer = train_tokenizer(
            text_file,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            tokenizer_type=tokenizer_type
        )

        tokenizer_file = os.path.join(temp_dir, "tokenizer.json")
        save_tokenizer(tokenizer, tokenizer_file)
        test_tokenizer(tokenizer)

        # Evaluate tokenizer
        logger.info("Step 3: Evaluating tokenizer")
        vocab = tokenizer.get_vocab()
        vocab_size_actual = tokenizer.get_vocab_size()
        logger.info(f"Actual vocab size: {vocab_size_actual}")
        logger.info(f"Special tokens: end_of_text={vocab.get('<|end_of_text|>')}, pad={vocab.get('<|pad|>')}, unk={vocab.get('<unk>')}")

        train_metrics = evaluate_tokenizer_dataset(tokenizer, train_val_dataset['train'])
        val_metrics = evaluate_tokenizer_dataset(tokenizer, train_val_dataset['validation'])

        # Upload to Hugging Face
        logger.info("Step 4: Uploading to Hugging Face Hub")
        VERSION = f"vb-{tokenizer.get_vocab_size()}-{tokenizer_type}-{version_suffix}"
        upload_tokenizer_to_hub(tokenizer_file, tokenizer_repo, VERSION)

        logger.info("Tokenizer training pipeline completed successfully")
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train tokenizer')
    parser.add_argument('--dataset-name', type=str, default="karpathy/tinystories-gpt4-clean", help='Dataset name')
    parser.add_argument('--vocab-size', type=int, default=None, help='Vocabulary size')
    parser.add_argument('--min-frequency', type=int, default=2, help='Minimum frequency for tokens')
    parser.add_argument('--max-text-length', type=int, default=1500, help='Maximum text length')
    parser.add_argument('--train-samples', type=int, default=100000, help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=20000, help='Number of validation samples')
    parser.add_argument('--tokenizer-type', type=str, default='char_bpe', choices=['byte_level_bpe', 'char_bpe', 'bpe', 'word_level'], help='Type of tokenizer to train')
    parser.add_argument('--tokenizer-repo', type=str, default="aditya-6122/tinystories-tokenizer", help='Tokenizer repo on Hugging Face')
    parser.add_argument('--version-suffix', type=str, default="v1-test", help='Version string for the tokenizer upload')

    args = parser.parse_args()
    main(args.dataset_name, args.vocab_size, args.min_frequency, args.max_text_length, args.train_samples, args.val_samples, args.tokenizer_type, args.tokenizer_repo, args.version_suffix)
