"""
Data preparation module for tokenizer training.

This module handles loading datasets, filtering, and preparing text data for tokenizer training.
"""

import logging
import random
from datasets import load_dataset
from config import HF_TOKEN

logger = logging.getLogger(__name__)


def load_and_split_dataset(dataset_name="karpathy/tinystories-gpt4-clean", test_size=0.2, val_size=0.2, seed=123):
    """
    Load dataset and split into train/validation/test sets.

    Args:
        dataset_name (str): Name of the dataset to load.
        test_size (float): Proportion of data for test set.
        val_size (float): Proportion of data for validation set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, token=HF_TOKEN)
    logger.info(f"Dataset loaded with {len(dataset['train'])} records")

    # Split train into train+val and test
    train_test_dataset = dataset['train'].train_test_split(test_size=test_size, shuffle=True, seed=seed)

    # Split train into train and val
    train_val_dataset = train_test_dataset['train'].train_test_split(test_size=val_size, shuffle=True, seed=seed)

    train_dataset = train_val_dataset['train']
    val_dataset = train_val_dataset['test']
    test_dataset = train_test_dataset['test']

    logger.info(f"Dataset split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def filter_and_sample_datasets(train_dataset, val_dataset, max_length=1500, train_samples=100000, val_samples=20000):
    """
    Filter short texts and sample datasets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        max_length (int): Maximum text length to keep.
        train_samples (int): Number of training samples to select.
        val_samples (int): Number of validation samples to select.

    Returns:
        tuple: Filtered and sampled (train_dataset, val_dataset)
    """
    logger.info(f"Filtering texts shorter than {max_length} characters")

    def filter_short_text(example):
        return len(example["text"]) < max_length

    train_dataset = train_dataset.filter(filter_short_text, num_proc=8)
    val_dataset = val_dataset.filter(filter_short_text, num_proc=8)

    logger.info(f"After filtering: train={len(train_dataset)}, val={len(val_dataset)}")

    # Sample datasets
    if len(train_dataset) > train_samples:
        train_dataset = train_dataset.select(range(train_samples))
        logger.info(f"Sampled train dataset to {train_samples} examples")

    if len(val_dataset) > val_samples:
        val_dataset = val_dataset.select(range(val_samples))
        logger.info(f"Sampled val dataset to {val_samples} examples")

    return train_dataset, val_dataset


def prepare_text_file(train_dataset, output_file="full_text.txt"):
    """
    Combine all training texts into a single file for tokenizer training.

    Args:
        train_dataset: Training dataset.
        output_file (str): Output file path.

    Returns:
        str: Path to the output file.
    """
    logger.info("Preparing text file for tokenizer training")
    full_text = "\n".join(train_dataset['text'])

    with open(output_file, 'w') as f:
        f.write(full_text)

    logger.info(f"Text file written to {output_file} with {len(full_text)} characters")
    return output_file


def show_example(train_dataset):
    """
    Display a random example from the dataset.

    Args:
        train_dataset: Dataset to sample from.
    """
    example = train_dataset[random.randint(0, len(train_dataset) - 1)]['text']
    logger.info(f"Example story:\n\n{example}")