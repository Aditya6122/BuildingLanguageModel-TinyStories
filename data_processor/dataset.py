"""
Dataset module for loading and preparing data for training.

This module provides functions to:
- Load tokenizers from Hugging Face Hub
- Load datasets
- Prepare datasets by adding padding and creating input/output sequences
- Create DataLoaders with custom collation
"""

import logging
import os
import shutil
import uuid
import torch
from functools import partial
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from config import HF_TOKEN

logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_name, file_name='tokenizer.json', repo_type='model'):
    """
    Load a tokenizer from Hugging Face Hub.

    Args:
        tokenizer_name (str): The name of the tokenizer repository.
        file_name (str): The filename of the tokenizer file.
        repo_type (str): The type of repository.

    Returns:
        Tokenizer: The loaded tokenizer object.
    """
    logger.info(f"Loading tokenizer from {tokenizer_name}")
    file_path = hf_hub_download(
        repo_id=tokenizer_name,
        filename=file_name,
        repo_type=repo_type,
        token=HF_TOKEN
    )
    tokenizer = Tokenizer.from_file(file_path)
    logger.info("Tokenizer loaded successfully")
    return tokenizer


def load_datasets(dataset_name):
    """
    Load the dataset from Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        DatasetDict: The loaded dataset.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, token=HF_TOKEN)
    logger.info(f"Dataset loaded: {dataset}")
    logger.info(f"Tiny Stories dataset has {dataset['train'].num_rows} train and {dataset['validation'].num_rows} val records")
    return dataset


def load_raw_dataset(dataset_name, text_column='text'):
    """
    Load raw dataset (train + validation) from Hugging Face Hub.

    Args:
        dataset_name (str): Hugging Face dataset name.
        text_column (str): Column in dataset that contains text.

    Returns:
        DatasetDict: Loaded dataset with train/validation splits.
    """
    dataset = load_dataset(dataset_name, token=HF_TOKEN)
    if text_column not in dataset['train'].column_names:
        raise ValueError(f"Expected text column '{text_column}' not found in dataset")
    logger.info(f"Raw dataset '{dataset_name}' loaded with text column '{text_column}'")
    return dataset


def tokenize_dataset(dataset, tokenizer, eot_token_id, text_column='text', num_proc=4):
    """
    Tokenize raw text and add raw IDs + shifted input/output IDs.

    Args:
        dataset (DatasetDict): A dataset containing train and validation splits.
        tokenizer (Tokenizer): A tokenizers.Tokenizer instance.
        eot_token_id (int): End-of-text token ID.
        text_column (str): Column in dataset that contains text.
        num_proc (int): Number of processes for dataset.map.

    Returns:
        tuple: (train_dataset, validation_dataset) after tokenization and sequence creation.
    """
    target_splits = ['train', 'validation']
    if any(split not in dataset for split in target_splits):
        raise ValueError(f"Dataset must contain splits: {target_splits}")

    logger.info("Tokenizing dataset splits")

    tokenized = {}
    for split in target_splits:
        tokenized[split] = dataset[split].map(
            lambda x: {
                'raw_ids': tokenizer.encode(x[text_column]).ids
            },
            num_proc=num_proc
        )

        tokenized[split] = tokenized[split].map(
            lambda x: {
                'ids': x['raw_ids'] + [eot_token_id]
            },
            num_proc=num_proc
        )

        tokenized[split] = tokenized[split].map(
            lambda x: {
                'input_ids': x['ids'][:-1],
                'output_ids': x['ids'][1:]
            },
            num_proc=num_proc
        )

    logger.info("Tokenization complete")

    return tokenized['train'], tokenized['validation']


def process_and_upload_dataset(dataset_name,
                               tokenizer_name,
                               dataset_repo_name,
                               version='v1',
                               text_column='text',
                               max_text_length=1500,
                               max_train_samples=100000,
                               max_val_samples=20000,
                               validation_split=0.2,
                               seed=123,
                               num_proc=4):
    """
    Full dataset processing pipeline: load raw data -> tokenize -> upload to HF Hub.

    This function creates a temporary directory (same pattern as tokenization module),
    and always cleans it up at the end.

    Args:
        dataset_name (str): Hugging Face dataset name.
        tokenizer_name (str): Hugging Face tokenizer repo name.
        dataset_repo_name (str): Base repository name to push processed dataset.
        version (str): Version suffix for destination repo.
        text_column (str): Column containing raw text.
        num_proc (int): Number of processes for .map.

    Returns:
        DatasetDict: Prepared dataset dict pushed to hub.
    """
    temp_dir = f"temp_dataset_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Created temporary dataset processing directory: {temp_dir}")

    try:
        tokenizer = load_tokenizer(tokenizer_name)

        if '<|end_of_text|>' in tokenizer.get_vocab():
            eot_token_id = tokenizer.token_to_id('<|end_of_text|>')
        else:
            eot_token_id = tokenizer.token_to_id('</s>') if tokenizer.token_to_id('</s>') is not None else 2

        train_val_dataset = load_and_split_dataset(dataset_name, val_size=validation_split, seed=seed)
        train_val_dataset = filter_and_sample_datasets(
            train_val_dataset,
            max_length=max_text_length,
            train_samples=max_train_samples,
            val_samples=max_val_samples
        )

        train_dataset, validation_dataset = tokenize_dataset(
            train_val_dataset,
            tokenizer,
            eot_token_id=eot_token_id,
            text_column=text_column,
            num_proc=num_proc
        )

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset
        })

        upload_path = f"{dataset_repo_name}-{tokenizer.get_vocab_size()}-{version}"
        logger.info(f"Uploading processed dataset to {upload_path}")
        dataset_dict.push_to_hub(upload_path, token=HF_TOKEN)
        logger.info("Processed dataset uploaded successfully")

        return dataset_dict

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


def _truncate_example(example, fixed_length):
    """Truncate input/output sequences to a fixed length."""
    return {
        "input_ids": example["input_ids"][:fixed_length],
        "output_ids": example["output_ids"][:fixed_length],
    }


def prepare_datasets(dataset, fixed_input_length=None):
    """
    Prepare the datasets by selecting input/output columns and optionally enforcing a fixed sequence length.

    Args:
        dataset (DatasetDict): The loaded dataset.
        fixed_input_length (int, optional): If set, truncates all input/output sequences to this length.

    Returns:
        tuple: Final train and validation datasets.
    """
    logger.info("Preparing datasets with padding and sequence creation")

    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    if fixed_input_length is not None:
        if fixed_input_length <= 0:
            raise ValueError("fixed_input_length must be a positive integer")

        logger.info("Enforcing fixed sequence length: %d", fixed_input_length)
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids"]) >= fixed_input_length,
            num_proc=1
        ).map(
            partial(_truncate_example, fixed_length=fixed_input_length),
            num_proc=1
        )
        validation_dataset = validation_dataset.filter(
            lambda x: len(x["input_ids"]) >= fixed_input_length,
            num_proc=1
        ).map(
            partial(_truncate_example, fixed_length=fixed_input_length),
            num_proc=1
        )

    train_dataset = train_dataset.select_columns(["input_ids", "output_ids"])
    validation_dataset = validation_dataset.select_columns(["input_ids", "output_ids"])

    logger.info("Datasets prepared successfully")
    return train_dataset, validation_dataset


def collate_fn(batch, eot_token_id=2):
    """
    Custom collation function for DataLoader to handle variable-length sequences.

    Args:
        batch (list): List of samples from the dataset.
        pad_token_id (int): Token ID for padding.

    Returns:
        tuple: Padded input and output tensors.
    """
    # 1. Extract sequences
    x_seqs = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    y_seqs = [torch.tensor(item["output_ids"], dtype=torch.long) for item in batch]

    # 2. Find the max length of the sequences ALREADY in the batch
    max_len = max(seq.size(0) for seq in x_seqs)

    # 3. Initialize tensors with 0 because padding will be done with 0s. We will add 'eot_token_id' after the actual sequence if there is space.
    x_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    y_padded = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i in range(len(batch)):
        x_len = x_seqs[i].size(0)
        y_len = y_seqs[i].size(0)

        # --- Process Input IDs (X) ---
        x_padded[i, :x_len] = x_seqs[i]
        # If there is space left, put a 'eot_token_id' in the very next slot
        if x_len < max_len:
            x_padded[i, x_len] = eot_token_id
        # The rest remains 0 from initialization

        # --- Process Output IDs (Y) ---
        y_padded[i, :y_len] = y_seqs[i]
        # The rest remains 0 from initialization

    return x_padded, y_padded


def create_dataloaders(final_train_dataset, final_validation_dataset, batch_size=128, num_workers=2, eot_token_id=2, device="cpu"):
    """
    Create DataLoaders for training and validation.

    Args:
        final_train_dataset: Prepared training dataset.
        final_validation_dataset: Prepared validation dataset.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        eot_token_id (int): Token ID for end of text.
        device (str): Device type ("cpu", "cuda", or "mps").

    Returns:
        tuple: Train and validation DataLoaders.
    """
    logger.info("Creating DataLoaders")
    pin_memory = True
    
    # Disable multiprocessing on MPS as it doesn't support worker processes properly
    if device == "mps" or device == "cpu":
        logger.info("MPS or CPU device detected. Disabling multiprocessing workers for DataLoader.")
        num_workers = 0
        pin_memory = False

    
    train_loader = DataLoader(
        final_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, eot_token_id=eot_token_id),
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        final_validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, eot_token_id=eot_token_id),
        pin_memory=pin_memory
    )

    logger.info(f"DataLoaders created: train batches={len(train_loader)}, val batches={len(val_loader)}")
    return train_loader, val_loader

def load_and_split_dataset(dataset_name="karpathy/tinystories-gpt4-clean", val_size=0.2, seed=123):
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
    train_val_dataset = dataset['train'].train_test_split(test_size=val_size, shuffle=True, seed=seed)
    train_val_dataset['validation'] = train_val_dataset.pop('test')

    logger.info(f"Dataset split: train={train_val_dataset['train'].num_rows}, val={train_val_dataset['validation'].num_rows}")
    return train_val_dataset


def filter_and_sample_datasets(train_val_dataset, max_length=1500, train_samples=100000, val_samples=20000):
    """
    Filter short texts and sample datasets.

    Args:
        train_val_dataset: Training and validation dataset.
        max_length (int): Maximum text length to keep.
        train_samples (int): Number of training samples to select.
        val_samples (int): Number of validation samples to select.

    Returns:
        tuple: Filtered and sampled (train_dataset, val_dataset)
    """
    logger.info(f"Filtering texts shorter than {max_length} characters")

    def filter_short_text(example):
        return len(example["text"]) < max_length

    train_val_dataset['train'] = train_val_dataset['train'].filter(filter_short_text, num_proc=8)
    train_val_dataset['validation'] = train_val_dataset['validation'].filter(filter_short_text, num_proc=8)

    logger.info(f"After filtering: train={len(train_val_dataset['train'])}, val={len(train_val_dataset['validation'])}")

    # Sample datasets
    if len(train_val_dataset['train']) > train_samples:
        train_val_dataset['train'] = train_val_dataset['train'].select(range(train_samples))
        logger.info(f"Sampled train dataset to {train_samples} examples")

    if len(train_val_dataset['validation']) > val_samples:
        train_val_dataset['validation'] = train_val_dataset['validation'].select(range(val_samples))
        logger.info(f"Sampled val dataset to {val_samples} examples")

    return train_val_dataset