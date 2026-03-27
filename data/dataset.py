"""
Dataset module for loading and preparing data for training.

This module provides functions to:
- Load tokenizers from Hugging Face Hub
- Load datasets
- Prepare datasets by adding padding and creating input/output sequences
- Create DataLoaders with custom collation
"""

import logging
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_name, file_name='bpe_tokenizer.json', repo_type='model'):
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
        repo_type=repo_type
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
    dataset = load_dataset(dataset_name)
    logger.info(f"Dataset loaded: {dataset}")
    logger.info(f"Tiny Stories dataset has {dataset['train'].num_rows} train and {dataset['validation'].num_rows} val records")
    return dataset


def prepare_datasets(dataset, pad_token_id):
    """
    Prepare the datasets by adding padding and creating input/output sequences.

    Args:
        dataset (DatasetDict): The loaded dataset.
        pad_token_id (int): The token ID for padding.

    Returns:
        tuple: Final train and validation datasets.
    """
    logger.info("Preparing datasets with padding and sequence creation")
    train_dataset = dataset['train'].select_columns(["raw_ids"])
    validation_dataset = dataset['validation'].select_columns(["raw_ids"])

    padded_train_dataset = train_dataset.map(lambda x: {"ids": x['raw_ids'] + [pad_token_id]}, num_proc=4)
    padded_validation_dataset = validation_dataset.map(lambda x: {"ids": x['raw_ids'] + [pad_token_id]}, num_proc=4)

    final_train_dataset = padded_train_dataset.map(lambda x: {"input_ids": x['ids'][:-1], "output_ids": x['ids'][1:]}, num_proc=4)
    final_validation_dataset = padded_validation_dataset.map(lambda x: {"input_ids": x['ids'][:-1], "output_ids": x['ids'][1:]}, num_proc=4)

    logger.info("Datasets prepared successfully")
    return final_train_dataset, final_validation_dataset


def collate_fn(batch, pad_token_id=2):
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

    # 3. Initialize tensors with 0
    x_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    y_padded = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i in range(len(batch)):
        x_len = x_seqs[i].size(0)
        y_len = y_seqs[i].size(0)

        # --- Process Input IDs (X) ---
        x_padded[i, :x_len] = x_seqs[i]
        # If there is space left, put a 'pad_token_id' in the very next slot
        if x_len < max_len:
            x_padded[i, x_len] = pad_token_id
        # The rest remains 0 from initialization

        # --- Process Output IDs (Y) ---
        y_padded[i, :y_len] = y_seqs[i]
        # The rest remains 0 from initialization

    return x_padded, y_padded


def create_dataloaders(final_train_dataset, final_validation_dataset, batch_size=128, num_workers=2, pad_token_id=2):
    """
    Create DataLoaders for training and validation.

    Args:
        final_train_dataset: Prepared training dataset.
        final_validation_dataset: Prepared validation dataset.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        pad_token_id (int): Token ID for padding.

    Returns:
        tuple: Train and validation DataLoaders.
    """
    logger.info("Creating DataLoaders")
    train_loader = DataLoader(
        final_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
        pin_memory=True
    )

    val_loader = DataLoader(
        final_validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
        pin_memory=True
    )

    logger.info(f"DataLoaders created: train batches={len(train_loader)}, val batches={len(val_loader)}")
    return train_loader, val_loader