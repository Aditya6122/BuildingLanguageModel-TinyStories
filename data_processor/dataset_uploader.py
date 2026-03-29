"""
Dataset and tokenizer upload module.

This module handles uploading trained tokenizers and processed datasets to Hugging Face Hub.
"""

import logging
from huggingface_hub import create_repo, upload_file
from datasets import DatasetDict
from config import HF_TOKEN

logger = logging.getLogger(__name__)


def upload_tokenizer_to_hub(tokenizer_file, repo_name, version="v1"):
    """
    Upload tokenizer to Hugging Face Hub.

    Args:
        tokenizer_file (str): Path to tokenizer file.
        repo_name (str): Base repository name.
        version (str): Version suffix.
    """
    full_repo_name = f"{repo_name}-{version}"
    logger.info(f"Uploading tokenizer to {full_repo_name}")

    create_repo(
        full_repo_name,
        repo_type="model",
        exist_ok=True,
        token=HF_TOKEN
    )

    upload_file(
        path_or_fileobj=tokenizer_file,
        path_in_repo="bpe_tokenizer.json",
        repo_id=full_repo_name,
        repo_type="model",
        token=HF_TOKEN
    )

    logger.info(f"Tokenizer uploaded to {full_repo_name}")


def prepare_and_upload_dataset(train_dataset, val_dataset, tokenizer, vocab, repo_name, version="v1"):
    """
    Prepare tokenized dataset and upload to Hugging Face Hub.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        tokenizer: Trained tokenizer.
        vocab (dict): Tokenizer vocabulary.
        repo_name (str): Base repository name.
        version (str): Version suffix.
    """
    logger.info("Preparing tokenized datasets")

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(
        lambda x: {"raw_ids": tokenizer.encode(x["text"]).ids},
        num_proc=8
    )
    tokenized_val_dataset = val_dataset.map(
        lambda x: {"raw_ids": tokenizer.encode(x["text"]).ids},
        num_proc=8
    )

    # Add end token
    end_token_id = vocab['<|end_of_text|>']
    padded_train_dataset = tokenized_train_dataset.map(
        lambda x: {"ids": x['raw_ids'] + [end_token_id]},
        num_proc=8
    )
    padded_validation_dataset = tokenized_val_dataset.map(
        lambda x: {"ids": x['raw_ids'] + [end_token_id]},
        num_proc=8
    )

    # Create input/output sequences
    final_train_dataset = padded_train_dataset.map(
        lambda x: {"input_ids": x['ids'][:-1], "output_ids": x['ids'][1:]},
        num_proc=8
    )
    final_validation_dataset = padded_validation_dataset.map(
        lambda x: {"input_ids": x['ids'][:-1], "output_ids": x['ids'][1:]},
        num_proc=8
    )

    # Log token length statistics
    token_lengths = [len(ids) for ids in final_train_dataset['ids']]
    logger.info(f"Token length stats: max={max(token_lengths)}, min={min(token_lengths)}")

    # Create dataset dict and upload
    dataset_dict = DatasetDict({
        "train": final_train_dataset,
        "validation": final_validation_dataset
    })

    full_repo_name = f"{repo_name}-{version}"
    logger.info(f"Uploading dataset to {full_repo_name}")
    dataset_dict.push_to_hub(full_repo_name, token=HF_TOKEN)
    logger.info(f"Dataset uploaded to {full_repo_name}")