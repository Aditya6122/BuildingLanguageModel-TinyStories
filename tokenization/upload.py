"""
Tokenizer upload module.

This module handles uploading trained tokenizers to Hugging Face Hub.
"""

import logging
from huggingface_hub import create_repo, upload_file
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
        path_in_repo="tokenizer.json",
        repo_id=full_repo_name,
        repo_type="model",
        token=HF_TOKEN
    )

    logger.info(f"Tokenizer uploaded to {full_repo_name}")