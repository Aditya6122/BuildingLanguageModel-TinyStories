"""
Main script for training the language model.

This script orchestrates the entire training pipeline:
- Loads dataset and tokenizer
- Prepares data for training
- Initializes the model
- Runs the training loop with validation and logging
"""

import logging
import numpy as np
import torch
import wandb

from config import *
from model import LanguageModel
from data.dataset import load_tokenizer, load_datasets, prepare_datasets, create_dataloaders
from training.train import train_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load dataset
logger.info("Loading dataset...")
dataset = load_datasets(DATASET_NAME)

# Load tokenizer
logger.info("Loading tokenizer...")
tokenizer = load_tokenizer(TOKENIZER_NAME)

# Calculate total tokens
total_tokens = int(np.sum([len(ids) for ids in dataset['train']["ids"]]))
total_estimate_tokens = total_tokens * EPOCHS  # no of epochs * block size
logger.info(f"Total Trainable tokens : {total_estimate_tokens / 1e6:.2f}M")
logger.info(f"Required Parameters : {total_estimate_tokens / (20 * 1e6):.2f}M")

vocab_size = tokenizer.get_vocab_size()
logger.info(f"Vocab Size: {vocab_size}")

pad_token_id = tokenizer.token_to_id("<|pad|>")
logger.info(f"Pad Token Id : {pad_token_id}")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

# Prepare datasets
logger.info("Preparing datasets...")
final_train_dataset, final_validation_dataset = prepare_datasets(dataset, pad_token_id)

# Create dataloaders
logger.info("Creating dataloaders...")
train_loader, val_loader = create_dataloaders(final_train_dataset, final_validation_dataset, BATCH_SIZE, NUM_WORKERS, pad_token_id)

# Instantiate model
logger.info("Instantiating model...")
model = LanguageModel(vocab_size=vocab_size, embed_dim=EMBEDDING_DIMENSION, hidden_dim=HIDDEN_DIMENSION)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total params: {total_params:,}")

# Initialize wandb
logger.info("Initializing wandb...")
wandb.init(
    project="tinystories-training",
    name=RUN_NAME,
    config={
        "epochs": EPOCHS,
        "dataset_name": DATASET_NAME,
        "tokenizer_name": TOKENIZER_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "hidden_dimension": HIDDEN_DIMENSION,
        "total_params": total_params,
        "vocab_size": vocab_size,
        "sequence_len": -1,
        "sequence_overlap_ratio": -1,
        "train_data_points": len(train_loader),
        "val_data_points": len(val_loader),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model": MODEL_NAME,
    }
)

# Train the model
logger.info("Starting training...")
train_model(model, train_loader, val_loader, tokenizer, LEARNING_RATE, EPOCHS, device, RUN_NAME)
logger.info("Training completed.")