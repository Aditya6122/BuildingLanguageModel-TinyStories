"""
Configuration module for the language model training.

This module loads environment variables and defines training hyperparameters.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# Hugging Face and Weights & Biases credentials
HF_TOKEN = os.environ.get('HF_TOKEN')
WANDB_API_KEY = os.environ.get('WANDB_API_KEY')

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment")

if WANDB_API_KEY is None:
    raise ValueError("WANDB_API_KEY not found in environment")

# Dataset and tokenizer configuration
DATASET_NAME = "aditya-6122/tinystories-custom-dataset-18542-v2-test"
TOKENIZER_NAME = "aditya-6122/tinystories-tokenizer-vb-18542-byte_level_bpe-v3-test"

# Model hyperparameters
EMBEDDING_DIMENSION = 128
NUM_HEADS = 4
MAX_SEQ_LEN = 500
NUM_LAYERS = 3

# Training hyperparameters
LEARNING_RATE = 1e-3
EPOCHS = 1
BATCH_SIZE = 2
NUM_WORKERS = 2

# Experiment configuration
RUN_NAME = "test"
MODEL_NAME = "test"