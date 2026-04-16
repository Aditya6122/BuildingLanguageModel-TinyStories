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
EMBEDDING_DIMENSION = 256 # Default : 256
NUM_HEADS = 8 # Default : 8
MAX_SEQ_LEN = 500 # Default : 500
NUM_LAYERS = 4 # Default : 4

# Training hyperparameters
LEARNING_RATE = 1e-3 # Default : 1e-3
EPOCHS = 5 # Default : 5
BATCH_SIZE = 64 # Default : 64
NUM_WORKERS = 2 # Default : 2 for Colab, adjust as needed for local training

# Experiment configuration
RUN_NAME = "LM-TinyStories-AttentionMasking-PadFix" # Current : "LM-TinyStories-PadFix"
MODEL_NAME = "LM-TinyStories-AttentionMasking-PadFix" # Current : "LM-TinyStories-PadFix"