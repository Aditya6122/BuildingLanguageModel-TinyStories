"""
Main script for training the language model.

This script orchestrates the entire training pipeline:
- Loads dataset and tokenizer
- Prepares data for training
- Initializes the model
- Runs the training loop with validation and logging
"""

import argparse
import logging
import numpy as np
import torch
import wandb
import os
import uuid
import json
import shutil
from huggingface_hub import HfApi

from config import (
    DATASET_NAME,
    TOKENIZER_NAME,
    EMBEDDING_DIMENSION,
    HIDDEN_DIMENSION,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    NUM_WORKERS,
    RUN_NAME,
    MODEL_NAME,
    HF_TOKEN,
)
from models.language_model import LanguageModel
from data_processor.dataset import load_tokenizer, load_datasets, prepare_datasets, create_dataloaders
from training.train import train_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model card template
MODEL_CARD_TEMPLATE = """---
language: en
license: mit
tags:
- language-model
- pytorch
- rnn
- text-generation
datasets:
- {dataset_name}
pipeline_tag: text-generation
---

# {model_name}

## Model Details

### Model Description
This is a custom language model trained on a dataset of short stories, designed for text generation tasks.

### Model Sources
- **Repository**: [GitHub Repository](https://github.com/your-repo)  # Replace with actual repo if available
- **Paper**: N/A

## Uses

### Direct Use
This model can be used for generating short stories and text completion tasks.

### Downstream Use
Fine-tune the model on specific domains for specialized text generation.

### Out-of-Scope Use
Not intended for production use without further validation.

## Training Details

### Training Data
The model was trained on the [{dataset_name}](https://huggingface.co/datasets/{dataset_name}) dataset.

### Training Procedure
- **Training Regime**: Standard language model training with cross-entropy loss
- **Epochs**: {epochs}
- **Batch Size**: {batch_size}
- **Learning Rate**: {learning_rate}
- **Optimizer**: Adam (assumed)
- **Hardware**: Apple Silicon MPS (if available) or CPU

### Tokenizer
The model uses the [{tokenizer_name}](https://huggingface.co/{tokenizer_name}) tokenizer.

### Model Architecture
- **Architecture Type**: RNN-based language model with GRU cells
- **Embedding Dimension**: {embedding_dim}
- **Hidden Dimension**: {hidden_dim}
- **Vocabulary Size**: {vocab_size}
- **Architecture Diagram**: See `model_arch.jpg` for visual representation

## Files
- `model.bin`: The trained model weights in PyTorch format.
- `tokenizer.json`: The tokenizer configuration.
- `model_arch.jpg`: Architecture diagram showing the GRU model structure.

## How to Use

Since this is a custom model, you'll need to load it using the provided code:

```python
import torch
from your_language_model import LanguageModel  # Replace with actual import
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Load model
vocab_size = tokenizer.get_vocab_size()
model = LanguageModel(vocab_size=vocab_size, embedding_dimension={embedding_dim}, hidden_dimension={hidden_dim})
model.load_state_dict(torch.load("model.bin"))
model.eval()

# Generate text
input_text = "Once upon a time"
# Tokenize and generate (implement your generation logic)
```

## Limitations
- This is a basic RNN model and may not perform as well as transformer-based models.
- Trained on limited data, may exhibit biases from the training dataset.
- Not optimized for production deployment.

## Ethical Considerations
Users should be aware of potential biases in generated text and use the model responsibly.

## Citation
If you use this model, please cite:
```
@misc{{{model_name_slug}}},
  title={{{model_name}}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{repo_name}}}
}}
```"""


def cleanup_wandb():
    """Remove wandb directory after training."""
    wandb_dir = "wandb"
    if os.path.exists(wandb_dir):
        shutil.rmtree(wandb_dir)
        logger.info("Cleaned up wandb directory.")


def cleanup_model_artifacts(checkpoint_dir):
    """Remove local model artifacts after uploading to Hugging Face Hub."""
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        logger.info(f"Cleaned up local model artifacts in {checkpoint_dir}.")


def upload_model_to_hf(checkpoint_dir, model_name, dataset_name, tokenizer_name, tokenizer, vocab_size, embedding_dim, hidden_dim, epochs, batch_size, learning_rate):
    """Upload model artifacts and tokenizer to Hugging Face Hub."""
    api = HfApi()
    user_info = api.whoami(token=HF_TOKEN)
    username = user_info['name']
    repo_name = f"{username}/{model_name}"
    
    # Create repository
    api.create_repo(repo_name, token=HF_TOKEN, private=False, exist_ok=True)
    
    # Save tokenizer to checkpoint_dir
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # Create README.md with description
    model_name_slug = model_name.replace(' ', '-').lower()
    readme_content = MODEL_CARD_TEMPLATE.format(
        dataset_name=dataset_name,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tokenizer_name=tokenizer_name,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        repo_name=repo_name,
        model_name_slug=model_name_slug
    )
    readme_path = os.path.join(checkpoint_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    model_config = {
        "vocab_size": vocab_size,
        "embedding_dimension": embedding_dim,
        "hidden_dimension": hidden_dim,
    }
    model_config_path = os.path.join(checkpoint_dir, "model_config.json")
    with open(model_config_path, "w") as f:
        json.dump(model_config, f)
    
    # Upload files
    api.upload_file(
        path_or_fileobj=os.path.join(checkpoint_dir, "model.bin"),
        path_in_repo="pytorch_model.bin",
        repo_id=repo_name,
        token=HF_TOKEN
    )
    api.upload_file(
        path_or_fileobj=tokenizer_path,
        path_in_repo="tokenizer.json",
        repo_id=repo_name,
        token=HF_TOKEN
    )
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_name,
        token=HF_TOKEN
    )
    api.upload_file(
        path_or_fileobj=os.path.join(checkpoint_dir, "model_config.json"),
        path_in_repo="model_config.json",
        repo_id=repo_name,
        token=HF_TOKEN
    )
    
    # Upload architecture diagram if it exists
    arch_diagram_path = "model_arch.jpg"
    if os.path.exists(arch_diagram_path):
        api.upload_file(
            path_or_fileobj=arch_diagram_path,
            path_in_repo="model_arch.jpg",
            repo_id=repo_name,
            token=HF_TOKEN
        )
        logger.info(f"Uploaded architecture diagram to https://huggingface.co/{repo_name}")
    
    logger.info(f"Uploaded model, tokenizer, README, and architecture diagram to https://huggingface.co/{repo_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train language model with CLI arguments")

    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME)
    parser.add_argument("--tokenizer-name", type=str, default=TOKENIZER_NAME)
    parser.add_argument("--embedding-dimension", type=int, default=EMBEDDING_DIMENSION)
    parser.add_argument("--hidden-dimension", type=int, default=HIDDEN_DIMENSION)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--run-name", type=str, default=RUN_NAME)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--train-sample", action='store_true', help="Run quick training sample")
    parser.add_argument("--checkpoint-dir", type=str, default="./model_artifacts")
    parser.add_argument("--wandb-project", type=str, default="tinystories-training")
    parser.add_argument("--validation-prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)

    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_datasets(args.dataset_name)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_name)

    # Calculate total tokens
    total_tokens = int(np.sum([len(ids) for ids in dataset['train']["ids"]]))
    total_estimate_tokens = total_tokens * args.epochs
    logger.info(f"Total Trainable tokens : {total_estimate_tokens / 1e6:.2f}M")
    logger.info(f"Required Parameters : {total_estimate_tokens / (20 * 1e6):.2f}M")

    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocab Size: {vocab_size}")

    pad_token_id = tokenizer.token_to_id("<|pad|>")
    logger.info(f"Pad Token Id : {pad_token_id}")

    eot_token_id = tokenizer.token_to_id("<|end_of_text|>")
    logger.info(f"EOT Token Id : {eot_token_id}")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS backend is available. You can use Apple Silicon GPU for training.")
        device = "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA is available. You can use NVIDIA GPU for training.")
        device = "cuda"
    else:
        logger.info("No GPU available. Training will be performed on CPU, which may be slow.")
        device = "cpu"

    # Prepare datasets
    logger.info("Preparing datasets...")
    final_train_dataset, final_validation_dataset = prepare_datasets(dataset)

    if args.train_sample:
        logger.info("Running a quick training sample for testing the pipeline...")
        final_train_dataset = final_train_dataset.select(range(10))
        final_validation_dataset = final_validation_dataset.select(range(2))

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        final_train_dataset,
        final_validation_dataset,
        args.batch_size,
        args.num_workers,
        eot_token_id,
        device,
    )

    # Instantiate model
    logger.info("Instantiating model...")
    model = LanguageModel(vocab_size=vocab_size, embedding_dimension=args.embedding_dimension, hidden_dimension=args.hidden_dimension)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}")

    # Initialize wandb
    logger.info("Initializing wandb...")
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "epochs": args.epochs,
            "dataset_name": args.dataset_name,
            "tokenizer_name": args.tokenizer_name,
            "embedding_dimension": args.embedding_dimension,
            "hidden_dimension": args.hidden_dimension,
            "total_params": total_params,
            "vocab_size": vocab_size,
            "sequence_len": -1,
            "sequence_overlap_ratio": -1,
            "train_data_points": len(train_loader),
            "val_data_points": len(val_loader),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model": args.model_name,
        },
    )

    checkpoint_dir = args.checkpoint_dir + "/" + str(uuid.uuid4())
    # Train the model
    logger.info("Starting training...")
    train_model(
        model,
        train_loader,
        val_loader,
        tokenizer,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=device,
        checkpoint_dir=checkpoint_dir,
        validation_prompt=args.validation_prompt,
        max_new_tokens=args.max_new_tokens,
    )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "model.bin")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    wandb.finish()
    logger.info("Training completed.")

    # Cleanup wandb
    cleanup_wandb()

    # Upload model to Hugging Face
    upload_model_to_hf(checkpoint_dir, args.model_name, args.dataset_name, args.tokenizer_name, tokenizer, vocab_size, args.embedding_dimension, args.hidden_dimension, args.epochs, args.batch_size, args.learning_rate)

    # Cleanup local model artifacts
    cleanup_model_artifacts(checkpoint_dir)


if __name__ == "__main__":
    main()