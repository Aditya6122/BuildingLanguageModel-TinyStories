"""
Training module for the language model.

This module contains the training loop, validation, and generation functions.
"""

import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import os

logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, tokenizer, learning_rate=1e-3, epochs=10, device="cuda", run_name="LM"):
    """
    Train the language model with validation and logging.

    Args:
        model: The language model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        tokenizer: Tokenizer for text generation.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.
        device (str): Device to run training on.
        run_name (str): Name for the run (for logging).
    """
    logger.info(f"Starting training for {epochs} epochs on {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    last_val_train_diff = 0

    # Create directory for checkpoints
    os.makedirs("./model_artifacts", exist_ok=True)

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs} started")
        # ===================== TRAIN =====================
        model.train()
        total_train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch_num, (x_batch, y_batch) in enumerate(pbar, 1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _, loss = model(x_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            wandb.log({
                "train/batch_loss": loss.item(),
                "train/step": (epoch * len(train_loader)) + batch_num
            })

            avg_so_far = total_train_loss / batch_num
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_so_far:.4f}"
            })

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} training completed. Avg loss: {avg_train_loss:.4f}")

        # SAVE PER EPOCH MODEL CHECKPOINT
        checkpoint_path = f"./model_artifacts/pytorch_model_checkpoint_epoch_{epoch+1}.bin"
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Model checkpoint saved: {checkpoint_path}")

        # ===================== VALIDATION =====================
        model.eval()
        total_val_loss = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

        with torch.no_grad():
            for batch_num, (x_batch, y_batch) in enumerate(pbar, 1):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits, _, loss = model(x_batch, y_batch)
                total_val_loss += loss.item()

                avg_val_so_far = total_val_loss / batch_num
                pbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "avg_val": f"{avg_val_so_far:.4f}"
                })

            # Generate sample
            input_word = "Once upon a time"
            input_token_id = tokenizer.encode(input_word).ids

            input_tensor = torch.tensor([input_token_id]).to(device)
            output_tokens = model.generate(input_tensor, max_new_tokens=100)

            output_words = input_word + " " + tokenizer.decode(output_tokens[0].tolist())
            logger.info(f"Generated Story: {output_words}")

        avg_val_loss = total_val_loss / len(val_loader)
        last_val_train_diff = avg_val_loss - avg_train_loss

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/train difference": last_val_train_diff
        })

        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    logger.info("Training completed")
    wandb.finish()