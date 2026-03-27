# Language Model Training Pipeline

A modular pipeline for training language models with custom tokenizers.

## Project Structure

- `main.py`: Main training script
- `config.py`: Configuration and hyperparameters
- `models/`: Model architectures
- `data/`: Data loading and preprocessing
- `training/`: Training loops and utilities
- `tokenization/`: Tokenizer training pipeline
- `inference/`: Scripts for testing trained models and tokenizers

## Training a Model

1. Train a tokenizer:
```bash
python tokenization/main_tokenizer.py --tokenizer-type byte_level_bpe
```

2. Train the model:
```bash
python main.py
```

## Supported Tokenizer Types

- `byte_level_bpe`: Byte-level BPE (default)
- `char_bpe`: Character-level BPE
- `bpe`: Word-level BPE
- `word_level`: Word-level tokenizer

## Inference

See `inference/README.md` for testing trained models and tokenizers.