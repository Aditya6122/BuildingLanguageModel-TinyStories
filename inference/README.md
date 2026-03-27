# Inference Scripts

This directory contains scripts for testing trained models and tokenizers downloaded from Hugging Face.

## Tokenizer Inference

Test a tokenizer from Hugging Face:

```bash
python inference/tokenizer_inference.py --repo aditya-6122/tinystories-tokenizer-vb-4000-byte_level_bpe-v1
```

For interactive testing:
```bash
python inference/tokenizer_inference.py --repo aditya-6122/tinystories-tokenizer-vb-4000-byte_level_bpe-v1 --interactive
```

## Model Inference

Generate text with a trained model:

```bash
python inference/model_inference.py \
  --model-repo aditya-6122/tinystories-model \
  --tokenizer-repo aditya-6122/tinystories-tokenizer-vb-4000-byte_level_bpe-v1 \
  --prompt "Once upon a time" \
  --max-tokens 100
```

For interactive generation:
```bash
python inference/model_inference.py \
  --model-repo aditya-6122/tinystories-model \
  --tokenizer-repo aditya-6122/tinystories-tokenizer-vb-4000-byte_level_bpe-v1 \
  --interactive
```