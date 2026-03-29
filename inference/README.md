# Inference Scripts

This directory contains scripts for testing trained models downloaded from Hugging Face.

## Model Inference

Generate text with a trained model:

```bash
python inference/model_inference.py \
  --model-repo aditya-6122/tiny-stories-vb-17831-cbpe-v1 \
  --start-text "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9 \
  --stream
```

### Arguments

- `--model-repo`: Hugging Face model repository (default: aditya-6122/tiny-stories-vb-17831-cbpe-v1)
- `--start-text`: Starting text for generation (default: "Once there was a")
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 1000)
- `--temperature`: Sampling temperature (0 for greedy, higher for more random, default: 0.2)
- `--top-k`: Top-k sampling (optional, default: None)
- `--top-p`: Top-p (nucleus) sampling (optional, default: None)
- `--stream`: Stream output token by token (default: enabled)
- `--no-stream`: Disable streaming, output all at once

For interactive generation or custom prompts, modify the arguments as needed.