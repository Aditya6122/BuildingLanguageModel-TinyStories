"""
Model inference script.

Downloads a trained model from Hugging Face and provides text generation functionality.
"""

import logging
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from config import HF_TOKEN

logger = logging.getLogger(__name__)


def load_model_from_hub(model_repo, model_file='pytorch_model.bin', repo_type='model', vocab_size=4000):
    """
    Load a PyTorch model from Hugging Face Hub.

    Args:
        model_repo (str): Repository name (e.g., 'user/repo-name').
        model_file (str): Model filename.
        repo_type (str): Type of repository.
        vocab_size (int): Vocabulary size for the model.

    Returns:
        torch.nn.Module: Loaded model.
    """
    logger.info(f"Downloading model from {model_repo}")
    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_file,
        repo_type=repo_type,
        token=HF_TOKEN
    )

    # Note: This assumes the model is saved with torch.save(model.state_dict(), ...)
    # You might need to adjust based on how the model was saved
    from models.language_model import LanguageModel
    model = LanguageModel(vocab_size=vocab_size)  # Adjust vocab_size as needed
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    logger.info("Model loaded successfully")
    return model


def load_tokenizer_for_model(tokenizer_repo, file_name='bpe_tokenizer.json'):
    """
    Load tokenizer for the model.

    Args:
        tokenizer_repo (str): Tokenizer repository.
        file_name (str): Tokenizer filename.

    Returns:
        Tokenizer: Loaded tokenizer.
    """
    from tokenizers import Tokenizer
    file_path = hf_hub_download(
        repo_id=tokenizer_repo,
        filename=file_name,
        repo_type="model",
        token=HF_TOKEN
    )
    tokenizer = Tokenizer.from_file(file_path)
    return tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=50, device='cpu'):
    """
    Generate text using the trained model.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt (str): Input prompt.
        max_new_tokens (int): Maximum tokens to generate.
        device (str): Device to run on.

    Returns:
        str: Generated text.
    """
    model.to(device)

    # Encode prompt
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    logger.info(f"Generating text for prompt: {prompt}")

    with torch.no_grad():
        generated_ids = model.generate(input_tensor, max_new_tokens=max_new_tokens)[0]

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_ids.tolist())
    full_text = prompt + " " + generated_text

    logger.info(f"Generated text: {full_text}")
    return full_text


def interactive_generation(model, tokenizer, device='cpu'):
    """
    Interactive text generation.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        device (str): Device to run on.
    """
    print("Interactive Text Generation")
    print("Enter a prompt (or 'quit' to exit):")

    while True:
        prompt = input("> ")
        if prompt.lower() == 'quit':
            break

        generated = generate_text(model, tokenizer, prompt, device=device)
        print(f"Generated: {generated}")
        print()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Generate text with trained model from Hugging Face')
    parser.add_argument('--model-repo', type=str, required=True, help='Model repository (e.g., user/repo-name)')
    parser.add_argument('--tokenizer-repo', type=str, required=True, help='Tokenizer repository')
    parser.add_argument('--vocab-size', type=int, default=4000, help='Vocabulary size of the model')
    parser.add_argument('--model-file', type=str, default='pytorch_model.bin', help='Model filename')
    parser.add_argument('--tokenizer-file', type=str, default='bpe_tokenizer.json', help='Tokenizer filename')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Generation prompt')
    parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens to generate')
    parser.add_argument('--interactive', action='store_true', help='Run interactive generation')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu/cuda)')

    args = parser.parse_args()

    model = load_model_from_hub(args.model_repo, args.model_file, vocab_size=args.vocab_size)
    tokenizer = load_tokenizer_for_model(args.tokenizer_repo, args.tokenizer_file)

    if args.interactive:
        interactive_generation(model, tokenizer, args.device)
    else:
        generate_text(model, tokenizer, args.prompt, args.max_tokens, args.device)