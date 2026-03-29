import logging
import json
import time
import argparse

import torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from config import HF_TOKEN
from models.language_model import LanguageModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model_from_hub(model_repo, model_file='pytorch_model.bin',  tokenizer_file="tokenizer.json", repo_type='model', model_config_file="model_config.json", device='cpu'):
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

    files = {
        "model_file":model_file,
        "model_config":model_config_file,
        "tokenizer_json":tokenizer_file
    }


    downloaded_paths = {}
    for file_name, file_path in files.items():
        path = hf_hub_download(
            repo_id=model_repo,
            filename=file_path,
            repo_type=repo_type,
            token=HF_TOKEN
        )
        downloaded_paths[file_name] = path

    tokenizer = Tokenizer.from_file(path=downloaded_paths["tokenizer_json"])

    with open(downloaded_paths["model_config"], 'r') as f:
        model_config = json.load(f)

    model = LanguageModel(**model_config)
    model.load_state_dict(torch.load(downloaded_paths["model_file"], map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text with a trained language model from Hugging Face Hub.')
    parser.add_argument('--model-repo', type=str, default='aditya-6122/tiny-stories-vb-17831-cbpe-v1', help='Hugging Face model repository (e.g., user/repo-name)')
    parser.add_argument('--start-text', type=str, default='Once there was a', help='Starting text for generation')
    parser.add_argument('--max-new-tokens', type=int, default=1000, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature (0 for greedy, higher for more random)')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling (optional)')
    parser.add_argument('--top-p', type=float, default=None, help='Top-p (nucleus) sampling (optional)')
    parser.add_argument('--stream', action='store_true', default=False, help='Stream output token by token (default: True)')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming, output all at once')

    args = parser.parse_args()

    # Handle stream flag
    stream = args.stream and not args.no_stream

    device = get_device()
    logger.info(f"Using device: {device}")
    
    model, tokenizer = load_model_from_hub(
        model_repo=args.model_repo,
        device=device
    )

    test_text = args.start_text
    start_time = time.time()

    encoded = tokenizer.encode(test_text)
    input_ids = torch.tensor(encoded.ids).unsqueeze(0).to(device)

    generated_text = test_text
    if stream:
        print(generated_text, end=" ", flush=True)

        for chunk in model.generate(
            input_ids,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stream=stream,
        ):
            generated_text += chunk
            print(chunk, end=" ", flush=True)

        print()  # newline after streaming
    else:
        generated_text += model.generate(
            input_ids,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stream=stream,
        )
        print(generated_text)

    logger.info(f"Generated text length: {len(generated_text)}")
    logger.info(f"Total generation time: {time.time() - start_time:.2f} seconds")