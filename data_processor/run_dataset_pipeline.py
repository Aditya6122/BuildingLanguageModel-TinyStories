"""
Run dataset processor script.

Usage: python -m data_processor.run_dataset_pipeline --dataset-name ...
"""

import argparse
import logging

from data_processor.dataset import process_and_upload_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Process raw dataset and push to Hugging Face Hub')
    parser.add_argument('--dataset-name', type=str, default="karpathy/tinystories-gpt4-clean", help='Source dataset name on Hugging Face')
    parser.add_argument('--tokenizer-name', type=str, default="aditya-6122/tinystories-tokenizer-vb-17783-char_bpe-v1-test", help='Tokenizer repo name on Hugging Face')
    parser.add_argument('--output-repo', type=str, default="aditya-6122/tinystories-custom-dataset", help='Destination processed dataset repo base name')
    parser.add_argument('--version', type=str, default='v1-test', help='Version suffix for output repo')
    parser.add_argument('--text-column', type=str, default='text', help='Text column in source dataset')
    parser.add_argument('--num-proc', type=int, default=8, help='Number of processes for dataset mapping')

    args = parser.parse_args()

    logger.info('Starting dataset processing pipeline')
    dataset_dict = process_and_upload_dataset(
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        dataset_repo_name=args.output_repo,
        version=args.version,
        text_column=args.text_column,
        num_proc=args.num_proc
    )

    logger.info('Dataset processing pipeline completed')
    logger.info(f"Processed dataset contains train shape {len(dataset_dict['train'])}, validation shape {len(dataset_dict['validation'])}")


if __name__ == '__main__':
    main()
