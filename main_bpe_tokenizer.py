"""
Legacy tokenizer training script.

This script has been modularized. Use tokenization/main_tokenizer.py instead.
"""

import logging
from tokenization.main_tokenizer import main

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    main()
