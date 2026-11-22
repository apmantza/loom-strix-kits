#!/bin/bash
# Download this kit's pinned weights and tokenizer (pins.json) into ./models and ./tok, or into WORK from the machine env,
# and verify every sha256. Needs the Hugging Face CLI (`pip install -U huggingface_hub`, command `hf`). ~19.7 GB.
exec python3 "$(dirname "$0")/../../common/fetch.py" qwen38-27b "$@"
