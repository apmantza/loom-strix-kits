#!/bin/bash
# Download the pinned weights into ./models and the tokenizer into ./tok, then verify their sha256.
# Needs the Hugging Face CLI (`pip install -U huggingface_hub`, command `hf`). ~19.7 GB.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models tok
UNSLOTH_REV=4af51b3542f775b0a8e4b310bfb3f17e01c306c9   # unsloth/Qwen3.8-27B-GGUF (the UD3.1 re-upload)
ZLAB_REV=2d9571f8ce46e151f61c6499c99dee6079e1d610      # z-lab/Qwen3.8-27B-DFlash2-GGUF
QWEN_REV=37561816e66b5b9153b1b64c10968c18c1e6f4a0      # Qwen/Qwen3.8-27B (tokenizer only)
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-F16.gguf --revision "$UNSLOTH_REV" --local-dir models
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf --revision "$ZLAB_REV" --local-dir models
hf download Qwen/Qwen3.8-27B tokenizer.json --revision "$QWEN_REV" --local-dir tok
sha256sum -c <<'EOF'
2a5f3565f39cf885103da591182c1266da64745737b42b452fd3e675369f1211  models/Qwen3.8-27B-UD-Q4_K_XL.gguf
cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e  models/mmproj-F16.gguf
1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd  models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3  tok/tokenizer.json
EOF
