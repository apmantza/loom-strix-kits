#!/bin/bash
# Build a loom-serve package with this kit's current recipe.
# usage: [EXTRA="SWITCH=1 ..."] build.sh PACKAGE_DIR
#   needs: python3 with gguf, numpy, tokenizers; hrx-loom 0.2.0 (`pip install hrx-loom==0.2.0`);
#   LOOM_KB_ROOT = skills/loom-kb of a loom-skills checkout (tested with loom-fun/loom-skills 7952d7f);
#   ./fetch.sh run first. The package is ~programs + model.json; it references the weights in ./models.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -eq 1 ] || { echo "usage: [EXTRA=...] build.sh PACKAGE_DIR"; exit 1; }
: "${LOOM_KB_ROOT:?set LOOM_KB_ROOT to skills/loom-kb of a loom-skills checkout}"
RECIPE="PREFILL_F16_SCRATCH=1 ATTN_L17_VS=1 FLASH2_PF11=1 FLASH2_PF11_SPLIT=1 FLASH2_PF11_NOIF=1 FLASH2_PF11_CARRY=1 FLASH2_KVH11=1 KV8=1 KV8_LAYOUT=2 GDN_FOLD=1 FLASH2_VS_KVH=1 FLASH2_KV8_KCARRY=1 FLASH2_KV8_VLATE=2 FLASH2_SPLITK_ANY=1 DFL_SPLIT_W=1 PF16_PAD=64 GDN_AB_R=1 DRAFT_MM_FROM=17"
env $RECIPE ${EXTRA:-} "${PYTHON:-python3}" kit/build.py config.json "$1" --tokenizer tok/tokenizer.json --model-id Qwen/Qwen3.8-27B \
  --pool 524544 --rows 4 --prefill 4096 --temperature 1.0 --top-p 0.95 --top-k 20
