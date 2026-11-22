#!/bin/bash
# loom-skills' quick gates (loom-benchmark/scripts/gates.sh: outputs vs llama.cpp on the kit's reference GGUF, chat, tools,
# vision, concurrency) on a package served with the kit's flags. Exit status = the gates' verdict.
# usage: gates.sh KIT PKG OUT_DIR        machine env: LOOM_SKILLS, LLAMACPP_DIR (llama-server for the reference outputs)
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: gates.sh KIT PKG OUT_DIR}"
p=$(lsk_pkg "${2:?package}"); OUT=$(realpath -m "${3:?out dir}"); mkdir -p "$OUT"
[ -n "${LOOM_SKILLS:-}" ] && [ -f "$LOOM_SKILLS/skills/loom-benchmark/scripts/gates.sh" ] || die "set LOOM_SKILLS to a loom-skills checkout"
[ -n "${LLAMACPP_DIR:-}" ] || die "set LLAMACPP_DIR (llama-server for the reference outputs)"
G=$(realpath "$KIT_REF_GGUF") || die "no reference GGUF $KIT_REF_GGUF"
trap 'bash $LSK/common/serve_as.sh $KIT --restore' EXIT
bash "$LSK/common/serve_as.sh" "$KIT" "$p" || exit 1
# the package names its vision projector without a path; resolve it against the package's parameter/ links
mm=$(python3 -c "import json,sys; print((json.load(open(sys.argv[1])).get('vision') or {}).get('mmproj', ''))" "$p/llm_config.json")
case "$mm" in ""|/*) ;; *) export MMPROJ; MMPROJ=$(readlink -f "$p/parameter/$mm"); [ -f "$MMPROJ" ] || die "$mm not in $p/parameter";; esac
export LLAMACPP_DIR VENV
LEVEL=quick bash "$LOOM_SKILLS/skills/loom-benchmark/scripts/gates.sh" "$p" "http://127.0.0.1:$PORT" "$G" "$OUT" > "$OUT/gates_run.log" 2>&1; rc=$?
tail -12 "$OUT/gates_run.log"; echo "gates exit $rc"; exit $rc
