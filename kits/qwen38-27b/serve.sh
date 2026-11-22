#!/bin/bash
# Serve a package built by build.sh (OpenAI-compatible API).  usage: serve.sh PACKAGE_DIR [loom-serve args...]
# Fixed pool backing: KV8 is only correct with it, and elastic costs prefill on Strix Halo. Needs unlimited memlock
# (io_uring pinning): `ulimit -l unlimited`, or DefaultLimitMEMLOCK=infinity in systemd. Resident ~53 GB.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: serve.sh PACKAGE_DIR [loom-serve args...]"; exit 1; }
P=$1; shift
[ "$(ulimit -l)" = unlimited ] || ulimit -l unlimited || { echo "memlock is $(ulimit -l) KiB; loom-serve needs unlimited"; exit 1; }
exec "${LOOM_SERVE:-loom-serve}" --device='amdgpu://?completion_spin_ns=4294967295' --listen="${LISTEN:-127.0.0.1:8160}" \
  --pool_backing=fixed --workers=8 --model="$P/model.json" "$@"
