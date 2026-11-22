#!/bin/bash
# Serve a package in the foreground with the kit's flags (OpenAI-compatible API on 127.0.0.1:$PORT).
# usage: serve.sh KIT PKG [extra loom-serve args]          needs unlimited memlock (io_uring pinning)
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: serve.sh KIT PKG [args]}"
p=$(lsk_pkg "${2:?package dir}"); shift 2
[ "$(ulimit -l)" = unlimited ] || ulimit -l unlimited 2>/dev/null || die "memlock is $(ulimit -l) KiB; loom-serve needs unlimited"
read -ra sa <<< "$KIT_SERVE_ARGS"
exec "$LOOM_SERVE" --listen=127.0.0.1:$PORT --model="$p/model.json" "${sa[@]}" "$@"
