#!/bin/bash
# What serves on $PORT. Read-only: no request reaches the model.
# usage: state.sh KIT            one line: package, pid, backing, phase, active, queued requests
#        state.sh KIT --pkg      only the served model.json; exit 1 when nothing serves
#        state.sh KIT --idle     exit 0 only when a server answers with 0 active and 0 queued requests
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: state.sh KIT [--pkg|--idle]}"
pkg=$(lsk_served) || die "nothing serves on :$PORT"
[ "${2:-}" = --pkg ] && { echo "$pkg"; exit 0; }
st=$(curl -s -m 5 127.0.0.1:$PORT/v1/status | python3 -c "import json,sys; s=json.load(sys.stdin)['data'][0]['loom']['status']; print(s['phase'], s['active_requests'], s['queued_requests'])") \
  || die "$pkg is served but /v1/status did not answer"
read -r phase act q <<< "$st"
if [ "${2:-}" = --idle ]; then [ "$act" = 0 ] && [ "$q" = 0 ]; exit; fi
pid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
echo "package $pkg pid $pid backing $(tr '\0' '\n' < /proc/$pid/cmdline | sed -n 's/^--pool_backing=//p') phase $phase active $act queued $q"
