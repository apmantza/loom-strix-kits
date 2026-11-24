#!/bin/bash
# Serve a package for a measurement, one server at a time, with the kit's flags (the configuration the kit was measured
# with: a harness default is a different configuration, e.g. elastic backing where the kit serves fixed). Stops our servers
# and any managed one (LSK_STOP_HOOK) first, waits for a quiet machine, and fails closed: exit 1 unless $PORT answers AND
# serves exactly PKG (a stale server answering /v1/models once made an A/B compare a package with itself).
# usage: serve_as.sh KIT PKG          serve_as.sh KIT --restore   (stop ours; LSK_RESTORE_HOOK brings the managed server back)
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: serve_as.sh KIT PKG | --restore}"
if [ "${2:-}" = --restore ]; then lsk_restore && echo "restored $(date +%T)"; exit; fi
p=$(lsk_pkg "${2:?package dir}")
lsk_stop || exit 1
bash "$LSK/common/wait_quiet.sh" | tail -1; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "serve_as: not quiet / memory not admitted (wait_quiet above); not serving"; exit 1; }
lsk_serve_bg bench "$p"
echo "serving $p ($KIT_SERVE_ARGS) $(date +%T)"
