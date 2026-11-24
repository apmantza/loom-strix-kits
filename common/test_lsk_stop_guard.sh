#!/bin/bash
# lsk_stop's LSK_GUARD (E438: a dry run's lsk_stop killed the pid files of another run's profiling servers before its refused stop
# hook). Runs no server: pid files point at sleep processes this test starts; PORT is a port nothing listens on.
set -u
H=$(cd "$(dirname "$0")" && pwd); T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; fails=0
. "$H/env.sh"
STATE=$T; PORT=1; LSK_STOP_HOOK=""
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', want '$3'"; fails=$((fails + 1)); fi; }
sleep 300 & p=$!; echo $p > "$T/x.pid"
LSK_GUARD=false; lsk_stop 2>/dev/null; rc=$?
check "guard refuses: exit 1" "$rc" 1
check "guard refuses: process alive" "$(kill -0 $p 2>/dev/null && echo alive)" alive
check "guard refuses: pid file kept" "$([ -f "$T/x.pid" ] && echo kept)" kept
LSK_GUARD=true; lsk_stop 2>/dev/null; rc=$?; sleep 0.2
check "guard allows: exit 0" "$rc" 0
check "guard allows: process stopped" "$(kill -0 $p 2>/dev/null && echo alive || echo gone)" gone
sleep 300 & p=$!; echo $p > "$T/y.pid"; unset LSK_GUARD; lsk_stop 2>/dev/null; rc=$?; sleep 0.2
check "no guard (other machines): stops as before" "$rc/$(kill -0 $p 2>/dev/null && echo alive || echo gone)" 0/gone
kill $p 2>/dev/null
[ $fails = 0 ] && echo "lsk_stop guard tests: all passed" || { echo "lsk_stop guard tests: $fails failed"; exit 1; }
