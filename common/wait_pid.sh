#!/bin/bash
# Wait until PID exits, after checking that PID runs exactly ARGS (its `ps -o args=`). Exits 2 on an empty or dead PID
# and 3 on an args mismatch: a waiter that returns at once chains the next GPU job onto a running one (11-21: an awk
# field error gave an empty PID and the waiter "finished" immediately; AGENTS section 7: never `pgrep -f`).
# usage: wait_pid.sh PID "EXACT ARGS" [POLL_SECONDS]      e.g.  bash q31.sh & wait_pid.sh $! "bash q31.sh"
pid=${1:-}; want=${2:-}; poll=${3:-20}
[[ "$pid" =~ ^[0-9]+$ ]] || { echo "wait_pid: no PID given ('$pid')" >&2; exit 2; }
have=$(ps -o args= -p "$pid" 2>/dev/null) || { echo "wait_pid: PID $pid is not running" >&2; exit 2; }
[ "$have" = "$want" ] || { echo "wait_pid: PID $pid runs '$have', not '$want'" >&2; exit 3; }
# a finished child whose parent has not reaped it yet is a zombie: kill -0 still succeeds on it, so read its state too
while kill -0 "$pid" 2>/dev/null && [[ "$(ps -o stat= -p "$pid" 2>/dev/null)" != Z* ]]; do sleep "$poll"; done
