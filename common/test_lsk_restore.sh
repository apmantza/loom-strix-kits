#!/bin/bash
# lsk_restore / lsk_stop (env.sh) and the restore line of its callers. Recurrence (HRX-like 11-26 20:56, q214): a step killed by
# SIGTERM ran its EXIT trap `lsk_restore; echo "live restored"`; the restore hook's output went to /dev/null (live_hook.sh had refused
# the start: its window.sh was gone), nothing answered on the port, and the log said 'live restored' while live stayed down until a
# manual restore. Shapes covered (each case names the one it prevents):
#   ok      : a stopped server of ours (pid file), the hook starts the managed server: exit 0, the port answers, no failure line.
#   guard   : LSK_GUARD refuses the stop (the window step died before the trap): the guard's reason reaches the one loud line.
#   refused : the hook itself fails (live_hook.sh exits 3 when no window step is among the callers); its reason was discarded.
#   silent  : the hook returns 0 and nothing comes up: one 'LIVE NOT RESTORED' line within LSK_RESTORE_WAIT, never 'restored'.
#   busy    : something we do not stop keeps the port: lsk_stop gives up after its 60 s wait and the hook is not called.
#   stale   : a pid file names a reused pid of an unrelated program (11-27 review probe G killed a `sleep`): left running, reported.
#   caller  : events_ab.sh's EXIT trap prints 'restored' only when lsk_restore succeeded.
# Network-isolated (re-execs itself under `unshare -rn`, loopback only); hooks are stubs that log to $T/hook.log; no real hook runs.
# usage: test_lsk_restore.sh        ENV_SH=FILE / EVENTS_AB=FILE test other copies (mutants); exit 0 when every check passes
set -u
H=$(cd "$(dirname "$0")" && pwd)
if [ "${LSK_RESTORE_TEST_NS:-}" != 1 ]; then exec unshare -rn env LSK_RESTORE_TEST_NS=1 bash "$0" "$@"; fi
ip link set lo up || { echo "FAIL: no loopback in the test's network namespace"; exit 1; }
[ "$(ip -o link show | grep -vc ': lo:')" = 0 ] || { echo "FAIL: not network-isolated"; exit 1; }
SUT=${ENV_SH:-$H/env.sh}; EAB=${EVENTS_AB:-$H/events_ab.sh}
T=$(mktemp -d); trap 'pkill -P $$ 2>/dev/null; for f in "$T"/*.kill; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done; rm -rf "$T"' EXIT; fails=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', want '$3'"; fails=$((fails + 1)); fi; }
has() { grep -qF -- "$2" "$1" 2>/dev/null && echo yes || echo no; }
starts() { cat "$T/hook.log" 2>/dev/null | grep -c '^start'; }   # restore-hook calls in this case (0 when no hook ran at all)
alive() { kill -0 "$1" 2>/dev/null && [ "$(ps -o stat= -p "$1" | cut -c1)" != Z ] && echo alive || echo gone; }

# a server answering /v1/models, run through $T/bin/loom-serve (a link to python3) so its comm is loom-serve, as lsk_serve_bg's
# exec'd server; `srv PORT` starts one in the background
mkdir -p "$T/bin"; ln -s "$(command -v python3)" "$T/bin/loom-serve"; cat > "$T/fake_server.py" <<'PY'
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = b'{"object":"list","data":[]}'; self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
cat > "$T/hook_start.sh" <<'SH'
echo "start" >> "$T/hook.log"
setsid "$T/bin/loom-serve" "$T/fake_server.py" "$PORT" < /dev/null > /dev/null 2>&1 & echo $! > "$T/live.kill"; echo '{"started":"live"}'
SH
PORT=18160
waitport() { for i in $(seq 1 50); do curl -s -m 1 127.0.0.1:$PORT/v1/models | grep -q list && return 0; sleep 0.1; done; return 1; }
freeport() { for i in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || return 0; sleep 0.1; done; return 1; }
# one case: fresh state, a server of ours on the port (pid file) unless NOOLD=1, then lsk_restore in a child shell with the given env
run() {   # NAME [VAR=VALUE ...]
  local n=$1; shift
  [ -f "$T/live.kill" ] && kill "$(cat "$T/live.kill")" 2>/dev/null; rm -f "$T/live.kill"
  [ -f "$T/hook.log" ] && cat "$T/hook.log" >> "$T/all_hooks.log"; rm -f "$T/hook.log"; rm -rf "$T/state"; mkdir -p "$T/state"
  [ "${KEEPPORT:-0}" = 1 ] || freeport
  if [ "${NOOLD:-0}" != 1 ]; then "$T/bin/loom-serve" "$T/fake_server.py" $PORT & echo $! > "$T/state/prof.pid"; cp "$T/state/prof.pid" "$T/old.pid"; waitport; fi
  t0=$(date +%s)
  env T="$T" PORT=$PORT STATE="$T/state" LOOM_SERVE="$T/bin/loom-serve" LSK_GUARD= LSK_STOP_HOOK="echo stop >> $T/hook.log" \
    LSK_RESTORE_HOOK="bash $T/hook_start.sh" LSK_RESTORE_WAIT=${LSK_RESTORE_WAIT:-20} "$@" \
    bash -c '. "$1"; lsk_restore' _ "$SUT" > "$T/$n.out" 2> "$T/$n.err"
  echo $? > "$T/$n.rc"; echo $(( $(date +%s) - t0 )) > "$T/$n.s"
}
answers() { curl -s -m 2 127.0.0.1:$PORT/v1/models | grep -q '"object":"list"' && echo yes || echo no; }

# ok: our server is stopped, the hook brings the managed one up
run ok
check "ok: lsk_restore exit 0" "$(cat "$T/ok.rc")" 0
check "ok: our server (pid file) was stopped" "$(alive "$(cat "$T/old.pid")")" gone
check "ok: the port answers after it" "$(answers)" yes
check "ok: the hook ran once" "$(starts)" 1
check "ok: no failure line" "$(has "$T/ok.err" 'LIVE NOT RESTORED')" no
check "ok: the hook's output is kept" "$(has "$T/state/restore_hook.log" '"started":"live"')" yes

# guard: the stop's guard refuses (the window died before the step's trap ran): one line, carrying the guard's own reason
run guard LSK_GUARD="bash -c 'echo \"live_hook: REFUSED check: no window step among the callers\" >&2; exit 3'"
check "guard: exit 1" "$(cat "$T/guard.rc")" 1
check "guard: stderr is one line" "$(wc -l < "$T/guard.err")" 1
check "guard: it is the LIVE NOT RESTORED line with the guard's reason" "$(grep -c '^LIVE NOT RESTORED: .*REFUSED check: no window step' "$T/guard.err")" 1
check "guard: the hook was not called" "$(starts)" 0
kill "$(cat "$T/old.pid")" 2>/dev/null; wait 2>/dev/null

# refused: the hook exits 3 with a reason on stderr (live_hook.sh's refusal)
run refused LSK_RESTORE_HOOK="bash -c 'echo \"live_hook: REFUSED start: no window step among the callers\" >&2; exit 3'"
check "refused: exit 1" "$(cat "$T/refused.rc")" 1
check "refused: one LIVE NOT RESTORED line" "$(grep -c '^LIVE NOT RESTORED: ' "$T/refused.err")" 1
check "refused: the line carries the hook's reason" "$(has "$T/refused.err" 'REFUSED start: no window step')" yes
check "refused: and its exit status" "$(has "$T/refused.err" 'exit 3')" yes
check "refused: no wait for an answer after a failed hook (< 10 s)" "$([ "$(cat "$T/refused.s")" -lt 10 ] && echo yes || echo no)" yes

# silent: the hook says ok and nothing comes up
NOOLD=1 run silent LSK_RESTORE_WAIT=3 LSK_RESTORE_HOOK='echo "{\"started\":\"x\"}"'
check "silent: exit 1" "$(cat "$T/silent.rc")" 1
check "silent: one LIVE NOT RESTORED line naming the port" "$(grep -c "^LIVE NOT RESTORED: .*nothing answers on $PORT" "$T/silent.err")" 1
check "silent: bounded by LSK_RESTORE_WAIT (< 12 s)" "$([ "$(cat "$T/silent.s")" -lt 12 ] && echo yes || echo no)" yes

# busy: a listener no pid file names and no hook stops keeps the port (lsk_stop's port wait is 60 s)
freeport; "$T/bin/loom-serve" "$T/fake_server.py" $PORT & echo $! > "$T/busy.kill"; waitport
KEEPPORT=1 NOOLD=1 run busy
check "busy: exit 1" "$(cat "$T/busy.rc")" 1
check "busy: the hook was not called" "$(starts)" 0
check "busy: one LIVE NOT RESTORED line naming the busy port" "$(grep -c "^LIVE NOT RESTORED: .*port $PORT still busy" "$T/busy.err")" 1
kill "$(cat "$T/busy.kill")" 2>/dev/null; rm -f "$T/busy.kill"; wait 2>/dev/null

# stale: a pid file naming an unrelated program (a reused pid) is not ours to kill
sleep 300 & sp=$!; echo $sp > "$T/sleep.kill"
NOOLD=1 run stale_prep LSK_RESTORE_HOOK=true LSK_RESTORE_WAIT=1 > /dev/null   # (only resets the state dir and the hook log)
echo $sp > "$T/state/prof.pid"
env T="$T" PORT=$PORT STATE="$T/state" LOOM_SERVE="$T/bin/loom-serve" LSK_GUARD= LSK_STOP_HOOK= LSK_RESTORE_HOOK="bash $T/hook_start.sh" \
  LSK_RESTORE_WAIT=20 bash -c '. "$1"; lsk_restore' _ "$SUT" > "$T/stale.out" 2> "$T/stale.err"; echo $? > "$T/stale.rc"
check "stale: the unrelated process is left running" "$(alive $sp)" alive
check "stale: it is reported with its pid and program" "$(grep -c "lsk_stop: .*prof.pid names pid $sp (sleep), not a loom-serve" "$T/stale.err")" 1
check "stale: the stale pid file is removed" "$([ -f "$T/state/prof.pid" ] && echo kept || echo removed)" removed
check "stale: the restore goes on (exit 0)" "$(cat "$T/stale.rc")" 0
kill $sp 2>/dev/null

# caller: events_ab.sh exits through its trap after a failed restore (the guard refuses the stop): it must not print 'restored'
K=$T/kit; mkdir -p "$K/common" "$K/kits/k" "$K/work/pkg-a"; cp "$SUT" "$K/common/env.sh"; cp "$EAB" "$K/common/events_ab.sh"
echo '{}' > "$K/work/pkg-a/model.json"; printf 'KIT_MODEL_ID=m\nKIT_SERVE_ARGS=\n' > "$K/kits/k/kit.env"
printf 'WORK=%s\nSTATE=%s\nPORT=%s\nIREE_PROFILE=/bin/true\nLSK_GUARD=false\nLSK_RESTORE_HOOK="echo start >> %s"\n' \
  "$K/work" "$K/work/state" $PORT "$T/hook.log" > "$T/env"
touch "$T/all_hooks.log"; [ -f "$T/hook.log" ] && cat "$T/hook.log" >> "$T/all_hooks.log"; rm -f "$T/hook.log"
env -u WORK -u STATE LSK_ENV="$T/env" bash "$K/common/events_ab.sh" k out 1000 pkg-a > "$T/caller.out" 2>&1
check "caller: events_ab exits nonzero" "$([ $? -ne 0 ] && echo yes || echo no)" yes
check "caller: no 'restored' line after a failed restore" "$(grep -c '^restored ' "$T/caller.out")" 0
check "caller: the LIVE NOT RESTORED line is in its output" "$(grep -c '^LIVE NOT RESTORED: ' "$T/caller.out")" 1
[ -f "$T/hook.log" ] && cat "$T/hook.log" >> "$T/all_hooks.log"
check "stubs only: every hook call in the stub log is a stub's line" "$(grep -cv '^\(start\|stop\)\( \|$\)' "$T/all_hooks.log")" 0
check "stubs only: the stubs were called (the log is live)" "$([ -s "$T/all_hooks.log" ] && echo yes || echo no)" yes

[ $fails = 0 ] && echo "lsk_restore tests: all passed" || { echo "lsk_restore tests: $fails failed"; exit 1; }
