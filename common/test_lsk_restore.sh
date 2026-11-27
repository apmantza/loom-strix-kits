#!/bin/bash
# lsk_restore (env.sh) and the restore line of its callers. Recurrence (HRX-like 11-26 20:56, q214): a step killed by SIGTERM ran its
# EXIT trap `lsk_restore; echo "live restored"`; the restore hook's output went to /dev/null, nothing answered on the port, and the
# log said 'live restored' while live stayed down until a manual restore. Shapes covered (each case names the one it prevents):
#   release : the stopped server closes its port at once but holds its resources (GPU memory) a few seconds more; a hook started in
#             that gap fails (the managed start is refused) and nothing retries. lsk_stop waits for the stopped pids to exit.
#   guard   : LSK_GUARD refuses the stop (the window step died before the trap): the guard's reason reaches the one loud line.
#   refused : the hook itself fails (HRX-like live_hook.sh exits 3 when no window step is among the callers); its reason was discarded.
#   silent  : the hook returns 0 and nothing comes up: one 'LIVE NOT RESTORED' line within LSK_RESTORE_WAIT, never 'restored'.
#   stuck   : a stopped pid outlives LSK_EXIT_WAIT: the hook is not called.
#   caller  : events_ab.sh's EXIT trap prints 'restored' only when lsk_restore succeeded.
# Network-isolated (re-execs itself under `unshare -rn`, loopback only); hooks are stubs that log to $T/hook.log; no real hook runs.
# usage: test_lsk_restore.sh        ENV_SH=FILE / EVENTS_AB=FILE test other copies (mutants); exit 0 when every check passes
set -u
H=$(cd "$(dirname "$0")" && pwd)
if [ "${LSK_RESTORE_TEST_NS:-}" != 1 ]; then exec unshare -rn env LSK_RESTORE_TEST_NS=1 bash "$0" "$@"; fi
ip link set lo up || { echo "FAIL: no loopback in the test's network namespace"; exit 1; }
[ "$(ip -o link show | grep -vc ': lo:')" = 0 ] || { echo "FAIL: not network-isolated"; exit 1; }
SUT=${ENV_SH:-$H/env.sh}; EAB=${EVENTS_AB:-$H/events_ab.sh}
T=$(mktemp -d); trap 'pkill -P $$ 2>/dev/null; [ -f "$T/live.pid" ] && kill "$(cat "$T/live.pid")" 2>/dev/null; rm -rf "$T"' EXIT; fails=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', want '$3'"; fails=$((fails + 1)); fi; }
has() { grep -qF -- "$2" "$1" && echo yes || echo no; }
starts() { cat "$T/hook.log" 2>/dev/null | grep -c '^start'; }   # restore-hook calls in this case (0 when no hook ran at all)

# a server answering /v1/models like loom-serve; with HOLD=N, SIGTERM closes the listener at once and the process exits N s later
cat > "$T/fake_server.py" <<'PY'
import http.server, os, signal, sys, threading, time
port, hold = int(sys.argv[1]), float(os.environ.get("HOLD", "0"))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = b'{"object":"list","data":[]}'; self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", port), H)
def term(*_):
    threading.Thread(target=s.shutdown).start(); s.server_close(); time.sleep(hold); os._exit(0)
signal.signal(signal.SIGTERM, term)
s.serve_forever()
PY
# the managed start: refused while the old server's resources are held (its pid alive), like a live start beside a resident model
cat > "$T/hook_start.sh" <<'SH'
echo "start $(date +%s.%N)" >> "$T/hook.log"
o=$(cat "$T/old.pid" 2>/dev/null); s=$(ps -o stat= -p "$o" 2>/dev/null)
if [ -n "$o" ] && [ -n "$s" ] && [ "${s:0:1}" != Z ]; then echo '{"error":{"message":"engine exited (out of memory)"}}'; exit 0; fi
setsid python3 "$T/fake_server.py" "$PORT" < /dev/null > /dev/null 2>&1 & echo $! > "$T/live.pid"; echo '{"started":"live"}'
SH
PORT=18160
# one case: fresh state, an old server (pid file) unless NOOLD=1, then lsk_restore in a child shell with the given env
run() {   # NAME [VAR=VALUE ...]
  local n=$1; shift
  [ -f "$T/live.pid" ] && kill "$(cat "$T/live.pid")" 2>/dev/null; rm -f "$T/live.pid" "$T/old.pid"; [ -f "$T/hook.log" ] && cat "$T/hook.log" >> "$T/all_hooks.log"; rm -f "$T/hook.log"; rm -rf "$T/state"; mkdir -p "$T/state"
  for i in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || break; sleep 0.1; done
  if [ "${NOOLD:-0}" != 1 ]; then
    HOLD=${HOLD:-4} python3 "$T/fake_server.py" $PORT & echo $! > "$T/state/prof.pid"; cp "$T/state/prof.pid" "$T/old.pid"
    for i in $(seq 1 50); do curl -s -m 1 127.0.0.1:$PORT/v1/models | grep -q list && break; sleep 0.1; done
  fi
  t0=$(date +%s)
  env T="$T" PORT=$PORT STATE="$T/state" LSK_GUARD= LSK_STOP_HOOK="echo stop >> $T/hook.log" LSK_RESTORE_HOOK="bash $T/hook_start.sh" \
    LSK_EXIT_WAIT=${LSK_EXIT_WAIT:-30} LSK_RESTORE_WAIT=${LSK_RESTORE_WAIT:-20} "$@" \
    bash -c '. "$1"; lsk_restore' _ "$SUT" > "$T/$n.out" 2> "$T/$n.err"
  echo $? > "$T/$n.rc"; echo $(( $(date +%s) - t0 )) > "$T/$n.s"
}
answers() { curl -s -m 2 127.0.0.1:$PORT/v1/models | grep -q '"object":"list"' && echo yes || echo no; }

# release: old server lingers 4 s after closing its port
run release
check "release: lsk_restore exit 0" "$(cat "$T/release.rc")" 0
check "release: the port answers after it" "$(answers)" yes
check "release: the hook ran once" "$(starts)" 1
check "release: no failure line" "$(has "$T/release.err" 'LIVE NOT RESTORED')" no

# refused: the hook exits 3 with a reason on stderr (live_hook.sh's refusal)
run refused LSK_RESTORE_HOOK="bash -c 'echo \"live_hook: REFUSED start: no window step among the callers\" >&2; exit 3'"
check "refused: exit 1" "$(cat "$T/refused.rc")" 1
check "refused: one LIVE NOT RESTORED line" "$(grep -c '^LIVE NOT RESTORED: ' "$T/refused.err")" 1
check "refused: the line carries the hook's reason" "$(has "$T/refused.err" 'REFUSED start: no window step')" yes
check "refused: and its exit status" "$(has "$T/refused.err" 'exit 3')" yes
check "refused: no wait for an answer after a failed hook (< 10 s)" "$([ "$(cat "$T/refused.s")" -lt 10 ] && echo yes || echo no)" yes

# guard: the stop's guard refuses (the window died before the step's trap ran): one line, carrying the guard's own reason
run guard LSK_GUARD="bash -c 'echo \"live_hook: REFUSED check: no window step among the callers\" >&2; exit 3'"
check "guard: exit 1" "$(cat "$T/guard.rc")" 1
check "guard: stderr is one line" "$(wc -l < "$T/guard.err")" 1
check "guard: it is the LIVE NOT RESTORED line with the guard's reason" "$(grep -c '^LIVE NOT RESTORED: .*REFUSED check: no window step' "$T/guard.err")" 1
check "guard: the hook was not called" "$(starts)" 0
kill "$(cat "$T/old.pid")" 2>/dev/null; wait 2>/dev/null

# silent: the hook says ok and nothing comes up
NOOLD=1 run silent LSK_RESTORE_WAIT=3 LSK_RESTORE_HOOK='echo "{\"started\":\"x\"}"'
check "silent: exit 1" "$(cat "$T/silent.rc")" 1
check "silent: one LIVE NOT RESTORED line naming the port" "$(grep -c "^LIVE NOT RESTORED: .*nothing answers on $PORT" "$T/silent.err")" 1
check "silent: bounded by LSK_RESTORE_WAIT (< 12 s)" "$([ "$(cat "$T/silent.s")" -lt 12 ] && echo yes || echo no)" yes

# stuck: the old server outlives LSK_EXIT_WAIT
HOLD=8 run stuck LSK_EXIT_WAIT=2
check "stuck: exit 1" "$(cat "$T/stuck.rc")" 1
check "stuck: the hook was not called" "$(starts)" 0
check "stuck: one LIVE NOT RESTORED line naming the pid" "$(grep -c "^LIVE NOT RESTORED: .*$(cat "$T/old.pid")" "$T/stuck.err")" 1
kill "$(cat "$T/old.pid")" 2>/dev/null; wait 2>/dev/null

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
