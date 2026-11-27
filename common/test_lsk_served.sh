#!/bin/bash
# lsk_served's identity rule (env.sh) and lsk_serve_bg's check on it. The rule: the package a server serves is its --model's directory
# resolved (realpath, as lsk_pkg resolves a package) with the model.json leaf kept, because loom-serve resolves a model card's relative
# program paths against the --model path's own directory (hrx-system v0.2.0 serve/server/configuration.c:958), not the leaf's target.
# Recurrences (each case names the one it prevents):
#   symlink : HRX-like E685 / E687: pkg-gdnnoo's model.json is a symlink to pkg-e480's; lsk_served followed it and lsk_serve_bg
#             died with ':8160 serves pkg-e480/model.json, not pkg-gdnnoo/model.json' on a correct server, stopping q245.
#   real    : a package with a real model.json still passes (the 11-22 check's normal path).
#   wrong   : a server for A answers when B is expected: lsk_serve_bg must die (resolving both sides made HRX-like serve_as_live.sh
#             accept pkg-e480 for pkg-gdnnoo; the rule must not make this check vacuous).
#   gone    : an unresolvable --model or cwd (package or cwd deleted, model.json removed, no --model): exit 1 and no output, never a path
#             (the echo "$(realpath ...)/model.json" form printed '/model.json' with exit 0, which state.sh and window.sh read as served).
# Network-isolated (re-execs itself under `unshare -rn`, loopback only); the server is a local stub; no hook is set.
# usage: test_lsk_served.sh         ENV_SH=FILE tests another copy of env.sh (mutants); exit 0 when every check passes
set -u
H=$(cd "$(dirname "$0")" && pwd)
if [ "${LSK_SERVED_TEST_NS:-}" != 1 ]; then exec unshare -rn env LSK_SERVED_TEST_NS=1 bash "$0" "$@"; fi
ip link set lo up || { echo "FAIL: no loopback in the test's network namespace"; exit 1; }
[ "$(ip -o link show | grep -vc ': lo:')" = 0 ] || { echo "FAIL: not network-isolated"; exit 1; }
SUT=${ENV_SH:-$H/env.sh}
T=$(mktemp -d); fails=0
trap 'for f in "$T"/state/*.pid "$T"/*.kill; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done; rm -rf "$T"' EXIT
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', want '$3'"; fails=$((fails + 1)); fi; }
PORT=18161
mkdir -p "$T/bin" "$T/work/pkg-a" "$T/work/pkg-b" "$T/state"
echo '{"schema_version":1,"a":1}' > "$T/work/pkg-a/model.json"; ln -s "$T/work/pkg-a/model.json" "$T/work/pkg-b/model.json"
cat > "$T/bin/loom-serve" <<'EOF'
#!/usr/bin/env python3
import http.server, sys
port = int(next(x for x in sys.argv[1:] if x.startswith("--listen=")).rsplit(":", 1)[1])
class Q(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = b'{"object":"list","data":[]}'; self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *_): pass
http.server.HTTPServer(("127.0.0.1", port), Q).serve_forever()
EOF
chmod +x "$T/bin/loom-serve"
E=(env PORT=$PORT WORK="$T/work" STATE="$T/state" LOOM_SERVE="$T/bin/loom-serve" KIT_SERVE_ARGS=)
lsk() { "${E[@]}" bash -c '. "$1"; shift; "$@"' _ "$SUT" "$@"; }   # run one env.sh function in a child shell
waitport() { for i in $(seq 1 50); do curl -s -m 1 127.0.0.1:$PORT/v1/models | grep -q list && return 0; sleep 0.1; done; return 1; }
stopall() {
  for f in "$T"/state/*.pid "$T"/*.kill; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
  for i in $(seq 1 50); do (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || return 0; sleep 0.1; done
}
# a server started by hand: DIR (its cwd) and its --model argument as given
srv() { (cd "$1" && exec "$T/bin/loom-serve" --listen=127.0.0.1:$PORT --model="$2") & echo $! > "$T/srv.kill"; waitport; }

# symlink: B's model.json links to A's; serving B must pass and name B
lsk lsk_serve_bg sb "$T/work/pkg-b" > "$T/sym.out" 2> "$T/sym.err"; echo $? > "$T/sym.rc"
check "symlink: lsk_serve_bg B exit 0" "$(cat "$T/sym.rc")" 0
check "symlink: no mismatch message" "$(cat "$T/sym.err")" ""
check "symlink: lsk_served names B's model.json, leaf kept" "$(lsk lsk_served)" "$T/work/pkg-b/model.json"
check "symlink: state.sh-style caller gets exit 0" "$(lsk lsk_served > /dev/null; echo $?)" 0
stopall

# real: A, a real file
lsk lsk_serve_bg sa "$T/work/pkg-a" > /dev/null 2> "$T/real.err"; echo $? > "$T/real.rc"
check "real: lsk_serve_bg A exit 0" "$(cat "$T/real.rc")" 0
check "real: lsk_served names A's model.json" "$(lsk lsk_served)" "$T/work/pkg-a/model.json"

# wrong: A keeps the port (from the case above); lsk_serve_bg B finds A answering and must die naming both
lsk lsk_serve_bg sb "$T/work/pkg-b" > /dev/null 2> "$T/wrong.err"; echo $? > "$T/wrong.rc"
check "wrong: lsk_serve_bg B while A answers exits nonzero" "$([ "$(cat "$T/wrong.rc")" != 0 ] && echo yes || echo no)" yes
check "wrong: the message names A as served and B as expected" \
  "$(grep -c "serves '$T/work/pkg-a/model.json', not $T/work/pkg-b/model.json" "$T/wrong.err")" 1
stopall

# relative --model through a symlinked package directory: the dir resolves as lsk_pkg resolves it
ln -s "$T/work/pkg-a" "$T/work/pkg-link"
srv "$T/work" pkg-link/model.json
check "dirlink: lsk_served resolves the directory as lsk_pkg does" "$(lsk lsk_served)" "$(lsk lsk_pkg "$T/work/pkg-link")/model.json"
stopall

# gone: unresolvable model or cwd; each must exit 1 and print nothing
mkdir -p "$T/cwdgone/pkg-c"; cp "$T/work/pkg-a/model.json" "$T/cwdgone/pkg-c/"
srv "$T/cwdgone" pkg-c/model.json; rm -rf "$T/cwdgone"
check "gone cwd: lsk_served prints nothing" "$(lsk lsk_served 2>/dev/null)" ""
check "gone cwd: lsk_served exit 1" "$(lsk lsk_served > /dev/null 2>&1; echo $?)" 1
stopall
mkdir -p "$T/work/pkg-d"; cp "$T/work/pkg-a/model.json" "$T/work/pkg-d/"
srv "$T/work" "$T/work/pkg-d/model.json"; rm -rf "$T/work/pkg-d"
check "gone package dir: lsk_served prints nothing" "$(lsk lsk_served 2>/dev/null)" ""
check "gone package dir: lsk_served exit 1" "$(lsk lsk_served > /dev/null 2>&1; echo $?)" 1
stopall
mkdir -p "$T/work/pkg-e"; cp "$T/work/pkg-a/model.json" "$T/work/pkg-e/"
srv "$T/work" "$T/work/pkg-e/model.json"; rm -f "$T/work/pkg-e/model.json"
check "gone model.json: lsk_served prints nothing" "$(lsk lsk_served 2>/dev/null)" ""
check "gone model.json: lsk_served exit 1" "$(lsk lsk_served > /dev/null 2>&1; echo $?)" 1
stopall
(cd "$T/work" && exec "$T/bin/loom-serve" --listen=127.0.0.1:$PORT --model "$T/work/pkg-a/model.json") & echo $! > "$T/srv.kill"; waitport
check "no --model=: lsk_served exit 1" "$(lsk lsk_served > /dev/null 2>&1; echo $?)" 1
stopall
check "nothing listens: lsk_served exit 1" "$(lsk lsk_served > /dev/null 2>&1; echo $?)" 1

[ $fails = 0 ] && echo "lsk_served tests: all passed" || { echo "lsk_served tests: $fails failed"; exit 1; }
