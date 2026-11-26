#!/bin/bash
# events_ab.sh end to end on stubs: no GPU, no real server, no managed-server hook. A temporary kit tree holds copies of
# events_ab.sh and of the tools it runs for real (env.sh, events_summary.py, ceiling.py, context_label_check.py, cool_wait.sh)
# beside stubs for the rest: loom-serve (a local HTTP stub on a free port that writes its profile bundle only when it exits),
# iree-profile (refuses an incomplete bundle), gpu_telemetry.py, env_block.py, mem_admit.py, wait_quiet.sh, profile_req.py.
# Recurrences it guards:
#   - HRX-like E485: the served profiles of a whole day were taken by a fork of this script that wrote no memfrag and no
#     environment block: every run must leave its start, environment, memfrag, telemetry, request and events files.
#   - HRX-like E415: CPU work beside a timed run slows it: a run whose CPU line is not 'clean' (CONTAMINATED, or unknown)
#     makes the profile UNTRUSTED (exit 2), never a silent pass.
#   - HRX-like E631: other GPU clients beside the server: a gpu line that is not 'clean' makes the profile UNTRUSTED (exit 2).
#   - the profile bundle is written when the server exits, after its port has closed: the export waits for the process.
# usage: test_events_ab.sh          EVENTS_AB=FILE tests another copy of events_ab.sh (mutants); exit 0 when every check passes
set -u
H=$(cd "$(dirname "$0")" && pwd); SUT=${EVENTS_AB:-$H/events_ab.sh}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT; fails=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
mkdir -p "$T/common" "$T/kits/k" "$T/work/pkg-a" "$T/work/other" "$T/bin"
cp "$SUT" "$T/common/events_ab.sh"
for f in env.sh events_summary.py ceiling.py context_label_check.py cool_wait.sh; do cp "$H/$f" "$T/common/"; done
cp "$H/../kits/qwen38-27b/lanes.json" "$T/kits/k/lanes.json"
echo '{}' > "$T/work/pkg-a/model.json"; echo '{}' > "$T/work/other/model.json"
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
cat > "$T/kits/k/kit.env" <<'EOF'
KIT_MODEL_ID=test/model
KIT_SERVE_ARGS="--device=amdgpu://?completion_spin_ns=4294967295 --pool_backing=fixed --workers=8"
KIT_LANES=lanes.json
EOF
cat > "$T/env" <<EOF
WORK=$T/work
STATE=$T/state
PORT=$PORT
PYTHON=python3
IREE_PROFILE=$T/bin/iree-profile
LOOM_SERVE=$T/bin/loom-serve
EOF
cat > "$T/bin/loom-serve" <<'EOF'
#!/usr/bin/env python3
import http.server, os, signal, sys, time
a = sys.argv[1:]
open(os.environ["STUB_LOG"], "a").write("loom-serve " + " ".join(a) + "\n")
port = int(next(x for x in a if x.startswith("--listen=")).rsplit(":", 1)[1])
out = next((x.split("=", 1)[1] for x in a if x.startswith("--device_profiling_output=")), None)
class Q(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = b'{"object":"list","data":[]}'
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *_):
        pass
srv = http.server.HTTPServer(("127.0.0.1", port), Q)
def term(*_):
    srv.socket.close()                                   # the port closes first ...
    time.sleep(float(os.environ.get("STUB_EXIT_DELAY", "0")))
    if out:
        open(out, "w").write("bundle complete\n")        # ... the bundle is complete only when the process ends
    os._exit(0)
signal.signal(signal.SIGTERM, term)
srv.serve_forever()
EOF
cat > "$T/bin/iree-profile" <<'EOF'
#!/bin/bash
echo "iree-profile $*" >> "$STUB_LOG"
grep -q '^bundle complete$' "${@: -1}" 2>/dev/null || { echo "incomplete bundle ${@: -1}" >&2; exit 1; }
for i in 1 2; do
  printf '{"type":"dispatch_event","key":"g_attention_vsplit","submission_id":%d,"duration_ns":1600000,"workgroup_count":[8,4,64],"valid":true}\n' $i
  printf '{"type":"dispatch_event","key":"g_attention_flash2","submission_id":9,"duration_ns":50000000,"workgroup_count":[8,4,64],"valid":true}\n'
done
EOF
cat > "$T/common/gpu_telemetry.py" <<'EOF'
#!/usr/bin/env python3
import os, sys, time
a = sys.argv[1:]
open(os.environ["STUB_LOG"], "a").write("gpu_telemetry.py " + " ".join(a) + "\n")
if a[0] == "temp":
    print(40)
elif a[0] == "record":
    open(a[1], "w").close()   # empty: events_summary imports the real gpu_telemetry only for a tel.tsv with samples
    while True:
        time.sleep(1)
elif a[0] == "summary":
    tag = a[2]
    print(f"  telemetry {tag}: 10 samples, busy >= 50 % in 99%; while busy: power 85.0 (p10 85.0, p90 85.0) W, "
          f"clock 2.15 (p10 2.15, p90 2.15) GHz, temp 80 (p10 80, p90 80) C")
    g = dict(kv.split("=") for kv in os.environ.get("STUB_GPU", "").split(",") if kv).get(tag, "clean")
    print(f"  gpu {tag}: unknown (no GPU client record)" if g == "unknown" else f"  gpu {tag}: {g}: other GPU clients stub")
    flag = dict(kv.split("=") for kv in os.environ.get("STUB_CPU", "").split(",") if kv).get(tag, "clean")
    print(f"  cpu {tag}: unknown (no CPU columns)" if flag == "unknown" else f"  cpu {tag}: {flag}: other CPU stub")
EOF
cat > "$T/common/env_block.py" <<'EOF'
#!/usr/bin/env python3
import filecmp, os, sys
a = sys.argv[1:]
open(os.environ["STUB_LOG"], "a").write("env_block.py " + " ".join(a) + "\n")
if a[0] == "--diff":
    same = filecmp.cmp(a[1], a[2], shallow=False)
    print("" if same else "kernel: differs")
    sys.exit(0 if same else 1)
print("kernel: stub")
EOF
printf '#!/usr/bin/env python3\nimport os, sys\nopen(os.environ["STUB_LOG"], "a").write("mem_admit.py " + " ".join(sys.argv[1:]) + "\\n")\nprint("memfrag order10 100 order10_gib 0.4 stub")\n' > "$T/common/mem_admit.py"
printf '#!/bin/bash\necho "wait_quiet.sh $*" >> "$STUB_LOG"; echo "memory free: stub"\n' > "$T/common/wait_quiet.sh"
cat > "$T/common/profile_req.py" <<'EOF'
#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
open(os.environ["STUB_LOG"], "a").write("profile_req.py " + " ".join(a) + "\n")
print(json.dumps({"requests": [{"prompt_tokens": int(os.environ.get("STUB_PROMPT_TOKENS", a[2]))}]}))
EOF
chmod +x "$T/bin/"* "$T/common/"*

run() {   # NAME [env ...]: one events_ab run over pkg-a, other, other, pkg-a into $T/NAME
  local n=$1; shift
  env LSK_ENV="$T/env" STUB_LOG="$T/$n.log" "$@" bash "$T/common/events_ab.sh" k "$n" 1000 pkg-a "$T/work/other" other pkg-a \
    > "$T/$n.out" 2>&1; echo $? > "$T/$n.rc"
}
files() { (cd "$1" && find . -type f | sort | tr '\n' ' '); }
RUNF="./env.diff ./env.err ./env.txt ./events.err ./events.jsonl ./mem.txt ./run.ireeprof ./run.req.err ./run.req.json ./start.txt ./tel.tsv ./tel.txt "

run clean
check "clean: exit 0, EVENTS_AB_DONE, no UNTRUSTED" "[ \$(cat $T/clean.rc) = 0 ] && grep -q EVENTS_AB_DONE $T/clean.out && ! grep -q UNTRUSTED $T/clean.out"
check "run dirs named <n>-<package dir without 'pkg-'>" "[ \"\$(cd $T/work/clean && ls -d */ | tr '\n' ' ')\" = '1-a/ 2-other/ 3-other/ 4-a/ ' ]"
check "every run leaves start, environment, memfrag, telemetry, request and events files" \
  "(for d in 1-a 2-other 3-other 4-a; do [ \"\$(files $T/work/clean/\$d 2>/dev/null)\" = \"$RUNF\" ] || exit 1; done)"
check "OUT holds summary.md, env.md, starts.txt" "[ \"\$(cd $T/work/clean && ls -p | grep -v / | tr '\n' ' ')\" = 'env.md starts.txt summary.md ' ]"
check "starts.txt: one line per run, thermal start and memfrag" \
  "[ \$(wc -l < $T/work/clean/starts.txt) = 4 ] && grep -q '^1-a start: edge 40 C after [0-9]* s of cooling; memfrag order10 100' $T/work/clean/starts.txt"
check "summary.md: the cycles table read each run's busy clock from tel.txt" "grep -q '1-a @ 2.150 GHz' $T/work/clean/summary.md"
check "summary.md: the environment section and a memory line per run" \
  "grep -q '^environment:' $T/work/clean/summary.md && [ \$(grep -c '^memory before ' $T/work/clean/summary.md) = 4 ]"
check "served-context lines" "[ \$(grep -c '^### [1-4] .* served context 1000 prompt tokens' $T/clean.out) = 4 ]"
check "telemetry recorded with the server's pid, judged with --cpu-max" \
  "[ \$(grep -c '^gpu_telemetry.py record .*/tel.tsv --pid [0-9][0-9]*\$' $T/clean.log) = 4 ] && [ \$(grep -c '^gpu_telemetry.py summary .* --cpu-max 3\$' $T/clean.log) = 4 ]"
check "served with the kit's flags plus profiling" \
  "[ \$(grep -c -- '^loom-serve --listen=127.0.0.1:$PORT --model=$T/work/.*/model.json --device=amdgpu://?completion_spin_ns=4294967295 --pool_backing=fixed --workers=8 --device_profiling_mode=dispatch-events,executable-metadata' $T/clean.log) = 4 ]"

run contam STUB_CPU=3-other=CONTAMINATED
check "a CONTAMINATED run: exit 2 after the summary, names the run" \
  "[ \$(cat $T/contam.rc) = 2 ] && grep -q EVENTS_AB_DONE $T/contam.out && grep -q \"^EVENTS: UNTRUSTED: CPU telemetry not 'clean' .* in run(s) 3-other \" $T/contam.out && tail -1 $T/work/contam/summary.md | grep -q 'EVENTS: UNTRUSTED'"
run unknown STUB_CPU=1-a=unknown
check "an unknown CPU line: exit 2 (fail closed)" "[ \$(cat $T/unknown.rc) = 2 ] && grep -q 'in run(s) 1-a ' $T/unknown.out"

# HRX-like E631: other GPU clients (desktop compositor, browser, a second compute process) time-share the GPU beside the server
run gcontam STUB_GPU=2-other=GPU-CONTAMINATED
check "a GPU-CONTAMINATED run: exit 2 after the summary, names the run as GPU (not CPU)" \
  "[ \$(cat $T/gcontam.rc) = 2 ] && grep -q \"^EVENTS: UNTRUSTED: GPU clients not 'clean' .* in run(s) 2-other \" $T/gcontam.out && ! grep -q 'CPU telemetry not' $T/gcontam.out"
run gunknown STUB_GPU=4-a=unknown
check "an unknown gpu line (no client record): exit 2 (fail closed)" "[ \$(cat $T/gunknown.rc) = 2 ] && grep -q 'GPU clients not .* in run(s) 4-a ' $T/gunknown.out"

run slowexit STUB_EXIT_DELAY=1.5
check "a server that writes its bundle 1.5 s after its port closes: every run exported" \
  "[ \$(cat $T/slowexit.rc) = 0 ] && [ \$(grep -c '^iree-profile ' $T/slowexit.log) = 4 ] && [ -s $T/work/slowexit/4-a/events.jsonl ]"

run label STUB_PROMPT_TOKENS=500
check "served context off the label: exit 1 after run 1, no export" \
  "[ \$(cat $T/label.rc) = 1 ] && grep -q 'events_ab STOP: run 1 served another context' $T/label.out && ! grep -q '^iree-profile' $T/label.log"
check "no stub server left behind" "! ss -ltn | grep -q \":$PORT \""
[ $fails = 0 ] && echo "events_ab tests: all passed" || { echo "events_ab tests: $fails failed"; exit 1; }
