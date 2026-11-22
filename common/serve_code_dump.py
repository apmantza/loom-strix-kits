#!/usr/bin/env python3
"""Extract the GPU code objects a loom-serve process actually loaded (E138: serving's vsplit runs 2.3 ms while every compile
of the same kernel we can build runs 1.5-2.0 ms; loom-serve compiles .loom at runtime and has no dump flag).

Starts loom-serve as a CHILD (yama ptrace_scope 1 lets a parent read its descendants' /proc/PID/mem), waits for readiness,
scans its readable private mappings (skipping regions over --max-region bytes, e.g. GPU pools) for ELF images whose bytes
contain --symbol, writes each distinct one to OUT/<n>.hsaco, then stops the server. Read-only on the child.
usage: serve_code_dump.py OUT --model PKG/model.json [--symbol g_attention_vsplit] [--port 8161] [-- extra loom-serve flags]"""
import argparse
import hashlib
import os
import struct
import subprocess
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--model", required=True)
ap.add_argument("--symbol", default="g_attention_vsplit")
ap.add_argument("--port", type=int, default=8161)
ap.add_argument("--serve", default=os.path.join(os.environ["VENV"], "bin", "loom-serve") if os.environ.get("VENV") else "loom-serve")
ap.add_argument("--max-region", type=int, default=1 << 31)
ap.add_argument("extra", nargs="*")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cmd = [a.serve, "--device=amdgpu://?completion_spin_ns=4294967295", f"--listen=127.0.0.1:{a.port}", "--pool_backing=fixed",
       "--workers=8", f"--model={a.model}"] + a.extra
log = open(os.path.join(a.out, "serve.log"), "w")
p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
try:
    t0 = time.time()
    while True:
        if p.poll() is not None:
            sys.exit(f"loom-serve exited {p.returncode} (see {a.out}/serve.log)")
        try:
            body = urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/status", timeout=5).read().decode()
            if '"registration":"ready"' in body.replace(" ", ""):
                break
        except Exception:
            pass
        if time.time() - t0 > 900:
            sys.exit("not ready after 900 s")
        time.sleep(2)
    print(f"ready after {time.time() - t0:.0f} s; scanning pid {p.pid}", flush=True)
    sym = a.symbol.encode()
    found, seen = 0, set()
    with open(f"/proc/{p.pid}/maps") as mf, open(f"/proc/{p.pid}/mem", "rb", 0) as mem:
        for ln in mf:
            f = ln.split()
            lo, hi = (int(x, 16) for x in f[0].split("-"))
            if "r" not in f[1] or hi - lo > a.max_region or (len(f) > 5 and f[5].startswith("/") and not f[5].startswith("/memfd")):
                continue
            try:
                mem.seek(lo)
                buf = mem.read(hi - lo)
            except OSError:
                continue
            i = buf.find(b"\x7fELF")
            while i >= 0:
                if buf[i + 4] == 2 and buf[i + 18:i + 20] == struct.pack("<H", 224):   # ELF64, e_machine EM_AMDGPU
                    shoff, = struct.unpack_from("<Q", buf, i + 40)
                    shentsize, shnum = struct.unpack_from("<HH", buf, i + 58)
                    end = i + shoff + shentsize * shnum
                    img = buf[i:end]
                    if 0 < len(img) < (64 << 20) and sym in img:
                        h = hashlib.sha256(img).hexdigest()[:16]
                        if h not in seen:
                            seen.add(h)
                            open(os.path.join(a.out, f"{found:02d}-{h}.hsaco"), "wb").write(img)
                            found += 1
                i = buf.find(b"\x7fELF", i + 4)
    print(f"{found} distinct AMDGPU ELF image(s) containing {a.symbol}")
finally:
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()
