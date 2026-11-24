#!/usr/bin/env python3
"""Tests of mem_admit.py on synthetic /proc and /sys trees. Each case names the recurrence it prevents.
usage: python3 test_mem_admit.py (exit 1 on a failure)"""
import os
import subprocess
import sys
import tempfile
import time

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import mem_admit  # noqa: E402

fails, ran = [], []
GIB = 2**30


def case(name, cond, detail=""):
    ran.append(name)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {detail}"))
    if not cond:
        fails.append(name)


def fake(total_gib=124, avail_gib=100, cma_free_gib=0, gtt_total_gib=124, gtt_used_gib=0.05, procs=()):
    """a root dir with proc/meminfo, one card's mem_info_gtt_*, and procs = [(pid, comm, gtt_gib)] with an amdgpu fdinfo each"""
    r = tempfile.mkdtemp(prefix="memadmit-", dir=os.environ.get("TMPDIR"))
    os.makedirs(f"{r}/proc")
    kb = lambda g: int(g * 1048576)
    with open(f"{r}/proc/meminfo", "w") as f:
        f.write(f"MemTotal: {kb(total_gib)} kB\nMemFree: {kb(avail_gib)} kB\nMemAvailable: {kb(avail_gib)} kB\n"
                f"CmaTotal: {kb(cma_free_gib)} kB\nCmaFree: {kb(cma_free_gib)} kB\n")
    d = f"{r}/sys/class/drm/card1/device"
    os.makedirs(d)
    open(f"{d}/mem_info_gtt_total", "w").write(str(int(gtt_total_gib * GIB)))
    open(f"{d}/mem_info_gtt_used", "w").write(str(int(gtt_used_gib * GIB)))
    os.makedirs(f"{r}/sys/class/drm/card1-DP-1")   # a connector dir with no mem_info_*
    for i, (pid, comm, g) in enumerate(procs):
        os.makedirs(f"{r}/proc/{pid}/fdinfo")
        open(f"{r}/proc/{pid}/comm", "w").write(comm + "\n")
        open(f"{r}/proc/{pid}/fdinfo/7", "w").write(
            f"pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{100 + i}\ndrm-total-gtt:\t{kb(g)} KiB\n"
            f"drm-resident-gtt:\t{kb(g)} KiB\ndrm-resident-vram:\t636 KiB\n")
    return r


def run(*args, env=None):
    p = subprocess.run([sys.executable, "-I", f"{H}/mem_admit.py", *args], capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    return p.returncode, p.stdout + p.stderr


for k in ("LSK_MEM_NEED_GIB", "LSK_MEM_RESERVE_GIB", "LSK_RESIDENT_GIB", "LSK_MEM_FLOOR_GIB", "LSK_MEM_ROOT"):
    os.environ.pop(k, None)

# a quiet box with only the desktop on the GPU is admitted (the gate must not block every run on this machine)
desk = [(2604, "gnome-shell", 0.04), (3217, "Xwayland", 0.01)]
rc, out = run("admit", "--root", fake(avail_gib=110, gtt_used_gib=0.05, procs=desk))
case("admit: quiet box, desktop only", rc == 0 and "admit:" in out, out)

# a run declaring more than fits beside the reserve is refused before it starts (a runaway allocation takes the desktop)
r = fake(avail_gib=50, gtt_used_gib=0.05, procs=desk)
rc, out = run("admit", "--root", r, env={"LSK_MEM_NEED_GIB": "45"})
case("refuse: need + reserve (max(8, 8 % of 124) = 9.9 GiB) > usable 50", rc == 1 and "REFUSE" in out and "short by" in out, out)
rc, out = run("admit", "--root", r, "--need", "40")
case("admit: need 40 + reserve 9.9 <= usable 50", rc == 0, out)
rc, out = run("admit", "--root", r, "--need", "40", env={"LSK_MEM_RESERVE_GIB": "12"})
case("refuse: the reserve is configurable (40 + 12 > 50)", rc == 1, out)
rc, out = run("admit", "--root", fake(total_gib=60, avail_gib=10, gtt_total_gib=60, gtt_used_gib=0))
case("admit: the reserve is max(8 GiB, 8 %): 8.0 on a 60 GiB box (8 % = 4.8), 10 usable", rc == 0 and "reserve 8.0" in out, out)
rc, out = run("admit", "--root", fake(total_gib=60, avail_gib=7.5, gtt_total_gib=60, gtt_used_gib=0))
case("refuse: 7.5 GiB usable < the 8 GiB floor reserve", rc == 1, out)

# CMA pages show in MemAvailable but a driver may not get them (DwarfStar QA section 9): counted out
rc, out = run("admit", "--root", fake(avail_gib=30, cma_free_gib=15), "--need", "15")
case("refuse: CmaFree is subtracted (30 - 15 = 15 < 15 + 9.9)", rc == 1 and "REFUSE" in out, out)
rc, out = run("admit", "--root", fake(avail_gib=30, cma_free_gib=0), "--need", "15")
case("admit: same box without CMA (30 >= 24.9)", rc == 0, out)

# the device side: GTT nearly full while the CPU side looks free is refused too
rc, out = run("admit", "--root", fake(avail_gib=100, gtt_total_gib=124, gtt_used_gib=118), env={"LSK_RESIDENT_GIB": "200"})
case("refuse: GTT free 6 GiB < reserve, though MemAvailable is 100", rc == 1 and "GTT free 6.0" in out, out)

# a resident model (a server, paused or not) beside a timed run: refused and named (E146-style wrong-configuration runs)
rc, out = run("admit", "--root", fake(avail_gib=100, gtt_used_gib=46.7, procs=desk + [(3272600, "loom-serve", 46.65)]))
case("refuse: loom-serve resident, named with its pid", rc == 1 and "loom-serve pid 3272600" in out, out)
rc, out = run("admit", "--root", fake(avail_gib=100, gtt_used_gib=6, procs=desk + [(4242, "python3", 6)]))
case("refuse: any GPU holder >= 1 GiB counts, not only loom-serve", rc == 1 and "python3 pid 4242" in out, out)
# another user's process cannot be read: its GTT shows only in mem_info_gtt_used, and fails closed
rc, out = run("admit", "--root", fake(avail_gib=100, gtt_used_gib=20, procs=desk))
case("refuse: GTT used that no readable process holds", rc == 1 and "unreadable process" in out, out)
rc, out = run("admit", "--root", fake(avail_gib=100, gtt_used_gib=20, procs=desk), env={"LSK_RESIDENT_GIB": "50"})
case("admit: the resident threshold is configurable", rc == 0, out)

# no card reports GTT: the gate cannot see the device and must not pass
r = fake()
for f in ("mem_info_gtt_total", "mem_info_gtt_used"):
    os.remove(f"{r}/sys/class/drm/card1/device/{f}")
rc, out = run("admit", "--root", r)
case("refuse: no mem_info_gtt_* found (fail closed)", rc == 1 and "cannot see" in out, out)

# the watchdog kills a run (and its children) when usable memory falls below the floor, and says why
r = fake(avail_gib=2.5)
child = subprocess.Popen(["bash", "-c", "sleep 60 & wait"])
time.sleep(0.3)
grand = [int(p) for p in os.listdir("/proc") if p.isdigit() and mem_admit.alive(int(p))
         and open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1] == str(child.pid)]
log = os.path.join(os.path.dirname(r), os.path.basename(r) + ".tsv")
t0 = time.time()
rc, out = run("watch", str(child.pid), "--root", r, "--log", log, "--every", "0.1")
child.wait(timeout=10)
case("watch: trips below the floor (2.5 < 3 GiB), exit 3, names why", rc == 3 and "KILLED bash pid" in out and "< floor 3" in out, out)
case("watch: the child process is killed too", grand and not any(mem_admit.alive(p) for p in grand), grand)
case("watch: logs a sample (t usable gtt_used)", open(log).read().count("\t") == 2, open(log).read())

r = fake(avail_gib=50)
child = subprocess.Popen(["sleep", "1"])
rc, out = run("watch", str(child.pid), "--root", r, "--every", "0.1")
child.wait()
case("watch: above the floor the run is left alone and the watcher exits 0 when it ends", rc == 0 and "ended" in out, out)
r = fake(avail_gib=10)
child = subprocess.Popen(["sleep", "30"])
rc, out = run("watch", str(child.pid), "--root", r, "--every", "0.1", env={"LSK_MEM_FLOOR_GIB": "12"})
child.wait(timeout=10)
case("watch: the floor is configurable (10 < 12 trips)", rc == 3, out)

# frag: free large physical blocks before a server starts (HRX-like E468 / E471: the same package's decode matvecs read 79.2 ms / step
# with ~21-30k free order-10 blocks and 95.5 with ~900 after a day of server restarts; nothing in a bench recorded it, so two hours
# of decode verdicts (q166, q179, q187) were judged without it)
r = fake()
open(f"{r}/proc/buddyinfo", "w").write(
    "Node 0, zone      DMA      0      0      0      0      0      0      0      0      1      1      2 \n"
    "Node 0, zone    DMA32     11     11     11     10     11     11      9      7      8     11    395 \n"
    "Node 0, zone   Normal    912   1810   1463   1721   1144    457    272    173     72     26  21579 \n")
open(f"{r}/proc/vmstat", "w").write("nr_free_pages 1\ncompact_stall 5\ncompact_fail 3\ncompact_success 2\n")
rc, out = run("frag", "--root", r)
case("frag: order 10 / order 9 summed over every zone (2 + 395 + 21579; 1 + 11 + 26)",
     rc == 0 and "order10 21976 " in out and "order9 38 " in out, out)
case("frag: order-10 GiB at 4 KiB pages (21976 x 4 MiB = 85.84 GiB)", "order10_gib 85.84 " in out, out)
case("frag: compaction counters from vmstat", "compact_stall 5 compact_fail 3 compact_success 2" in out, out)
r = fake()
rc, out = run("frag", "--root", r)
case("frag: no buddyinfo reads 'unknown', never a number (an absent reading must not look like a fragmented or a clean machine)",
     rc == 0 and out.startswith("memfrag unknown") and "order10 " not in out, out)

r = fake()
open(f"{r}/proc/buddyinfo", "w").write("")
rc, out = run("frag", "--root", r)
case("frag: an empty buddyinfo reads 'unknown', not order10 0 (zero would read as a fully fragmented machine)",
     rc == 0 and out.startswith("memfrag unknown") and "order10 " not in out, out)

print(f"{len(ran) - len(fails)}/{len(ran)} passed")
sys.exit(1 if fails else 0)
