#!/usr/bin/env python3
"""Memory admission and a watchdog for GPU runs on unified memory (Strix Halo: CPU and GPU share one RAM, no OOM boundary
between them, so a runaway GPU allocation takes the desktop and every other agent's work with it). HRX-like E403 / E409 item 5;
ideas from DwarfStar's QA_BEFORE_RELEASES.md section 9 and Strata's STRIX_HALO.md section 5b (both MIT).

usable    = max(0, MemAvailable - CmaFree)   (CMA pages show as available but a driver may not get them)
room      = min(usable, GTT total - GTT used) (the device side: /sys/class/drm/card*/device/mem_info_gtt_*)
admit     = no model resident on the GPU, and room >= need + reserve
  need    = LSK_MEM_NEED_GIB or --need (the run's own peak; unset = 0, only the reserve)
  reserve = LSK_MEM_RESERVE_GIB, else max(8 GiB, 8 % of MemTotal) (Strata's desktop reserve)
resident  = any process holding >= LSK_RESIDENT_GIB (default 1) GiB of GPU memory (drm fdinfo, common/checks.py drm_memory),
            paused or not; or GTT used that no readable process accounts for (another user's process): fails closed
watch     = sample usable and GTT used every --every s (default 0.5); when usable < floor (LSK_MEM_FLOOR_GIB, default 3 GiB)
            kill the watched PID and its descendants (TERM, then KILL after 3 s), print why, exit 3; exit 0 when PID ends

usage: mem_admit.py admit [--need GIB]                       exit 0 admits, 1 refuses; one line either way
       mem_admit.py watch PID [--floor GIB] [--log TSV] [--every S]
                                                             TSV: t usable_gib gtt_used_gib (keep it outside /tmp)
       --root DIR (or LSK_MEM_ROOT): read DIR/proc/meminfo, DIR/proc/PID/fdinfo and DIR/sys/class/drm instead (tests).
       The watched PID is always a real process.
"""
import glob
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks  # noqa: E402

GIB_KB = 1048576


def meminfo(root):
    out = {}
    for line in open(f"{root}/proc/meminfo"):
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0])
    return out


def usable_gib(mi):
    return max(0, mi["MemAvailable"] - mi.get("CmaFree", 0)) / GIB_KB


def gtt(root):
    """(total, used) GiB of the first card that reports GTT; None when no card does"""
    for t in sorted(glob.glob(f"{root}/sys/class/drm/card*/device/mem_info_gtt_total")):
        d = t.rsplit("/", 1)[0]
        return int(open(t).read()) / 2**30, int(open(d + "/mem_info_gtt_used").read()) / 2**30
    return None


def reserve_gib(mi):
    e = os.environ.get("LSK_MEM_RESERVE_GIB")
    return float(e) if e else max(8.0, 0.08 * mi["MemTotal"] / GIB_KB)


def residents(root, min_gib, gtt_used):
    """[(pid, comm, GiB)] of processes holding >= min_gib of GPU memory, plus one ('?', ...) row for GTT no readable process holds"""
    found, seen_gtt = [], 0.0
    for p in os.listdir(f"{root}/proc"):
        if not p.isdigit():
            continue
        try:
            m = checks.drm_memory(int(p), proc=f"{root}/proc")
            comm = open(f"{root}/proc/{p}/comm").read().strip()
        except OSError:   # gone, or another user's: the unattributed GTT below covers it
            continue
        seen_gtt += m.get("gtt", 0) / GIB_KB
        held = sum(m.values()) / GIB_KB
        if held >= min_gib:
            found.append((int(p), comm, held))
    if gtt_used is not None and gtt_used - seen_gtt >= min_gib:
        found.append(("?", "unreadable process", gtt_used - seen_gtt))
    return sorted(found, key=lambda r: -r[2])


def admit(root, need):
    mi = meminfo(root)
    g = gtt(root)
    if g is None:
        return False, f"mem_admit: REFUSE: no card under {root}/sys/class/drm reports mem_info_gtt_* (cannot see the GPU's memory)"
    min_res = float(os.environ.get("LSK_RESIDENT_GIB", "1"))
    res = residents(root, min_res, g[1])
    if res:
        who = ", ".join(f"{c} pid {p} {h:.1f} GiB" for p, c, h in res)
        return False, f"mem_admit: REFUSE: a model is resident on the GPU ({who}); stop it, even a paused one, before a timed run"
    use, gfree, rsv = usable_gib(mi), g[0] - g[1], reserve_gib(mi)
    room = min(use, gfree)
    line = (f"usable {use:.1f} GiB (MemAvailable - CmaFree {mi.get('CmaFree', 0) / GIB_KB:.1f}), GTT free {gfree:.1f} GiB; "
            f"need {need:g} + reserve {rsv:.1f} GiB")
    if room < need + rsv:
        return False, f"mem_admit: REFUSE: {line}: short by {need + rsv - room:.1f} GiB"
    return True, f"mem_admit: admit: {line}"


def alive(pid):
    try:
        st = open(f"/proc/{pid}/stat").read()
    except OSError:
        return False
    return st.rsplit(")", 1)[1].split()[0] != "Z"


def tree(pid):
    """pid and all its descendants (real /proc), children first"""
    kids = {}
    for s in glob.glob("/proc/[0-9]*/stat"):
        try:
            f = open(s).read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        kids.setdefault(int(f[1]), []).append(int(s.split("/")[2]))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += kids.get(p, [])
    return out[::-1]


def kill_tree(pid):
    ps = tree(pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in ps:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        end = time.time() + 3
        while time.time() < end and any(alive(p) for p in ps):
            time.sleep(0.1)
        if not any(alive(p) for p in ps):
            return


def watch(root, pid, floor, log, every):
    try:
        comm = open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        comm = "?"
    low = None
    f = open(log, "w", buffering=1) if log else None
    while alive(pid):
        use = usable_gib(meminfo(root))
        g = gtt(root)
        low = use if low is None else min(low, use)
        if f:
            f.write(f"{time.time():.2f}\t{use:.2f}\t{g[1]:.2f}\n" if g else f"{time.time():.2f}\t{use:.2f}\tNA\n")
        if use < floor:
            print(f"mem_admit watch: KILLED {comm} pid {pid} and its children: usable {use:.1f} GiB < floor {floor:g} GiB "
                  f"(MemAvailable - CmaFree; GTT used {g[1]:.1f} GiB)" if g else
                  f"mem_admit watch: KILLED {comm} pid {pid} and its children: usable {use:.1f} GiB < floor {floor:g} GiB",
                  flush=True)
            kill_tree(pid)
            return 3
        time.sleep(every)
    print(f"mem_admit watch: {comm} pid {pid} ended; lowest usable {'-' if low is None else f'{low:.1f}'} GiB")
    return 0


def opt(args, name, default):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default


def main(argv):
    args = list(argv)
    root = opt(args, "--root", os.environ.get("LSK_MEM_ROOT", ""))
    if args[:1] == ["admit"]:
        ok, line = admit(root, float(opt(args, "--need", os.environ.get("LSK_MEM_NEED_GIB") or "0")))
        print(line)
        return 0 if ok else 1
    if args[:1] == ["watch"]:
        floor = float(opt(args, "--floor", os.environ.get("LSK_MEM_FLOOR_GIB", "3")))
        log, every = opt(args, "--log", None), float(opt(args, "--every", "0.5"))
        if len(args) == 2 and args[1].isdigit():
            return watch(root, int(args[1]), floor, log, every)
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
