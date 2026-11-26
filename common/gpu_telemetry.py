#!/usr/bin/env python3
"""GPU telemetry beside a timed run: busy %, power (W), shader clock (GHz) and temperature (C) every 0.1 s, plus the active
fabric / memory / SoC clock levels and the CPU's Tctl when the driver exposes them.

Why: a package whose kernels were faster on every profiled and isolated timing decoded 7-9 % slower in the seeded agent bench
(Qwen3.8-27B kit, E351-E353). Device timers cannot see a lower clock under a power or thermal limit, nor host-bound idle; this
records both beside every A/B and bench run, so a "faster kernel, slower step" result shows its cause. The first half of a bench
after an idle gap decoded ~7 % faster than one run straight after other GPU work with the shader clock unchanged (E368): the
fabric clock sets memory bandwidth, so it is recorded too.

Card: the first /sys/class/drm/card*/device with gpu_busy_percent (connector entries such as card1-DP-1 have none); its first
hwmon gives power, clock and the edge temperature. Tctl: the hwmon named k10temp, if any.

CPU (user 11-24, E415): CPU and GPU share one DRAM system and the power budget, so CPU work beside a timed run slows it (a builder's
7-core python during a reference half, load 9.4). Each sample carries the 1-min load average, the whole machine's busy CPU and the
busy CPU of everything except --pid PID (the bench's own server), in % of one core (top's convention: 250 = 2.5 cores busy).
A summary flags CONTAMINATED when that other CPU exceeds --cpu-max cores (default 2) in more than 10 % of the samples.

Other GPU clients (HRX-like E631: served dispatches ran 1.2-1.8x slow on some nights with memory-placement leads refuted, and busy %
cannot say whose work it was): the recorder snapshots, at start, every CLIENT_PERIOD s and on SIGTERM, every amdgpu DRM client's
cumulative engine time (/proc/PID/fdinfo/N: drm-engine-<engine> ns, one record per drm-pdev + drm-client-id however many fds or
processes hold it; only fds that link to /dev/dri/) and every KFD process (/sys/class/kfd/kfd/proc/PID), except --pid, into
OUT.clients.jsonl. KFD compute (loom-serve, HIP) shows in no engine counter: loom-serve holds a render node for its memory, but its
fdinfo shows DMA only. A process whose fd/ cannot be listed (another user's) is counted 'unreadable N', never skipped silently.
The summary's `gpu LABEL:` line, printed before the cpu line: other clients' engine time between the first and last snapshot (a
client new during the run from 0), as a share of that wall time; GPU-CONTAMINATED when the share exceeds --gpu-max % (default 1)
or another KFD process existed (its time cannot be measured); unknown without two snapshots (no .clients.jsonl: recorded before
11-26, or killed before its final snapshot). Engine time is time a client's jobs held a ring, an upper bound on interference.

usage: gpu_telemetry.py [--root DIR] record OUT.tsv [--pid PID]   runs until killed; one line per sample:
           t busy power_w clock_ghz temp_c fclk_mhz mclk_mhz socclk_mhz cpu_tctl_c load1 cpu_pct other_cpu_pct
           (a column the machine does not expose is empty; the first sample's CPU columns are empty: they are deltas)
       gpu_telemetry.py [--root DIR] summary OUT.tsv [LABEL] [--cpu-max CORES] [--gpu-max PCT]   medians and p10 / p90 over the samples with
           busy >= 50 %, the busy share, and the CPU line (`cpu: unknown` for a file recorded without CPU columns); the clock
           with 3 decimals (the recorded 1 MHz: 2 decimals stepped 0.46 %, HRX-like E624; summaries before 11-26 have 2)
       gpu_telemetry.py [--root DIR] temp                       the edge temperature now, in whole C (a bench's thermal start gate)
--root: the directory that holds sys/ and proc/ (default /; a test points it at a fake tree). Exit 2 when no GPU card is found."""
import glob
import json
import os
import re
import signal
import statistics
import sys
import time


class NoGpu(Exception):
    pass


def card(root="/"):
    """the drm device directory of the first card with gpu_busy_percent"""
    hits = sorted(glob.glob(os.path.join(root, "sys/class/drm/card*/device/gpu_busy_percent")))
    if not hits:
        raise NoGpu(f"no {os.path.join(root, 'sys/class/drm/card*/device/gpu_busy_percent')}")
    return hits[0].rsplit("/", 1)[0]


def paths(root="/"):
    dev = card(root)
    hw = sorted(glob.glob(dev + "/hwmon/hwmon*"))
    if not hw:
        raise NoGpu(f"no hwmon under {dev}")
    hw = hw[0]
    return dev + "/gpu_busy_percent", hw + "/power1_average", hw + "/freq1_input", hw + "/temp1_input"


def extra_paths(root="/"):
    """the active fabric / memory / SoC clock tables and the CPU's Tctl (None when there is no k10temp hwmon)"""
    dev = card(root)
    tctl = None
    for h in sorted(glob.glob(os.path.join(root, "sys/class/hwmon/hwmon*"))):
        try:
            if open(h + "/name").read().strip() == "k10temp":
                tctl = h + "/temp1_input"
                break
        except OSError:
            continue
    return [dev + "/pp_dpm_fclk", dev + "/pp_dpm_mclk", dev + "/pp_dpm_socclk"], tctl


def active_mhz(p):
    """the level marked '*' in a pp_dpm_* table, in MHz (0 when none: a table caught mid-transition)"""
    for line in open(p):
        if line.rstrip().endswith("*"):
            return int(line.split(":")[1].strip().split("Mhz")[0])
    return 0


def read(p):
    with open(p) as f:
        return int(f.read().strip())


def opt(f, p):
    """f(p) as text, '' when the file is missing or unreadable"""
    try:
        return str(f(p))
    except (OSError, ValueError, IndexError):
        return ""


def cpu_counters(root="/", pid=None):
    """(wall s, busy jiffies, total jiffies, cpus, PID's utime + stime jiffies or None) from /proc"""
    with open(os.path.join(root, "proc/stat")) as f:
        lines = f.read().split("\n")
    v = [int(x) for x in lines[0].split()[1:]]
    cpus = sum(1 for l in lines if l.startswith("cpu") and l[3:4].isdigit())
    idle = v[3] + (v[4] if len(v) > 4 else 0)
    own = None
    if pid:
        try:
            st = open(os.path.join(root, f"proc/{pid}/stat")).read()
            fs = st[st.rindex(")") + 2:].split()   # fields after the command name (which may hold spaces)
            own = int(fs[11]) + int(fs[12])         # utime, stime (stat fields 14, 15)
        except (OSError, ValueError, IndexError):
            own = None
    return time.time(), sum(v) - idle, sum(v), cpus, own


def cpu_columns(prev, cur, tck=100):
    """(cpu_pct, other_cpu_pct) between two cpu_counters() readings, in % of one core; other = minus the PID's own CPU"""
    dt, dbusy, dtot = cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]
    if dtot <= 0 or dt <= 0:
        return "", ""
    pct = 100.0 * cur[3] * dbusy / dtot
    other = pct
    if prev[4] is not None and cur[4] is not None:
        other = max(0.0, pct - 100.0 * (cur[4] - prev[4]) / tck / dt)
    return f"{pct:.0f}", f"{other:.0f}"


def sample(busy, power, freq, temp, clocks, tctl, cpu=("", "", "")):
    ex = "\t".join(opt(active_mhz, c) for c in clocks) + "\t" + (opt(lambda p: f"{read(p) / 1e3:.0f}", tctl) if tctl else "")
    return (f"{time.time():.2f}\t{read(busy)}\t{read(power) / 1e6:.1f}\t{read(freq) / 1e9:.3f}\t{read(temp) / 1e3:.0f}\t{ex}\t"
            + "\t".join(cpu) + "\n")


CLIENT_PERIOD = 5.0   # s between client snapshots: a scan of ~820 processes took 14 ms (0.3 % of one core at 5 s), and a client that
                      # exits mid-run loses at most its last 5 s


def clients_path(tsv):
    return (tsv[:-4] if tsv.endswith(".tsv") else tsv) + ".clients.jsonl"


ENGINE = re.compile(r"drm-engine-(?!capacity-)([\w-]+):\s*(\d+) ns")


def gpu_clients(root="/", exclude=None):
    """one snapshot: {"t", "clients": {pdev/client-id: {pid, comm, ns: {engine: ns}}}, "kfd": {pid: comm}, "unreadable": N}.
    A client held by several processes belongs to the lowest pid; amdgpu fdinfo without a client id is keyed by its pid/fd
    (counted, never merged); an engine line that does not parse makes the snapshot an error (the summary says unknown)."""
    proc = os.path.join(root, "proc")
    comm = lambda p: (opt(lambda q: open(q).read().strip(), f"{proc}/{p}/comm") or "?")
    clients, unreadable = {}, 0
    for p in sorted((x for x in os.listdir(proc) if x.isdigit()), key=int):
        if p == str(exclude):
            continue
        try:
            fds = os.listdir(f"{proc}/{p}/fd")
        except PermissionError:
            unreadable += 1
            continue
        except OSError:   # exited
            continue
        for fd in fds:
            try:
                if not os.readlink(f"{proc}/{p}/fd/{fd}").startswith("/dev/dri/"):
                    continue
                txt = open(f"{proc}/{p}/fdinfo/{fd}").read()
            except PermissionError:
                unreadable += 1
                break
            except OSError:
                continue
            kv = dict(l.split(":", 1) for l in txt.splitlines() if ":" in l)
            if kv.get("drm-driver", "").strip() != "amdgpu":
                continue
            eng = [l for l in txt.splitlines() if l.startswith("drm-engine-") and not l.startswith("drm-engine-capacity-")]
            ns = {m.group(1): int(m.group(2)) for m in map(ENGINE.fullmatch, (l.strip() for l in eng)) if m}
            if len(ns) != len(eng):
                raise ValueError(f"unparsed engine line in {proc}/{p}/fdinfo/{fd}: {eng}")
            cid = kv.get("drm-client-id", "").strip()
            key = f"{kv.get('drm-pdev', '').strip()}/{cid}" if cid else f"pid{p}/fd{fd}"
            clients.setdefault(key, {"pid": int(p), "comm": comm(p), "ns": ns})
    kd = os.path.join(root, "sys/class/kfd/kfd/proc")
    kfd = {p: comm(p) for p in (sorted(os.listdir(kd)) if os.path.isdir(kd) else []) if p.isdigit() and p != str(exclude)}
    return {"t": round(time.time(), 3), "clients": clients, "kfd": kfd, "unreadable": unreadable}


def snapshot(f, root, pid):
    """one JSON line; an error is recorded as such (the summary then says unknown), never dropped"""
    try:
        d = gpu_clients(root, pid)
    except (OSError, ValueError) as e:
        d = {"t": round(time.time(), 3), "error": str(e)[:300]}
    f.write(json.dumps(d, sort_keys=True) + "\n")


def record(out, root="/", period=0.1, pid=None):
    busy, power, freq, temp = paths(root)
    clocks, tctl = extra_paths(root)
    tck = os.sysconf("SC_CLK_TCK")
    prev = None
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # the callers' `kill $TEL`: leave through the final client snapshot
    with open(out, "w", buffering=1) as f, open(clients_path(out), "w", buffering=1) as cf:
        snapshot(cf, root, pid)
        last = time.time()
        try:
            while True:
                if time.time() - last >= CLIENT_PERIOD:
                    snapshot(cf, root, pid)
                    last = time.time()
                try:
                    cur = cpu_counters(root, pid)
                    load = open(os.path.join(root, "proc/loadavg")).read().split()[0]
                    cpu = (load,) + (cpu_columns(prev, cur, tck) if prev else ("", ""))
                    prev = cur
                except (OSError, ValueError, IndexError):
                    cpu = ("", "", "")
                try:
                    f.write(sample(busy, power, freq, temp, clocks, tctl, cpu))
                except OSError:
                    pass
                time.sleep(period)
        finally:
            snapshot(cf, root, pid)


CONTAMINATED_SHARE = 0.10   # share of samples above --cpu-max that flags a run


def samples(path):
    """(all rows, rows with busy >= 50 %) of a recorded tsv"""
    rows = [l.rstrip("\n").split("\t") for l in open(path) if l.count("\t") >= 4]
    return rows, [r for r in rows if int(r[1]) >= 50]


def busy_values(rows, busy, i):
    """column i over the busy samples (all samples when none is busy), as the summary's medians use them"""
    xs = [float(r[i]) for r in busy] or [float(r[i]) for r in rows]
    if i >= 5:   # a pp_dpm_* table caught mid-transition shows no active level (0): not a clock (E371)
        xs = [x for x in xs if x > 0] or [0.0]
    return xs


def busy_clock(path):
    """the median shader clock while busy (GHz) at the recorded 1 MHz resolution, or None without samples
    (events_summary's cycles table: HRX-like E624, tel.txt's 2-decimal clock stepped 0.46 % against a 0.27 % lane floor)"""
    rows, busy = samples(path)
    return statistics.median(busy_values(rows, busy, 3)) if rows else None


def summary(path, label="", cpu_max=2.0, gpu_max=1.0):
    rows, busy = samples(path)
    if not rows:
        print(f"  telemetry {label}: no samples")
        return 1
    q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))]

    def col(i, d=2):
        xs = busy_values(rows, busy, i)
        return f"{statistics.median(xs):.{d}f} (p10 {q(xs, 0.1):.{d}f}, p90 {q(xs, 0.9):.{d}f})"
    more = ""
    if all(len(r) >= 9 and r[5] for r in rows):
        more = f"; fclk {col(5)} MHz, mclk {col(6)} MHz, socclk {col(7)} MHz"
        if all(r[8] for r in rows):
            more += f", cpu {col(8)} C"
    print(f"  telemetry {label}: {len(rows)} samples, busy >= 50 % in {len(busy) / len(rows):.0%}; while busy: power {col(2)} W, "
          f"clock {col(3, 3)} GHz, temp {col(4)} C" + more)
    print(gpu_line(clients_path(path), label, gpu_max))
    print(cpu_line(rows, label, cpu_max))
    return 0


def gpu_line(cpath, label, gpu_max):
    """the other-GPU-clients line: engine time of every client but the server between the first and last snapshot, per process,
    and the flag (unknown without two good snapshots: no record, an error snapshot, or a recorder killed before its final one)"""
    try:
        snaps = [json.loads(l) for l in open(cpath)]
    except FileNotFoundError:
        return f"  gpu {label}: unknown (no GPU client record {os.path.basename(cpath)})"
    except (OSError, ValueError) as e:
        return f"  gpu {label}: unknown (unreadable GPU client record: {e})"
    bad = [s.get("error", "no clients") for s in snaps if "clients" not in s]
    if bad:
        return f"  gpu {label}: unknown (a client snapshot failed: {bad[0]})"
    wall = snaps[-1]["t"] - snaps[0]["t"] if len(snaps) >= 2 else 0
    if wall <= 0:
        return f"  gpu {label}: unknown ({len(snaps)} client snapshot(s): no start and end)"
    first, seen = snaps[0]["clients"], {}
    for s in snaps:
        seen.update(s["clients"])   # a client's last-seen counters (one that exited counts up to its last snapshot)
    per = {}
    for key, c in seen.items():
        base = first.get(key, {}).get("ns", {})
        for eng, ns in c["ns"].items():
            d = ns - base.get(eng, 0)
            if d > 0:
                who = per.setdefault((c["comm"], c["pid"]), {})
                who[eng] = who.get(eng, 0) + d
    tot = {w: sum(e.values()) for w, e in per.items()}
    total = sum(tot.values())
    kfd = {}
    for s in snaps:
        kfd.update(s["kfd"])
    share = 100.0 * total / 1e9 / wall
    top = sorted(tot, key=lambda w: -tot[w])
    who = ", ".join(f"{c} {p} {tot[(c, p)] / 1e6:.1f} ms (" + ", ".join(f"{e} {v / 1e6:.1f}" for e, v in sorted(per[(c, p)].items()))
                    + ")" for c, p in top[:5]) or "none"
    unread = max(s["unreadable"] for s in snaps)
    body = (f"other GPU clients {total / 1e6:.1f} ms engine time in {wall:.1f} s ({share:.2f} % of wall, flag above {gpu_max:g} %): "
            f"{who}; other KFD processes {', '.join(f'{c} {p}' for p, c in sorted(kfd.items())) or 'none'}; {len(snaps)} snapshots; "
            f"unreadable {unread} processes (another user's: not counted)")
    over = [f"{c} {p} {tot[(c, p)] / 1e6:.1f} ms" for c, p in top[:3]] if share > gpu_max else []
    over += [f"kfd {c} {p} unmeasured" for p, c in sorted(kfd.items())]
    return f"  gpu {label}: " + (f"GPU-CONTAMINATED ({', '.join(over)}): " if over else "clean: ") + body


def cpu_line(rows, label, cpu_max):
    """the CPU line: other CPU (all but the bench's server) in cores, and the CONTAMINATED flag (unknown without CPU columns)"""
    oth = [float(r[11]) / 100 for r in rows if len(r) >= 12 and r[11]]
    load = [float(r[9]) for r in rows if len(r) >= 12 and r[9]] or [0.0]
    if not oth:
        return f"  cpu {label}: unknown (no CPU columns)"
    q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))]
    share = sum(1 for x in oth if x > cpu_max) / len(oth)
    flag = "CONTAMINATED" if share > CONTAMINATED_SHARE else "clean"
    return (f"  cpu {label}: {flag}: other CPU (beside the server) median {statistics.median(oth):.2f} cores (p90 {q(oth, 0.9):.2f}), "
            f"> {cpu_max:g} cores in {share:.0%} of samples (flag above {CONTAMINATED_SHARE:.0%}); load1 median "
            f"{statistics.median(load):.2f} (max {max(load):.2f})")


def main(argv):
    root, pid, cpu_max, gpu_max = "/", None, 2.0, 1.0
    argv = list(argv)
    for flag in ("--root", "--pid", "--cpu-max", "--gpu-max"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                print(f"gpu_telemetry: {flag} needs a value", file=sys.stderr)
                return 2
            v = argv[i + 1]
            del argv[i:i + 2]
            if flag == "--root":
                root = v
            elif flag == "--pid":
                pid = int(v) if v.isdigit() else None   # no server pid found: other CPU = all CPU (stricter)
            elif flag == "--cpu-max":
                cpu_max = float(v)
            else:
                gpu_max = float(v)
    try:
        if len(argv) >= 2 and argv[0] == "record":
            record(argv[1], root, pid=pid)
        elif len(argv) >= 2 and argv[0] == "summary":
            return summary(argv[1], argv[2] if len(argv) > 2 else "", cpu_max, gpu_max)
        elif argv[:1] == ["temp"]:
            print(round(read(paths(root)[3]) / 1e3))
            return 0
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except NoGpu as e:
        print(f"gpu_telemetry: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
