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

usage: gpu_telemetry.py [--root DIR] record OUT.tsv [--pid PID]   runs until killed; one line per sample:
           t busy power_w clock_ghz temp_c fclk_mhz mclk_mhz socclk_mhz cpu_tctl_c load1 cpu_pct other_cpu_pct
           (a column the machine does not expose is empty; the first sample's CPU columns are empty: they are deltas)
       gpu_telemetry.py [--root DIR] summary OUT.tsv [LABEL] [--cpu-max CORES]   medians and p10 / p90 over the samples with
           busy >= 50 %, the busy share, and the CPU line (`cpu: unknown` for a file recorded without CPU columns)
       gpu_telemetry.py [--root DIR] temp                       the edge temperature now, in whole C (a bench's thermal start gate)
--root: the directory that holds sys/ and proc/ (default /; a test points it at a fake tree). Exit 2 when no GPU card is found."""
import glob
import os
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


def record(out, root="/", period=0.1, pid=None):
    busy, power, freq, temp = paths(root)
    clocks, tctl = extra_paths(root)
    tck = os.sysconf("SC_CLK_TCK")
    prev = None
    with open(out, "w", buffering=1) as f:
        while True:
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


CONTAMINATED_SHARE = 0.10   # share of samples above --cpu-max that flags a run


def summary(path, label="", cpu_max=2.0):
    rows = [l.rstrip("\n").split("\t") for l in open(path) if l.count("\t") >= 4]
    if not rows:
        print(f"  telemetry {label}: no samples")
        return 1
    busy = [r for r in rows if int(r[1]) >= 50]
    q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))]

    def col(i):
        xs = [float(r[i]) for r in busy] or [float(r[i]) for r in rows]
        if i >= 5:   # a pp_dpm_* table caught mid-transition shows no active level (0): not a clock (E371)
            xs = [x for x in xs if x > 0] or [0.0]
        return f"{statistics.median(xs):.2f} (p10 {q(xs, 0.1):.2f}, p90 {q(xs, 0.9):.2f})"
    more = ""
    if all(len(r) >= 9 and r[5] for r in rows):
        more = f"; fclk {col(5)} MHz, mclk {col(6)} MHz, socclk {col(7)} MHz"
        if all(r[8] for r in rows):
            more += f", cpu {col(8)} C"
    print(f"  telemetry {label}: {len(rows)} samples, busy >= 50 % in {len(busy) / len(rows):.0%}; while busy: power {col(2)} W, "
          f"clock {col(3)} GHz, temp {col(4)} C" + more)
    print(cpu_line(rows, label, cpu_max))
    return 0


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
    root, pid, cpu_max = "/", None, 2.0
    argv = list(argv)
    for flag in ("--root", "--pid", "--cpu-max"):
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
            else:
                cpu_max = float(v)
    try:
        if len(argv) >= 2 and argv[0] == "record":
            record(argv[1], root, pid=pid)
        elif len(argv) >= 2 and argv[0] == "summary":
            return summary(argv[1], argv[2] if len(argv) > 2 else "", cpu_max)
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
