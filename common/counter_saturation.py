#!/usr/bin/env python3
"""Flag hardware counters that saturated at 32 bits. Each counter instance (per SE / WGP / channel) is a 32-bit register on gfx1151:
on a long launch it stops at 2^32 - 1, the summed value reads k x 2^32 for k saturated instances, and every ratio built on it is
wrong (a flash2 launch of 4,096 tokens at 1.14 s: SQ_WAIT_ANY = SQ_WAIT_INST_ANY = SQ_WAVE_CYCLES = 20 x 2^32, so rocprof-compute's
Wave Dependency Wait and Wave Issue Wait read 100 % and Wavefront Occupancy was off; HRX-like E425).
Two checks per dispatch, on raw counters only (derived metrics defined in common/counters/*.yaml are skipped):
  multiple  a value within --tol (relative) of k x 2^32, integer k >= 1. The tolerance absorbs printing precision only
            (rocprof-compute writes 6 significant digits: 5e-6); 0.1 % flagged a healthy GL1C_CYCLE_sum at 2.9972 x 2^32 (q171b).
            Instance counts are not in either output (rocprof-compute writes only _sum), so any k counts.
  group     >= 2 different counters (name suffixes _sum/_avr/_min/_max and [i] ignored, so X and X_sum are one counter) with
            values > 2^31 equal within --tol in the same pass and dispatch.
Inputs: a rocprof-compute workload dir (results_pmc_perf_*.csv[.gz]) or common/counters.sh's OUT dir (*_counter_collection.csv,
one per preset), or such files directly.
Verdict line `SATURATION: NONE` (exit 0) or `SATURATION: FOUND <counters>` (exit 1); exit 2 when no counter rows are found.
usage: counter_saturation.py PATH [PATH...] [--tol 1e-5] [--kernel REGEX]"""
import argparse, collections, csv, glob, gzip, os, re, sys

H = os.path.dirname(os.path.abspath(__file__))
TWO32 = float(2 ** 32)
SUFFIX = re.compile(r"(_sum|_avr|_min|_max)?(\[\d+\])?$")


def derived_names():
    out = set()
    for y in glob.glob(os.path.join(H, "counters", "*.yaml")):
        out |= set(re.findall(r"^\s*- name:\s*(\S+)", open(y).read(), re.M))
    return out


def files(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "results_pmc_perf_*.csv*")))
            out += sorted(glob.glob(os.path.join(p, "**", "*counter_collection.csv"), recursive=True))
        elif os.path.exists(p):
            out.append(p)
    return out


def rows(paths, kernel=None, skip=frozenset()):
    """(file, dispatch, kernel, counter, value) for every raw counter row"""
    for f in files(paths):
        with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
            for r in csv.DictReader(fh):
                r = {k.lower(): v for k, v in r.items() if k}
                name, k = r.get("counter_name"), r.get("kernel_name", "")
                if not name or name in skip or (kernel and not re.search(kernel, k)):
                    continue
                try:
                    v = float(r["counter_value"])
                except (KeyError, ValueError):
                    continue
                yield f, r.get("dispatch_id") or r.get("correlation_id"), k, name, v


def near(a, b, tol):
    return abs(a - b) <= tol * max(abs(a), abs(b))


def check(rs, tol=1e-5):
    """{counter: [reason, ...]}"""
    found = collections.defaultdict(set)
    per = collections.defaultdict(list)
    for f, d, k, c, v in rs:
        q = round(v / TWO32)
        if q >= 1 and near(v, q * TWO32, tol):   # an instance stopped at 2^32 - 1 is within tol too
            found[c].add(f"{v:.6g} = {q} x 2^32 ({k[:40]}, dispatch {d})")
        if v > TWO32 / 2:
            per[(f, d)].append((c, v, k))
    for (f, d), cs in per.items():
        for i, (c, v, k) in enumerate(cs):
            same = {c2 for c2, v2, _ in cs if SUFFIX.sub("", c2) != SUFFIX.sub("", c) and near(v, v2, tol)}
            if same:
                found[c].add(f"{v:.6g} equals {', '.join(sorted(same))} ({k[:40]}, dispatch {d}, {os.path.basename(f)})")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--kernel")
    a = ap.parse_args()
    rs = list(rows(a.paths, a.kernel, derived_names()))
    if not rs:
        print(f"SATURATION: ERROR no counter rows under {' '.join(a.paths)}")
        sys.exit(2)
    found = check(rs, a.tol)
    for c in sorted(found):
        why = sorted(found[c])
        print(f"  {c}: {why[0]}" + (f" (+{len(why) - 1} more)" if len(why) > 1 else ""))
    print("SATURATION: " + ("FOUND " + " ".join(sorted(found)) if found else f"NONE ({len(rs)} counter rows)"))
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
