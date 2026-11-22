#!/usr/bin/env python3
"""Summarise rocprofv3 counter CSVs (common/counters.sh): per kernel, the mean of each counter over its dispatches, the mean
dispatch time, and MemReadBytes as GB/s. Column names are matched case-insensitively (rocprofv3 writes Kernel_Name /
Counter_Name, the rocpd exporter kernel_name / counter_name).
--expect-bytes KERNEL_REGEX=BYTES (calibrate): print MemReadBytes / BYTES for matching kernels, the check that the size-weighted
L2 read requests count real memory bytes (bw_probe's read kernel reads its whole buffer once per dispatch).
usage: counters_summary.py DIR [--kernel REGEX] [--expect-bytes 'k_read=4294967296' ...]"""
import argparse, collections, csv, glob, os, re, sys


def rows(d):
    files = sorted(glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True))
    if not files:
        sys.exit(f"counters_summary: no *counter_collection.csv under {d}")
    for f in files:
        for r in csv.DictReader(open(f)):
            yield {k.lower(): v for k, v in r.items()}


def summarise(d, kernel=None):
    """kernel -> {"n": dispatches, "ns": mean dispatch ns, counter: mean value}"""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    span = collections.defaultdict(dict)
    for r in rows(d):
        k = r["kernel_name"]
        if kernel and not re.search(kernel, k):
            continue
        did = r.get("dispatch_id") or r.get("correlation_id")
        acc[k][r["counter_name"]].append((did, float(r["counter_value"])))
        span[k][did] = int(r["end_timestamp"]) - int(r["start_timestamp"])
    out = {}
    for k, cs in acc.items():
        s = {"n": len(span[k]), "ns": sum(span[k].values()) / len(span[k])}
        for c, vals in cs.items():
            per = collections.defaultdict(float)          # a counter may come as several rows per dispatch (dimensions)
            for did, v in vals:
                per[did] += v
            s[c] = sum(per.values()) / len(per)
        out[k] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--kernel")
    ap.add_argument("--expect-bytes", action="append", default=[])
    a = ap.parse_args()
    S = summarise(a.dir, a.kernel)
    for k, s in sorted(S.items()):
        print(f"{k[:90]}  dispatches {s['n']}  mean {s['ns'] / 1e3:.1f} us")
        for c in sorted(x for x in s if x not in ("n", "ns")):
            extra = f"   ({s[c] / s['ns']:.1f} GB/s)" if c == "MemReadBytes" and s["ns"] else ""
            print(f"    {c:28s} {s[c]:16.4g}{extra}")
    bad = 0
    for e in a.expect_bytes:
        rx, b = e.rsplit("=", 1)
        hits = [(k, s) for k, s in S.items() if re.search(rx, k) and "MemReadBytes" in s]
        if not hits:
            print(f"calibrate {rx}: no kernel with MemReadBytes"); bad = 1
        for k, s in hits:
            print(f"calibrate {k[:60]}: MemReadBytes / expected = {s['MemReadBytes'] / float(b):.3f}")
    sys.exit(bad)


if __name__ == "__main__":
    main()
