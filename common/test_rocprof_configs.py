#!/usr/bin/env python3
"""Static check of common/counters/rocprof-compute-gfx115x against the device file (no GPU). Recurrences it prevents:
  - % of peak at a clock the chip does not hold: AMD's configs use $max_sclk 2,900 MHz, the 85 W cap holds ~2.14 GHz (E425), and a
    device-file change (a new cap profile) that is not regenerated into the copy;
  - GL2-Fabric peaks of 128 GB/s ($max_mclk x 4 x 8 x 4 B) while the bus streams 240 GB/s (q171b read 112 % of peak);
  - SPI 6.2 rows on SPI_CSN_WAVE, which reads 0 on gfx1151, shown as a measured zero;
  - metric ids or names drifting from AMD's (when the AMD install is present: the copy must be AMD's plus the documented edits).
usage: python3 common/test_rocprof_configs.py   (ROCM_SDK from the machine env for the AMD comparison; skipped, and said so, without it)"""
import glob, json, os, subprocess, sys, tempfile

import yaml

H = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(H, "counters", "rocprof-compute-gfx115x")
sys.path.insert(0, os.path.join(H, "counters"))
import make_rocprof_configs as mk  # noqa: E402


def tables(d):
    """table id -> metric names, over every panel yaml in d"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.yaml"))):
        for ds in (yaml.safe_load(open(f)).get("Panel Config", {}).get("data source") or []):
            t = ds.get("metric_table")
            if t:
                out[t["id"]] = list((t.get("metric") or {}).keys())
    return out


def main():
    fails = 0

    def ok(cond, what):
        nonlocal fails
        print(("ok  " if cond else "FAIL") + " " + what)
        fails += not cond

    prof, mhz, gbs = mk.peaks(os.path.join(H, "devices", "gfx1151.json"))
    sol = yaml.safe_load(open(os.path.join(CFG, "0200_system_speed_of_light.yaml")))["Panel Config"]["data source"][0]["metric_table"]["metric"]
    ok(sol["VALU FLOPs"]["peak"] == f"{mhz} * $cu_per_gpu * 128 / 1000", f"VALU FLOPs peak at the {prof} clock {mhz} MHz: {sol['VALU FLOPs']['peak']}")
    for m in ("GL2-Fabric Read BW", "GL2-Fabric Write BW"):
        ok(str(sol[m]["peak"]) == f"{gbs:g}e9", f"{m} peak = {prof} bandwidth {gbs:g} GB/s: {sol[m]['peak']}")
    left = [f for f in glob.glob(os.path.join(CFG, "*.yaml")) if "$max_sclk" in open(f).read() or "$max_mclk" in open(f).read()]
    ok(not left, f"no $max_sclk / $max_mclk left ({[os.path.basename(f) for f in left]})")
    spi = yaml.safe_load(open(os.path.join(CFG, "0600_workgroup_manager_spi.yaml")))["Panel Config"]["data source"]
    rows = {k: v for ds in spi for k, v in (ds["metric_table"].get("metric") or {}).items()}
    bad = [k for k, v in rows.items() if any("SPI_CSN_WAVE" in str(v.get(f)) for f in ("avg", "min", "max"))]
    ok(not bad and all("unsupported" in rows[k]["unit"] for k in mk.SPI_ROWS), f"SPI 6.2 rows marked unsupported, no SPI_CSN_WAVE expression ({bad})")
    amd = mk.amd_dir()
    if amd and os.path.isdir(amd):
        ok(tables(amd) == tables(CFG), "metric tables and names identical to AMD's")
        with tempfile.TemporaryDirectory() as t:
            subprocess.run([sys.executable, os.path.join(H, "counters", "make_rocprof_configs.py"), "--out", t], check=True, capture_output=True)
            diff = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(t, "*.yaml")))
                    if open(f).read() != open(os.path.join(CFG, os.path.basename(f))).read()]
        ok(not diff, f"the copy equals a fresh generation ({diff}): regenerate with common/counters/make_rocprof_configs.py")
    else:
        print(f"skip AMD comparison: no AMD configs at {amd!r} (not a pass for that part)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
