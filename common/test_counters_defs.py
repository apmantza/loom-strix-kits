#!/usr/bin/env python3
"""Static check of common/counters/<gfx>.yaml and counters.sh's presets against AMD's counter definitions for the chip (ROCm
10.2 rocprofv3's share/rocprofiler-sdk/config.yaml; no GPU). Recurrences it prevents:
  - a derived metric that names a counter the chip does not have (rocprofv3 fails the whole preset at run time);
  - a preset naming a metric that neither our yaml nor rocprofv3 defines for the chip;
  - SQ_WAIT_* divided by 4 * SQ_WAVE_CYCLES: the first yaml did so for VMVS and DELAY_ALU from rocprofv3-avail's unit text,
    and read 24 % for a streaming read that waits ~97 % of its time (q98); AMD's panels divide by SQ_WAVE_CYCLES alone;
  - an average latency whose accumulated level counts FLAT/global loads but whose count does not (VmemLoadLatency read 0 on
    every global-load kernel, q98): SQ_INST_LEVEL_TEX_LOAD includes FLAT, SQ_INSTS_TEX_LOAD does not.
usage: python3 common/test_counters_defs.py [YAML]   (ROCM_SDK from the machine env, as counters.sh)"""
import os, re, sys

import yaml

H = os.path.dirname(os.path.abspath(__file__))
GFX = os.environ.get("GFX", "gfx1151")
FUNCS = {"reduce", "accumulate", "sum", "max", "min", "avr", "HIGH_RES", "NONE", "LOW_RES"}


def sdk_config():
    sdk = os.environ.get("ROCM_SDK")
    if not sdk:
        env = os.environ.get("LSK_ENV", os.path.expanduser("~/.config/loom-strix-kits/env"))
        if os.path.exists(env):
            m = re.search(r"^ROCM_SDK=(\S+)", open(env).read(), re.M)
            sdk = m and m.group(1)
    p = sdk and os.path.join(sdk, "share/rocprofiler-sdk/config.yaml")
    if not p or not os.path.exists(p):
        sys.exit(f"test_counters_defs: no rocprofiler-sdk config.yaml (ROCM_SDK={sdk!r}): cannot check, not a pass")
    return p


def defined(counters):
    """name -> gfx expression ('' for a hardware counter) for every counter the list defines for GFX"""
    out = {}
    for c in counters:
        for d in c.get("definitions", []):
            if GFX in d.get("architectures", []):
                out[c["name"]] = d.get("expression") or ""
    return out


def names(expr):
    return {t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr) if t not in FUNCS}


def presets(sh):
    return {m.group(1): m.group(2).split() for m in re.finditer(r'^\s*\[(\w+)\]="([^"]*)"', open(sh).read(), re.M)}


def check(ours, amd, pre):
    errs = []
    for n, e in ours.items():
        for t in sorted(names(e) - set(amd) - set(ours)):
            errs.append(f"{n}: names {t}, which {GFX} does not define")
        if re.search(r"SQ_WAIT_\w+.*?/\s*\(\s*4\s*\*\s*reduce\(SQ_WAVE_CYCLES", e):
            errs.append(f"{n}: SQ_WAIT_* over 4 * SQ_WAVE_CYCLES (same unit: AMD divides by SQ_WAVE_CYCLES alone)")
        if "SQ_INST_LEVEL_TEX_LOAD" in e and "SQ_INSTS_TEX_LOAD" in e and "SQ_INSTS_FLAT_LOAD" not in e:
            errs.append(f"{n}: TEX_LOAD level (includes FLAT) over a TEX-only count")
    for p, ms in pre.items():
        for m in ms:
            if m not in ours and m not in amd:
                errs.append(f"preset {p}: {m} is defined neither by our yaml nor by rocprofv3 for {GFX}")
    return errs


def main():
    yp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "counters", f"{GFX}.yaml")
    amd = defined(yaml.safe_load(open(sdk_config()))["rocprofiler-sdk"]["counters"])
    ours = defined(yaml.safe_load(open(yp))["rocprofiler-sdk"]["counters"])
    pre = presets(os.path.join(H, "counters.sh"))
    fails = 0

    def ok(cond, what):
        nonlocal fails
        print(("ok  " if cond else "FAIL") + " " + what)
        fails += not cond

    errs = check(ours, amd, pre)
    for e in errs:
        print("     " + e)
    ok(not errs, f"{len(ours)} derived metrics and {len(pre)} presets in {os.path.basename(yp)} resolve on {GFX}")
    ok(len(pre) >= 5 and all(pre.values()), "counters.sh presets parsed")
    # negative controls: the checker must catch each recurrence on a synthetic metric
    bad = {"X1": "100*reduce(SQ_WAIT_CNT_VMVS,sum)/(4*reduce(SQ_WAVE_CYCLES,sum))",
           "X2": "reduce(accumulate(SQ_INST_LEVEL_TEX_LOAD,HIGH_RES),sum)/reduce(SQ_INSTS_TEX_LOAD,sum)",
           "X3": "reduce(SQ_NO_SUCH_COUNTER,sum)"}
    for k, e in bad.items():
        ok(len(check({k: e}, amd, {})) == 1, f"negative control {k} caught")
    ok(len(check({}, amd, {"p": ["NoSuchMetric"]})) == 1, "negative control: unknown preset metric caught")
    # Recurrence (HRX-like T7 / E483): lint_counters prices its rules on raw SQ_WAIT_* / SQ_WAVE_CYCLES, while the presets collected
    # only derived percentages, so q197 ran a hand-written rocprofv3 pass beside counters.sh. Every raw counter it reads must be in a preset.
    sys.path.insert(0, H)
    import lint_counters
    raw = set(lint_counters.WAITS.values()) | {"SQ_WAVE_CYCLES"}
    have = {c for v in pre.values() for c in v}
    ok(raw <= have, f"lint_counters' raw wait counters are collected by a counters.sh preset (missing: {sorted(raw - have)})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
