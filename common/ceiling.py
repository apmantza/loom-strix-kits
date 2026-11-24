#!/usr/bin/env python3
"""Hardware floors for the lanes of a Loom LLM on gfx1151, from a constants file (kits/<kit>/lanes.json).

A floor is what the lane would cost at the hardware limit: decode lanes are DRAM-bound (bytes read per verify step /
practical bandwidth), prefill lanes compute-bound (useful FLOPs / f16 WMMA peak). Lanes without a floor (drafter, dequant,
GDN, producers) are fixed costs: judge them against zero. Floors are per verify step (decode, ms) or per prompt (prefill, s).

Latency chain (optional lane field "chain"): a lane whose work is a chain of dependent steps cannot run faster than that
chain, whatever its bytes or FLOPs (SGLang's performance-cuda.md: time >= max(critical path, most-loaded resource)). The
field lists the lane's dispatches per step / prompt as items {"what", "dispatches", "trips", "serial_cycles", "_src"}:
dispatches on one queue run back to back (each pays the device's launch_gap_us), trips = dependent memory round trips on
one dispatch's critical path (a load whose address needs the previous load: page record -> page map -> K/V), each costing
the device's global_load_latency_ns; serial_cycles = dependent issue cycles on one wave's critical path (a loop carried
from step to step), at the profile's clock_ghz. Count what the algorithm forces, not what the current code happens to
serialise: tiles a kernel could prefetch are not a chain. A count or device constant may be "unknown" (or absent): that part
is left out and named, so the chain term stays a lower bound and the old floor stands unless the known part exceeds it.
The lane's floor is max(resource floor, chain) and the output names the term that binds.

The hardware block comes from MODEL.json unless --device names a device file (common/devices/<gfx>.json): its profile
(--profile, default the file's default_profile) replaces it, so the floors are the ones the machine reaches at that power
setting (on gfx1151 the 85 W cap: ~2.14 GHz, 43.8 TFLOP/s, 240 GB/s). --clock-ghz re-derives the WMMA peak for another clock.

usage: ceiling.py MODEL.json [--context 81432] [--prompt 81432] [--device common/devices/gfx1151.json [--profile cap85w]]
                  [--clock-ghz 2.14] [--trip-ns 800]
  --trip-ns: what-if global load latency per dependent round trip (replaces the device file's global_load_latency_ns)
"""
import argparse
import json
import math


def hardware(d, device=None, profile=None, clock_ghz=None):
    """The hardware block floors use: MODEL.json's, or a device file's profile; clock_ghz re-derives the WMMA peak."""
    if device is None:
        if profile or clock_ghz:
            raise SystemExit("ceiling.py: --profile / --clock-ghz need --device (the clock-to-peak constants live there)")
        return d["hardware"]
    dev = json.load(open(device)) if isinstance(device, str) else device
    name = profile or dev.get("default_profile")
    if name not in dev.get("profiles", {}):
        raise SystemExit(f"ceiling.py: device has no profile {name!r} (has: {', '.join(dev.get('profiles', {}))})")
    hw = dict(dev["profiles"][name], name=f"{dev['name']} [{name}]")
    if clock_ghz:
        hw["clock_ghz"] = clock_ghz
        hw.pop("wmma_f16_tflops", None)
    if "wmma_f16_tflops" not in hw:
        if "clock_ghz" not in hw:
            raise SystemExit(f"ceiling.py: profile {name!r} gives neither wmma_f16_tflops nor clock_ghz")
        hw["wmma_f16_tflops"] = round(dev["simds"] * dev["wmma_flop"] / dev["wmma_cycles"] * hw["clock_ghz"] * 1e9 / 1e12, 2)
    return hw


def floor(kind, m, hw, context=0, prompt=0):
    """ms per verify step (decode kinds) or s per prompt (prefill kinds); None for lanes without a floor"""
    bw = hw["bw_practical_gbs"] * 1e9
    peak = hw["wmma_f16_tflops"] * 1e12
    if kind == "decode_weights":
        return m["decode_weight_bytes"] / bw * 1e3
    if kind == "decode_drafter":   # DFlash2 per verify step: propose pass (own tensors + proposal-head copy), warm pass
        # (fc + K/V projections of the committed taps), and its sliding-window context K/V
        kvb = m["drafter_layers"] * min(context, m["drafter_window"]) * m["drafter_kv_bytes_per_pos_layer"]
        return (m["drafter_weight_bytes"] + m["drafter_head_bytes"] + m["drafter_warm_bytes"] + kvb) / bw * 1e3
    if kind == "decode_gdn_state":   # the recurrent state of every GDN layer read and written once per verify step
        return 2 * m["gdn_state_bytes"] / bw * 1e3
    if kind == "prefill_dequant":   # design floor of the f16-scratch prefill: per chunk, read the quantized GEMM weights and
        # write them as f16 (2 B per linear parameter); ideal is 0 (a GEMM that reads quantized weights directly)
        chunks = -(-prompt // m["prefill_chunk"])
        return chunks * (m["gemm_weight_bytes"] + 2 * m["linear_params"]) / bw
    if kind == "decode_kv":   # hybrids (make_lanes --global-layers): global slots read the context, sliding slots their window
        if "attn_slots_global" in m:
            pos = m["attn_slots_global"] * context + m["attn_slots_sliding"] * min(context, m["sliding_window"])
        else:
            pos = m["attn_layers"] * context
        return pos * m["kv_bytes_per_pos_layer"] / bw * 1e3
    if kind == "prefill_linear":
        return 2 * m["linear_params"] * prompt / peak
    if kind == "prefill_attn":   # causal: query i attends i + 1 keys; QK^T and PV are 2 FLOPs per multiply-add each
        full = prompt * (prompt + 1) / 2
        if "attn_slots_global" in m:   # hybrids: a sliding slot's query i attends min(i + 1, window) keys
            w = m["sliding_window"]
            win = w * (w + 1) / 2 + max(prompt - w, 0) * w if prompt > w else full
            pairs = m["attn_slots_global"] * full + m["attn_slots_sliding"] * win
        else:
            pairs = m["attn_layers"] * full
        return 4 * m["q_heads"] * m["head_dim"] * pairs / peak
    if kind is None:
        return None
    raise SystemExit(f"ceiling.py: unknown floor kind {kind!r}")


RESOURCE = {"decode_weights": "bandwidth", "decode_drafter": "bandwidth", "decode_gdn_state": "bandwidth",
            "decode_kv": "bandwidth", "prefill_dequant": "bandwidth", "prefill_linear": "compute", "prefill_attn": "compute"}


def known(v, what):
    """a count or device constant: a number >= 0, or None when unknown (null, absent, or the string 'unknown')"""
    if v is None or v == "unknown":
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise SystemExit(f"ceiling.py: {what} must be a number >= 0 or 'unknown', got {v!r}")
    return float(v)


def chain_floor(chain, hw):
    """(seconds of the chain's known parts, or None when no part is known; [the unknown parts, named])"""
    dev = {"launch": (known(hw.get("launch_gap_us"), "launch_gap_us"), 1e-6, "launch_gap_us"),
           "trips": (known(hw.get("global_load_latency_ns"), "global_load_latency_ns"), 1e-9, "global_load_latency_ns"),
           "serial_cycles": (known(hw.get("clock_ghz"), "clock_ghz"), None, "clock_ghz")}
    total, seen, unknown = 0.0, False, []
    for it in chain:
        what = it.get("what", "?")
        n = known(it.get("dispatches"), f"chain {what}: dispatches")
        if n is None:
            unknown.append(f"{what}: dispatches")
            continue
        for part, count in (("launch", 1.0), ("trips", it.get("trips")), ("serial_cycles", it.get("serial_cycles"))):
            if part != "launch" and part not in it:
                continue   # not declared: contributes nothing (a lower bound stays a lower bound)
            c = known(count, f"chain {what}: {part}")
            rate, scale, dname = dev[part]
            if c is None:
                unknown.append(f"{what}: {part}")
            elif rate is None:
                if c:
                    unknown.append(f"{dname} (device)")
            else:
                total += n * c * (rate * scale if scale else 1 / (rate * 1e9))
                seen = True
    return (total if seen else None), list(dict.fromkeys(unknown))


def lane_floor(ln, m, hw, context=0, prompt=0):
    """the lane's floor in its phase's unit (decode ms/step, prefill s/prompt): {"floor", "binds", "terms", "unknown"},
    or None for a lane with neither a resource floor nor a known chain. KeyError when lanes.json lacks a constant."""
    terms, unknown = {}, []
    res = floor(ln.get("floor"), m, hw, context, prompt)
    if res is not None:
        terms[RESOURCE[ln["floor"]]] = res
    if ln.get("chain"):
        c, unknown = chain_floor(ln["chain"], hw)
        if c is not None:
            terms["chain"] = c * (1e3 if ln["phase"] == "decode" else 1)
    if not terms:
        return None
    binds = max(terms, key=terms.get)
    return {"floor": terms[binds], "binds": binds, "terms": terms, "unknown": unknown}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--context", type=int, default=81432, help="positions a verify step attends (decode floors)")
    ap.add_argument("--prompt", type=int, default=81432, help="prompt tokens (prefill floors)")
    ap.add_argument("--device", help="device file (common/devices/<gfx>.json): its profile replaces MODEL.json's hardware")
    ap.add_argument("--profile", help="the device file's profile (default: its default_profile)")
    ap.add_argument("--clock-ghz", type=float, help="shader clock for the WMMA peak (needs --device)")
    ap.add_argument("--trip-ns", type=float, help="global load latency per dependent round trip (what-if)")
    a = ap.parse_args()
    d = json.load(open(a.model))
    m, hw = d["model"], dict(hardware(d, a.device, a.profile, a.clock_ghz))
    if a.trip_ns is not None:
        hw["global_load_latency_ns"] = a.trip_ns
    print(f"{m['name']} on {hw['name']}: {hw['bw_practical_gbs']} GB/s practical, {hw['wmma_f16_tflops']} TFLOP/s f16 WMMA"
          + (f" (clock {hw['clock_ghz']} GHz" + (f", {hw['power_w']} W cap)" if hw.get("power_w") else ")") if hw.get("clock_ghz") else ""))
    for ln in d["lanes"]:
        try:
            f = lane_floor(ln, m, hw, a.context, a.prompt)
        except KeyError as e:   # a new kit's lanes.json before every constant is known (common/new_kit.sh): say which one
            print(f"  {ln['lane']:45s} floor unknown: lanes.json model has no {e.args[0]!r}")
            continue
        unit = "ms/step" if ln["phase"] == "decode" else "s/prompt"
        why = ""
        if ln.get("chain"):
            c = f["terms"].get("chain") if f else None
            unk = f["unknown"] if f else chain_floor(ln["chain"], hw)[1]
            why = (f"  binds: {f['binds']}; " if f else "  ") + "chain " + (f"{c:.3f}" if c is not None else "unknown") \
                + (f" ({'lower bound; ' if c is not None else ''}unknown: {', '.join(unk)})" if unk else "")
        print(f"  {ln['lane']:45s} " + (f"floor {f['floor']:9.2f} {unit}" if f is not None else "no floor (fixed cost: judge against 0)")
              + why + (f"  [{ln['note']}]" if ln.get("note") else ""))


if __name__ == "__main__":
    main()
