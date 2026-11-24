#!/usr/bin/env python3
"""Hardware floors for the lanes of a Loom LLM on gfx1151, from a constants file (kits/<kit>/lanes.json).

A floor is what the lane would cost at the hardware limit: decode lanes are DRAM-bound (bytes read per verify step /
practical bandwidth), prefill lanes compute-bound (useful FLOPs / f16 WMMA peak). Lanes without a floor (drafter, dequant,
GDN, producers) are fixed costs: judge them against zero. Floors are per verify step (decode, ms) or per prompt (prefill, s).

The hardware block comes from MODEL.json unless --device names a device file (common/devices/<gfx>.json): its profile
(--profile, default the file's default_profile) replaces it, so the floors are the ones the machine reaches at that power
setting (on gfx1151 the 85 W cap: ~2.14 GHz, 43.8 TFLOP/s, 240 GB/s). --clock-ghz re-derives the WMMA peak for another clock.

usage: ceiling.py MODEL.json [--context 81432] [--prompt 81432] [--device common/devices/gfx1151.json [--profile cap85w]]
                  [--clock-ghz 2.14]
"""
import argparse
import json


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--context", type=int, default=81432, help="positions a verify step attends (decode floors)")
    ap.add_argument("--prompt", type=int, default=81432, help="prompt tokens (prefill floors)")
    ap.add_argument("--device", help="device file (common/devices/<gfx>.json): its profile replaces MODEL.json's hardware")
    ap.add_argument("--profile", help="the device file's profile (default: its default_profile)")
    ap.add_argument("--clock-ghz", type=float, help="shader clock for the WMMA peak (needs --device)")
    a = ap.parse_args()
    d = json.load(open(a.model))
    m, hw = d["model"], hardware(d, a.device, a.profile, a.clock_ghz)
    print(f"{m['name']} on {hw['name']}: {hw['bw_practical_gbs']} GB/s practical, {hw['wmma_f16_tflops']} TFLOP/s f16 WMMA"
          + (f" (clock {hw['clock_ghz']} GHz" + (f", {hw['power_w']} W cap)" if hw.get("power_w") else ")") if hw.get("clock_ghz") else ""))
    for ln in d["lanes"]:
        try:
            f = floor(ln["floor"], m, hw, a.context, a.prompt)
        except KeyError as e:   # a new kit's lanes.json before every constant is known (common/new_kit.sh): say which one
            print(f"  {ln['lane']:45s} floor unknown: lanes.json model has no {e.args[0]!r}")
            continue
        unit = "ms/step" if ln["phase"] == "decode" else "s/prompt"
        print(f"  {ln['lane']:45s} " + (f"floor {f:9.2f} {unit}" if f is not None else "no floor (fixed cost: judge against 0)")
              + (f"  [{ln['note']}]" if ln.get("note") else ""))


if __name__ == "__main__":
    main()
