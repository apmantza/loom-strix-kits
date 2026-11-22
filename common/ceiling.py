#!/usr/bin/env python3
"""Hardware floors for the lanes of a Loom LLM on gfx1151, from a constants file (kits/<kit>/lanes.json).

A floor is what the lane would cost at the hardware limit: decode lanes are DRAM-bound (bytes read per verify step /
practical bandwidth), prefill lanes compute-bound (useful FLOPs / f16 WMMA peak). Lanes without a floor (drafter, dequant,
GDN, producers) are fixed costs: judge them against zero. Floors are per verify step (decode, ms) or per prompt (prefill, s).

usage: ceiling.py MODEL.json [--context 81432] [--prompt 81432]
"""
import argparse
import json


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
    if kind == "decode_kv":
        return m["attn_layers"] * context * m["kv_bytes_per_pos_layer"] / bw * 1e3
    if kind == "prefill_linear":
        return 2 * m["linear_params"] * prompt / peak
    if kind == "prefill_attn":   # causal: query i attends i + 1 keys; QK^T and PV are 2 FLOPs per multiply-add each
        return 4 * m["attn_layers"] * m["q_heads"] * m["head_dim"] * (prompt * (prompt + 1) / 2) / peak
    if kind is None:
        return None
    raise SystemExit(f"ceiling.py: unknown floor kind {kind!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--context", type=int, default=81432, help="positions a verify step attends (decode floors)")
    ap.add_argument("--prompt", type=int, default=81432, help="prompt tokens (prefill floors)")
    a = ap.parse_args()
    d = json.load(open(a.model))
    m, hw = d["model"], d["hardware"]
    print(f"{m['name']} on {hw['name']}: {hw['bw_practical_gbs']} GB/s practical, {hw['wmma_f16_tflops']} TFLOP/s f16 WMMA")
    for ln in d["lanes"]:
        f = floor(ln["floor"], m, hw, a.context, a.prompt)
        unit = "ms/step" if ln["phase"] == "decode" else "s/prompt"
        print(f"  {ln['lane']:45s} " + (f"floor {f:9.2f} {unit}" if f is not None else "no floor (fixed cost: judge against 0)")
              + (f"  [{ln['note']}]" if ln.get("note") else ""))


if __name__ == "__main__":
    main()
