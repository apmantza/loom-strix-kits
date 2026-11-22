#!/usr/bin/env python3
"""Tests of make_lanes.py on synthetic safetensors headers (no weights needed). Each case names the recurrence it prevents.
usage: python3 test_make_lanes.py (exit 1 on a failure)"""
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
fails = []


def case(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def st(path, tensors):   # name -> (shape, bytes): a header with offsets; the payload is never read
    h, off = {}, 0
    for n, (shape, b) in tensors.items():
        h[n] = {"dtype": "U8", "shape": shape, "data_offsets": [off, off + b]}
        off += b
    j = json.dumps(h).encode()
    open(path, "wb").write(struct.pack("<Q", len(j)) + j)


def run(path, *args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_lanes.py"), path, "--q-heads", "8", "--kv-heads", "2", "--head-dim", "64", *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


with tempfile.TemporaryDirectory() as d:
    dense = {"model.embed_tokens.weight": ([100, 16], 1000), "lm_head.weight": ([100, 16], 800),
             "model.layers.0.self_attn.k_proj.weight": ([16, 16], 10), "model.layers.0.mlp.up_proj.weight": ([32, 16], 20),
             "model.layers.1.linear_attn.in_proj.weight": ([16, 16], 10), "model.layers.1.mlp.up_proj.weight": ([32, 16], 20),
             "model.layers.1.input_layernorm.weight": ([16], 4),
             "mtp.layers.0.mlp.up_proj.weight": ([32, 16], 500)}
    st(f"{d}/dense.safetensors", dense)
    m = run(f"{d}/dense.safetensors")
    # a dense model: decode reads every matrix but the embedding (head included), not norms
    case("dense: decode = matrices + head, no embedding, no norm, no MTP", m["decode_weight_bytes"] == 10 + 20 + 10 + 20 + 800)
    # the stale floor: an MTP layer nobody reads during DFlash serving inflated the decode floor by 2 % (Qwen3.8, 351 MB)
    case("MTP tensors excluded by default (their 500 B absent)", m["decode_weight_bytes"] == 860)
    case("--include-mtp counts them", run(f"{d}/dense.safetensors", "--include-mtp")["decode_weight_bytes"] == 860 + 500)
    # prefill computes the head for one row per chunk: not a per-token GEMM
    case("linear_params excludes the head", m["linear_params"] == 256 + 512 + 256 + 512)
    # a GDN layer's fused projection is not full attention (the first version counted 65 attention layers for 16)
    case("attention layers = layers with a K projection", m["attn_layers"] == 1 and m["layers"] == 2)
    case("kv bytes f16 = 2 x kv_heads x head_dim x 2", m["kv_bytes_per_pos_layer"] == 2 * 2 * 64 * 2)
    moe = {"model.layers.0.self_attn.k_proj.weight": ([16, 16], 10), "model.layers.0.mlp.experts.up_proj": ([4, 32, 16], 400)}
    st(f"{d}/moe.safetensors", moe)
    m1 = run(f"{d}/moe.safetensors", "--experts", "4", "--top-k", "1", "--verify-tokens", "1")
    m8 = run(f"{d}/moe.safetensors", "--experts", "4", "--top-k", "1", "--verify-tokens", "8")
    # MoE decode reads the experts the step's tokens pick: 1 token -> k/n, 8 tokens -> the expected union
    case("MoE, one token: experts x k/n", m1["decode_weight_bytes"] == 10 + 100)
    case("MoE, 8-token verify: experts x expected union", m8["decode_weight_bytes"] == 10 + round(400 * (1 - 0.75 ** 8)))
    case("MoE prefill gemm bytes count every expert", m1["gemm_weight_bytes"] == 410)
    # prefill FLOPs per token run only the routed experts (a dense count overstates an MoE prefill floor n/k times)
    case("MoE linear_params count top-k of n experts", m1["linear_params"] == 16 * 16 + (4 * 32 * 16) // 4)
    # NVFP4: U8 payload with an F8 scale sibling holds 2 values per byte; scales are bytes, not multiply-adds
    h, off = {}, 0
    for nme, (shape, b, dt) in {"model.layers.0.mlp.experts.w13_weight": ([2, 8, 4], 64, "U8"),
                                "model.layers.0.mlp.experts.w13_weight_scale": ([2, 8, 1], 16, "F8_E4M3")}.items():
        h[nme] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + b]}; off += b
    j = json.dumps(h).encode(); open(f"{d}/fp4.safetensors", "wb").write(struct.pack("<Q", len(j)) + j)
    mf = run(f"{d}/fp4.safetensors", "--experts", "2", "--top-k", "1", "--verify-tokens", "1")
    case("NVFP4: packed elements x2, scales not multiply-adds, both in bytes", mf["linear_params"] == (2 * 8 * 4 * 2) // 2 and mf["decode_weight_bytes"] == (64 + 16) // 2)
    # looped layers: weights read again on every pass (they do not fit any cache)
    st(f"{d}/loop.safetensors", {f"model.layers.{i}.mlp.up_proj.weight": ([8, 8], 10) for i in range(4)})
    ml = run(f"{d}/loop.safetensors", "--loop", "1:2:3")
    case("loop 1:2:3 -> layers 1-2 counted three times", ml["decode_weight_bytes"] == 10 + 30 + 30 + 10 and ml["gemm_weight_bytes"] == 40)
print(f"{13 - len(fails)}/13 passed" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
