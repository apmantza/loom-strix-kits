#!/usr/bin/env python3
"""Model constants for the hardware floors (common/ceiling.py, events_summary.py) computed from the weights, not by hand.

Reads tensor names, shapes and byte sizes from a GGUF (the gguf package) or a safetensors file (its JSON header only: a local
file, or a Hugging Face URL read with two HTTP range requests, so a 20 GB checkpoint is never downloaded). Prints the "model"
block of kits/<kit>/lanes.json with a _src line saying how each number was made, or writes it into a lanes.json (--write).

  decode_weight_bytes   bytes one decode step reads: every matrix except the token embedding (read one row per token); MoE
                        expert tensors count the experts a verify step of --verify-tokens tokens touches (expected union of
                        top-k picks, independent routing assumed), looped layers count --loop times
  gemm_weight_bytes     the same minus the output head and with every expert (prefill touches them all); looped layers once
  linear_params         multiply-adds per token: matrix elements (experts: top-k of them), looped layers x --loop
  layers, attn_layers, q_heads, kv_heads, head_dim, kv_bytes_per_pos_layer (--kv f16 | kv8)
  prefill_chunk, gdn_state_bytes (--gdn-state-mib from a build log), drafter_* (--drafter GGUF)
Floors drift with the model file (an 8 % stale floor, E-log): rerun this whenever a kit's weights change.
usage: make_lanes.py WEIGHTS [--config config.json] [--experts N --top-k K] [--loop FIRST:LAST:TIMES] [--verify-tokens 8]
                     [--kv kv8] [--prefill-chunk 4096] [--gdn-state-mib X] [--drafter GGUF] [--name NAME] [--write lanes.json]"""
import argparse
import json
import os
import re
import struct
import sys
import urllib.request

EMBED = re.compile(r"(^|\.)(token_embd|embed_tokens|wte|tok_embeddings|embed)(\.|$)")
HEAD = re.compile(r"(^|\.)(output|lm_head|unembed)\.weight$|^output\.weight$")
EXPERT = re.compile(r"_exps\.|\.experts\.")
LAYER = re.compile(r"(?:^|\.)(?:blk|layers|h)\.(\d+)\.")
ATTN = re.compile(r"attn_k\.weight$|\.k_proj\.weight$|\.wk_dv\.weight$")
VISION = re.compile(r"(^|\.)(visual|vision_tower|vision_model|vision|mm_projector|v\.blk)\.|^v\.")   # image encoders run per image, not per token   # full attention has its own K projection (a GDN layer's fused attn_qkv does not count)
MTP = re.compile(r"nextn|(^|\.)mtp[._]")


def read_gguf(path):
    from gguf import GGUFReader
    r = GGUFReader(path)
    meta = {}
    for k, f in r.fields.items():
        try:
            v = f.parts[f.data[0]] if f.data else None
            meta[k] = v.tolist()[0] if hasattr(v, "tolist") and len(v) == 1 else (bytes(v).decode() if hasattr(v, "dtype") and v.dtype.kind == "u" and len(v) > 1 else v)
        except Exception:   # noqa: BLE001 - metadata we cannot decode is not needed for byte counts
            pass
    tensors = {t.name: {"shape": [int(x) for x in t.shape], "bytes": int(t.n_bytes)} for t in r.tensors}
    return tensors, meta


def _header(path):
    if path.startswith("http"):
        get = lambda a, b: urllib.request.urlopen(urllib.request.Request(path, headers={"Range": f"bytes={a}-{b}", "User-Agent": "make_lanes"}), timeout=60).read()
        n = struct.unpack("<Q", get(0, 7))[0]
        return json.loads(get(8, 8 + n - 1))
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def read_safetensors(path):
    h = _header(path)
    tensors = {k: {"shape": v["shape"], "bytes": v["data_offsets"][1] - v["data_offsets"][0], "dtype": v["dtype"]}
               for k, v in h.items() if k != "__metadata__"}
    return tensors, h.get("__metadata__", {})


def expected_union(n, k, t):
    """expected number of distinct experts t tokens touch when each picks k of n independently"""
    return n * (1 - (1 - k / n) ** t)


def constants(tensors, a, cfg):
    loop = None
    if a.loop:
        f, l, times = (int(x) for x in a.loop.split(":"))
        loop = (f, l, times)
    layer_of = lambda name: int(LAYER.search(name).group(1)) if LAYER.search(name) else None
    times_of = lambda name: (loop[2] if loop and layer_of(name) is not None and loop[0] <= layer_of(name) <= loop[1] else 1)
    # a GGUF MTP layer is a whole blk.N carrying nextn tensors; safetensors MTP tensors are named mtp.* (their 'layers.0' is not
    # the model's layer 0, so only the name excludes them)
    mtp_layers = {layer_of(n) for n in tensors if "nextn" in n and n.startswith("blk.")}
    skip_mtp = lambda n: (not a.include_mtp and (bool(MTP.search(n)) or layer_of(n) in mtp_layers)) or bool(VISION.search(n))
    used = None
    if a.program:   # only what the served program references (command.parameter keys)
        txt = open(a.program).read()
        used = {n for n in tensors if f'"{n}"' in txt}
    dec = gemm = params = head = 0.0
    n_exp, k_exp = a.experts, a.top_k
    for name, t in tensors.items():
        if len(t["shape"]) < 2 or EMBED.search(name) or "norm" in name or skip_mtp(name) or (used is not None and name not in used):
            continue
        is_exp = bool(EXPERT.search(name))
        elems = 1
        for x in t["shape"]:
            elems *= x
        # block-scaled formats: scale / amax tensors are bytes read, not multiply-adds; packed 4-bit payloads (U8 with a sibling
        # *_scale, e.g. NVFP4 e2m1) hold two values per byte (the first Inkling count read 3.2e9 against the card's ~5.4e9 with the loop)
        meta_only = bool(re.search(r"_(scale|amax|outer_scale_denominator|scale_inv)$", name))
        if t.get("dtype") == "U8" and f"{name}_scale" in tensors and tensors[f"{name}_scale"].get("dtype", "").startswith("F8"):
            elems *= 2
        if is_exp:
            assert n_exp and k_exp, f"{name} is an expert tensor: pass --experts and --top-k"
            frac_step = expected_union(n_exp, k_exp, a.verify_tokens) / n_exp
            frac_tok = k_exp / n_exp
        else:
            frac_step = frac_tok = 1.0
        times = times_of(name)
        dec += t["bytes"] * frac_step * times
        if HEAD.search(name):
            head += t["bytes"]   # prefill needs the head for one row per chunk: not in gemm bytes nor linear_params
        else:
            gemm += t["bytes"]
            if not meta_only:
                params += elems * frac_tok * times
    layers = len({layer_of(n) for n in tensors if layer_of(n) is not None and not skip_mtp(n)})
    attn = sorted({layer_of(n) for n in tensors if ATTN.search(n) and layer_of(n) is not None and not skip_mtp(n)})
    q_heads = a.q_heads or cfg.get("q_heads")
    kv_heads = a.kv_heads or cfg.get("kv_heads")
    head_dim = a.head_dim or cfg.get("head_dim")
    globals_ = [int(x) for x in a.global_layers.split(",")] if a.global_layers else None
    slots = lambda ls: sum(times_of(f"blk.{l}.") for l in ls)   # one KV slot per layer and pass
    kvb = None
    if kv_heads and head_dim:
        kvb = 2 * kv_heads * head_dim * 2 if a.kv == "f16" else 2 * kv_heads * head_dim + 2 * kv_heads * 8
    m = {"name": a.name, "layers": layers, "attn_layers": len(attn), "q_heads": q_heads, "kv_heads": kv_heads, "head_dim": head_dim,
         "decode_weight_bytes": round(dec), "gemm_weight_bytes": round(gemm), "linear_params": round(params),
         "kv_bytes_per_pos_layer": kvb, "prefill_chunk": a.prefill_chunk}
    if globals_ is not None:   # hybrid attention: global slots read the whole context, sliding slots their window
        m["attn_slots_global"] = slots([l for l in attn if l in globals_])
        m["attn_slots_sliding"] = slots([l for l in attn if l not in globals_])
        m["sliding_window"] = a.window
    if a.gdn_state_mib:
        m["gdn_state_bytes"] = round(a.gdn_state_mib * 1048576)
    src = (f"make_lanes.py on {os.path.basename(a.weights)}: decode = all matrices but the token embedding"
           + (f", experts x E[union of {a.verify_tokens} tokens' top-{k_exp} of {n_exp}]" if n_exp else "")
           + (f", layers {loop[0]}-{loop[1]} x{loop[2]}" if loop else "") + "; gemm = the same without the head, every expert, loops once"
           + f"; linear_params = matrix elements per token without the head; kv = {a.kv}; attention layers {attn[:3]}...{attn[-2:] if attn else ''}"
           + ("" if a.include_mtp else f"; MTP layers {sorted(x for x in mtp_layers if x is not None)} excluded (not read when serving without MTP)")
           + (f"; restricted to the {len(used)} tensors {os.path.basename(a.program)} references" if used is not None else ""))
    return m, src, head


def drafter_constants(path, verify_tokens):
    tensors, meta = read_gguf(path)
    w = sum(t["bytes"] for n, t in tensors.items() if not EMBED.search(n))
    arch = meta.get("general.architecture", "")
    layers = 1 + max((int(LAYER.search(n).group(1)) for n in tensors if LAYER.search(n)), default=-1)
    window = meta.get(f"{arch}.attention.sliding_window")
    return {"drafter_weight_bytes": w, "drafter_layers": layers, "drafter_window": window}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="a .gguf, a .safetensors, or an https URL of a .safetensors file")
    ap.add_argument("--config", help="HF config.json (heads, head_dim) when the weights do not carry them")
    ap.add_argument("--experts", type=int); ap.add_argument("--top-k", type=int)
    ap.add_argument("--loop", help="FIRST:LAST:TIMES, e.g. 6:17:3 (layers run TIMES per token, shared weights)")
    ap.add_argument("--verify-tokens", type=int, default=8, help="tokens per decode step (speculative verify width; 1 = plain decode)")
    ap.add_argument("--kv", choices=("f16", "kv8"), default="f16")
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--gdn-state-mib", type=float)
    ap.add_argument("--q-heads", type=int); ap.add_argument("--kv-heads", type=int); ap.add_argument("--head-dim", type=int)
    ap.add_argument("--drafter")
    ap.add_argument("--global-layers", help="comma list of full-context attention layers (the rest slide over --window)")
    ap.add_argument("--window", type=int, help="sliding attention window in positions")
    ap.add_argument("--include-mtp", action="store_true", help="count MTP / nextn layers (serving that drafts with them)")
    ap.add_argument("--program", help="a served llm.loom: count only the tensors it references")
    ap.add_argument("--name", default="")
    ap.add_argument("--write", help="a lanes.json whose model block to replace (hardware and lanes kept)")
    a = ap.parse_args()
    if a.weights.endswith(".gguf"):
        tensors, meta = read_gguf(a.weights)
        arch = meta.get("general.architecture", "")
        cfg = {"q_heads": meta.get(f"{arch}.attention.head_count"), "kv_heads": meta.get(f"{arch}.attention.head_count_kv"),
               "head_dim": meta.get(f"{arch}.attention.key_length")}
        a.experts = a.experts or meta.get(f"{arch}.expert_count")
        a.top_k = a.top_k or meta.get(f"{arch}.expert_used_count")
    else:
        tensors, _ = read_safetensors(a.weights)
        cfg = {}
    if a.config:
        c = json.load(open(a.config))
        c = c.get("text_config", c)
        cfg = {"q_heads": c.get("num_attention_heads"), "kv_heads": c.get("num_key_value_heads"), "head_dim": c.get("head_dim")}
        a.experts = a.experts or c.get("n_routed_experts") or c.get("num_experts")
        a.top_k = a.top_k or c.get("num_experts_per_tok")
    m, src, head = constants(tensors, a, cfg)
    if a.drafter:
        m.update(drafter_constants(a.drafter, a.verify_tokens))
        src += f"; drafter_* from {os.path.basename(a.drafter)} (weights without embedding; head copy and warm bytes are kit-specific: add by hand)"
    m["_src"] = src
    print(json.dumps(m, indent=1))
    print(f"(output head {head / 1e9:.3f} GB counted in decode, not in gemm)", file=sys.stderr)
    if a.write:
        d = json.load(open(a.write))
        keep = {k: v for k, v in d.get("model", {}).items() if k.startswith("drafter_") and k not in m}
        d["model"] = {**keep, **m}
        json.dump(d, open(a.write, "w"), indent=1)
        open(a.write, "a").write("\n")
        print(f"wrote the model block of {a.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
