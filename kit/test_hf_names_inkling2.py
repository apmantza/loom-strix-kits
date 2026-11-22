#!/usr/bin/env python3
"""hf_names/inkling2.json against the real tensor lists (fixtures/inkling2_tensors.json: the native header's text keys and
the Unsloth UD-Q8_K_XL GGUF's names). Recurrences it prevents: a text tensor the map drops or reads twice (a key renamed
upstream), an internal name the GGUF route has and the native route lacks (or the reverse), and a pairing with the wrong
tensor, e.g. attn_res queries off by one layer (only the [3, 2304] block-boundary queries tell them apart: the shape check).
expand() is the map's meaning (inspect_safetensors.py has no MoE / loop / NVFP4 path yet; it takes this over when it does).
usage: python3 kit/test_hf_names_inkling2.py"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def expand(m, layers_kind=None):
    """internal name -> (spec, hf key) for every layer of the map."""
    out = {}
    for n, s in m["global"].items():
        out[n] = (s, s["hf"])
    p = m["layer"]["prefix"]
    for i in range(m["layers"]):
        parts = ["common", "global" if i in m["global_layers"] else "local"]
        if m["loop"]["first"] <= i <= m["loop"]["last"]:
            parts.append("loop")
        parts.append("nvfp4" if i in m["expert_formats"]["nvfp4"] else "mxfp8")
        for part in parts:
            for n, s in m["layer"][part].items():
                if not n.startswith("_"):
                    out[f"blk.{i}.{n}"] = (s, p.format(i=i) + s["hf"])
                    if "gguf_scale" in s:                # derived (amax, outer): no native key of its own
                        out[f"blk.{i}.{s['gguf_scale']}"] = ({"derived": True}, None)
        for n, s in m["indexed"].items():
            if not n.startswith("_") and i >= s.get("first", 0):
                out[f"blk.{i}.{n}"] = (s, s["hf"].format(i=i, im1=i - 1))
    return out


def shape_ok(spec, hf, gg):
    (hd, hs), (gd, gs) = hf, gg
    if "amax" in spec:                                   # NVFP4: HF [E, rows, K/2] U8 -> GGUF [K, rows, E]
        return gd == "NVFP4" and gs == [hs[2] * 2, hs[1], hs[0]]
    if "scale" in spec:                                  # MXFP8: HF [E, rows, K] f8 -> GGUF (bf16 upcast) [K, rows, E]
        return gs == [hs[2], hs[1], hs[0]]
    if spec.get("transpose"):                            # transposed by the converter: GGUF ne order = the HF shape
        return gs == hs
    if spec.get("shape"):                                # conv1d [C, 1, K] -> [K, C]
        return gs == [hs[2], hs[0]]
    return gs == list(reversed(hs))


def check(m, fx):
    errs, used = [], {}
    names = expand(m)
    for n, (spec, hf) in names.items():
        if spec.get("derived"):
            if fx["gguf"].get(n, [None, None])[1] != [128]:
                errs.append(f"{n}: derived per-expert scale missing or not [128] in the GGUF")
            continue
        if hf not in fx["hf"]:
            errs.append(f"{n}: {hf} not in the native header")
            continue
        used.setdefault(hf, []).append(n)
        for extra in ("scale", "amax", "outer"):
            if extra in spec:
                k = m["layer"]["prefix"].format(i=int(n.split(".")[1])) + spec[extra]
                (used.setdefault(k, []).append(n)) if k in fx["hf"] else errs.append(f"{n}: {k} not in the native header")
        if n not in fx["gguf"]:
            errs.append(f"{n}: not a GGUF name")
        elif not shape_ok(spec, fx["hf"][hf], fx["gguf"][n]):
            errs.append(f"{n}: shapes disagree, native {hf} {fx['hf'][hf]} vs GGUF {fx['gguf'][n]}")
    errs += [f"{k}: native key not mapped" for k in fx["hf"] if k not in used]
    errs += [f"{k}: read by {v}" for k, v in used.items() if len(v) > 1]
    errs += [f"{g}: GGUF name without a map entry" for g in fx["gguf"] if g not in names]
    return errs


def main():
    m = json.load(open(os.path.join(HERE, "hf_names", "inkling2.json")))
    fx = json.load(open(os.path.join(HERE, "fixtures", "inkling2_tensors.json")))
    assert len(fx["hf"]) > 800 and len(fx["gguf"]) > 700, "fixture lost its tensor lists"
    errs = check(m, fx)
    for e in errs[:40]:
        print("FAIL", e)
    # negative controls: each must be caught
    bad = json.loads(json.dumps(m)); bad["indexed"]["attn_res_query.weight"]["hf"] = "model.llm.attn_res.attn_queries.{i}"
    bad["indexed"]["attn_res_query.weight"]["first"] = 0
    bad2 = json.loads(json.dumps(m)); del bad2["layer"]["loop"]["loop_k_gain.weight"]
    bad3 = json.loads(json.dumps(m)); bad3["layer"]["common"]["attn_k.weight"]["hf"] = "attn.wv_dv.weight"
    ctl = [("attn_res off by one", bad), ("a loop tensor dropped", bad2), ("k read as v", bad3)]
    missed = [name for name, mm in ctl if not check(mm, fx)]
    for name in missed:
        print("FAIL negative control passed:", name)
    n = len(expand(m))
    print(f"hf_names/inkling2: {n} internal names, {len(fx['hf'])} native keys, {len(fx['gguf'])} GGUF names; "
          f"{len(errs)} errors; {len(ctl) - len(missed)}/{len(ctl)} negative controls caught")
    sys.exit(1 if errs or missed else 0)


if __name__ == "__main__":
    main()
