#!/usr/bin/env python3
"""hf_names/nemotron_h_moe.json against the real tensor tables (fixtures/nemotron_h_moe_tensors.json: the shipping 24GB GGUF and the
BF16 release's safetensors headers). Recurrences it prevents (as test_hf_names_inkling2.py): a tensor the map drops or reads twice
(a key renamed upstream), a block given the wrong mixer kind, and a pairing with the wrong tensor, e.g. the latent projections
swapped (fc1 [1024, 4096] and fc2 [4096, 1024] are each other's transpose: only the shape check tells them apart) or a stacked
expert tensor fed from the shared expert. expand() is the map's meaning.
usage: python3 kit/test_hf_names_nemotron_h_moe.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def expand(m, experts):
    """GGUF name -> (spec, [HF keys]) for every block; experts: MoE layer -> expert count"""
    out = {n: (s, [s["hf"]]) for n, s in m["global"].items()}
    pre = m["layer"]["prefix"]
    for i, kind in enumerate(m["kinds"]):
        for part in ("common", kind):
            for n, s in m["layer"][part].items():
                if "gguf_only" in s:
                    keys = []
                elif "stack" in s:
                    keys = [pre.format(i=i) + s["hf"].format(e=e) for e in range(experts[str(i)])]
                else:
                    keys = [pre.format(i=i) + s["hf"]]
                out[f"blk.{i}.{n}"] = (s, keys)
    return out


def ggml_shape(spec, hf_shape, count=None):
    """the ggml shape the map's prep and stack rules give for an HF shape"""
    s = list(hf_shape)
    prep = spec.get("prep", "")
    if prep == "squeeze":
        s = [d for d in s if d != 1]
    elif prep.startswith("groups:"):
        n = int(prep.split(":")[1])
        s = [n, s[0] // n]
    elif prep in ("row", "neg_exp"):
        s = s + [1]                              # ggml [1, n]: ne0 = 1
    if "stack" in spec:
        s = [count] + s
    return s[::-1]


def main():
    m = json.load(open(os.path.join(HERE, "hf_names", "nemotron_h_moe.json")))
    fx = json.load(open(os.path.join(HERE, "fixtures", "nemotron_h_moe_tensors.json")))
    ok(m["kinds"] == fx["layers_block_type"] and m["layers"] == len(m["kinds"]), f"block kinds = the config's ({m['layers']} blocks)")
    experts = {L: e["count"] for L, e in fx["hf_experts"].items()}
    exp = expand(m, experts)
    gg = {n: (s, t) for n, s, t in fx["gguf"]}
    ok(set(exp) == set(gg), f"every GGUF tensor mapped once, nothing extra (missing {sorted(set(gg) - set(exp))[:4]}, "
                            f"extra {sorted(set(exp) - set(gg))[:4]})")
    hf = dict(fx["hf"])
    for L, e in fx["hf_experts"].items():
        for k in range(e["count"]):
            hf[f"language_model.backbone.layers.{L}.mixer.experts.{k}.up_proj.weight"] = e["up"]
            hf[f"language_model.backbone.layers.{L}.mixer.experts.{k}.down_proj.weight"] = e["down"]
    used = [k for _, keys in exp.values() for k in keys]
    ok(len(used) == len(set(used)) and set(used) == set(hf),
       f"every HF tensor read once ({len(used)} reads, {len(hf)} keys; unread {sorted(set(hf) - set(used))[:3]}, "
       f"twice {len(used) - len(set(used))})")
    bad = []
    for n, (spec, keys) in exp.items():
        if not keys or n not in gg:
            continue
        shapes = {tuple(hf[k]) for k in keys if k in hf}
        if len(shapes) != 1:
            bad.append((n, "mixed or missing HF shapes"))
            continue
        want = ggml_shape(spec, list(shapes.pop()), len(keys))
        if want != gg[n][0]:
            bad.append((n, want, gg[n][0]))
    ok(not bad, f"every pairing has the GGUF shape its rule gives ({len(bad)} wrong: {bad[:3]})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
