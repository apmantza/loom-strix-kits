#!/usr/bin/env python3
"""Do N concurrent streams' histories fit a served package's KV pool without eviction? Static: reads the package, runs nothing.

Model-agnostic: every quantity comes from the package (model.json preset, the storage program, llm_config.json), the kit's
serve flags (kits/<KIT>/kit.env KIT_SERVE_ARGS), or an argument. A quantity none of them supplies is printed 'unknown'; the
verdict is then UNKNOWN (exit 2), never a guess.

How loom-serve sizes the pool (hrx-system loom/src/loom/serve/server/model_registration.c):
  SDK 0.1.x   the preset's `pool_capacity` (positions shared by all rows; doc default 65536)
  SDK 0.2/0.3 the preset field is gone; fixed backing: pool = `context_capacity` (one context shared by every resident row);
              elastic backing: pages from the backing pool's physical capacity or --memory_bytes (pass --pool: the served
              figure is text_memory.pool.capacity_tokens in the server's telemetry)
Demand (loom serve/service/text/README.md, "Admission and packing"): each admitted request holds its history plus its
page-rounded completion high-water (max_tokens) and speculative credit (draft_depth + 1 positions). Drafter K/V planes use the
target's page ids (the storage program: 'draft K/V ... over the pool's page ids'), so a drafter costs bytes, not positions.
Idle retained histories and prefix endpoints are evictable: they are what gets evicted, so the question is whether the ACTIVE
streams fit. A continuation miss (the next request does not extend the retained history exactly) can hold the old history and
the new request at once until the old one is evicted: --miss-tolerant budgets one such fork (the largest stream twice); the
exact model is the default.
Recurrent state (hybrid models: one fixed-size slot per row, the storage program's %recurrent_bytes) is bytes outside the
position pool, so it is reported beside the verdict, not in it. loom serve/service/text/source.c sizes the slots: rows when the
deployment has no cache endpoints, else 2 x rows (a fork slot per row) + endpoints, where endpoints = checkpoint_capacity +
prefix_cache_capacity + (session_cache_capacity + 1 when nonzero) (cache_policy.c); each endpoint is a recurrent-state checkpoint
a resumed turn restores. An automatic prefix or session count is chosen by cache_target_bytes (model-documents.md: default
2 GiB): its bytes are then bounded by that target, printed as 'at most'.

usage: kv_budget.py PKG --streams CTX[,CTX...] [--kit KIT | --backing fixed|elastic] [--pool N] [--completion N]
                    [--retained N] [--preset NAME] [--miss-tolerant]
       exit 0 FIT, 1 NO_FIT, 2 usage / invalid / UNKNOWN
"""
import argparse
import json
import os
import re
import subprocess
import sys

LSK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def storage_geometry(text):
    """{page_positions, kv_position_bytes, planes, draft_planes, draft_position_bytes} from a storage program (kit/ generator
    names); a field the program does not spell is None"""
    def const(name):
        m = re.search(r"%" + name + r" = scalar\.constant (\d+) : i64", text)
        return int(m.group(1)) if m else None
    g = {"kv_position_bytes": const("kv_position_bytes"), "planes": const("plane_count"),
         "draft_planes": const("draft_planes"), "draft_position_bytes": const("draft_pos_bytes"), "page_positions": None,
         "recurrent_bytes": const("recurrent_bytes"), "slot_stride": const("slot_stride")}
    # the page size the source declares: geometry record field 0 (block positions); cross-checked against block bytes
    m = re.search(r"view\.store %(\w+), %geometry_view\[0\]", text)
    page = const(m.group(1)) if m else None
    block = const("block_bytes")
    by_bytes = block // g["kv_position_bytes"] if block and g["kv_position_bytes"] and block % g["kv_position_bytes"] == 0 else None
    # two spellings that disagree: unknown, not either one
    g["page_positions"] = None if (page and by_bytes and page != by_bytes) else (page or by_bytes)
    return g


def kit_serve_args(kit):
    env = os.path.join(LSK, "kits", kit, "kit.env")
    if not os.path.isfile(env):
        sys.exit(f"kv_budget: no kit manifest {env}")
    r = subprocess.run(["bash", "-c", 'set -a; . "$1" >/dev/null 2>&1; printf %s "$KIT_SERVE_ARGS"', "-", env],
                       capture_output=True, text=True)
    return r.stdout


def backing_from(args_text):
    m = re.search(r"--pool_backing[= ](\w+)", args_text or "")
    return m.group(1) if m else None


def load_package(pkg, preset=None):
    mj = os.path.join(pkg, "model.json")
    if not os.path.isfile(mj):
        raise ValueError(f"no {mj}")
    M = json.load(open(mj))
    presets = M.get("presets") or {}
    if preset is None:
        if len(presets) != 1:
            raise ValueError(f"{len(presets)} presets ({', '.join(presets)}): pass --preset")
        preset = next(iter(presets))
    if preset not in presets:
        raise ValueError(f"no preset {preset!r}")
    ex = presets[preset].get("execution") or {}
    storage = None
    for src in (M.get("programs") or {}).values():
        for inp in src.get("inputs") or []:
            p = inp["path"] if isinstance(inp, dict) else inp
            if os.path.basename(p) == "storage.loom":
                storage = os.path.join(pkg, p)
    geo = storage_geometry(open(storage).read()) if storage and os.path.isfile(storage) else storage_geometry("")
    cfg = {}
    lc = os.path.join(pkg, "llm_config.json")
    if os.path.isfile(lc):
        cfg = json.load(open(lc))
    return {"preset": preset, "execution": ex, "geometry": geo, "storage": storage, "llm_config": cfg}


def kv_format(geo, cfg):
    """bytes per K or V element: kv_position_bytes / (kv_heads x head_dim) when the config names both"""
    kh, hd, b = cfg.get("kv_heads"), cfg.get("head_dim"), geo.get("kv_position_bytes")
    if not (kh and hd and b) or b % (kh * hd):
        return "unknown"
    return {1: "8-bit", 2: "f16"}.get(b // (kh * hd), f"{b // (kh * hd)} B/element")


def round_up(n, page):
    return -(-n // page) * page


def recurrent(ex, geo):
    """recurrent-state bytes: per slot, per stream (its row slot + its fork slot when endpoints exist), checkpoint endpoints,
    and the whole bank (source.c: slots = endpoints ? 2 x rows + endpoints : rows); None when the program spells no slot size"""
    per = geo.get("recurrent_bytes")
    if not per:
        return None
    if geo.get("slot_stride"):
        per = round_up(per, geo["slot_stride"])
    rows = ex.get("rows", 4)   # model-documents.md default
    prefix, session, explicit = ex.get("prefix_cache_capacity"), ex.get("session_cache_capacity"), ex.get("checkpoint_capacity", 0)
    endpoints = explicit + (prefix or 0) + (session + 1 if session else 0)
    target = ex.get("cache_target_bytes", 2 << 30) if prefix is None or session is None else 0   # automatic counts' bound
    slots = 2 * rows + endpoints if endpoints else rows
    return {"slot_bytes": per, "rows": rows, "endpoints": endpoints, "automatic_bound_bytes": target,
            "per_stream_bytes": per * (2 if endpoints or target else 1), "endpoint_bytes": endpoints * per,
            "bank_bytes": slots * per, "bank_max_bytes": slots * per + target, "slots": slots}


def budget(pkg_info, streams, backing=None, pool=None, completion=None, retained=0, miss_tolerant=False):
    """dict with every input, its source, the demand, and verdict FIT / NO_FIT / UNKNOWN"""
    ex, geo = pkg_info["execution"], pkg_info["geometry"]
    unknown, src = [], {}
    ctx_cap = ex.get("context_capacity")
    if pool is not None:
        src["pool"] = "argument"
    elif "pool_capacity" in ex:
        pool, src["pool"] = ex["pool_capacity"], "preset pool_capacity (read by SDK 0.1.x; 0.2+ rejects the field)"
    elif backing == "fixed" and ctx_cap:
        pool, src["pool"] = ctx_cap, "fixed backing: pool = context_capacity (loom serve/server/model_registration.c, SDK 0.2/0.3)"
    elif backing == "elastic":
        unknown.append("pool (elastic backing sizes it from physical memory: pass --pool, from text_memory.pool.capacity_tokens)")
    else:
        unknown.append("pool (no preset pool_capacity and no backing: pass --kit, --backing or --pool)")
    if completion is not None:
        src["completion"] = "argument"
    elif ex.get("max_tokens"):
        completion, src["completion"] = ex["max_tokens"], "preset max_tokens (a request's own max_tokens overrides it)"
    else:
        unknown.append("completion credit (no preset max_tokens: pass --completion)")
    depth = ex.get("draft_depth", 0)
    spec = depth + 1
    page = geo.get("page_positions")
    if not page:
        unknown.append("page size (the storage program does not spell it)")
    res = {"pool": pool, "pool_source": src.get("pool"), "context_capacity": ctx_cap, "completion": completion,
           "completion_source": src.get("completion"), "draft_depth": depth, "page_positions": page, "backing": backing,
           "streams": streams, "retained": retained, "miss_tolerant": miss_tolerant, "unknown": unknown,
           "kv_format": kv_format(geo, pkg_info["llm_config"])}
    res["recurrent"] = recurrent(ex, geo)
    if geo.get("kv_position_bytes") and geo.get("planes"):
        per = geo["kv_position_bytes"] * geo["planes"] + ((geo.get("draft_planes") or 0) * (geo.get("draft_position_bytes") or 0) if depth else 0)
        res["kv_bytes_per_position"] = per
        res["kv_gib"] = round(pool * per / 2**30, 1) if pool else None
    if unknown:
        res["verdict"] = "UNKNOWN"
        return res
    over = [c for c in streams if ctx_cap and c + completion > ctx_cap]
    demand = [round_up(c + completion + spec, page) for c in streams]
    total = sum(demand) + round_up(retained, page) + (max(demand) if miss_tolerant and demand else 0)
    res.update(demand=demand, total=total, headroom=pool - total, over_context=over)

    def largest(n):   # the largest history each of n equal streams can keep (exact continuations), within the row limit
        each = (pool - round_up(retained, page)) // n // page * page - completion - spec
        return max(0, min(each, (ctx_cap - completion) if ctx_cap else each))
    res["largest_each"] = {n: largest(n) for n in sorted({1, 2, 3, len(streams)})}
    res["verdict"] = "FIT" if total <= pool and not over else "NO_FIT"
    return res


def report(r, out=print):
    out(f"pool {r['pool'] if r['pool'] is not None else 'unknown'} positions ({r['pool_source'] or 'unknown'})"
        + (f", KV {r['kv_gib']} GiB at {r['kv_bytes_per_position']} B/position" if r.get("kv_gib") else ""))
    out(f"page {r['page_positions'] or 'unknown'} positions, KV format {r['kv_format']}, draft depth {r['draft_depth']}, "
        f"completion credit {r['completion'] if r['completion'] is not None else 'unknown'} ({r['completion_source'] or 'unknown'}), "
        f"context_capacity {r['context_capacity']}, backing {r['backing'] or 'unknown'}")
    rc = r.get("recurrent")
    if rc:
        mib = lambda b: f"{b / 2**20:,.0f} MiB"   # noqa: E731
        out(f"recurrent state {mib(rc['slot_bytes'])} per slot: {mib(rc['per_stream_bytes'])} per stream (its row"
            + (" + fork slot" if rc["per_stream_bytes"] > rc["slot_bytes"] else "") + f"), {rc['endpoints']} configured checkpoint endpoints (pinned + prefix + session) "
            f"{mib(rc['endpoint_bytes'])}, bank {rc['slots']} slots {mib(rc['bank_bytes'])}"
            + (f", at most {mib(rc['bank_max_bytes'])} with the automatic cache counts" if rc["automatic_bound_bytes"] else "")
            + " (bytes outside the position pool)")
    else:
        out("recurrent state: the storage program spells no %recurrent_bytes (attention-only, or another spelling): unknown")
    if r["verdict"] == "UNKNOWN":
        for u in r["unknown"]:
            out(f"unknown: {u}")
        out("KV_BUDGET: UNKNOWN (" + "; ".join(r["unknown"]) + ")")
        return 2
    for c, d in zip(r["streams"], r["demand"]):
        out(f"stream {c:>8} history -> {d:>8} positions (+ completion + speculative credit, page-rounded)")
    if r["retained"]:
        out(f"retained (must survive) {r['retained']}")
    if r["miss_tolerant"]:
        out(f"one continuation miss: + {max(r['demand'])} (the largest stream held twice until its old history is evicted)")
    for c in r["over_context"]:
        out(f"stream {c} + completion exceeds context_capacity {r['context_capacity']}: refused, not evicted")
    out("largest history each: " + ", ".join(f"{n} stream{'s' if n > 1 else ''} {v}" for n, v in r["largest_each"].items()))
    line = (f"KV_BUDGET: {r['verdict']} ({len(r['streams'])} streams need {r['total']} of {r['pool']} positions, "
            + (f"headroom {r['headroom']})" if r["headroom"] >= 0 else f"short by {-r['headroom']}: eviction, a full re-prefill per evicted turn)"))
    out(line)
    return 0 if r["verdict"] == "FIT" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pkg")
    ap.add_argument("--streams", required=True, help="comma-separated history lengths (positions), one per concurrent stream")
    ap.add_argument("--kit", help="read the backing from kits/KIT/kit.env KIT_SERVE_ARGS")
    ap.add_argument("--backing", choices=["fixed", "elastic"], help="the served --pool_backing (overrides --kit)")
    ap.add_argument("--pool", type=int, help="pool positions as served (overrides the package and backing)")
    ap.add_argument("--completion", type=int, help="completion credit per request (default: the preset's max_tokens)")
    ap.add_argument("--retained", type=int, default=0, help="positions of other histories that must survive (pins, a third session)")
    ap.add_argument("--preset")
    ap.add_argument("--miss-tolerant", action="store_true", help="budget one continuation miss (the largest stream held twice)")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        streams = [int(x) for x in a.streams.split(",") if x.strip()]
        if not streams or min(streams) < 0:
            raise ValueError("--streams needs one or more non-negative lengths")
        info = load_package(a.pkg, a.preset)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"kv_budget: {e}")
        print("KV_BUDGET: UNKNOWN (invalid input)")
        return 2
    backing = a.backing or (backing_from(kit_serve_args(a.kit)) if a.kit else None)
    return report(budget(info, streams, backing, a.pool, a.completion, a.retained, a.miss_tolerant))


if __name__ == "__main__":
    sys.exit(main())
