#!/usr/bin/env python3
"""Tests of kv_budget.py and concurrency_audit.py. Each case names the recurrence it prevents.
usage: python3 test_concurrency.py (exit 1 on a failure)"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import concurrency_audit as ca  # noqa: E402
import kv_budget as kb  # noqa: E402

fails, ran = [], []


def machine_env(key):
    """$KEY, else KEY= from the machine env file common/env.sh reads ($LSK_ENV or ~/.config/loom-strix-kits/env); '' if neither"""
    if os.environ.get(key):
        return os.environ[key]
    f = os.environ.get("LSK_ENV") or os.path.expanduser("~/.config/loom-strix-kits/env")
    for line in open(f) if os.path.isfile(f) else []:
        if line.startswith(key + "="):
            return line.split("=", 1)[1].split("#")[0].strip().strip('"')
    return ""


# the Qwen kit's work dir (real-data cases: E47's package, E48's rows, E59's trace); they skip when it is unset or absent
W = os.environ.get("LSK_WORK") or machine_env("WORK")


def case(name, cond):
    ran.append(name)
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


STORAGE = """// KV geometry
  %kv_position_bytes = scalar.constant {kvb} : i64
  %plane_count = scalar.constant 32 : i64
  %draft_planes = scalar.constant 10 : i64
  %draft_pos_bytes = scalar.constant {kvb} : i64
  %c16_i64 = scalar.constant {page} : i64
  view.store %c16_i64, %geometry_view[0] : i64, view<13xi64>
  %block_bytes = scalar.constant {block} : i64
"""


def package(root, execution, kvb=2048, page=16, block=None, storage=True):
    os.makedirs(os.path.join(root, "program"), exist_ok=True)
    m = {"programs": {"source": {"inputs": ["program/storage.loom"]}}, "presets": {"chat": {"execution": execution}}}
    json.dump(m, open(os.path.join(root, "model.json"), "w"))
    json.dump({"kv_heads": 4, "head_dim": 256}, open(os.path.join(root, "llm_config.json"), "w"))
    if storage:
        open(os.path.join(root, "program", "storage.loom"), "w").write(
            STORAGE.format(kvb=kvb, page=page, block=block if block is not None else page * kvb))
    return root


tmp = tempfile.mkdtemp()
E47 = [140439, 122812]   # LEDGER E47: the two histories that thrashed (pool 327,680 on SDK 0.1.0, pkg-pf16-live)
old = package(os.path.join(tmp, "old"), {"rows": 4, "context_capacity": 262144, "pool_capacity": 327680, "max_tokens": 4096, "draft_depth": 7})
new = package(os.path.join(tmp, "new"), {"rows": 4, "context_capacity": 262144, "max_tokens": 4096, "draft_depth": 7}, kvb=1024)

# --- kv_budget
# E47: position arithmetic alone says the two histories fit the 0.1.0 pool (exact continuations); only a continuation miss
# (both copies of one history held at once) overflows it. Recurrence: the ledger's capacity explanation taken as arithmetic.
r = kb.budget(kb.load_package(old), E47)
case("kv_budget: E47 histories fit the 327,680 pool with exact continuations (headroom 56,208)", r["verdict"] == "FIT" and r["headroom"] == 56208)
r = kb.budget(kb.load_package(old), E47, miss_tolerant=True)
case("kv_budget: E47 with one continuation miss does not fit (eviction, E47's verdict)", r["verdict"] == "NO_FIT" and r["headroom"] == -88336)
# SDK 0.2+: no pool_capacity; fixed backing pools ONE context for every row (model_registration.c). Recurrence: a pool figure
# from the build's --pool taken as what loom-serve allocates.
r = kb.budget(kb.load_package(new), E47, backing="fixed")
case("kv_budget: fixed backing = context_capacity: E47 short by 9,328", r["pool"] == 262144 and r["verdict"] == "NO_FIT" and r["headroom"] == -9328)
case("kv_budget: two streams each keep at most 126,968 on a 262,144 fixed pool", r["largest_each"][2] == 126968)
case("kv_budget: KV format from bytes per position (1024 / (4 x 256) = 8-bit)", r["kv_format"] == "8-bit")
# fail closed: a quantity nobody supplies is unknown, never a Qwen default
r = kb.budget(kb.load_package(new), E47, backing="elastic")
case("kv_budget: elastic without --pool is UNKNOWN", r["verdict"] == "UNKNOWN")
r = kb.budget(kb.load_package(new), E47)
case("kv_budget: no backing and no preset pool is UNKNOWN", r["verdict"] == "UNKNOWN")
nopage = package(os.path.join(tmp, "nopage"), {"context_capacity": 262144, "max_tokens": 4096, "draft_depth": 7}, storage=False)
case("kv_budget: no storage program -> page size unknown", kb.budget(kb.load_package(nopage), E47, backing="fixed")["verdict"] == "UNKNOWN")
clash = package(os.path.join(tmp, "clash"), {"context_capacity": 262144, "max_tokens": 4096}, page=16, block=32 * 2048)
case("kv_budget: page spellings that disagree -> unknown, not either", kb.load_package(clash)["geometry"]["page_positions"] is None)
nomax = package(os.path.join(tmp, "nomax"), {"context_capacity": 262144, "pool_capacity": 327680})
case("kv_budget: no max_tokens and no --completion is UNKNOWN", kb.budget(kb.load_package(nomax), E47)["verdict"] == "UNKNOWN")
r = kb.budget(kb.load_package(old), [260000])
case("kv_budget: a history + completion over context_capacity is NO_FIT (refused)", r["verdict"] == "NO_FIT" and r["over_context"] == [260000])
rc, out = quiet(kb.main, [old, "--streams", "140439,122812"])
case("kv_budget: CLI exit 0 and one verdict line on FIT", rc == 0 and out.strip().splitlines()[-1].startswith("KV_BUDGET: FIT"))
rc, _ = quiet(kb.main, [new, "--streams", "140439,122812"])
case("kv_budget: CLI exit 2 on UNKNOWN", rc == 2)
rc, _ = quiet(kb.main, [new, "--streams", "x"])
case("kv_budget: CLI exit 2 on invalid streams", rc == 2)
# Recurrent-state checkpoints (E409 survey: kv_budget counted KV positions only; E105: each prefix endpoint holds a 157 MB GDN
# snapshot). Slots per loom source.c: 2 x rows + endpoints when any endpoint exists; bytes outside the position pool.
REC = "  %slot_stride = scalar.constant 256 : i64\n  %recurrent_bytes = scalar.constant 156893184 : i64\n"
hyb = package(os.path.join(tmp, "hyb"), {"rows": 4, "context_capacity": 262144, "max_tokens": 4096, "draft_depth": 7, "prefix_cache_capacity": 8,
                                         "session_cache_capacity": 0})
open(os.path.join(hyb, "program", "storage.loom"), "a").write(REC)
rh = kb.budget(kb.load_package(hyb), E47, backing="fixed")
rr = rh["recurrent"] or {}
case("kv_budget: recurrent bank = (2 x 4 rows + 8 prefix endpoints) slots, 2 slots per stream, verdict unchanged",
     rr.get("slots") == 16 and rr.get("bank_bytes") == 16 * 156893184 and rr.get("per_stream_bytes") == 2 * 156893184
     and rr.get("endpoint_bytes") == 8 * 156893184 and rr.get("automatic_bound_bytes") == 0 and rh["headroom"] == -9328)
auto = package(os.path.join(tmp, "auto"), {"rows": 4, "context_capacity": 262144, "max_tokens": 4096})
open(os.path.join(auto, "program", "storage.loom"), "a").write(REC)
ra = kb.budget(kb.load_package(auto), E47, backing="fixed")["recurrent"] or {}
case("kv_budget: automatic cache counts -> bank bounded by cache_target_bytes (2 GiB), printed as 'at most'",
     ra.get("automatic_bound_bytes") == 2 << 30 and ra.get("bank_max_bytes") == 4 * 156893184 + (2 << 30))
rc, out = quiet(kb.main, [new, "--streams", "1000", "--backing", "fixed"])
case("kv_budget: no %recurrent_bytes in the program -> recurrent state unknown, said", "recurrent state: " in out and "unknown" in out)
if os.path.isfile(f"{W}/pkg-pf16-live/model.json"):
    r = kb.budget(kb.load_package(f"{W}/pkg-pf16-live"), E47)
    case("kv_budget: the real E47 package (pkg-pf16-live) parses to pool 327,680, page 16, KV 26.2 GiB (E51's figure)",
         r["pool"] == 327680 and r["page_positions"] == 16 and r["kv_gib"] == 26.2)


# --- concurrency_audit
def row(turn, send, ttft, gen, out, prompt, cached=None, cap=4096, sha=None, epoch=True):
    cached = prompt // 2 if cached is None else cached
    r = {"turn": turn, "prompt": prompt, "cached": cached, "new": prompt - cached, "ttft": ttft, "gen_s": gen, "out": out,
         "wall": ttft + gen, "cap": cap, "decode_tps": (out - 1) / gen if gen else None, "prefill_tps": (prompt - cached) / ttft}
    if epoch:
        r.update(t_send=send, t_first=send + ttft, t_end=send + ttft + gen)
    if sha is not None:
        r["out_sha"] = sha
    return r


def rec(label, rows):
    return {"summary": {"label": label}, "rows": rows}


def write(name, recs):
    p = os.path.join(tmp, name)
    with open(p, "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    return p


# A: prefill 0-10, decode 10-20 (101 tokens). B: turn 0 prefill 0-2 decode 2-10 (during A's prefill: 41 tokens),
# turn 1 sent at 10, prefill 10-11, decode 11-20 (during A's decode: 91 tokens); first turns cached 0 (distinct prompts)
A = [row(0, 0, 10, 10, 101, 1000, cached=0, sha="a0"), row(1, 20, 1, 4, 41, 1200, sha="a1"), row(2, 25, 1, 4, 41, 1400, sha="a2")]
B = [row(0, 0, 2, 8, 41, 1000, cached=0, sha="b0"), row(1, 10, 1, 9, 91, 1200, sha="b1"), row(2, 20, 1, 4, 41, 1400, sha="b2")]
c = ca.cell({"A": rec("A", A), "B": rec("B", B)})
bp = c["streams"]["B"]["by_phase"]
# Recurrence: E49's split (decode while the other prefills vs decodes) computed by hand on a reconstructed timeline
case("audit: B's turn decoding during A's prefill is classed 'prefill' at 5.0 tok/s", bp["prefill"]["tok_s"] == 5.0 and bp["idle"]["tok_s"] is None)
case("audit: B's turn decoding during A's decode is classed 'decode' at 10.0 tok/s", bp["decode"]["tok_s"] == 10.0)
case("audit: the aggregate is all decode tokens over the cell wall (30 s)", c["wall_s"] == 30.0 and abs(c["aggregate_decode_tok_s"] - (100 + 40 + 40 + 40 + 90 + 40) / 30) < 0.01)
case("audit: epoch rows are not marked as a rebuilt timeline", not c["rebuilt_timeline"])
noep = [dict((k, v) for k, v in r.items() if not k.startswith("t_")) for r in A]
case("audit: rows without t_send rebuild the timeline from walls (said in the report)", ca.timeline(noep)[1] and ca.timeline(noep)[0][1][1] == 20)

# three or more streams: one other prefilling while another decodes counts as 'prefill' (the disruptive phase wins)
case("audit: overlapping other-stream phases -> prefill beats decode", ca.segments([(0, 10, "decode"), (2, 6, "prefill")]) ==
     [(0, 2, "decode"), (2, 6, "prefill"), (6, 10, "decode")])
mix = ca.cell({"A": rec("A", A), "C": rec("C", [row(0, 5, 1, 10, 51, 1000, cached=0)])})   # C decodes 6-16: A prefill to 10, then decode
case("audit: a turn overlapping both another's prefill and decode is classed 'prefill'", mix["streams"]["C"]["by_phase"]["prefill"]["tok_s"] == 5.0
     and mix["streams"]["C"]["sliced_prefill_overlap"] == 0.4)

cell_p = write("cell.jsonl", [rec("A", A), rec("B", B)])
solo_p = write("solo.jsonl", [rec("A", [dict(r, t_send=r["t_send"] + 100, t_first=r["t_first"] + 100, t_end=r["t_end"] + 100) for r in A]),
                              rec("B", B)])
rc, out = quiet(ca.main, [cell_p, "--solo", solo_p])
case("audit: identical replies, no eviction, overlap -> CONCURRENCY: OK exit 0", rc == 0 and out.strip().endswith("CONCURRENCY: OK"))
bad = [dict(r) for r in B]
bad[2]["out_sha"] = "zz"
rc, out = quiet(ca.main, [cell_p, "--solo", write("solo_bad.jsonl", [rec("A", A), rec("B", bad)])])
case("audit: one reply differing from solo -> FAIL naming the turn", rc == 1 and "first 2" in out)
orig = ca.exactness
ca.exactness = lambda conc, solo: {lab: {"compared": 3, "mismatch": 0, "first": None, "missing": 0} for lab in conc}
rc, out = quiet(ca.main, [cell_p, "--solo", solo_p])
ca.exactness = orig
# Recurrence (compare_outputs --diff, retro 11-22): a comparator that cannot see a difference passed everything
case("audit: a blind comparator is caught by the planted control (exit 2 CONTROL FAILED)", rc == 2 and "CONTROL FAILED" in out)
rc, out = quiet(ca.main, [cell_p])
case("audit: no --solo -> UNKNOWN exit 2 (exactness not checked is not OK)", rc == 2 and "UNKNOWN" in out)
nosha = [dict((k, v) for k, v in r.items() if k != "out_sha") for r in A]
rc, out = quiet(ca.main, [write("cell_nosha.jsonl", [rec("A", nosha), rec("B", B)]), "--solo", solo_p])
case("audit: rows without reply hashes -> exactness UNKNOWN", rc == 2 and "out_sha" in out)
# E47: a turn that re-prefills its whole history although the client's history was exact is eviction
ev = [dict(r) for r in B]
ev[2] = row(2, 20, 30, 4, 41, 1400, cached=0, sha="b2")
rc, out = quiet(ca.main, [write("cell_ev.jsonl", [rec("A", A), rec("B", ev)]), "--solo", solo_p])
case("audit: a whole-history re-prefill after an exact turn -> FAIL eviction", rc == 1 and "eviction" in out)
miss = [dict(r) for r in ev]
miss[1] = dict(miss[1], out=4096, cap=4096)
case("audit: after a reply that hit its cap the re-prefill is a continuation miss, not eviction",
     (ca.reprefills(miss) or [{"kind": ""}])[0]["kind"].startswith("continuation miss"))
late = [dict(r, t_send=r["t_send"] + 1000, t_first=r["t_first"] + 1000, t_end=r["t_end"] + 1000) for r in B]
rc, out = quiet(ca.main, [write("cell_seq.jsonl", [rec("A", A), rec("B", late)]), "--solo", solo_p])
case("audit: streams that never overlapped -> FAIL (not a concurrent cell)", rc == 1 and "never ran at the same time" in out)
shared = [dict(r) for r in B]
shared[0] = row(0, 0, 2, 8, 41, 1000, cached=900, sha="b0")
rc, out = quiet(ca.main, [write("cell_shared.jsonl", [rec("A", A), rec("B", shared)]), "--solo", solo_p])
case("audit: a first turn reusing a cached prefix -> FAIL (prompts not salted apart)", rc == 1 and "not distinct" in out)
rc, out = quiet(ca.main, [cell_p, "--solo", solo_p, "--ref", cell_p])
case("audit: --ref without --floor is a usage error (no Qwen floor default)", rc == 2)
slow = [dict(r, decode_tps=r["decode_tps"] * 0.9) for r in A]
slowB = [dict(r, decode_tps=r["decode_tps"] * 0.9) for r in B]
rc, out = quiet(ca.main, [write("cell_slow.jsonl", [rec("A", slow), rec("B", slowB)]), "--solo", solo_p, "--ref", cell_p, "--floor", "0.01"])
case("audit: a cell 10 % slower per turn than --ref beyond the floor -> FAIL", rc == 1 and "decode_tps 0.9" in out)
rc, _ = quiet(ca.main, [write("one.jsonl", [rec("A", A)])])
case("audit: a single stream is not a cell (exit 2)", rc == 2)
rc, _ = quiet(ca.main, [write("dup.jsonl", [rec("A", A), rec("A", B)])])
case("audit: a label twice is invalid input (exit 2)", rc == 2)

# gap_stats is decode_while_prefill.py's phase classifier: E59's saved trace must come out as recorded
seg = [(float("-inf"), 0, "before"), (0, 5, "during"), (5, float("inf"), "after")]
g = ca.gap_stats([-2, -1, 0.5, 4.5, 6, 6.5], seg)
case("gap_stats: gaps labelled by the phase at their end (L19 method)", g["during"]["chunks"] == 2 and g["during"]["max_gap_ms"] == 4000.0)
p = f"{W}/agent-runs/l19_trace_40k.json"
if os.path.isfile(p):
    d = json.load(open(p))
    g = ca.gap_stats(d["b_stamps"], [(float("-inf"), 0, "before"), (0, d["a_first"], "during"), (d["a_first"], float("inf"), "after")])
    case("gap_stats: E59's 8192-chunk trace: B advanced 6 times during the prefill, max gap 41.5 s (as l19_trace recorded)",
         g["during"]["chunks"] == d["result"]["during"]["chunks"] == 6 and g["during"]["max_gap_ms"] == d["result"]["during"]["max_gap_ms"])

# positive check on real data: E48's two-stream rows reproduce E49's split (decode while the other prefills vs decodes)
p = f"{W}/agent-runs/replay-ab2-l17.jsonl"
if os.path.isfile(p):
    for pkg, pre, dec in (("pkg-vsfull2", (9.5, 11.6), 18.5), ("pkg-pf16-live", (8.0, 8.8), 13.5)):
        c = ca.cell(ca.load(p, [f"{pkg}-s1", f"{pkg}-s2"]))
        ps = [c["streams"][f"{pkg}-s{i}"]["by_phase"] for i in (1, 2)]
        case(f"audit: E49 reproduced on {pkg}: other-prefill {[x['prefill']['tok_s'] for x in ps]} in {pre}, other-decode >= {dec}",
             all(pre[0] <= (x["prefill"]["tok_s"] or 0) <= pre[1] and (x["decode"]["tok_s"] or 0) >= dec for x in ps) and c["rebuilt_timeline"])


# replay.py writes what the audit needs: epoch send / first / end stamps and a reply hash that ignores server-made call ids.
# Recurrence: replay_ab2's rows had neither (E49 rebuilt the timeline from walls; exactness was not checkable). A fake
# OpenAI SSE server on a loopback port, so no model or GPU is involved.
def fake_server(reply_text, call_id):
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ch in [{"choices": [{"delta": {"content": w}}]} for w in reply_text.split(" ")] + [
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": call_id, "function": {"name": "f", "arguments": "{}"}}]}}]},
                    {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 0}}}]:
                self.wfile.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if machine_env("LOOM_SKILLS"):
    import subprocess
    sess = os.path.join(tmp, "s.session.json")
    json.dump({"name": "t", "tools": [], "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
               "turns": [{"upto": 2, "effort": "none", "rec_new": 50, "rec_cached": 0, "rec_out": 7, "rec_wall_s": 1.0}]}, open(sess, "w"))
    env = dict(os.environ, LOOM_SKILLS=machine_env("LOOM_SKILLS"))
    shas = []
    for text, cid in (("a b c", "id1"), ("a b c", "id2"), ("a b d", "id1")):
        srv = fake_server(text, cid)
        log = os.path.join(tmp, f"r{len(shas)}.jsonl")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_bench", "replay.py"), sess,
                        f"http://127.0.0.1:{srv.server_address[1]}", "m", "--log", log], capture_output=True, text=True, env=env)
        srv.shutdown()
        r = json.load(open(log))["rows"][0] if os.path.isfile(log) else {}
        shas.append(r.get("out_sha"))
        if len(shas) == 1:
            case("replay: rows carry t_send <= t_first <= t_end on the epoch clock",
                 all(k in r for k in ("t_send", "t_first", "t_end")) and r["t_send"] <= r["t_first"] <= r["t_end"] and r["t_send"] > 1.7e9)
    case("replay: the reply hash ignores the server's call id", shas[0] is not None and shas[0] == shas[1])
    case("replay: the reply hash sees a one-word difference", shas[0] != shas[2])

print(f"{len(ran) - len(fails)}/{len(ran)} passed")
sys.exit(1 if fails else 0)
