#!/usr/bin/env python3
"""Does a served turn that resumes from the server's cache give exactly the reply the same turn gives prefilled fresh?

Three arms of common/agent_bench/replay.py rows, same package, same seed and sampling (greedy or seeded), same prompts:
  --fresh    each turn sent to a server that holds nothing (a just-started process): it prefills from position 0. Its turns are
             the turns judged. A fresh row that reports cached tokens is no reference (NOT-EXERCISED).
  --resumed  the session replayed in order on one server (own mode, --keep-replies): each turn continues the retained row.
             The first row of each record (and a row after an error) is sent cold on purpose: paired with the fresh row it is
             an A/A control of the fresh path itself (both cold; a difference there is non-determinism, not a resume defect).
  --fork     a turn re-sent later on a server that has since moved past it (a branch from an earlier cached point): it can
             only resume from an intermediate prefix endpoint, for a hybrid model a recurrent-state checkpoint.
The fresh and fork arms re-send the resumed arm's own history (replay.py --history-from), so every arm's prompt for turn k is
the same token sequence; equal prompt token counts are checked, a difference makes the pair invalid.

Evidence that the restore path ran is the server's cached-token count per turn (OpenAI usage.prompt_tokens_details.cached_tokens,
replay.py's `cached`): a resumed (non-cold) or fork row with cached 0 did not exercise it and is NOT-EXERCISED, never a pass.
Per judged pair: EXACT (same reply hash), MISMATCH (first differing position: part and character, when the rows carry the
reply text), NOT-EXERCISED (with its reason). Before a pass is trusted, a planted one-character difference in one reply must be
reported as a MISMATCH at the planted position (CONTROL FAILED otherwise), as compare_outputs.py --diff does.

Model-agnostic: nothing here knows the model; the hit depths come from the server.

usage: resume_audit.py --fresh F.jsonl --resumed R.jsonl [--fork K.jsonl] [--json OUT]
       RESUME: EXACT (exit 0) | MISMATCH (exit 1) | NOT-EXERCISED (exit 2) | UNKNOWN (exit 2: invalid input, control failed)
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks  # noqa: E402  reply_sha: the hash replay.py writes

SAME = ("model", "session", "mode", "sampling", "seed", "effort")   # summary fields every arm must share


def load(path):
    """[(summary, rows)] of a replay.py --log file; each row gets 'cold_expected' (first of its record or after an error)"""
    recs = []
    for n, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if "summary" not in d or "rows" not in d:
            raise ValueError(f"{path}:{n}: not a replay.py log record")
        prev_ok = False
        for r in d["rows"]:
            r["cold_expected"] = not prev_ok
            prev_ok = not r.get("error")
        recs.append((d["summary"], d["rows"]))
    if not recs:
        raise ValueError(f"{path}: no records")
    return recs


def by_turn(recs, path):
    """{turn: row} over every record of one arm; a turn twice is invalid (which one is the arm's?)"""
    out = {}
    for _, rows in recs:
        for r in rows:
            if r["turn"] in out:
                raise ValueError(f"{path}: turn {r['turn']} twice (one row per turn and arm)")
            out[r["turn"]] = r
    return out


def config_clash(arms):
    """the first summary field that differs between records of all arms, as text; None when they agree"""
    seen = {}
    for name, recs in arms.items():
        for s, _ in recs:
            for k in SAME:
                v = s.get(k)
                if k in seen and seen[k][0] != v:
                    return f"{k}: {seen[k][1]} has {seen[k][0]!r}, {name} has {v!r}"
                seen.setdefault(k, (v, name))
    return None


def parts(reply):
    """the reply as ordered (name, text) parts: reasoning, content, then each tool call's name and arguments"""
    p = [("reasoning", reply.get("reasoning_content") or ""), ("content", reply.get("content") or "")]
    for i, tc in enumerate(reply.get("tool_calls") or []):
        p += [(f"tool_calls[{i}].name", tc["function"]["name"]), (f"tool_calls[{i}].arguments", tc["function"]["arguments"])]
    return p


def first_difference(ra, rb):
    """'part char N' of the first differing character between two replies; None when equal"""
    pa, pb = parts(ra), parts(rb)
    for (na, ta), (nb, tb) in zip(pa, pb):
        if na != nb:
            return f"{na} vs {nb}"
        if ta != tb:
            i = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            return f"{na} char {i} ({ta[max(0, i - 12):i + 12]!r} vs {tb[max(0, i - 12):i + 12]!r})"
    if len(pa) != len(pb):
        return f"tool call count {(len(pa) - 2) // 2} vs {(len(pb) - 2) // 2}"
    return None


def compare(fresh, other, arm):
    """one judged pair -> {turn, arm, verdict, reason, cached, prompt, kind}; arm 'resumed' or 'fork'"""
    t = fresh["turn"]
    res = {"turn": t, "arm": arm, "prompt": fresh.get("prompt"), "cached": None, "kind": arm}
    if other is None:
        return dict(res, verdict="NOT-EXERCISED", reason=f"no {arm} row")
    res["cached"] = other.get("cached")
    for side, r in (("fresh", fresh), (arm, other)):
        if r.get("error"):
            return dict(res, verdict="NOT-EXERCISED", reason=f"{side} turn failed: {r['error'][:80]}")
        if "out_sha" not in r:
            return dict(res, verdict="NOT-EXERCISED", reason=f"{side} row has no out_sha (replay.py older than the hash)")
        if "reply" in r and checks.reply_sha(r["reply"]) != r["out_sha"]:
            return dict(res, verdict="NOT-EXERCISED", reason=f"{side} row's reply does not hash to its out_sha (edited row?)")
    if fresh.get("cached"):
        return dict(res, verdict="NOT-EXERCISED", reason=f"fresh row reused {fresh['cached']} cached tokens: not a fresh reference")
    if fresh.get("prompt") != other.get("prompt"):
        return dict(res, verdict="NOT-EXERCISED", reason=f"prompts differ ({fresh.get('prompt')} vs {other.get('prompt')} tokens): "
                    "not the same history (--history-from)")
    cold = not other.get("cached")
    if cold and not (arm == "resumed" and other.get("cold_expected")):
        return dict(res, verdict="NOT-EXERCISED", reason=f"{arm} row reused no cached tokens: the restore path did not run")
    res["kind"] = "cold pair (A/A)" if cold else ("tip" if arm == "resumed" else "fork")
    if other["out_sha"] == fresh["out_sha"]:
        return dict(res, verdict="EXACT", reason="")
    where = first_difference(fresh["reply"], other["reply"]) if "reply" in fresh and "reply" in other else \
        "position unknown (rows lack the reply text: replay.py --keep-replies)"
    return dict(res, verdict="MISMATCH", reason=("both cold: the fresh path itself does not reproduce; " if cold else "")
                + f"first difference {where}; out {fresh.get('out')} vs {other.get('out')} tokens")


def judge(fresh, resumed, fork=None):
    """[pair results] for every fresh turn against each given arm"""
    out = []
    for t in sorted(fresh):
        for arm, rows in (("resumed", resumed), ("fork", fork)):
            if rows is None or (arm == "fork" and t not in rows):
                continue   # the fork arm re-sends a chosen few turns; the resumed arm must cover every fresh turn
            out.append(compare(fresh[t], rows.get(t), arm))
    return out


def overall(pairs):
    """(verdict, exit) over the pairs: any MISMATCH -> MISMATCH; any NOT-EXERCISED or no exercised EXACT -> NOT-EXERCISED"""
    if any(p["verdict"] == "MISMATCH" for p in pairs):
        return "MISMATCH", 1
    if any(p["verdict"] == "NOT-EXERCISED" for p in pairs):
        return "NOT-EXERCISED", 2
    if not any(p["verdict"] == "EXACT" and not p["kind"].startswith("cold") for p in pairs):
        return "NOT-EXERCISED", 2
    return "EXACT", 0


def planted_control(fresh, resumed, fork):
    """a copy of one compared row with one character changed (and its hash re-made, as a real differing reply would have) must
    come back MISMATCH at that position; True when it does, or when nothing was compared (no pass to protect)"""
    for arm, rows in (("resumed", resumed), ("fork", fork)):
        for t, f in sorted(fresh.items()):
            o = (rows or {}).get(t)
            if o is None or compare(f, o, arm)["verdict"] != "EXACT":
                continue
            o = copy.deepcopy(o)
            if "reply" in o:
                name, text = max(((n, x) for n, x in parts(o["reply"]) if not n.endswith(".name")), key=lambda p: len(p[1]))
                i = len(text) // 2
                new = text[:i] + ("x" if text[i:i + 1] != "x" else "y") + text[i + 1:]
                if name == "reasoning":
                    o["reply"]["reasoning_content"] = new
                elif name == "content":
                    o["reply"]["content"] = new
                else:
                    o["reply"]["tool_calls"][int(name.split("[")[1].split("]")[0])]["function"]["arguments"] = new
                o["out_sha"] = checks.reply_sha(o["reply"])
                c = compare(f, o, arm)
                return c["verdict"] == "MISMATCH" and f"{name} char {i}" in c["reason"]
            o["out_sha"] = ("0" if o["out_sha"][:1] != "0" else "1") + o["out_sha"][1:]
            return compare(f, o, arm)["verdict"] == "MISMATCH"
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fresh", required=True, help="replay.py log: each turn on a server holding nothing (its turns are judged)")
    ap.add_argument("--resumed", required=True, help="replay.py log: the session in order on one server")
    ap.add_argument("--fork", help="replay.py log: earlier turns re-sent on a server that moved past them")
    ap.add_argument("--json", help="write the pairs and the verdict here")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    try:
        arms = {"fresh": load(a.fresh), "resumed": load(a.resumed)}
        if a.fork:
            arms["fork"] = load(a.fork)
        rows = {k: by_turn(v, getattr(a, k)) for k, v in arms.items()}
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"resume_audit: {e}")
        print("RESUME: UNKNOWN (invalid input)")
        return 2
    clash = config_clash(arms)
    if clash:
        print(f"RESUME: UNKNOWN (arms differ in {clash}: not the same configuration)")
        return 2
    if not rows["fresh"]:
        print("RESUME: UNKNOWN (no fresh rows)")
        return 2
    pairs = judge(rows["fresh"], rows["resumed"], rows.get("fork"))
    print(f"{'turn':>5} {'arm':<8} {'kind':<16} {'prompt':>8} {'cached':>8}  verdict")
    for p in pairs:
        print(f"{p['turn']:>5} {p['arm']:<8} {p['kind']:<16} {p['prompt'] if p['prompt'] is not None else '-':>8} "
              f"{p['cached'] if p['cached'] is not None else '-':>8}  {p['verdict']}" + (f": {p['reason']}" if p["reason"] else ""))
    if not planted_control(rows["fresh"], rows["resumed"], rows.get("fork")):
        print("RESUME: UNKNOWN (CONTROL FAILED: a planted one-character difference was not reported at its position)")
        return 2
    verdict, rc = overall(pairs)
    n = {v: sum(1 for p in pairs if p["verdict"] == v) for v in ("EXACT", "MISMATCH", "NOT-EXERCISED")}
    if a.json:
        json.dump({"pairs": pairs, "verdict": verdict}, open(a.json, "w"), indent=1)
    first = next((p for p in pairs if p["verdict"] == verdict), None) if verdict != "EXACT" else None
    print(f"RESUME: {verdict} ({n['EXACT']} exact, {n['MISMATCH']} mismatch, {n['NOT-EXERCISED']} not exercised"
          + (f"; first: turn {first['turn']} {first['arm']}: {first['reason'] or 'no exercised exact pair'}" if first else "")
          + ("; no exercised exact pair" if verdict == "NOT-EXERCISED" and not first else "") + ")")
    return rc


if __name__ == "__main__":
    sys.exit(main())
