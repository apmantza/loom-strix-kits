#!/usr/bin/env python3
"""Tests of resume_audit.py (and replay.py's --keep-replies / --history-from, which feed it). Each case names the recurrence it
prevents. Synthetic replay rows; the replay cases use a fake OpenAI SSE server on a loopback port (no model, no GPU).
usage: python3 test_resume_audit.py (exit 1 on a failure)"""
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks  # noqa: E402
import resume_audit as ra  # noqa: E402

fails, ran = [], []


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


tmp = tempfile.mkdtemp()


def reply(text, reasoning="think", args='{"a": 1}'):
    return {"role": "assistant", "content": text, "reasoning_content": reasoning,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": args}}]}


def row(turn, prompt, cached, rp, out=10):
    return {"turn": turn, "prompt": prompt, "cached": cached, "new": prompt - cached, "out": out, "reply": rp, "out_sha": checks.reply_sha(rp)}


def write(name, *records, cfg=None):
    p = os.path.join(tmp, name)
    with open(p, "w") as f:
        for rows in records:
            f.write(json.dumps({"summary": dict({"model": "m", "session": "s", "mode": "own", "sampling": "card", "seed": 1, "effort": None},
                                                **(cfg or {})), "rows": rows}) + "\n")
    return p


R = {k: reply(f"reply {k} " + "word " * 20) for k in range(4)}
# resumed arm: turn 0 cold (first row of its record), turns 1-2 continue the row (cached = the previous prompt + reply)
RES = [row(0, 1000, 0, R[0]), row(1, 1200, 1010, R[1]), row(2, 1400, 1210, R[2])]
FRESH = [[row(0, 1000, 0, R[0])], [row(1, 1200, 0, R[1])], [row(2, 1400, 0, R[2])]]   # one record per turn: one server per turn
FORK = [row(1, 1200, 1024, R[1])]                                                        # re-sent later: an intermediate endpoint
fresh_p, res_p, fork_p = write("fresh.jsonl", *FRESH), write("res.jsonl", RES), write("fork.jsonl", FORK)


def run(fresh=fresh_p, res=res_p, fork=fork_p):
    return quiet(ra.main, ["--fresh", fresh, "--resumed", res] + (["--fork", fork] if fork else []))


# Recurrence (E409 survey item 2): nothing tested that a turn restored from the prefix cache / GDN checkpoint equals a fresh prefill
rc, out = run()
case("resume: identical replies on tip resume and fork -> RESUME: EXACT exit 0", rc == 0 and out.strip().splitlines()[-1].startswith("RESUME: EXACT"))
case("resume: the first resumed row (cold) is reported as the A/A pair, not as a resume", "cold pair (A/A)" in out)

bad = copy.deepcopy(RES)
rp = copy.deepcopy(R[2])
rp["content"] = rp["content"][:9] + "X" + rp["content"][10:]
bad[2] = row(2, 1400, 1210, rp)
rc, out = run(res=write("res_bad.jsonl", bad))
case("resume: one changed character -> MISMATCH exit 1, naming turn and position (content char 9)",
     rc == 1 and "RESUME: MISMATCH" in out and "turn 2" in out and "content char 9" in out)

ev = copy.deepcopy(RES)
ev[2]["cached"], ev[2]["new"] = 0, 1400
rc, out = run(res=write("res_ev.jsonl", ev), fork=None)
# Recurrence: an evicted turn re-prefills from zero and trivially equals fresh; counting it as a pass would test nothing
case("resume: a resumed (non-first) turn with no cache hit -> NOT-EXERCISED exit 2, not a pass", rc == 2 and "RESUME: NOT-EXERCISED" in out
     and "restore path did not run" in out)
fk = [dict(FORK[0], cached=0, new=1200)]
rc, out = run(fork=write("fork0.jsonl", fk))
case("resume: a fork turn with no cache hit -> NOT-EXERCISED", rc == 2 and "fork" in out and "did not run" in out)

fr = copy.deepcopy(FRESH)
fr[1][0]["cached"] = 512
rc, out = run(fresh=write("fresh_hit.jsonl", *fr))
case("resume: a 'fresh' row that reused cached tokens is no reference -> NOT-EXERCISED", rc == 2 and "not a fresh reference" in out)
fr = copy.deepcopy(FRESH)
fr[2][0]["prompt"] = 1390
rc, out = run(fresh=write("fresh_prompt.jsonl", *fr))
# Recurrence: a fresh arm that rebuilt the history from the recording (not the resumed run's replies) compares different prompts
case("resume: different prompt token counts -> NOT-EXERCISED (not the same history)", rc == 2 and "prompts differ" in out)
rc, out = run(res=write("res_cfg.jsonl", RES, cfg={"seed": 2}))
case("resume: arms with different seeds -> UNKNOWN exit 2", rc == 2 and "UNKNOWN" in out and "seed" in out)
rc, out = run(res=write("res_dup.jsonl", RES, RES[1:2]))
case("resume: a turn twice in one arm is invalid input (exit 2)", rc == 2 and "twice" in out)
only_cold = write("res_cold.jsonl", [RES[0]], [RES[1] | {"cached": 0, "new": 1200}])   # each turn first of its record: both cold
rc, out = run(res=only_cold, fork=None, fresh=write("fresh01.jsonl", *FRESH[:2]))
case("resume: only cold A/A pairs -> NOT-EXERCISED (no exercised exact pair)", rc == 2 and "no exercised exact pair" in out)
aa = copy.deepcopy(RES)
aa[0] = row(0, 1000, 0, reply("other " + "word " * 20))
rc, out = run(res=write("res_aa.jsonl", aa))
case("resume: a cold A/A pair that differs -> MISMATCH said as a fresh-path non-determinism", rc == 1 and "fresh path itself" in out)
edited = copy.deepcopy(RES)
edited[1]["reply"]["content"] += "!"
rc, out = run(res=write("res_edit.jsonl", edited), fork=None)
case("resume: a reply that does not hash to its out_sha -> NOT-EXERCISED (edited row)", rc == 2 and "does not hash" in out)
nr = [{k: v for k, v in r.items() if k != "reply"} for r in bad]
rc, out = run(res=write("res_noreply.jsonl", nr), fork=None)
case("resume: rows without reply text still MISMATCH by hash, position unknown", rc == 1 and "position unknown" in out)

# the planted control (compare_outputs.py --diff's): a comparator blind to differences must not pass
orig = ra.compare
ra.compare = lambda f, o, arm: dict(orig(f, o, arm), verdict="EXACT", reason="") if orig(f, o, arm)["verdict"] == "MISMATCH" else orig(f, o, arm)
rc, out = run()
ra.compare = orig
case("resume: a blind comparator is caught by the planted control (UNKNOWN, CONTROL FAILED)", rc == 2 and "CONTROL FAILED" in out)
case("resume: the planted control passes on the real comparator, with and without reply text",
     ra.planted_control(ra.by_turn(ra.load(fresh_p), fresh_p), ra.by_turn(ra.load(res_p), res_p), None)
     and ra.planted_control({0: {k: v for k, v in FRESH[0][0].items() if k != "reply"}}, {0: {k: v for k, v in RES[0].items() if k != "reply"}}, None))
case("first_difference: tool-call arguments and call counts are positions too",
     "tool_calls[0].arguments char 6" in ra.first_difference(reply("t", args='{"a": 1}'), reply("t", args='{"a": 2}'))
     and "tool call count" in ra.first_difference(reply("t"), dict(reply("t"), tool_calls=[])))


# replay.py: --keep-replies stores the reply that out_sha hashes; --history-from re-sends ANOTHER run's replies as the history
# (so a fresh server sees the resumed run's exact prompt) and refuses a turn whose earlier replies are missing.
def fake_server(seen):
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            n = len(seen)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ch in [{"choices": [{"delta": {"reasoning_content": f"r{n}", "content": f"reply number {n}"}}]},
                       {"choices": [], "usage": {"prompt_tokens": 50 * n, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 0}}}]:
                self.wfile.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def machine_env(key):
    if os.environ.get(key):
        return os.environ[key]
    f = os.environ.get("LSK_ENV") or os.path.expanduser("~/.config/loom-strix-kits/env")
    for line in open(f) if os.path.isfile(f) else []:
        if line.startswith(key + "="):
            return line.split("=", 1)[1].split("#")[0].strip().strip('"')
    return ""


if machine_env("LOOM_SKILLS"):
    REPLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_bench", "replay.py")
    env = dict(os.environ, LOOM_SKILLS=machine_env("LOOM_SKILLS"))
    sess = os.path.join(tmp, "s.session.json")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u0"}, {"role": "assistant", "content": "REC0"},
            {"role": "user", "content": "u1"}, {"role": "assistant", "content": "REC1"}, {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "REC2"}]
    T = [{"upto": u, "effort": "none", "rec_new": 10, "rec_cached": 0, "rec_out": 3, "rec_wall_s": 1.0} for u in (2, 4, 6)]
    json.dump({"name": "t", "tools": [], "messages": msgs, "turns": T}, open(sess, "w"))

    def replay(args, log):
        seen = []
        srv = fake_server(seen)
        p = subprocess.run([sys.executable, REPLAY, sess, f"http://127.0.0.1:{srv.server_address[1]}", "m", "--mode", "own",
                            "--log", log] + args, capture_output=True, text=True, env=env)
        srv.shutdown()
        recs = [json.loads(x) for x in open(log)] if os.path.isfile(log) else []
        return p, seen, recs

    p, seen, recs = replay(["--turns", "0:3", "--keep-replies"], os.path.join(tmp, "b.jsonl"))
    rows = recs[0]["rows"] if recs else []
    case("replay --keep-replies: each row carries its reply, and out_sha is that reply's hash",
         len(rows) == 3 and all(r.get("reply") and checks.reply_sha(r["reply"]) == r["out_sha"] for r in rows))
    p, seen, recs = replay(["--turns", "2:3", "--history-from", os.path.join(tmp, "b.jsonl")], os.path.join(tmp, "a.jsonl"))
    sent = [m.get("content") for m in (seen[0]["messages"] if seen else []) if m["role"] == "assistant"]
    case("replay --history-from: a later turn re-sends the frozen run's replies (not the recording's, not its own)",
         sent == ["reply number 1", "reply number 2"])
    p, seen, recs = replay(["--turns", "1:3", "--history-from", os.path.join(tmp, "b.jsonl")], os.path.join(tmp, "a1.jsonl"))
    sent = [m.get("content") for m in (seen[1]["messages"] if len(seen) > 1 else []) if m["role"] == "assistant"]
    case("replay --history-from over two turns: the second still carries the frozen reply 1, not this run's own reply",
         sent == ["reply number 1", "reply number 2"])
    bonly = os.path.join(tmp, "b12.jsonl")
    with open(bonly, "w") as f:   # the frozen run without turn 1: turn 2's prompt cannot be rebuilt
        f.write(json.dumps(dict(json.load(open(os.path.join(tmp, "b.jsonl"))), rows=[r for r in rows if r["turn"] != 1])) + "\n")
    p, seen, recs = replay(["--turns", "2:3", "--history-from", bonly], os.path.join(tmp, "a2.jsonl"))
    case("replay --history-from: a turn whose earlier frozen replies are missing fails (nothing sent)", not seen and p.returncode != 0)
    p, seen, recs = replay(["--turns", "0:1", "--mode", "fixed", "--history-from", bonly], os.path.join(tmp, "a3.jsonl"))
    case("replay --history-from without own mode is a usage error", p.returncode != 0 and not seen)

print(f"{len(ran) - len(fails)}/{len(ran)} passed")
sys.exit(1 if fails else 0)
