#!/usr/bin/env python3
"""common/autotune.py on real compiles of a fixture generator (kit/fixtures/autotune_gen.py: AT_LIVE values kept live per lane in
at_hold, 8 -> 16 waves, 100 -> 9 waves, 300 -> a 1,964 B spill; AT_SCALE a constant in at_scale; AT_OTHER a kernel outside the
family; AT_BAD a compile error), on synthetic rows for the gates no cheap fixture reaches, and on a stub harness at the GPU process
boundary. Recurrences it prevents:
  - a tile or rows knob that spills ranked by its instruction count (E230: a rewrite's 2,464 B spill; spills share vmcnt with the
    loads): scratch above the base disqualifies;
  - a knob value that costs residency passing a static screen (E236: a variant's residency drop): fewer waves disqualify unless the
    knob declares trades_occupancy;
  - a knob that reintroduces a kernel_lint defect shape (E229/E230 select chains, E179 copies): a flag the base lacked disqualifies;
  - a misspelt or retired knob searched as if it did something (a switch that did not apply, binary_gate's E372 lesson): a knob no
    generator reads by a literal name is refused (exit 2), a name quoted only in prose is not a read, and a knob that reaches no
    family kernel is reported KNOB NO-OP;
  - the base dropped from the comparison (every ratio is against it): the base is evaluated first and survives;
  - a timed finalist whose outputs differ, or whose gain is inside the A/A spread, handed on as faster.
usage: python3 kit/test_autotune.py   (needs loom-compile: VENV from the environment or the machine env file)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LSK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LSK, "common"))
import autotune as at  # noqa: E402

TOOL = os.path.join(LSK, "common", "autotune.py")
GEN = {"argv": ["{python}", "{lsk}/kit/fixtures/autotune_gen.py", "{out}/program/fixture.loom"], "sources": ["kit/fixtures/autotune_gen.py"]}
fails = 0


def ok(cond, what):
    global fails
    print(("ok  " if cond else "FAIL") + " " + what)
    fails += not cond


def search(t, name, knobs, **kw):
    sp = dict({"family": "fixture", "kernels": "^at_(hold|scale)$", "generator": GEN, "knobs": knobs, "budget": 12}, **kw)
    path = os.path.join(t, name + ".json")
    json.dump(sp, open(path, "w"))
    r = subprocess.run([sys.executable, TOOL, "search", path, "--out", os.path.join(t, name)], capture_output=True, text=True)
    res = os.path.join(t, name, "result.json")
    doc = json.load(open(res)) if os.path.exists(res) else None
    by = {c["label"]: c for c in doc["candidates"]} if doc else {}
    return r.returncode, r.stdout + r.stderr, by, res


def main():
    with tempfile.TemporaryDirectory(prefix="test_autotune.") as t:
        rc, out, by, _ = search(t, "unknown", {"AT_LIVE": ["100"], "AT_NOPE": ["1"]})
        ok(rc == 2 and "unknown knob(s) ['AT_NOPE']" in out, f"a knob no generator reads is refused, exit 2 ({rc}: {out.strip()[-120:]})")
        rc, out, by, _ = search(t, "prose", {"AT_RETIRED": ["1"]})
        ok(rc == 2 and "AT_RETIRED" in out, f"a knob named only in a docstring and a comment is not known, exit 2 ({rc})")
        rc, out, by, _ = search(t, "badbase", {"AT_LIVE": ["100"]}, base_env={"AT_BAD": "1"})
        ok(rc == 2 and "REFUSED: the base" in out and "compile error" in out, f"a base that does not compile refuses, exit 2 ({rc}: {out.strip()[-120:]})")
        rc, out, by, _ = search(t, "nofamily", {"AT_LIVE": ["100"]}, kernels="^nothing$")
        ok(rc == 2, f"a family regex matching no base kernel refuses, exit 2 ({rc})")

        rc, out, by, res = search(t, "grid", {"AT_LIVE": ["100", "300"], "AT_SCALE": ["3.0"], "AT_BAD": ["1"], "AT_OTHER": ["5.0"]},
                                  budget=24)
        b = by.get("base", {})
        ok(rc == 0 and b.get("id") == "c00" and b.get("status") == "ok", f"the base is evaluated first and survives ({b.get('id')} {b.get('status')})")
        c = by.get("AT_LIVE=300", {})
        ok(c.get("status") == "DQ" and any("scratch 0 -> " in w for w in c.get("why", [])),
           f"AT_LIVE=300 (a spill) is disqualified on scratch ({c.get('why')})")
        c = by.get("AT_LIVE=100", {})
        ok(c.get("status") == "DQ" and c.get("why") == ["at_hold: waves 16 -> 9 (no knob declares trades_occupancy)"],
           f"AT_LIVE=100 (9 waves, no spill) is disqualified on waves alone ({c.get('why')})")
        c = by.get("AT_SCALE=3.0", {})
        ok(c.get("status") == "ok", f"AT_SCALE=3.0 (same cost) survives ({c.get('status')} {c.get('why')})")
        c = by.get("AT_BAD=1", {})
        ok(c.get("status") == "DQ" and any("compile error" in w for w in c.get("why", [])), f"a compile error disqualifies ({c.get('why')})")
        c = by.get("AT_OTHER=5.0", {})
        ok(c.get("status") == "no-op" and c.get("outside") == ["at_other"], f"a knob outside the family is a no-op, its kernel reported ({c.get('status')} {c.get('outside')})")
        ok("KNOB NO-OP: AT_OTHER" in out and "KNOB NO-OP: AT_LIVE" not in out and "OUTSIDE: at_other" in out, "KNOB NO-OP names only AT_OTHER; OUTSIDE names at_other")
        ok("AUTOTUNE: " in out and "survivors, best base" in out, f"verdict line, ties go to the base ({out.strip().splitlines()[-1]})")

        rc, out, by, _ = search(t, "waiver", {"AT_LIVE": {"values": ["100", "300"], "trades_occupancy": "fixture: more values live per lane"}})
        ok(by.get("AT_LIVE=100", {}).get("status") == "ok", f"trades_occupancy waives the waves gate ({by.get('AT_LIVE=100', {}).get('why')})")
        ok(by.get("AT_LIVE=300", {}).get("status") == "DQ", "... but not the scratch gate")
        rc, out, by, _ = search(t, "none", {"AT_LIVE": ["300"]})
        ok(rc == 1 and "AUTOTUNE: NONE" in out, f"only a spilling candidate: AUTOTUNE: NONE, exit 1 ({rc})")
        rc, out, by, _ = search(t, "gen", {"AT_LIVE": ["x"]})
        ok(by.get("AT_LIVE=x", {}).get("status") == "DQ" and "generation exit" in " ".join(by.get("AT_LIVE=x", {}).get("why", [])),
           "a generator error disqualifies")
        rc, out, by, _ = search(t, "coord", {"AT_SCALE": ["3.0"], "AT_LIVE": ["100", "300"]}, strategy="coordinate", budget=6)
        ok(by.get("base", {}).get("id") == "c00" and len(by) <= 6, f"coordinate: base first, budget kept ({sorted(by)})")

        # gates no cheap fixture reaches, through the same judge
        base = {"k": {"scratch": 0, "flags": [], "waves": 12}, "m": {"scratch": 64, "flags": ["copies"], "waves": 8}}
        ok(at.judge(base, base) == [], "the base judged against itself passes (it keeps its own scratch and flags)")
        ok(at.judge(base, dict(base, k=dict(base["k"], flags=["select"]))) == ["k: new lint flag select"], "a lint flag the base lacked disqualifies")
        ok(at.judge(base, {"k": base["k"]}) == ["m: missing"], "a family kernel missing in the candidate disqualifies")
        ok(at.judge(base, dict(base, n={"scratch": 0, "flags": ["serial"], "waves": 12})) == ["n: new lint flag serial"],
           "a new family kernel with any flag disqualifies")
        ok(at.judge(base, dict(base, k=dict(base["k"], waves=None))) == ["k: no waves in the compile report"], "no residency report fails closed")

        # timed stage through a stub harness (the GPU process boundary): ms per code object from AT_STUB
        stub = os.path.join(t, "stub.py")
        open(stub, "w").write(
            "import json, os, sys\nms = json.loads(os.environ['AT_STUB'])\nobjs = [a.split('=', 1) for a in sys.argv[1:] if '=' in a and a.endswith('.hsaco')]\n"
            "cmp = '--compare' in sys.argv\nrc = 0\n"
            "for lab, p in objs:\n    v = ms.get(os.path.basename(os.path.dirname(p)), 1.0)\n    v = v * ms.get(lab, 1.0) if lab == 'aa' else v\n    print(f'{lab} median {abs(v):.4f} ms  best 0 ms (20 reps)  ratio 1')\n"
            "    if cmp and v < 0 and lab != objs[0][0]:\n        print(f'compare y: {lab} != {objs[0][0]}: max |diff| 1, 1 elements differ'); rc = 1\nsys.exit(rc)\n")
        rc, out, by, res = search(t, "timed", {"AT_SCALE": ["3.0", "4.0"]},
                                  timed={"top": 2, "rounds": 2, "kernels": {"at_scale": {"workload": ["n=4096"], "buf": ["w=16384", "y=16384"], "compare": ["y:f32"]}}})
        d3 = os.path.basename(os.path.dirname(by["AT_SCALE=3.0"]["rows"]["at_scale"]["hsaco"]))
        d4 = os.path.basename(os.path.dirname(by["AT_SCALE=4.0"]["rows"]["at_scale"]["hsaco"]))
        # the key covers the disassembly too (verify r2 V3): a copy of autotune.py beside kernel_lint.py and a touched loop_census.py
        lv = os.path.join(t, "lv"); os.makedirs(lv)
        for f in ("autotune.py", "kernel_lint.py", "loop_census.py"):
            shutil.copy(os.path.join(os.path.dirname(at.__file__), f), lv)
        ver = lambda: subprocess.run([sys.executable, "-c", f"import importlib.util as u; s = u.spec_from_file_location('a', {lv + '/autotune.py'!r}); "
                                      "m = u.module_from_spec(s); s.loader.exec_module(m); print(m.lint_version())"],
                                     capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=os.path.dirname(at.__file__))).stdout.strip()
        v0 = ver()
        open(os.path.join(lv, "loop_census.py"), "a").write("\n# touched\n")
        v1 = ver()
        ok(v0 == at.lint_version() and v1 and v1 != v0, f"lint_version changes when loop_census.py (the disassembly) changes ({v0} -> {v1})")
        ok(d3.endswith("." + at.lint_version()) and len(at.lint_version()) == 12, f"the compile cache is keyed by kernel_lint's version: "
           f"a lint change never reuses an older lint's row.json (review r1 F6) ({d3})")
        def timed(ms):
            r = subprocess.run([sys.executable, TOOL, "time", res, "--hsaco-run", stub], capture_output=True, text=True,
                               env=dict(os.environ, AT_STUB=json.dumps(ms)))
            return r.returncode, r.stdout + r.stderr, json.load(open(os.path.join(os.path.dirname(res), "timed.json")))
        rc, out, doc = timed({d3: 0.90, d4: 0.95})
        ok(rc == 0 and doc["best_env"] == "AT_SCALE=3.0", f"the faster finalist beyond the A/A spread is best ({rc} {doc['best_env']})")
        rc, out, doc = timed({d3: -0.90, d4: 1.0})
        v = {r["label"]: r["verdict"] for r in doc["rows"]}
        ok(rc == 1 and v == {"AT_SCALE=3.0": "DQ", "AT_SCALE=4.0": "within A/A"}, f"a finalist whose outputs differ is disqualified, an equal one is not faster ({rc} {v})")
        rc, out, doc = timed({d3: 0.98, d4: 1.0, "aa": 1.03})
        v = {r["label"]: r["verdict"] for r in doc["rows"]}
        ok(rc == 1 and v["AT_SCALE=3.0"] == "within A/A" and "NONE" in out, f"a 2 % gain inside a 3 % A/A spread is not handed on ({rc} {v})")

        # the step script (a .cmd recipe; nothing is run)
        cmd = os.path.join(t, "pkg-x.cmd")
        open(cmd, "w").write("env KV8=1 MV_PF=2 kit/build.py kits/qwen38-27b/config.json pkg-x --tokenizer tok/tokenizer.json --rows 4  # loom-strix-kits abc\n")
        sp = {"family": "mv", "kernels": "^g_mv_", "kit": "qwen38-27b", "base_cmd": cmd, "knobs": {"MV_CHUNK": {"values": ["8"]}},
              "strategy": "grid", "budget": 2, "timed": {"kernels": {"g_mv_x": {}}, "chain": "bash chain.sh {pkg}"}}
        s = at.emit_step({"space": sp}, os.path.join(t, "r.json"), "pkg-at")
        ok("EXTRA='KV8=1 MV_PF=2'\" $BEST\"" in s and "--allow '^g_mv_'" in s and 'bash chain.sh "$WORK/$CAND"' in s
           and "KIT_CONFIG=config.json" in s and "KIT_BUILD_ARGS='--tokenizer tok/tokenizer.json --rows 4'" in s,
           "step: the best's switches after the base recipe, the binary gate on the family, then the chain")
        try:
            at.emit_step({"space": {"generator": GEN}}, "r.json", "p")
            ok(False, "step refuses a generator-mode space")
        except at.Refuse:
            ok(True, "step refuses a generator-mode space")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
