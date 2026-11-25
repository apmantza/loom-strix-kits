#!/usr/bin/env python3
"""common/lint_counters.py on fixture workloads (rocprof-compute's results_pmc_perf_*.csv.gz, log.txt, clock.txt) and fixture
code objects. Recurrences each case prevents (HRX-like T4):
  - a bandwidth-bound kernel's vmcnt waits read as backing a 'serial' flag (q171b's matvec: 75 % counter wait at 92 % of the DRAM
    ceiling, a full bus, not serial loads); each threshold is straddled (24.9 / 25.0 %, 1.9 / 2.0 %, 0.69 / 0.70 of the ceiling);
  - a candidate's counters priced under the live kernel's name (q184's 'half' flash2 is named g_attention_flash2_f16o): counters
    attach only to a byte-identical code object, and a kernel without them says 'no counters', never a silent blank;
  - wait shares from a saturated SQ_WAVE_CYCLES (q171 / q171b flash2 at 20 x 2^32 read 100 %, E425) or from profile_standard (q171's
    matvec 16 % counter wait vs 75 % at auto) backing a flag;
  - two shapes of one kernel that disagree counted as backed (a kernel name covers several call shapes, E238);
  - served time split into phases by the kit's lanes file (events_summary.classify), not by name."""
import gzip, json, os, sys, tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import lint_counters as lc  # noqa: E402

HDR = "GPU_ID,Dispatch_ID,Grid_Size,Workgroup_Size,Kernel_Name,Start_Timestamp,End_Timestamp,Counter_Name,Counter_Value\n"
WC = 1.0e9          # SQ_WAVE_CYCLES of every fixture dispatch
DUR = 1_000_000     # ns per dispatch: bytes / DUR = GB/s


def workload(d, hsaco, level="auto", kernel="g_k", wait=None, bank=None, gbs=None, extra=None, hs_line=None):
    """a rocprof-compute workload dir: two dispatches with the given wait shares (% of WC), bank conflict %, GB/s"""
    os.makedirs(d)
    c = {"SQ_WAVE_CYCLES": WC}
    for w, pct in (wait or {}).items():
        c[lc.WAITS[w]] = WC * pct / 100
    if bank is not None:
        c["SQC_LDS_IDX_ACTIVE_sum"] = 1.0e6
        c["SQC_LDS_BANK_CONFLICT_sum"] = 1.0e6 * bank / 100
    if gbs is not None:
        for b in lc.EA:
            c[f"GL2C_EA_RDREQ_{b}B_sum"] = 0.0
        c["GL2C_EA_RDREQ_128B_sum"] = gbs * DUR / 128
    c.update(extra or {})
    with gzip.open(os.path.join(d, "results_pmc_perf_0.csv.gz"), "wt") as f:
        f.write(HDR + "".join(f"1,{i},393216,384,{kernel},{1000 * i},{1000 * i + DUR},{n},{v}\n" for i in (1, 2) for n, v in c.items()))
    with open(os.path.join(d, "log.txt"), "w") as f:
        f.write("Rocprofiler-Compute version: 3.10.0\n" + (hs_line or f"Command: python3 hsaco_run.py --kernel {kernel} f2={hsaco}\n"))
    if level:
        with open(os.path.join(d, "clock.txt"), "w") as f:
            f.write(f"sclk at {level}: 0: 600Mhz * 1: 1100Mhz  2: 2900Mhz\n")
    return d


def main():
    fails = 0

    def ok(cond, what, got):
        nonlocal fails
        print(("ok  " if cond else "FAIL") + f" {what}: {got}")
        fails += not cond

    with tempfile.TemporaryDirectory() as t:
        codes = os.path.join(t, "codes"); os.makedirs(codes)
        for k, b in (("g_k", b"served kernel"), ("g_j", b"another served kernel")):
            open(os.path.join(codes, k + ".hsaco"), "wb").write(b)
        live = os.path.join(t, "live_g_k.hsaco"); open(live, "wb").write(b"served kernel")
        cand = os.path.join(t, "zz_half_g_k.hsaco"); open(cand, "wb").write(b"a candidate kernel")   # sorts after live: an ambiguous
        # Command line must be refused, not resolved to its first path
        n = [0]

        def verdict(rule, bw=240.0, **kw):
            n[0] += 1
            wl = lc.load_workloads([workload(os.path.join(t, f"w{n[0]}"), live, **kw)], codes)
            return lc.back(rule, wl, "g_k", bw)[0]

        # serial: memory wait AND below 70 % of the ceiling
        ok(verdict("serial", wait={"cnt": 40}, gbs=100) == "backed", "serial: cnt 40 % at 100 GB/s backed", "")
        ok(verdict("serial", wait={"cnt": 40}, gbs=0.70 * 240) == "not backed", "serial: at 0.70 of the ceiling = bandwidth-bound", "")
        ok(verdict("serial", wait={"cnt": 40}, gbs=0.69 * 240) == "backed", "serial: at 0.69 of the ceiling backed", "")
        ok(verdict("serial", wait={"cnt": 24.9}, gbs=100) == "not backed", "serial: cnt 24.9 % not backed", "")
        ok(verdict("serial", wait={"cnt": 25.0}, gbs=100) == "backed", "serial: cnt 25.0 % backed", "")
        ok(verdict("serial", wait={"vmem": 10, "lgkm": 5, "cnt": 40}, gbs=100) == "not backed",
           "serial: vmem (10 %) wins over cnt when the split is collected", "")
        ok(verdict("serial", wait={"cnt": 40}) == "unusable counters", "serial: no byte counters -> unusable, not backed", "")
        # ldsbank: measured conflicts
        ok(verdict("ldsbank", bank=2.0) == "backed", "ldsbank: 2.0 % backed", "")
        ok(verdict("ldsbank", bank=1.9) == "not backed", "ldsbank: 1.9 % not backed", "")
        # instruction-count rules: issue wait
        for rule in ("copies", "select", "join", "addr64"):
            ok(verdict(rule, wait={"issue": 25.0}) == "backed", f"{rule}: issue 25.0 % backed", "")
            ok(verdict(rule, wait={"issue": 24.9}) == "not backed", f"{rule}: issue 24.9 % not backed", "")
        # level: profile_standard waits do not back; bank % (a count ratio) still does
        ok(verdict("copies", level="profile_standard", wait={"issue": 40}) == "unusable counters",
           "profile_standard waits are not used", "")
        ok(verdict("copies", level=None, wait={"issue": 40}) == "unusable counters", "unknown level: waits not used", "")
        ok(verdict("ldsbank", level="profile_standard", bank=8.0) == "backed", "profile_standard bank % used", "")

        # saturation: SQ_WAVE_CYCLES and INST_ANY at 20 x 2^32 (q171b flash2)
        sat = 20.0 * 2 ** 32
        n[0] += 1
        wl = lc.load_workloads([workload(os.path.join(t, "sat"), live, wait={"cnt": 40}, bank=8.0,
                                         extra={"SQ_WAVE_CYCLES": sat, "SQ_WAIT_INST_ANY": sat})], codes)
        ok(lc.back("copies", wl, "g_k", 240)[0] == "unusable counters", "saturated: issue share dropped", lc.back("copies", wl, "g_k", 240))
        ok(lc.back("ldsbank", wl, "g_k", 240)[0] == "backed", "saturated: bank % (unsaturated counters) kept", "")
        cell, warn = lc.counters_cell(wl, "g_k")
        ok("SATURATED" in warn and "SQ_WAVE_CYCLES" in warn and "wait saturated" in cell, "saturation warning in the row", (cell, warn))

        # identity: a candidate's code object under the live name is not used; no workload = 'no counters'
        wl = lc.load_workloads([workload(os.path.join(t, "cand"), cand, wait={"issue": 40})], codes)
        ok(lc.back("copies", wl, "g_k", 240)[0] == "no counters", "candidate code object not used", "")
        cell, _ = lc.counters_cell(wl, "g_k")
        ok(cell.startswith("no counters; not used:") and "differs from the served compile" in cell, "row names the unused workload", cell)
        ok(lc.counters_cell([], "g_j")[0] == "no counters", "no workload at all: 'no counters'", lc.counters_cell([], "g_j")[0])
        wl = lc.load_workloads([workload(os.path.join(t, "two"), live, wait={"issue": 40},
                                         hs_line=f"Command: run a={live} b={cand}\n")], codes)
        ok(lc.back("copies", wl, "g_k", 240)[0] == "no counters", "two code objects on the Command line: not used", "")
        wl = lc.load_workloads([workload(os.path.join(t, "two2"), live, wait={"issue": 40},
                                         hs_line=f"Command: run a={live} b={cand}\n") + f":hsaco={live}"], codes)
        ok(lc.back("copies", wl, "g_k", 240)[0] == "backed", "hsaco= in the spec resolves it", "")
        empty = os.path.join(t, "empty"); os.makedirs(empty)
        wl = lc.load_workloads([empty], codes)
        ok(wl[0]["why"] == "no counter rows" and lc.back("copies", wl, "g_k", 240)[0] == "no counters", "a dir without counters", wl[0]["why"])

        # two shapes that disagree: mixed, and not counted as backed
        wl = lc.load_workloads([workload(os.path.join(t, "s1"), live, wait={"issue": 40}),
                                workload(os.path.join(t, "s2"), live, wait={"issue": 10})], codes)
        ok(lc.back("copies", wl, "g_k", 240)[0] == "mixed", "two shapes disagree: mixed", "")
        lint = {"g_k": {"kernel": "g_k", "flags": ["copies"], "notes": ["maskedv"]}, "g_j": {"kernel": "g_j", "flags": ["copies"]}}
        srv = {("g_k", "prefill"): {"ms": 60.0, "calls": 2, "shapes": {1024}}, ("g_j", "decode"): {"ms": 40.0, "calls": 2, "shapes": {8}}}
        A, B, C = lc.build(lint, srv, 100.0, wl, 240)
        row = next(x for x in B if x.startswith("copies"))
        ok(row.split()[1:5] == ["2", "2", "1", "0"] and "0/1 (n=1" in row, "precision: mixed is a hit with counters, not backed", row)
        ok(any("maskedv" in x and "not priced" in x for x in B), "maskedv not priced", "")
        ok(any("g_j" in x and "no counters" in x for x in A), "table row: 'no counters' named", "")
        ok(any(x.split()[1:4] == ["15.000", "g_k", "copies"] for x in C), "price = 60 % x mean(40, 10) % = 15.000 %", C[2:])
        ok(any(x.strip().startswith("40.00 %  g_j") for x in C), "unpriced hit listed with its share", C[-2:])

        # served time: phases from the lanes file
        run = os.path.join(t, "run"); os.makedirs(run)
        ev = [("g_token_info", 1, [4096, 1, 1], 1000), ("g_attention_flash2", 1, [64, 2, 1], 5_000_000), ("g_k", 1, [8, 1, 1], 1_000_000),
              ("g_token_info", 2, [8, 1, 1], 1000), ("g_k", 2, [8, 1, 1], 2_000_000)]
        with open(os.path.join(run, "events.jsonl"), "w") as f:
            for k, s, wg, ns in ev:
                f.write(json.dumps({"type": "dispatch_event", "key": k, "submission_id": s, "workgroup_count": wg, "duration_ns": ns}) + "\n")
        open(os.path.join(run, "run.req.json"), "w").write(json.dumps({"requests": [{"prompt_tokens": 4096}]}) + "\n")
        lanes = os.path.join(t, "lanes.json")
        json.dump({"prefill_marker": "^g_attention_flash2", "prefill_min_tokens": 64,
                   "lanes": [{"lane": "d", "phase": "decode", "regex": ".", "floor": "decode_kv"}, {"lane": "p", "phase": "prefill", "regex": "."}]}, open(lanes, "w"))
        rows, total, prompt, steps = lc.served(run, lanes)
        ok(abs(rows[("g_k", "prefill")]["ms"] - 1.0) < 1e-9 and abs(rows[("g_k", "decode")]["ms"] - 2.0) < 1e-9 and prompt == 4096,
           "g_k split: 1 ms prefill (flash2 submission), 2 ms decode", {k: v["ms"] for k, v in rows.items()})
        ok(rows[("g_k", "prefill")]["lanes"] == {"p": 1.0} and rows[("g_k", "decode")]["lanes"] == {"d": 2.0} and steps == 1,
           "per-lane ms on each row; verify steps = decode submissions in the decode_kv lane (events_summary's count)",
           (rows[("g_k", "prefill")]["lanes"], steps))
        # a kit whose lanes.json has no verify-attention (decode_kv) lane served before (d) existed: it must still run
        json.dump({"prefill_marker": "^g_attention_flash2", "lanes": [{"lane": "d", "phase": "decode", "regex": "."},
                                                                       {"lane": "p", "phase": "prefill", "regex": "."}]}, open(lanes + "2", "w"))
        ok(lc.served(run, lanes + "2")[3] is None, "no decode_kv lane: verify steps None, not a crash", "")
        nf = os.path.join(t, "nf.json")
        json.dump({"d": 1.5}, open(nf, "w"))
        ok(lc.lane_noise(nf, json.load(open(lanes))) == {"d": 1.5}, "noise floor keys resolve to lane names", "")
        json.dump({"x": 1.5}, open(nf, "w"))
        try:
            lc.lane_noise(nf, json.load(open(lanes)))
            ok(False, "a noise key matching no lane must stop", "passed")
        except SystemExit:
            ok(True, "a noise key matching no lane stops", "")
        ok(rows[("g_attention_flash2", "prefill")]["shapes"] == {128}, "shape = workgroups product", rows[("g_attention_flash2", "prefill")]["shapes"])

    # counters.sh writes one CSV per preset, each numbering dispatches from 1, and SQ_WAVE_CYCLES sits in several presets: summed
    # per dispatch id alone it counted 3x and every wait share read a third (T6: q130's any-wait 65.0 % read 32.5 %)
    with tempfile.TemporaryDirectory() as t:
        H2 = '"Dispatch_Id","Grid_Size","Kernel_Name","Workgroup_Size","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"\n'
        WC = 1e9
        for name, cs in (("waits", {"SQ_WAVE_CYCLES": WC, "SQ_WAIT_CNT_VMVS": 0.4 * WC, "SQ_WAIT_ANY": 0.8 * WC}),
                         ("lds", {"SQ_WAVE_CYCLES": WC, "SQC_LDS_BANK_CONFLICT": 5e4, "SQC_LDS_IDX_ACTIVE": 1e6}),
                         ("split", {"SQ_WAVE_CYCLES": WC, "SQ_WAIT_INST_ANY": 0.3 * WC, "SQ_WAIT_ANY": 0.8 * WC})):
            os.makedirs(f"{t}/{name}")
            with open(f"{t}/{name}/{name}_counter_collection.csv", "w") as f:
                f.write(H2)
                for i in (1, 2):
                    for c, v in cs.items():
                        f.write(f'{i},327680,"g_k",256,"{c}",{v},{1000 * i},{1000 * i + 1000000}\n')
        m = lc.metrics(lc.read_rows(t)["g_k"], set())
        ok(abs(m["vmem"] - 40) < 1e-6 and abs(m["any"] - 80) < 1e-6 and abs(m["issue"] - 30) < 1e-6,
           "counters.sh presets: a counter in two presets is not summed twice (vmem 40, any 80, issue 30)",
           {k: round(m[k], 1) for k in ("vmem", "any", "issue")})

    # ------------------------------------------------------------ section (d): levers (user 11-25: 'make the linter rank, not
    # just flag'). Recurrences: flash2 sat at 6 waves held by VGPR and LDS at once while (a) showed one 'limit', so an occupancy
    # lever that trims registers alone looked open (both must move); a lever without a lint flag never appeared; levers were
    # priced against served share only, never against the lane's floor or the A/B noise floor (rules 2: price before building).
    with tempfile.TemporaryDirectory() as t:
        js = os.path.join(t, "k.json")
        json.dump({"target_resources": {"limiting_resource": "amdgpu.vgpr", "residency": {"current_tier": 6, "best_tier": 16,
                                                                                         "next_better_tier": 7},
                                        "vector": {"final": {"register_count": 240}}},
                   "residency_constraints": {"rows": [
                       {"name": "amdgpu.sgpr", "unit": "registers", "units": 90, "independent_tier": 16, "limiting": False},
                       {"name": "amdgpu.vgpr", "unit": "registers", "units": 240, "independent_tier": 6, "limiting": True,
                        "reduction_units_to_next_better_tier": 24},
                       {"name": "amdgpu.lds", "unit": "bytes", "units": 61952, "independent_tier": 6, "limiting": True,
                        "reduction_units_to_next_better_tier": 18944},
                       {"name": "amdgpu.workgroup_slots", "independent_tier": 15, "limiting": False}]}}, open(js, "w"))
        lim, tier, vg, lims, nt = lc.report_limit(js)
        ok(lim == "vgpr+lds" and tier == 6 and vg == 240 and nt == 7 and [(x["name"], x["reduction"]) for x in lims] == [("vgpr", 24), ("lds", 18944)],
           "compile report: both limiting resources with their reductions (flash2: VGPR and LDS)", (lim, tier, vg, lims))
        json.dump({"target_resources": {"limiting_resource": "amdgpu.lds", "residency": {"current_tier": 4}}}, open(js, "w"))
        ok(lc.report_limit(js)[0] == "lds" and lc.report_limit(js)[3] == [], "no residency rows: the summary's one limit, no rows",
           lc.report_limit(js))

        # lanes: decode weights 2.4 GB at 240 GB/s = 10 ms/step x 2 steps = 20 ms floor; prefill 2 x 1e9 params x 4096 tokens at
        # 8.192 TFLOP/s = 1.0 s; a lane without a floor is judged against 0 (ceiling.py)
        d = {"model": {"decode_weight_bytes": 2.4e9, "linear_params": 1e9}, "prefill_marker": "^g_fl",
             "lanes": [{"lane": "dec w", "phase": "decode", "regex": "^g_mv", "floor": "decode_weights"},
                       {"lane": "dec other", "phase": "decode", "regex": ".", "floor": None},
                       {"lane": "pre gemm", "phase": "prefill", "regex": "^g_gemm", "floor": "prefill_linear"},
                       {"lane": "pre attn", "phase": "prefill", "regex": "^g_fl", "floor": "prefill_attn"},
                       {"lane": "pre other", "phase": "prefill", "regex": ".", "floor": None}]}
        hw = {"bw_practical_gbs": 240, "wmma_f16_tflops": 8.192}
        srv = {("g_mv_a", "decode"): {"ms": 30.0, "lanes": {"dec w": 30.0}}, ("g_mv_b", "decode"): {"ms": 20.0, "lanes": {"dec w": 20.0}},
               ("g_x", "decode"): {"ms": 4.0, "lanes": {"dec other": 4.0}},
               ("g_gemm_a", "prefill"): {"ms": 1500.0, "lanes": {"pre gemm": 1500.0}},
               ("g_fl", "prefill"): {"ms": 100.0, "lanes": {"pre attn": 100.0}}}
        heads = lc.lane_headroom(srv, 2, 4096, d, hw)
        h = {k: (round(v["floor_ms"], 6) if v["floor_ms"] is not None else None, round(v["headroom_ms"], 6) if v["headroom_ms"] is not None
                 else None) for k, v in heads.items()}
        ok(h["dec w"] == (20.0, 30.0) and h["pre gemm"] == (1000.0, 500.0) and h["dec other"] == (0.0, 4.0),
           "lane headroom: served - floor (decode floor x verify steps; no floor = 0)", h)
        ok(h["pre attn"] == (None, None) and "attn_layers" in heads["pre attn"]["why"], "a floor missing a model constant: unknown, named",
           heads["pre attn"]["why"])
        ok(lc.lane_headroom(srv, 0, 4096, d, hw)["dec w"]["headroom_ms"] is None, "no verify steps: decode headroom unknown, not 0", "")
        ok(lc.lane_headroom({("g_mv_a", "decode"): {"ms": 15.0, "lanes": {"dec w": 15.0}}}, 2, 4096, d, hw)["dec w"]["headroom_ms"] == 0.0,
           "served below its floor: headroom 0", "")

        # levers: g_mv_a holds 30 / 50 of its lane's 30 ms headroom = 18 ms; a backed 'copies' (issue 40 %) prices 30 x 0.4 = 12 ms;
        # g_mv_b's 'join' has no counters (static only: the attribution 12 ms is its bound); g_x has no flag (no evidence)
        codes = os.path.join(t, "codes"); os.makedirs(codes)
        open(os.path.join(codes, "g_mv_a.hsaco"), "wb").write(b"a")
        hs = os.path.join(t, "a.hsaco"); open(hs, "wb").write(b"a")
        wl = lc.load_workloads([workload(os.path.join(t, "wa"), hs, kernel="g_mv_a", wait={"issue": 40, "vmem": 30, "lgkm": 5})], codes)
        lint = {"g_mv_a": {"flags": ["copies", "join"]}, "g_mv_b": {"flags": ["join"], "notes": ["maskedv"]}, "g_x": {},
                "g_gemm_a": {}, "g_fl": {"flags": ["copies"]}}
        lv = lc.levers(lint, srv, wl, 240, heads)
        got = {(x["kernel"], x["lever"]): (x["evidence"], x["upper_ms"] if x["upper_ms"] is None else round(x["upper_ms"], 6)) for x in lv}
        ok(got[("g_mv_a", "copies")] == ("counter-backed", 12.0), "backed lever: min(attribution 18, price 12) = 12 ms", got)
        ok(got[("g_mv_b", "join")] == ("static only", 12.0), "static-only lever: bounded by its headroom attribution", got)
        ok(got[("g_x", "unnamed (no lint flag)")] == ("no evidence", 4.0), "a kernel without a flag still appears (no evidence)", got)
        ok(got[("g_gemm_a", "unnamed (no lint flag)")] == ("no evidence", 500.0), "the whole lane's headroom on its one kernel", got)
        ok(got[("g_fl", "copies")] == ("static only", None), "headroom unknown and no price: no bound", got)
        ok(not any(x["lever"] == "maskedv" for x in lv), "unpriced rules (maskedv) are not speed levers", "")
        ok([x["upper_ms"] for x in lv if x["upper_ms"] is not None] == sorted((x["upper_ms"] for x in lv if x["upper_ms"] is not None),
                                                                              reverse=True) and lv[-1]["upper_ms"] is None,
           "ranked by upper bound, unbounded last", [(x["kernel"], x["lever"], x["upper_ms"]) for x in lv])

        # amdahl: one phase per lane, the lane's noise floor as its screen, one overlap group per kernel
        noise = {"dec w": 50.0, "pre gemm": 0.33}
        doc, picked, omitted = lc.amdahl_input(lv, heads, noise, 10)
        ph = {p["phase"]: p for p in doc["phases"]}
        ok(ph["dec w"]["screen_pct"] == 50.0 and ph["dec w"]["wall"] == 50.0 and
           not any("screen_pct" in p for p in lc.amdahl_input(lv, heads, {}, 10)[0]["phases"]),
           "lane = phase, wall = lane served ms, screen = the lane's noise floor (none: amdahl's default)", ph["dec w"])
        grp = [c["overlap_group"] for c in ph["dec w"]["candidates"] if c["name"].startswith("g_mv_a")]
        ok(len(grp) == 2 and set(grp) == {"g_mv_a"}, "two levers on one kernel share its overlap group", grp)
        ok(not any(x["upper_ms"] is None for x in picked), "unbounded levers are not sent to amdahl", "")
        ok(omitted and omitted[0][0] == "dec other", "a lane whose levers claim its whole time (floor 0) is left out, named", omitted)
        out, verdicts = lc.run_amdahl(doc, os.path.join(t, "amdahl_in.json"))
        ok(verdicts.get(("pre gemm", "g_gemm_a:unnamed (no lint flag)")) is True, "500 of 1500 ms clears a 0.33 % screen", verdicts)
        ok(verdicts.get(("dec w", "g_mv_a:copies")) is False, "12 of 50 ms is below a 50 % screen: not worth building", verdicts)
        ok("amdahl:" in out[-1], "amdahl's own output is in the report", out[-1:])
        lines = lc.section_d(lv, heads, verdicts, omitted, noise, lint, srv, wl, 10, "fixture hw")
        txt = "\n".join(lines)
        ok("not worth building" in txt and "an attribution, not a measurement" in txt, "verdicts and the attribution caveat printed", "")
        lint["g_fl"].update(_limits=lims, waves=6, _next_tier=7)
        txt = "\n".join(lc.section_d(lv, heads, verdicts, omitted, noise, lint, srv, wl, 10, "fixture hw"))
        occ = [x for x in txt.splitlines() if x.lstrip().startswith("g_fl")]
        ok(occ and "vgpr 240 registers -24" in occ[-1] and "lds 61952 bytes -18944" in occ[-1] and "all 2" in occ[-1],
           "occupancy lever lists every limiting resource and says it needs all of them", occ)
        lint["g_fl"]["_limits"] = [{"name": "max_waves", "units": None, "unit": "", "reduction": None, "tier": 16}]
        txt = "\n".join(lc.section_d(lv, heads, verdicts, omitted, noise, lint, srv, wl, 10, "fixture hw"))
        ok(not [x for x in txt.split("(d2)")[1].splitlines() if x.lstrip().startswith("g_fl")], "a kernel at the best tier has no lever", "")
        lint["g_mv_a"].update(_limits=lims, waves=6, _next_tier=7)
        txt = "\n".join(lc.section_d(lv, heads, verdicts, omitted, noise, lint, srv, wl, 10, "fixture hw"))
        occ = [x for x in txt.split("(d2)")[1].splitlines() if x.lstrip().startswith("g_mv_a")]
        ok(occ and "35.0 %" in occ[0], "occupancy lever shows the latency-wait share (vmem 30 + lgkm 5)", occ)

    print(f"{'FAILED ' + str(fails) if fails else 'all ok'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
