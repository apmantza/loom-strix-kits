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
    # just flag'). Recurrences: flash2 sat at 6 waves held by VGPR and LDS at once while (a) showed one 'limit'; a lever without a
    # lint flag never appeared; levers were priced against served share only, never against the lane's floor or the noise floor.
    # r3 (review 11-25): Loom 0.2.0's report rounds LDS and workgroup slots to whole workgroups but not VGPRs (HRX-like E533), so
    # (d2) relayed flash2 'tier 7' (384 lanes = 3 waves per SIMD: tiers 3/6/9 only) and the GEMMs 'tier 5' (256 lanes: 2/4/6);
    # a per-kernel attribution was called an upper bound and printed 'not worth building'; a 'not backed' price cut vsplit's bound.
    def report(path, lanes_, cur, rows, wave=32, cap=16):
        doc = {"target_resources": {"limiting_resource": "amdgpu.vgpr", "subgroup_size": wave, "max_subgroups_per_simd": cap,
                                    "residency": {"current_tier": cur, "best_tier": 16, "next_better_tier": cur + 1},
                                    "vector": {"final": {"register_count": 240}}},
               "residency_constraints": {"rows": rows}}
        if lanes_:
            doc["workload"] = {"workgroup_size": {"flat": lanes_}}
        json.dump(doc, open(path, "w"))
        return path

    def vgpr(units, tier, limiting=True):
        return {"name": "amdgpu.vgpr", "kind": "pooled_resource", "unit": "registers", "allocation_scope": "subgroup", "pool_scope": "SIMD",
                "pool_units": 1536, "allocation_granularity": 24, "units": units, "independent_tier": tier, "limiting": limiting,
                "reduction_units_to_next_better_tier": 24}

    def lds(units, tier, limiting=True):
        return {"name": "amdgpu.lds", "kind": "pooled_resource", "unit": "bytes", "allocation_scope": "workgroup",
                "pool_scope": "occupancy domain", "pool_units": 131072, "allocation_granularity": 1024, "units": units,
                "independent_tier": tier, "limiting": limiting, "reduction_units_to_next_better_tier": 18944}
    SG = {"name": "amdgpu.sgpr", "kind": "unconstrained_resource", "allocation_scope": "subgroup", "independent_tier": 16, "limiting": False}

    def slots(tier):
        return {"name": "amdgpu.workgroup_slots", "kind": "fixed_limit", "independent_tier": tier, "limiting": False}

    with tempfile.TemporaryDirectory() as t:
        js = report(os.path.join(t, "fl.json"), 384, 6, [SG, vgpr(240, 6), lds(61952, 6), slots(15)])
        lim, tier, vg, lims, occ = lc.report_limit(js)
        ok(lim == "vgpr+lds" and tier == 6 and vg == 240 and [(x["name"], x["reduction"]) for x in lims] == [("vgpr", 24), ("lds", 18944)],
           "compile report: both limiting resources (flash2: VGPR and LDS)", (lim, tier, vg, lims))
        o = lc.occupancy(occ, 4)
        ok(o.get("next") == 9 and o.get("q") == 3 and o.get("report_tier") == 6 and o.get("needs") == [("vgpr", 240, "registers", 168, 72), ("lds", 61952, "bytes", 43008, 18944)],
           "384-lane workgroup (3 waves per SIMD): next tier 9, VGPR <= 168 and LDS <= 43,008 B, not the report's tier 7 / -24", o)
        o = lc.occupancy(lc.report_limit(report(os.path.join(t, "g.json"), 256, 4, [SG, vgpr(256, 5, False), lds(55328, 4), slots(16)]))[4], 4)
        ok(o.get("next") == 6 and o.get("needs") == [("vgpr", 256, "registers", 240, 16), ("lds", 55328, "bytes", 43008, 12320)],
           "256-lane GEMM: next tier 6 needs the LDS cut AND VGPR <= 240 (VGPR not limiting at 4)", o)
        o = lc.occupancy(lc.report_limit(report(os.path.join(t, "p.json"), 384, 5, [SG, vgpr(252, 5), lds(8192, 16, False)]))[4], 4)
        ok(o.get("tier") == 3 and o.get("report_tier") == 5 and o.get("next") == 6 and o.get("needs") == [("vgpr", 252, "registers", 240, 12)],
           "the report's unrounded tier (flash2_partial: VGPR 5 at 3 waves per workgroup) is really 3; next 6 needs VGPR <= 240", o)
        o = lc.occupancy(lc.report_limit(report(os.path.join(t, "m.json"), 160, 12, [SG, vgpr(114, 12)]))[4], 4)
        ok(o.get("tier") == 12 and o.get("bound") is True and o.get("next") == 15, "160 lanes (5 waves over 4 SIMDs): verified at "
           "ceil(9 x 5 / 4) = 12, a bound (Loom's convention), next 15 at VGPR <= 96 (review r1 F1/F2: was 'unverified')", o)
        o = lc.occupancy(lc.report_limit(report(os.path.join(t, "n.json"), None, 6, [SG, vgpr(240, 6)]))[4], 4)
        ok("tier unverified" in o.get("why", ""), "no workgroup size in the report: tier unverified", o)
        o = lc.occupancy(lc.report_limit(report(os.path.join(t, "x.json"), 384, 6, [SG, vgpr(240, 6), lds(61952, 6)]))[4], None)
        ok("tier unverified" in o.get("why", ""), "device file without SIMDs per occupancy domain: tier unverified", o)
        o = lc.occupancy(lc.report_limit(report(os.path.join(t, "d.json"), 384, 7, [SG, vgpr(240, 6), lds(61952, 6)]))[4], 4)
        ok("tier unverified" in o.get("why", "") and "report" in o["why"], "computed tier disagrees with the report: unverified", o)
        o = lc.occupancy(lc.report_limit(report(os.path.join(t, "s.json"), 384, 6, [SG, vgpr(240, 6), lds(61952, 6), slots(7)]))[4], 4)
        ok("workgroup_slots" in o.get("why", "") and o["why"].startswith("next tier") and "blocked" in o["why"] and o.get("tier") == 6,
           "a fixed limit below the next tier blocks it (named; 'next tier ... blocked' is loom_suggest's phantom test)", o)
        best = lambda o: o is None                     # the contract loom_suggest (C6) relies on: None at the best tier
        ok(best(lc.occupancy(lc.report_limit(report(os.path.join(t, "b.json"), 128, 16, [SG, vgpr(24, 16, False)]))[4], 4)),
           "at the wave cap: no occupancy lever", "")
        ok(best(lc.occupancy(lc.report_limit(report(os.path.join(t, "z.json"), 256, 16, [SG, vgpr(24, 16, False), lds(0, 16, False)]))[4], 4)),
           "a kernel using no LDS (0 B) is at the best tier, not 'unverified' (live: every g_deq16_*)", "")
        ok(lc.occupancy(None, 4) == {"why": "no compile report"}, "no compile report: said, not dropped", "")
        json.dump({"target_resources": {"limiting_resource": "amdgpu.lds", "residency": {"current_tier": 4}}}, open(js, "w"))
        ok(lc.report_limit(js)[0] == "lds" and lc.report_limit(js)[3] == [], "no residency rows: the summary's one limit, no rows",
           lc.report_limit(js))

        # lanes: decode weights 2.4 GB at 240 GB/s = 10 ms/step x 2 steps = 20 ms floor; prefill 2 x 1e9 params x 4096 tokens at
        # 8.192 TFLOP/s = 1.0 s; a lane without a floor is judged against 0 (ceiling.py)
        d = {"model": {"decode_weight_bytes": 2.4e9, "linear_params": 1e9}, "prefill_marker": "^g_fl",
             "lanes": [{"lane": "dec w", "phase": "decode", "regex": "^g_mv", "floor": "decode_weights"},
                       {"lane": "dec other", "phase": "decode", "regex": ".", "floor": None},
                       {"lane": "pre gemm", "phase": "prefill", "regex": "^g_gemm", "floor": "prefill_linear", "note": "design floor; ideal 0"},
                       {"lane": "pre attn", "phase": "prefill", "regex": "^g_fl", "floor": "prefill_attn"},
                       {"lane": "pre other", "phase": "prefill", "regex": ".", "floor": None}]}
        hw = {"bw_practical_gbs": 240, "wmma_f16_tflops": 8.192}
        srv = {("g_mv_a", "decode"): {"ms": 30.0, "lanes": {"dec w": 30.0}}, ("g_mv_b", "decode"): {"ms": 20.0, "lanes": {"dec w": 20.0}},
               ("g_mv_c", "decode"): {"ms": 10.0, "lanes": {"dec w": 10.0}}, ("g_x", "decode"): {"ms": 4.0, "lanes": {"dec other": 4.0}},
               ("g_gemm_a", "prefill"): {"ms": 1500.0, "lanes": {"pre gemm": 1500.0}},
               ("g_fl", "prefill"): {"ms": 100.0, "lanes": {"pre attn": 100.0}}}
        heads = lc.lane_headroom(srv, 2, 4096, d, hw)
        h = {k: (round(v["floor_ms"], 6) if v["floor_ms"] is not None else None, round(v["headroom_ms"], 6) if v["headroom_ms"] is not None
                 else None) for k, v in heads.items()}
        ok(h["dec w"] == (20.0, 40.0) and h["pre gemm"] == (1000.0, 500.0) and h["dec other"] == (0.0, 4.0),
           "lane headroom: served - floor (decode floor x verify steps; no floor = 0)", h)
        ok(h["pre attn"] == (None, None) and "attn_layers" in heads["pre attn"]["why"], "a floor missing a model constant: unknown, named",
           heads["pre attn"]["why"])
        ok(lc.lane_headroom(srv, 0, 4096, d, hw)["dec w"]["headroom_ms"] is None, "no verify steps: decode headroom unknown, not 0", "")
        ok(lc.lane_headroom({("g_mv_a", "decode"): {"ms": 15.0, "lanes": {"dec w": 15.0}}}, 2, 4096, d, hw)["dec w"]["headroom_ms"] == 0.0,
           "served below its floor: headroom 0", "")
        dec_only = {k: v for k, v in srv.items() if k[1] == "decode"}
        hd = lc.lane_headroom(dec_only, 2, 4096, d, hw)
        ok(hd["pre gemm"]["floor_ms"] is None and hd["pre gemm"]["headroom_ms"] is None and "not served" in hd["pre gemm"]["why"]
           and not hd["pre gemm"]["floored"], "an unserved phase: 'not served', no floor or headroom (was floor 1000, headroom 0)", hd["pre gemm"])

        # levers: dec w = 60 ms over a 20 ms floor, 40 ms headroom. g_mv_a 30/60 -> attribution 20 ms; a backed 'copies' (issue 40 %)
        # prices 30 x 0.4 = 12 ms and bounds it. g_mv_c 10/60 -> 6.667 ms; its 'copies' is NOT backed (issue 10 %, price 1 ms):
        # the price must not cut the attribution (vsplit: 697 -> 293 ms on an unbacked pattern). g_mv_b 'join': no counters.
        codes = os.path.join(t, "codes"); os.makedirs(codes)
        open(os.path.join(codes, "g_mv_a.hsaco"), "wb").write(b"a")
        open(os.path.join(codes, "g_mv_c.hsaco"), "wb").write(b"c")
        hs = os.path.join(t, "a.hsaco"); open(hs, "wb").write(b"a")
        hc = os.path.join(t, "c.hsaco"); open(hc, "wb").write(b"c")
        wl = lc.load_workloads([workload(os.path.join(t, "wa"), hs, kernel="g_mv_a", wait={"issue": 40, "vmem": 30, "lgkm": 5}),
                                workload(os.path.join(t, "wc"), hc, kernel="g_mv_c", wait={"issue": 10})], codes)
        lint = {"g_mv_a": {"flags": ["copies", "join"]}, "g_mv_b": {"flags": ["join"], "notes": ["maskedv"]}, "g_mv_c": {"flags": ["copies"]},
                "g_x": {}, "g_gemm_a": {}, "g_fl": {"flags": ["copies"]}}
        lv = lc.levers(lint, srv, wl, 240, heads)
        UN = "unnamed (no lint flag)"
        got = {(x["kernel"], x["lever"]): (x["evidence"], x["bound_ms"] if x["bound_ms"] is None else round(x["bound_ms"], 3), x["bound_kind"])
               for x in lv}
        ok(got[("g_mv_a", "copies")] == ("counter-backed", 12.0, "price"), "backed lever: min(attribution 20, price 12) = 12 ms", got)
        ok(got[("g_mv_c", "copies")] == ("counters: not backed", 6.667, "attribution"),
           "an unbacked price (1 ms) does not cut the attribution (6.667 ms); its evidence label is kept", got)
        ok(got[("g_mv_b", "join")] == ("static only", 13.333, "attribution"), "static-only lever: its attribution", got)
        ok(got[("g_x", UN)] == ("no evidence", 4.0, "attribution"), "a kernel without a flag still appears (no evidence)", got)
        ok(got[("g_gemm_a", UN)] == ("no evidence", 500.0, "attribution"), "the whole lane's headroom on its one kernel", got)
        ok(got[("g_fl", "copies")] == ("static only", None, None), "headroom unknown and no price: no bound", got)
        ok(not any(x["lever"] == "maskedv" for x in lv), "unpriced rules (maskedv) are not speed levers", "")
        pc = {(x["kernel"], x["lever"]): (x["phase"], round(x["pct"], 3) if x["pct"] is not None else None, x["floored"]) for x in lv}
        ok(pc[("g_mv_a", "copies")] == ("decode", 18.75, True) and pc[("g_gemm_a", UN)] == ("prefill", 31.25, True),
           "pct = bound / the phase's served ms (decode 64 ms, prefill 1600 ms)", pc)
        ok(pc[("g_x", UN)][2] is False and pc[("g_fl", "copies")][2] is False, "no-floor and unknown-floor lanes are not floor-backed", pc)

        # amdahl: one phase per lane (wall = its served ms), the lane's noise floor as its screen, one overlap group per kernel
        noise = {"dec w": 50.0, "dec other": 1.0}
        doc, picked = lc.amdahl_input(lv, heads, noise, 10)
        ph = {p["phase"]: p for p in doc["phases"]}
        ok(ph["dec w"]["screen_pct"] == 50.0 and ph["dec w"]["wall"] == 60.0 and "screen_pct" not in ph["pre gemm"],
           "screen = the lane's noise floor; a lane without one gets amdahl's default", ph["dec w"])
        grp = [c["overlap_group"] for c in ph["dec w"]["candidates"] if c["name"].startswith("g_mv_a")]
        ok(len(grp) == 2 and set(grp) == {"g_mv_a"}, "two levers on one kernel share its overlap group", grp)
        ok(not any(x["kernel"] == "g_x" for x in picked) and "dec other" not in ph,
           "a no-floor lane's levers never reach amdahl (whole kernel time is not a floor-backed bound)", [x["kernel"] for x in picked])
        one = lc.amdahl_input(lv, heads, noise, 1)[1]
        ok(sorted((x["phase"], x["kernel"]) for x in one) == [("decode", "g_mv_b"), ("prefill", "g_gemm_a")],
           "--top applies per phase: decode's best lever is screened beside prefill's 500 ms", [(x["phase"], x["kernel"]) for x in one])
        path = os.path.join(t, "amdahl_in.json")
        out, verdicts = lc.run_amdahl(doc, path)
        ok(verdicts.get(("pre gemm", "g_gemm_a:" + UN)) == (True, 1.0), "500 of 1500 ms clears amdahl's default 1 % screen", verdicts)
        ok(verdicts.get(("dec w", "g_mv_a:copies")) == (False, 50.0), "12 of 60 ms is below a 50 % screen", verdicts)
        ok(any(x.startswith("## dec w: wall 60.00") for x in out), "amdahl's own table is in the report", out[:3])
        bad = {"phases": [{"phase": "x", "wall": 10, "candidates": [{"name": "a", "target": 5, "reduction": 1}, {"name": "a", "target": 5, "reduction": 1}]}]}
        out2, v2 = lc.run_amdahl(bad, path)   # same path: the previous run's result file is still there
        ok(v2 == {} and out2[-1] == "amdahl: INVALID", "amdahl INVALID: no verdicts, not the previous run's stale result", (out2[-1:], v2))

        def sec(verd, nz, lt=None):
            return "\n".join(lc.section_d(lv, heads, verd, nz, lt or lint, srv, wl, 10, "fixture hw", 4))

        def row(txt, kernel, lever, block=None):
            part = txt.split(block)[1] if block else txt
            return [x for x in part.splitlines() if f" {kernel} " in x and f" {lever} " in x]
        txt = sec(verdicts, noise)
        ok("not worth building" in row(txt, "g_mv_a", "copies")[0], "a backed price below its screen: not worth building", row(txt, "g_mv_a", "copies"))
        rb = row(txt, "g_mv_b", "join")[0]
        ok("attribution below screen (no per-kernel floor)" in rb and "not worth building" not in rb,
           "an attribution below the screen is not called 'not worth building'", rb)
        rc = row(txt, "g_mv_c", "copies")[0]
        ok("counters: not backed" in rc and "clears" not in rc, "an unbacked pattern never 'clears' on its price", rc)
        rg = row(txt, "g_gemm_a", UN)[0]
        ok("attribution clears default 1 %" in rg, "the screen cell names the bound and the screen it used (default here)", rg)
        ok("noise_floor.json" in txt and "provisional" in txt, "with --noise-floor: the provisional line is printed", "")
        txt0 = sec({k: (p, 1.0) for k, (p, _) in verdicts.items()}, {})
        ok("noise_floor.json" not in txt0 and "amdahl's default" in txt0 and " noise " not in txt0.split("(d) decode")[1].split("(d2)")[0],
           "without --noise-floor: the header says amdahl's default and no row cites a noise floor", "")
        txtv = sec({}, noise)
        ra = row(txtv, "g_mv_a", "copies")[0]
        ok("amdahl gave no verdict" in ra and "clears" not in ra, "a floor-backed lever without a verdict says so", ra)
        rf = row(txt, "g_fl", "copies", "floor unknown (see the lane table)")
        ok(rf and "floor unknown: not screened" in rf[0] and not row(txt.split("floor unknown (see the lane table)")[0], "g_fl", "copies"),
           "an unknown-floor lever is listed in its own block, not under 'no floor'", rf)
        ok("pre gemm" in txt and "[design floor; ideal 0]" in [x for x in txt.splitlines() if x.startswith("pre gemm")][0],
           "the lane table shows the lane's floor kind from lanes.json", "")
        firsts = [ln.split()[:2] for ln in txt.splitlines() if ln[:3].strip() == "1" and len(ln.split()) > 2]
        ok(txt.index("(d) decode") < txt.index("(d) prefill") and len(firsts) == 2,
           "two ranked sections, decode first, each with its own rank 1 (user 11-25: 'two separate ranks')", firsts)
        pre = txt.split("(d) prefill")[1].split("(d2)")[0]
        ok([ln for ln in pre.splitlines() if ln[:3].strip() == "1" and "g_gemm_a" in ln], "prefill's rank 1 is its own best lever", pre)
        dec = txt.split("(d) decode")[1].split("(d) prefill")[0]
        rk, nofl = dec.split("no floor: bound = whole kernel time")
        gx = [x for x in nofl.splitlines() if "g_x" in x]
        ok("g_mv_a" in rk and "g_x" not in rk and gx and "clears" not in gx[0] and "not screened" in gx[0],
           "decode ranked block, then the no-floor block; a no-floor lever is never marked as clearing the screen", (rk, gx))
        hd_txt = "\n".join(lc.section_d(lc.levers(lint, dec_only, wl, 240, hd), hd, {}, noise, lint, dec_only, wl, 10, "fixture hw", 4))
        ok("(d) prefill: not served" in hd_txt, "an unserved phase prints 'not served'", "")

        # (d2): whole-workgroup tiers, every served kernel accounted for
        lint["g_fl"]["_occ"] = lc.report_limit(report(os.path.join(t, "fl2.json"), 384, 6, [SG, vgpr(240, 6), lds(61952, 6), slots(15)]))[4]
        lint["g_gemm_a"]["_occ"] = lc.report_limit(report(os.path.join(t, "b2.json"), 128, 16, [SG, vgpr(24, 16, False)]))[4]
        lint["g_mv_a"]["_occ"] = lint["g_fl"]["_occ"]
        txt = sec(verdicts, noise).split("(d2)")[1]
        occ_ = [x for x in txt.splitlines() if x.startswith("g_fl ")]
        ok(occ_ and "tier 6 -> 9" in occ_[0] and "vgpr 240 -> <= 168 registers (-72)" in occ_[0] and "lds 61952 -> <= 43008 bytes (-18944)" in occ_[0]
           and "needs all 2" in occ_[0], "d2: the reachable next tier and every resource that must move", occ_)
        ok(not [x for x in txt.splitlines() if x.startswith("g_gemm_a ")], "d2: a kernel at the wave cap has no lever", "")
        nb = [x for x in txt.splitlines() if x.startswith("g_mv_b ")]
        ok(nb and "no compile report" in nb[0], "d2: a served kernel without a compile report is listed as such, not dropped", nb)
        # (a) and (d2) print one occupancy (review r1 F1: (a) said 11.25 and (d2) 'unverified' for g_mv_q5_k_k5120_t8_q8, 160 lanes)
        cd = os.path.join(t, "agree"); os.makedirs(cd)
        report(os.path.join(cd, "g_mv_q5.json"), 160, 12, [SG, vgpr(114, 12), lds(2560, 16, False), slots(15)])
        la = {"g_mv_q5": {"kernel": "g_mv_q5", "flags": []}}
        lc.attach_reports(la, cd, 4)
        sq = {("g_mv_q5", "decode"): {"ms": 10.0, "calls": 1, "shapes": {8}, "lanes": {"dec w": 10.0}}}
        ra = [x for x in lc.build(la, sq, 10.0, [], 240)[0] if x.startswith("g_mv_q5 ")]
        rd = [x for x in lc.section_d(lc.levers(la, sq, [], 240, hd), hd, {}, noise, la, sq, [], 10, "fixture hw", 4)
              if x.startswith("g_mv_q5 ")]
        ok(ra and rd and " 12 bound " in ra[0] and "tier 12 bound -> 15 bound" in rd[0],
           "(a) and (d2) agree on a 160-lane matvec: 12 bound, next 15", (ra, rd))
        # the --device file's SIMD count reaches the tiers (verify r2 V8: attach_reports used 4 whatever the device said): a 64-lane
        # workgroup with 16 KiB LDS seats 8 workgroups = 16 waves; over 2 SIMDs that is tier 8, over 4 it would be 4
        dv = os.path.join(t, "fake2.json")
        json.dump({"simds_per_occupancy_domain": 2}, open(dv, "w"))
        report(os.path.join(cd, "g_two.json"), 64, 8, [SG, vgpr(48, 16, False), lds(16384, 8), slots(16)])
        l2 = {"g_two": {"kernel": "g_two", "flags": []}, "g_none": {"kernel": "g_none", "flags": []}}
        lc.attach_reports(l2, cd, json.load(open(dv)).get("simds_per_occupancy_domain"))
        ok(l2["g_two"]["occ"].get("tier") == 8 and not l2["g_two"]["occ"].get("bound"),
           "a device file with 2 SIMDs per occupancy domain: tier 8, not 4 (the --device count is used)", l2["g_two"]["occ"])
        sn = {("g_none", "decode"): {"ms": 1.0, "calls": 1, "shapes": {8}, "lanes": {"dec w": 1.0}}}
        rn = [x for x in lc.build(l2, sn, 1.0, [], 240)[0] if x.startswith("g_none ")]
        ok(rn and rn[0].split()[7] == "-", "(a): a kernel without a compile report prints '-' (no report), not '?' (unverified)", rn)
        ok("placement caveat" in sec(verdicts, noise).split("(d2)")[1], "(d2)'s header says what 'bound' means there", "")
        ma = [x for x in txt.splitlines() if x.startswith("g_mv_a ")]
        ok(ma and "35.0 %" in ma[0], "occupancy lever shows the latency-wait share (vmem 30 + lgkm 5)", ma)
        wps = lc.load_workloads([workload(os.path.join(t, "wps"), hs, kernel="g_mv_a", level="profile_standard",
                                          wait={"vmem": 30, "lgkm": 5})], codes)
        ok(lc.latency_wait(wps, "g_mv_a").startswith("unknown"), "load waits at profile_standard are not the served latency (q171: 16 vs 75 %)",
           lc.latency_wait(wps, "g_mv_a"))

        # main(): the wiring (--noise-floor, --amdahl-out, --top) on a served fixture run, no code objects
        run = os.path.join(t, "run"); os.makedirs(run)
        with open(os.path.join(run, "events.jsonl"), "w") as f:
            for k, s_, wg, ns in (("g_token_info", 1, [4096, 1, 1], 1000), ("g_attention_flash2", 1, [64, 2, 1], 5_000_000),
                                  ("g_k", 1, [8, 1, 1], 1_000_000), ("g_token_info", 2, [8, 1, 1], 1000), ("g_k", 2, [8, 1, 1], 2_000_000)):
                f.write(json.dumps({"type": "dispatch_event", "key": k, "submission_id": s_, "workgroup_count": wg, "duration_ns": ns}) + "\n")
        open(os.path.join(run, "run.req.json"), "w").write(json.dumps({"requests": [{"prompt_tokens": 4096}]}) + "\n")
        ml = os.path.join(t, "ml.json")
        json.dump({"model": {"attn_layers": 1, "kv_bytes_per_pos_layer": 1000, "linear_params": 1e6}, "prefill_marker": "^g_attention_flash2",
                   "lanes": [{"lane": "dk", "phase": "decode", "regex": ".", "floor": "decode_kv"},
                             {"lane": "pl", "phase": "prefill", "regex": ".", "floor": "prefill_linear"}]}, open(ml, "w"))
        nf2 = os.path.join(t, "nf2.json"); json.dump({"dk": 0.5}, open(nf2, "w"))
        emp = os.path.join(t, "empty_codes"); os.makedirs(emp)
        am = os.path.join(t, "am.json")
        cmd = [sys.executable, os.path.join(H, "lint_counters.py"), "--codes", emp, "--events", run, "--lanes", ml]
        import subprocess
        p = subprocess.run(cmd + ["--noise-floor", nf2, "--amdahl-out", am], capture_output=True, text=True)
        phs = {x["phase"]: x for x in json.load(open(am))["phases"]} if os.path.exists(am) else {}
        ok(p.returncode == 0 and phs.get("dk", {}).get("screen_pct") == 0.5 and "(d) decode:" in p.stdout and "(d3)" in p.stdout,
           "main(): --noise-floor reaches amdahl's input, --amdahl-out is written, both phases ranked", (p.returncode, p.stderr[-300:], phs))
        p = subprocess.run(cmd + ["--top", "0"], capture_output=True, text=True)
        ok(p.returncode == 2 and "--top" in p.stderr, "main(): --top 0 is refused", (p.returncode, p.stderr[-200:]))

    print(f"{'FAILED ' + str(fails) if fails else 'all ok'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
