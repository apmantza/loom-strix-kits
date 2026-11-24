#!/usr/bin/env python3
"""common/counter_saturation.py on fixtures in both formats it reads (rocprof-compute's results_pmc_perf_*.csv.gz, counters.sh's
*_counter_collection.csv). Recurrence it prevents: a flash2 launch of 4,096 tokens saturated SQ_WAIT_ANY, SQ_WAIT_INST_ANY and
SQ_WAVE_CYCLES at 20 x 2^32 and rocprof-compute read Wave Dependency / Issue Wait 100 % with no warning (HRX-like E425); the values
below are that run's (q171b flash2, rocprof-compute writes 6 significant digits). Near misses that must stay clean: a healthy cycle
counter 0.095 % off 3 x 2^32 (q171b's GL1C_CYCLE_sum: a 0.1 % tolerance flagged it), a lone counter at exactly 2^31 (a multiple,
below 2^32), the same counter under two names (SQ_INSTS_INTERNAL and SQ_INSTS_INTERNAL_sum, both 0.85 x 2^32), and a derived
metric (MemReadBytes) at exactly 2^32 bytes, two cycle counters equal below 2^31 (GPU and SPI busy every cycle)."""
import gzip, os, subprocess, sys, tempfile

H = os.path.dirname(os.path.abspath(__file__))
RCH = "GPU_ID,Dispatch_ID,Kernel_Name,Start_Timestamp,End_Timestamp,Counter_Name,Counter_Value\n"
CSH = "Correlation_Id,Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value,Start_Timestamp,End_Timestamp\n"


def rc_file(path, counters):
    with gzip.open(path, "wt") as f:
        f.write(RCH + "".join(f"1,{d},g_attention_flash2_f16o,0,1,{c},{v}\n" for d in (1, 2) for c, v in counters))


def run(*paths):
    r = subprocess.run([sys.executable, f"{H}/counter_saturation.py", *paths], capture_output=True, text=True)
    return r.returncode, r.stdout.strip().split("\n")[-1]


def main():
    fails = 0

    def ok(cond, what, got):
        nonlocal fails
        print(("ok  " if cond else "FAIL") + f" {what}: {got}")
        fails += not cond

    with tempfile.TemporaryDirectory() as t:
        inc = f"{t}/incident"; os.makedirs(inc)   # q171b flash2: pass 0 and pass 1 as rocprof-compute wrote them
        rc_file(f"{inc}/results_pmc_perf_0.csv.gz", [("SQ_INSTS_ALL_sum", 31431600000.0), ("GL1C_CYCLE_sum", 12872700000.0),
                                                    ("SQ_INSTS_INTERNAL_sum", 3631630000.0)])
        rc_file(f"{inc}/results_pmc_perf_1.csv.gz", [("SQ_WAIT_ANY", 85899400000.0), ("SQ_WAIT_INST_ANY", 85899400000.0),
                                                    ("SQ_WAVE_CYCLES", 85899400000.0), ("TCP_REQ_READ_sum", 2969420000.0),
                                                    ("SQ_INSTS_INTERNAL", 3631632000.0)])
        code, last = run(inc)
        ok(code == 1 and last == "SATURATION: FOUND SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAVE_CYCLES", "incident: the three SQ counters, exit 1", last)

        near = f"{t}/near"; os.makedirs(near)
        rc_file(f"{near}/results_pmc_perf_0.csv.gz", [("GL1C_CYCLE_sum", 12872700000.0), ("SQ_WAVES", 2.0 ** 31),
                                                     ("SQ_INSTS_INTERNAL_sum", 3631630000.0), ("SQ_INSTS_INTERNAL", 3631632000.0),
                                                     ("GRBM_GUI_ACTIVE", 5e8), ("GRBM_SPI_BUSY", 5e8)])
        code, last = run(near)
        ok(code == 0 and last.startswith("SATURATION: NONE"), "near misses stay clean, exit 0", last)

        mult = f"{t}/mult"; os.makedirs(mult)   # one counter alone at 2 x 2^32: the multiple check alone must flag it
        rc_file(f"{mult}/results_pmc_perf_0.csv.gz", [("SQ_BUSY_CYCLES_sum", 2.0 * 2 ** 32), ("TCP_REQ_sum", 2975760000.0)])
        code, last = run(mult)
        ok(code == 1 and last == "SATURATION: FOUND SQ_BUSY_CYCLES_sum", "lone counter at 2 x 2^32", last)

        grp = f"{t}/grp"; os.makedirs(grp)   # two different counters, equal, > 2^31, not a multiple: the group check alone must flag
        rc_file(f"{grp}/results_pmc_perf_0.csv.gz", [("SQ_WAIT_ANY", 3.1e9), ("SQ_WAVE_CYCLES", 3.1e9)])
        code, last = run(grp)
        ok(code == 1 and last == "SATURATION: FOUND SQ_WAIT_ANY SQ_WAVE_CYCLES", "identical pair > 2^31", last)

        csh = f"{t}/csh/waits"; os.makedirs(csh)   # counters.sh layout, full precision, one instance at 2^32 - 1; a derived metric at 2^32
        open(f"{csh}/waits_counter_collection.csv", "w").write(CSH + "".join(f"1,1,k,{c},{v},0,1\n" for c, v in (
            ("SQ_WAIT_CNT_VMVS", 4294967295), ("SQ_BUSY_CYCLES", 794285851), ("MemReadBytes", 4294967296), ("WaitAnyPct", 99.3))))
        code, last = run(f"{t}/csh")
        ok(code == 1 and last == "SATURATION: FOUND SQ_WAIT_CNT_VMVS", "counters.sh layout: instance at 2^32 - 1, derived metric skipped", last)

        code, last = run(f"{t}/empty-does-not-exist")
        ok(code == 2 and "ERROR" in last, "no counter rows: exit 2", last)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
