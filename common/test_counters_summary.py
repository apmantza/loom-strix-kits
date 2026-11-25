#!/usr/bin/env python3
"""common/counters_summary.py on CSVs in both spellings rocprofv3 10.2 writes (rocprofv3's Kernel_Name / Counter_Name, the rocpd
exporter's kernel_name / counter_name). Recurrences: a header spelling change silently empties the summary; a counter reported
as several rows per dispatch (one per dimension) averaged instead of summed; the calibrate ratio read off the wrong kernel."""
import os, subprocess, sys, tempfile

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import counters_summary as cs  # noqa: E402

TITLE = "Correlation_Id,Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value,Start_Timestamp,End_Timestamp\n"
SNAKE = "guid,Correlation_Id,dispatch_id,kernel_name,counter_name,Counter_Value,Start_Timestamp,End_Timestamp\n"


def write(d, header, lines):
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "x_counter_collection.csv"), "w").write(header + "".join(l + "\n" for l in lines))


def main():
    fails = 0
    with tempfile.TemporaryDirectory() as t:
        # two dispatches of k_read; MemReadBytes per dispatch 1 GiB split over two dimension rows; 4 ms and 6 ms
        write(f"{t}/a", TITLE, ["1,1,_Z6k_readPK,MemReadBytes,536870912,0,4000000", "1,1,_Z6k_readPK,MemReadBytes,536870912,0,4000000",
                               "2,2,_Z6k_readPK,MemReadBytes,1073741824,10,6000010", "3,3,_Z7k_write,MemReadBytes,0,0,1000"])
        write(f"{t}/b", SNAKE, ["g,1,1,_Z6k_readPK,MemReadBytes,536870912,0,4000000", "g,1,1,_Z6k_readPK,MemReadBytes,536870912,0,4000000",
                               "g,2,2,_Z6k_readPK,MemReadBytes,1073741824,10,6000010", "g,3,3,_Z7k_write,MemReadBytes,0,0,1000"])
        for d in ("a", "b"):
            s = cs.summarise(f"{t}/{d}")["_Z6k_readPK"]
            ok = s["n"] == 2 and s["MemReadBytes"] == 1073741824 and s["ns"] == 5e6
            print(("ok  " if ok else "FAIL"), d, s); fails += not ok
        r = subprocess.run([sys.executable, f"{H}/counters_summary.py", f"{t}/a", "--expect-bytes", "k_read=1073741824"], capture_output=True, text=True)
        ok = r.returncode == 0 and "= 1.000" in r.stdout and "k_write" not in r.stdout.split("calibrate", 1)[1]
        print(("ok  " if ok else "FAIL"), "calibrate ratio 1.000 on k_read only"); fails += not ok
        r = subprocess.run([sys.executable, f"{H}/counters_summary.py", f"{t}/a", "--expect-bytes", "nomatch=1"], capture_output=True, text=True)
        ok = r.returncode == 1
        print(("ok  " if ok else "FAIL"), "calibrate with no matching kernel exits 1"); fails += not ok
        # Recurrence HRX-like E497: GCEA_RDRAM_SIZE_REQ (then named DramReadBytes) read 6.3 GB in 18.4 ms = 345 GB/s on a grouped GEMM,
        # above the chip's DRAM read ceiling: the fabric requests include MALL hits. A byte rate above the device's ceiling must say so.
        write(f"{t}/c", TITLE, ["1,1,_Z4gemm,MemReadBytes,6340000000,0,18400000", "2,2,_Z4slow,MemReadBytes,3800000000,0,18700000"])
        r = subprocess.run([sys.executable, f"{H}/counters_summary.py", f"{t}/c"], capture_output=True, text=True)
        g, sl = r.stdout.split("_Z4slow")[0], r.stdout.split("_Z4slow")[1]
        ok = r.returncode == 0 and "344.6 GB/s" in g and "MALL" in g and "MALL" not in sl
        print(("ok  " if ok else "FAIL"), "a byte rate above the DRAM ceiling is flagged as MALL-inclusive, one below is not"); fails += not ok
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
