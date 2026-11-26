#!/usr/bin/env python3
"""Tests of gpu_telemetry.py on a fake sysfs tree (--root). Each case names the recurrence it prevents.
usage: python3 test_gpu_telemetry.py (exit 1 on a failure)"""
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "gpu_telemetry.py")
fails = []


def case(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def put(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def fake_sysfs(root, fclk_active=True, k10temp=True):
    # a connector entry sorts before the GPU and has a device/ without gpu_busy_percent (card1-DP-1 on a real machine)
    os.makedirs(os.path.join(root, "sys/class/drm/card0-DP-1/device/hwmon/hwmon9"), exist_ok=True)
    put(root, "sys/class/drm/card0-DP-1/device/hwmon/hwmon9/temp1_input", "99000\n")
    d = "sys/class/drm/card1/device"
    put(root, d + "/gpu_busy_percent", "100\n")
    put(root, d + "/hwmon/hwmon6/power1_average", "85000000\n")
    put(root, d + "/hwmon/hwmon6/freq1_input", "2150000000\n")
    put(root, d + "/hwmon/hwmon6/temp1_input", "54000\n")
    put(root, d + "/pp_dpm_fclk", "0: 800Mhz\n1: 1600Mhz" + (" *" if fclk_active else "") + "\n")
    put(root, d + "/pp_dpm_mclk", "0: 1000Mhz\n1: 4000Mhz *\n")
    put(root, d + "/pp_dpm_socclk", "0: 400Mhz *\n1: 1200Mhz\n")
    put(root, "sys/class/hwmon/hwmon0/name", "acpitz\n")
    put(root, "sys/class/hwmon/hwmon0/temp1_input", "30000\n")
    if k10temp:
        put(root, "sys/class/hwmon/hwmon3/name", "k10temp\n")
        put(root, "sys/class/hwmon/hwmon3/temp1_input", "71000\n")
    os.makedirs(os.path.join(root, "sys/class/hwmon/hwmon5"), exist_ok=True)   # a hwmon without a name file (seen on some drivers)
    put(root, "proc/stat", "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 50 0 50 400 0 0 0 0 0 0\ncpu1 50 0 50 400 0 0 0 0 0 0\nintr 1\n")
    put(root, "proc/loadavg", "9.40 3.31 4.37 2/1904 3302311\n")


def run(*args):
    p = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def recorded(root, seconds=0.45):
    out = os.path.join(root, "tel.tsv")
    p = subprocess.Popen([sys.executable, TOOL, "--root", root, "record", out])
    time.sleep(seconds)
    p.kill(); p.wait()
    return out, [l.rstrip("\n").split("\t") for l in open(out)]


with tempfile.TemporaryDirectory() as root:
    fake_sysfs(root)
    # the thermal start gate must read the GPU's edge sensor, not a connector's or the first hwmon in the system
    rc, out, _ = run("--root", root, "temp")
    case("temp: the card with gpu_busy_percent, its hwmon edge sensor (54 C, not the connector's 99)", rc == 0 and out.strip() == "54")
    # the columns agent_ab.sh and the summary read: t busy power clock temp fclk mclk socclk tctl
    tsv, rows = recorded(root)
    case("record: samples written while running (killed like agent_ab kills it)", len(rows) >= 2)
    r = rows[0]
    case("record: busy / power W / clock GHz / temp C scaled from sysfs units", r[1:5] == ["100", "85.0", "2.150", "54"])
    case("record: active fclk / mclk / socclk level and the k10temp Tctl (E368: the fabric clock, not the shader clock, moved)",
         r[5:9] == ["1600", "4000", "400", "71"])
    rc, out, _ = run("--root", root, "summary", tsv, "h1-pkg")
    case("summary: one line with the busy share and every column's median", rc == 0 and "telemetry h1-pkg:" in out
         and "busy >= 50 % in 100%" in out and "power 85.00" in out and "fclk 1600.00" in out and "cpu 71.00" in out)

with tempfile.TemporaryDirectory() as root:
    # HRX-like E624: the summary printed the shader clock with 2 decimals (2.18 / 2.19 GHz: 0.46 % steps) while the cycles table it
    # feeds is compared with a 0.27 % lane floor; a 0.2 % clock difference (2.180 vs 2.184) must survive the summary
    outs = []
    for ghz in ("2.180", "2.184"):
        p = os.path.join(root, ghz + ".tsv")
        open(p, "w").write("".join(f"{1000 + i:.2f}\t100\t85.0\t{ghz}\t80\t1600\t1000\t1104\t80\t0.5\t900\t0\n" for i in range(9)))
        outs.append(run("summary", p, "x")[1])
    case("summary: the busy clock at the recorded 1 MHz (3 decimals), not 2", "clock 2.180 (p10 2.180" in outs[0]
         and "clock 2.184 (p10 2.184" in outs[1])

with tempfile.TemporaryDirectory() as root:
    # E371: a pp_dpm table caught mid-transition has no '*' level; a 0 clock must not drag the p10 to 0
    fake_sysfs(root, fclk_active=False)
    tsv, rows = recorded(root)
    case("record: a table with no active level records 0", rows and rows[0][5] == "0")
    with open(tsv, "a") as f:
        f.write("\t".join([rows[0][0], "100", "85.0", "2.150", "54", "1600", "4000", "400", "71"]) + "\n")
    rc, out, _ = run("summary", tsv)
    case("summary: 0 (no active level) is dropped from the clock column, not counted", "fclk 1600.00 (p10 1600.00" in out)

with tempfile.TemporaryDirectory() as root:
    # a machine without the k10temp driver still records the GPU columns
    fake_sysfs(root, k10temp=False)
    tsv, rows = recorded(root)
    case("record without k10temp: GPU columns and clocks, Tctl empty, CPU columns still at 9-11 (fixed positions)",
         rows and rows[0][1:8] == ["100", "85.0", "2.150", "54", "1600", "4000", "400"] and rows[0][8] == "" and len(rows[0]) == 12
         and rows[0][9] == "9.40")
    rc, out, _ = run("summary", tsv)
    case("summary without Tctl: clocks shown, no 'cpu .. C' part", rc == 0 and "fclk 1600.00" in out and " C\n" not in out.split("fclk")[1].split("\n")[0] + "\n")

with tempfile.TemporaryDirectory() as root:
    # no GPU: a clear refusal (exit 2), never a traceback a bench mistakes for a crash in its own code
    os.makedirs(os.path.join(root, "sys/class/drm"))
    rc, _, err = run("--root", root, "temp")
    case("no card: exit 2 with a message", rc == 2 and "no " in err and "Traceback" not in err)
    rc, _, err = run("--root", root, "record", os.path.join(root, "x.tsv"))
    case("no card, record: exit 2 at once (a bench's telemetry must not silently record nothing)", rc == 2)
    empty = os.path.join(root, "empty.tsv")
    open(empty, "w").close()
    rc, out, _ = run("summary", empty, "h2")
    case("summary of no samples: says so, exit 1", rc == 1 and "no samples" in out)

# CPU beside the bench (E415): a builder's 7-core python during a reference half. The columns are deltas of /proc/stat; the
# server's own CPU (/proc/PID/stat utime + stime) is subtracted so the flag measures other work only.
sys.path.insert(0, HERE)
import gpu_telemetry as gt  # noqa: E402
prev = (0.0, 0, 1000, 32, 0)
cur = (1.0, 250, 2000, 32, 300)    # 25 % of 32 CPUs busy = 8 cores; the server used 300 jiffies at 100/s = 3 cores
case("cpu_columns: busy in % of one core, other = minus the server's own", gt.cpu_columns(prev, cur, 100) == ("800", "500"))
case("cpu_columns: no server reading = other is all CPU (stricter, never laxer)", gt.cpu_columns(prev[:4] + (None,), cur, 100) == ("800", "800"))
case("cpu_columns: no time passed = empty, not a division by zero", gt.cpu_columns(prev, prev, 100) == ("", ""))


def tsv_rows(path, others, base="1.0\t100\t85.0\t2.150\t90\t1600\t4000\t400\t70\t3.0\t"):
    with open(path, "w") as f:
        for o in others:
            f.write(base + f"{o + 100}\t{o}\n")


with tempfile.TemporaryDirectory() as root:
    t = os.path.join(root, "t.tsv")
    tsv_rows(t, [50] * 85 + [400] * 15)      # 15 % of samples above 2 cores of other CPU
    rc, out, _ = run("summary", t, "h1-ref")
    case("summary: other CPU > 2 cores in 15 % of samples -> CONTAMINATED", rc == 0 and "cpu h1-ref: CONTAMINATED" in out)
    tsv_rows(t, [50] * 95 + [400] * 5)       # a 5 % blip (a cron job) is below the 10 % share
    rc, out, _ = run("summary", t, "h1-ref")
    case("summary: a 5 % blip stays clean", "cpu h1-ref: clean" in out)
    rc, out, _ = run("summary", t, "h1-ref", "--cpu-max", "0.4")
    case("summary: --cpu-max sets the core threshold", "CONTAMINATED" in out and "> 0.4 cores" in out)
    with open(t, "w") as f:                  # a file recorded before the CPU columns existed (11-23 benches)
        f.write("1.0\t100\t85.0\t2.150\t90\t1600\t4000\t400\t70\n")
    rc, out, _ = run("summary", t, "old")
    case("summary of an old file: GPU line as before, CPU unknown (never 'clean')", rc == 0 and "cpu old: unknown" in out
         and "fclk 1600.00" in out)
    fake_sysfs(root)
    p = subprocess.Popen([sys.executable, TOOL, "--root", root, "record", t, "--pid", "4242"])
    time.sleep(0.25)
    put(root, "proc/stat", "cpu  500 0 100 1200 0 0 0 0 0 0\ncpu0 1 0 0 0\ncpu1 1 0 0 0\n")   # 400 busy of 800 jiffies on 2 CPUs
    time.sleep(0.25)
    p.kill(); p.wait()
    rows = [l.rstrip("\n").split("\t") for l in open(t)]
    case("record: first sample's CPU columns empty, a later one shows the /proc/stat delta (1 core = 100 %)",
         rows[0][10:] == ["", ""] and ["100", "100"] in [r[10:] for r in rows])

print(f"{len(fails)} failed" if fails else "all passed")
sys.exit(1 if fails else 0)
