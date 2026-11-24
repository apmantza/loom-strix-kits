#!/usr/bin/env python3
"""Tests of env_block.py on a fake root (--root), a fake venv, ROCm SDK, package and kit repository. Each case names the
recurrence it prevents. usage: python3 test_env_block.py (exit 1 on a failure)"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "env_block.py")
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


def fake(t):
    r = os.path.join(t, "root")
    put(r, "proc/sys/kernel/osrelease", "7.0.0-38-generic\n")
    put(r, "proc/cmdline", "BOOT_IMAGE=/vmlinuz root=UUID=x ro quiet ttm.pages_limit=100 amd_iommu=off amdgpu.gttsize=126976 iommu=pt\n")
    # a connector entry sorts before the GPU and has a device/ without gpu_busy_percent (card1-DP-1 on a real machine)
    put(r, "sys/class/drm/card0-DP-1/device/vbios_version", "CONNECTOR\n")
    d = "sys/class/drm/card1/device/"
    put(r, d + "gpu_busy_percent", "0\n")
    put(r, d + "vbios_version", "113-STRXLGEN-001\n")
    put(r, d + "fw_version/smc_fw_version", "0x0a640700\n")
    put(r, d + "fw_version/mes_fw_version", "0x00000086\n")
    put(r, d + "power_dpm_force_performance_level", "auto\n")
    put(r, d + "pp_power_profile_mode", "PROFILE_INDEX(NAME)\n 0 BOOTUP_DEFAULT :\n 1 3D_FULL_SCREEN*:\n 2 POWER_SAVING  :\n")
    put(r, d + "hwmon/hwmon6/power1_cap", "85000000\n")
    put(r, "sys/firmware/acpi/platform_profile", "performance\n")
    for c, g in ((0, "powersave"), (1, "powersave"), (2, "performance")):
        put(r, f"sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor", g + "\n")
        put(r, f"sys/devices/system/cpu/cpu{c}/cpufreq/energy_performance_preference", "balance_performance\n")
    put(r, "var/lib/dpkg/status",
        "Package: libgl1-mesa-dri\nStatus: install ok installed\nVersion: 25.0.0-1\n\n"
        "Package: mesa-vulkan-drivers\nStatus: deinstall ok config-files\nVersion: 24.0.0-1\n\n"
        "Package: linux-firmware\nStatus: install ok installed\nVersion: 20260319.git217ca6e4\n\n"
        "Package: libdrm-amdgpu1\nStatus: install ok installed\nVersion: 2.4.131-1\n")
    sdk = os.path.join(t, "sdk")
    put(sdk, ".info/version", "10.2.0\n")
    venv = os.path.join(t, "venv")
    put(venv, "lib/python3.12/site-packages/hrx_loom-0.2.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: hrx-loom\nVersion: 0.2.0\n")
    put(venv, "lib/python3.12/site-packages/numpy-2.0.dist-info/METADATA", "Name: numpy\nVersion: 2.0\n")
    kit = os.path.join(t, "kit")
    put(kit, "a.txt", "a\n")
    g = ["git", "-C", kit, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", kit], check=True)
    subprocess.run(g + ["add", "a.txt"], check=True)
    subprocess.run(g + ["commit", "-qm", "x"], check=True)
    put(t, "w/pkg-a/model.json", '{"a": 1}\n')
    put(t, "w/pkg-a.cmd", "env KV8=1\n  kit/build.py pkg-a\n")   # a wrapped line must stay one block line
    put(t, "w2/pkg-a/model.json", '{"a": 2}\n')
    put(t, "w/pkg-a/program/llm.loom", "kernel A\n")   # E481: two packages with one model.json differ in their program
    put(t, "w2/pkg-a/program/llm.loom", "kernel B\n")   # another package with the same basename, in another directory
    dev = os.path.join(t, "gfx.json")
    put(t, "gfx.json", json.dumps({"default_profile": "cap85w", "profiles": {"cap85w": {}, "nominal": {}}}))
    return r, sdk, venv, kit, dev


def run(*args, env=None):
    p = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def block(out):   # a line that is not `key: value` (a value spilling onto a second line) is kept as its own key
    return dict(l.split(": ", 1) if ": " in l else ("?" + l, "") for l in out.splitlines())


with tempfile.TemporaryDirectory() as t:
    r, sdk, venv, kit, dev = fake(t)
    env = dict(os.environ, ROCM_SDK=sdk, VENV=venv, PYTHON=sys.executable, HOME=t)
    args = ["--root", r, "--kit", kit, "--pkg", f"{t}/w/pkg-a", "--pkg", f"{t}/w2/pkg-a", "--device", dev]
    rc, out, err = run(*args, env=env)
    b = block(out)
    case("block: exit 0, every line `key: value`", rc == 0 and len(b) == len(out.splitlines()))
    case("kernel from osrelease", b.get("kernel") == "7.0.0-38-generic")
    # the GPU boot parameters only (the UUID and quiet are not the GPU's environment), in boot order
    case("cmdline_gpu: amdgpu.* / ttm.* / iommu params kept, others dropped",
         b.get("cmdline_gpu") == "ttm.pages_limit=100 amd_iommu=off amdgpu.gttsize=126976 iommu=pt")
    # a firmware update is a different machine (#241's shape: a stack update under an unchanged package)
    case("firmware: the GPU card's vbios and fw_version files, not the connector's",
         b.get("gpu_vbios") == "113-STRXLGEN-001" and b.get("gpu_fw.smc") == "0x0a640700" and b.get("gpu_fw.mes") == "0x00000086")
    # a removed package whose config files remain is still in the dpkg database: it is not the installed Mesa
    case("mesa: an installed package only (mesa-vulkan-drivers is config-files: falls back to libgl1-mesa-dri)",
         b.get("mesa") == "25.0.0-1 (libgl1-mesa-dri)")
    case("linux_firmware / libdrm_amdgpu from dpkg", b.get("linux_firmware") == "20260319.git217ca6e4 (linux-firmware)"
         and b.get("libdrm_amdgpu") == "2.4.131-1 (libdrm-amdgpu1)")
    case("rocm from $ROCM_SDK/.info/version", b.get("rocm") == "10.2.0")
    case("loom: the venv's loom dist-info only", b.get("loom") == "hrx-loom 0.2.0")
    case("kit_commit: a short hash, clean", len(b.get("kit_commit", "")) >= 7 and not b["kit_commit"].endswith("-dirty"))
    case("pkg: model.json sha256 and the build line beside it",
         b.get("pkg.pkg-a.model_json_sha256") == hashlib.sha256(b'{"a": 1}\n').hexdigest()
         and b.get("pkg.pkg-a.cmd") == "env KV8=1 kit/build.py pkg-a")
    # an A/A bench of a package against a same-named copy elsewhere must not lose one package's line
    case("pkg: same basename in two directories stays two keys, a missing .cmd reads unknown",
         b.get("pkg.pkg-a#2.model_json_sha256") == hashlib.sha256(b'{"a": 2}\n').hexdigest()
         and b.get("pkg.pkg-a#2.cmd") == "unknown")
    case("pkg: program_sha256 tells packages with the same model.json apart (E481)",
         b.get("pkg.pkg-a.program_sha256", "unknown") != "unknown"
         and b.get("pkg.pkg-a#2.program_sha256", "unknown") != "unknown"
         and b.get("pkg.pkg-a.program_sha256") != b.get("pkg.pkg-a#2.program_sha256"))
    case("power: the '*' profile, the cap in W, the perf level, the platform profile",
         b.get("gpu_power_profile") == "3D_FULL_SCREEN" and b.get("gpu_power_cap_w") == "85" and b.get("gpu_perf_level") == "auto"
         and b.get("platform_profile") == "performance")
    case("device_file: the default profile and a hash prefix", b.get("device_file", "").startswith("gfx.json profile cap85w sha256:"))
    # one CPU on another governor is a different machine state; a first-CPU read would miss it
    case("cpu_governor: the set over all CPUs", b.get("cpu_governor") == "performance,powersave"
         and b.get("cpu_epp") == "balance_performance")
    case("python versions", b.get("python", "").count(".") == 2 and b.get("kit_python") == b.get("python"))
    rc2, out2, _ = run(*args, env=env)
    case("stable: a second read prints the same block (no timestamps)", rc2 == 0 and out2 == out)
    rc, out3, _ = run("--root", r, "--kit", kit, "--device", dev + ":nominal", env=env)
    case("device_file: an explicit profile", block(out3).get("device_file", "").startswith("gfx.json profile nominal "))
    rc, out3, _ = run("--root", r, "--kit", kit, "--device", dev + ":cap99w", env=env)
    case("device_file: a profile not in the file is named as such", "cap99w NOT IN FILE" in block(out3).get("device_file", ""))
    put(kit, "untracked.txt", "u\n")
    rc, out4, _ = run(*args, env=env)
    # the kit holds scratch files that are not part of any commit: they must not make every block '-dirty'
    case("kit_commit: an untracked file is not '-dirty'", block(out4).get("kit_commit") == b.get("kit_commit"))
    put(kit, "a.txt", "b\n")
    rc, out4, _ = run(*args, env=env)
    # a bench run from an edited kit is not the commit it names
    case("kit_commit: '-dirty' after a tracked edit", block(out4).get("kit_commit", "").endswith("-dirty"))
    with tempfile.TemporaryDirectory() as e:
        rc, out5, err5 = run("--root", e, "--kit", e, env=dict(os.environ, HOME=e, VENV="", ROCM_SDK="", PYTHON=""))
        b5 = block(out5)
        case("nothing readable: exit 0, every key unknown or none, no crash",
             rc == 0 and b5.get("kernel") == "unknown" and b5.get("gpu_vbios") == "unknown" and b5.get("mesa") == "unknown"
             and b5.get("rocm") == "unknown" and b5.get("loom") == "unknown" and b5.get("kit_commit") == "unknown"
             and b5.get("device_file") == "none" and b5.get("cpu_governor") == "unknown")
    # --diff
    A, B, C, E = (os.path.join(t, n) for n in ("A.txt", "B.txt", "C.txt", "E.txt"))
    put(t, "A.txt", out)
    put(t, "B.txt", out.replace("25.0.0-1 (libgl1-mesa-dri)", "25.1.0-1 (libgl1-mesa-dri)"))
    put(t, "C.txt", "\n".join(l for l in out.splitlines() if not l.startswith("rocm:")) + "\n")
    put(t, "E.txt", "\n")
    put(t, "G.txt", out + "garbage line\n")
    rc, o, _ = run("--diff", A, A)
    case("diff: identical blocks exit 0, print nothing", rc == 0 and o == "")
    rc, o, _ = run("--diff", A, B)
    # strix-llama.cpp #241: a Mesa update between two results must stop the comparison and name itself
    case("diff: a changed Mesa exits 1 and names the key with both values",
         rc == 1 and o.strip() == "mesa: 25.0.0-1 (libgl1-mesa-dri) -> 25.1.0-1 (libgl1-mesa-dri)")
    rc, o, _ = run("--diff", A, C)
    case("diff: a key missing from one block differs", rc == 1 and o.strip() == "rocm: 10.2.0 -> <absent>")
    rc, o, _ = run("--diff", C, A)
    case("diff: a key only in the second block differs", rc == 1 and o.strip() == "rocm: <absent> -> 10.2.0")
    rc, o, _ = run("--diff", A, os.path.join(t, "G.txt"))
    # a line that is not `key: value` (a traceback, a wrapped value) is part of the block, never skipped as a match
    case("diff: a line that is not key: value differs", rc == 1 and "garbage line" in o)
    # a collection that crashed leaves an empty file: two empty blocks must not read as a match
    rc, o, _ = run("--diff", E, E)
    case("diff: an empty block fails closed (exit 2)", rc == 2)
    rc, o, _ = run("--diff", A, os.path.join(t, "missing.txt"))
    case("diff: a missing block fails closed (exit 2)", rc == 2)

print(f"{len(fails)} failed")
sys.exit(1 if fails else 0)
