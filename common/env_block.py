#!/usr/bin/env python3
"""The environment a bench or profile ran in, as a fixed block of `key: value` lines, so two results are compared only when
their environments match (a Mesa update turned a measured speedup into a loss elsewhere: strix-llama.cpp #241).
Keys, in this order (a value that cannot be read is `unknown`, a setting that is absent is `none`):
  kernel                      /proc/sys/kernel/osrelease (uname -r)
  cmdline_gpu                 the GPU-relevant boot parameters from /proc/cmdline (amdgpu.*, ttm.*, iommu*, amd_iommu*)
  gpu_vbios, gpu_fw.<block>   the GPU card's vbios_version and the firmware versions sysfs exposes (device/fw_version/*)
  linux_firmware, mesa, libdrm_amdgpu   package versions from the dpkg database (no dpkg: unknown)
  rocm                        $ROCM_SDK/.info/version (else /opt/rocm/.info/version)
  loom                        the *loom*.dist-info versions in $VENV (the venv loom-serve runs from)
  kit_commit                  git rev-parse --short HEAD of the kit repository, '-dirty' when a tracked file is modified
  pkg.<name>.model_json_sha256, pkg.<name>.cmd   per --pkg: its model.json's sha256 and its build line (<pkg>.cmd beside it)
  gpu_perf_level, gpu_power_profile, gpu_power_cap_w, platform_profile   the GPU / platform power settings sysfs exposes
  device_file                 --device FILE[:PROFILE]: its name, the profile (else its default_profile) and a sha256 prefix
  cpu_governor, cpu_epp       the set of scaling governors and energy-performance preferences over all CPUs
  python, kit_python          the interpreter running this tool, and $PYTHON's (the one the bench tools run under)
usage: env_block.py [--root DIR] [--kit DIR] [--pkg PKG ...] [--device FILE[:PROFILE]]    print the block (exit 0)
       env_block.py --diff A B    the keys whose values differ between two saved blocks (a key missing from one side differs);
                                  exit 0 when none, 1 when any, 2 when a block is unreadable or empty (fails closed)
--root: the directory that holds proc/, sys/ and var/lib/dpkg/ (default /; a test points it at a fake tree).
--kit: the kit repository (default: the one holding this file)."""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CMDLINE_PREFIXES = ("amdgpu.", "ttm.", "iommu", "amd_iommu")
DPKG = {"linux_firmware": ["linux-firmware"], "mesa": ["mesa-vulkan-drivers", "libgl1-mesa-dri", "mesa-libgallium"],
        "libdrm_amdgpu": ["libdrm-amdgpu1"]}


def text(p):
    with open(p) as f:
        return f.read().strip()


def one_line(v):
    v = " ".join(str(v).split())
    return v if v else "unknown"


def guard(f, *a):
    """f(*a), or 'unknown' when its source is missing or unreadable"""
    try:
        return f(*a)
    except (OSError, ValueError, IndexError, KeyError, subprocess.SubprocessError):
        return "unknown"


def cmdline_gpu(root):
    toks = [t for t in text(os.path.join(root, "proc/cmdline")).split() if t.startswith(CMDLINE_PREFIXES)]
    return " ".join(toks) or "none"


def gpu_card(root):
    sys.path.insert(0, HERE)
    import gpu_telemetry   # the same card the telemetry samples (the first with gpu_busy_percent, not a connector)
    try:
        return gpu_telemetry.card(root)
    except gpu_telemetry.NoGpu:
        return None


def gpu_firmware(dev):
    out = [("gpu_vbios", guard(text, os.path.join(dev, "vbios_version")) if dev else "unknown")]
    files = sorted(glob.glob(os.path.join(dev, "fw_version/*_fw_version"))) if dev else []
    out += [(f"gpu_fw.{os.path.basename(f)[:-len('_fw_version')]}", guard(text, f)) for f in files]
    if dev and not files:
        out.append(("gpu_fw", "unknown"))
    return out


def dpkg_versions(root):
    """{package: version} for the packages DPKG names, from the dpkg status file (installed ones only)"""
    want = {p for ps in DPKG.values() for p in ps}
    got, pkg, ver, ok = {}, None, None, False
    with open(os.path.join(root, "var/lib/dpkg/status"), errors="replace") as f:
        for line in list(f) + ["\n"]:
            if line.startswith("Package: "):
                pkg = line.split(None, 1)[1].strip()
            elif line.startswith("Version: "):
                ver = line.split(None, 1)[1].strip()
            elif line.startswith("Status: "):
                ok = line.split()[-1] == "installed"
            elif not line.strip():
                if pkg in want and ver and ok:
                    got[pkg] = ver
                pkg, ver, ok = None, None, False
    return got


def packages(root):
    try:
        got = dpkg_versions(root)
    except OSError:
        return [(k, "unknown") for k in DPKG]
    out = []
    for key, names in DPKG.items():
        hit = next((n for n in names if n in got), None)
        out.append((key, f"{got[hit]} ({hit})" if hit else "unknown"))
    return out


def rocm(root):
    sdk = os.environ.get("ROCM_SDK")
    for p in ([os.path.join(sdk, ".info/version")] if sdk else []) + [os.path.join(root, "opt/rocm/.info/version")]:
        if os.path.exists(p):
            return text(p)
    return "unknown"


def loom():
    venv = os.environ.get("VENV")
    if not venv:
        return "unknown"
    vs = []
    for d in sorted(glob.glob(os.path.join(venv, "lib/python*/site-packages/*loom*.dist-info"))):
        meta = {k: v for k, _, v in (l.partition(": ") for l in text(os.path.join(d, "METADATA")).splitlines()[:20])}
        vs.append(f"{meta['Name']} {meta['Version']}")
    return ", ".join(vs) or "unknown"


def git(kit, *args):
    return subprocess.run(["git", "-C", kit, *args], capture_output=True, text=True, check=True, timeout=30).stdout.strip()


def kit_commit(kit):
    head = git(kit, "rev-parse", "--short", "HEAD")
    return head + ("-dirty" if git(kit, "status", "--porcelain", "--untracked-files=no") else "")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def pkg_lines(pkgs):
    out, seen = [], set()
    for p in pkgs:
        p = os.path.realpath(p)
        name = os.path.basename(p)
        while name in seen:   # two packages with one basename in different directories stay two keys
            name += "#2"
        seen.add(name)
        out.append((f"pkg.{name}.model_json_sha256", guard(sha256, os.path.join(p, "model.json"))))
        out.append((f"pkg.{name}.cmd", guard(text, p + ".cmd")))
    return out


def power(root, dev):
    def profile(p):   # the active line of pp_power_profile_mode carries a '*'
        for line in text(p).splitlines():
            m = re.match(r"\s*\d+\s+([A-Za-z0-9_]+)\s*\*", line)
            if m:
                return m.group(1)
        return "unknown"

    def cap(dev):
        return f"{int(text(sorted(glob.glob(os.path.join(dev, 'hwmon/hwmon*/power1_cap')))[0])) / 1e6:g}"
    return [("gpu_perf_level", guard(text, os.path.join(dev, "power_dpm_force_performance_level")) if dev else "unknown"),
            ("gpu_power_profile", guard(profile, os.path.join(dev, "pp_power_profile_mode")) if dev else "unknown"),
            ("gpu_power_cap_w", guard(cap, dev) if dev else "unknown"),
            ("platform_profile", guard(text, os.path.join(root, "sys/firmware/acpi/platform_profile")))]


def device_file(spec):
    if not spec:
        return "none"
    path, _, prof = spec.partition(":")
    d = json.load(open(path))
    prof = prof or d.get("default_profile", "")
    if prof not in d.get("profiles", {}):
        return f"{os.path.basename(path)} profile {prof or '?'} NOT IN FILE"
    return f"{os.path.basename(path)} profile {prof} sha256:{sha256(path)[:12]}"


def cpu_set(root, leaf):
    vals = sorted({text(p) for p in glob.glob(os.path.join(root, f"sys/devices/system/cpu/cpu[0-9]*/cpufreq/{leaf}"))})
    return ",".join(vals) or "unknown"


def kit_python():
    py = os.environ.get("PYTHON")
    if not py:
        return "unknown"
    return subprocess.run([py, "-c", "import platform; print(platform.python_version())"], capture_output=True, text=True,
                          check=True, timeout=30).stdout


def collect(root="/", kit=None, pkgs=(), device=None):
    """[(key, value)] in the documented order"""
    dev = guard(gpu_card, root)
    dev = None if dev == "unknown" else dev
    rows = [("kernel", guard(text, os.path.join(root, "proc/sys/kernel/osrelease"))),
            ("cmdline_gpu", guard(cmdline_gpu, root))]
    rows += gpu_firmware(dev)
    rows += packages(root)
    rows += [("rocm", guard(rocm, root)), ("loom", guard(loom)),
             ("kit_commit", guard(kit_commit, kit or os.path.dirname(HERE)))]
    rows += pkg_lines(pkgs)
    rows += power(root, dev)
    rows += [("device_file", guard(device_file, device)),
             ("cpu_governor", guard(cpu_set, root, "scaling_governor")),
             ("cpu_epp", guard(cpu_set, root, "energy_performance_preference")),
             ("python", ".".join(map(str, sys.version_info[:3]))), ("kit_python", guard(kit_python))]
    return [(k, one_line(v)) for k, v in rows]


def parse(path):
    """{key: value} of a saved block; a line that is not `key: value` is kept under its own text, so it can only differ"""
    d = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        k, sep, v = line.strip().partition(": ")
        d[k if sep else "unparsed: " + line.strip()] = v
    return d


def diff(a, b):
    """[(key, value in a or None, value in b or None)] for every key whose value differs, in a's order then b's"""
    keys = list(a) + [k for k in b if k not in a]
    return [(k, a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)]


def main(argv):
    if argv[:1] == ["--diff"]:
        if len(argv) != 3:
            print(__doc__, file=sys.stderr)
            return 2
        try:
            a, b = parse(argv[1]), parse(argv[2])
        except OSError as e:
            print(f"env_block: {e}")
            return 2
        if not a or not b:
            print(f"env_block: empty block ({argv[1] if not a else argv[2]})")
            return 2
        d = diff(a, b)
        for k, x, y in d:
            print(f"{k}: {'<absent>' if x is None else x} -> {'<absent>' if y is None else y}")
        return 1 if d else 0
    root, kit, pkgs, device = "/", None, [], None
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag in ("--root", "--kit", "--pkg", "--device") and i + 1 < len(argv):
            v = argv[i + 1]
            if flag == "--root":
                root = v
            elif flag == "--kit":
                kit = v
            elif flag == "--pkg":
                pkgs.append(v)
            else:
                device = v
            i += 2
        else:
            print(__doc__, file=sys.stderr)
            return 2
    for k, v in collect(root, kit, pkgs, device):
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
