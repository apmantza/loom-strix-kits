#!/usr/bin/env python3
"""GEMM256_KT=32 must generate, gfx12 twins included (HRX-like E407: the twin's final check refused any `vector<16xf16>,
vector<16xf16>` text, so the K-tile-32 kernel's 16-value activation prefetch pairs failed every KT=32 build; autotune found it,
E406). Also: KT=64 output is unchanged by the narrowed check, and GEMM256_FLATADDR with KT=32 refuses by name."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def gen(env_extra):
    env = dict(os.environ, **env_extra)
    return subprocess.run([sys.executable, os.path.join(HERE, "gen_gemm.py")], capture_output=True, text=True, env=env)


def main():
    fails = 0
    r32 = gen({"GEMM256_KT": "32"})
    ok = r32.returncode == 0 and "g_gemm_q4_k_store_m256_rdna4" in r32.stdout and "view<40x" in r32.stdout
    print(("ok  " if ok else "FAIL") + "  KT=32 generates with its gfx12 twins (exit %d)" % r32.returncode); fails += not ok
    r64 = gen({"GEMM256_KT": "64"}); rd = gen({})
    ok = r64.returncode == 0 and r64.stdout == rd.stdout and "view<72x" in r64.stdout
    print(("ok  " if ok else "FAIL") + "  KT=64 equals the default and keeps its 72-wide twin views"); fails += not ok
    sys.path.insert(0, HERE)
    os.environ["GEMM256_KT"] = "32"
    import gemm_flataddr
    try:
        gemm_flataddr.flataddr("kernel text"); ok = False
    except SystemExit as e:
        ok = "GEMM256_KT=64 only" in str(e)
    print(("ok  " if ok else "FAIL") + "  FLATADDR with KT=32 refuses by name"); fails += not ok
    print(f"{3 - fails}/3 passed"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
