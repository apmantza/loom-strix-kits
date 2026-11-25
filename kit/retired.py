"""Switches the kit no longer reads (switch census 11-25; rules section 4: a switch is a test fixture, so an adopted one is folded
into the code and a refuted one's code is deleted). check() refuses an environment that still sets one, unless its value asks for
exactly what the kit now builds: a folded switch at its folded value with the switches it modified set, a deleted switch off.
Fail closed: any other value (another spelling of off included) stops the generator by name, so a stale recipe or package .cmd
cannot build something other than it asks for. gen_llm.py, gen_flash2s.py and gen_gemm.py call it first.
The deciding ledger rows: switch-census/CENSUS.md (HRX-like E-numbers in the folding commits)."""
import os
import sys

OFF = ("", "0")
# name -> (accepted values, switches that must be "1" for the value to pass, scope: the entry applies only when these are "1")
RETIRED = {
    # folded into FLASH2_PF11 (E55 / E57)
    "FLASH2_PF11_SPLIT": (("1",), ("FLASH2_PF11",), ()),
    "FLASH2_PF11_NOIF": (("1",), ("FLASH2_PF11",), ()),
    "FLASH2_PF11_CARRY": (("1",), ("FLASH2_PF11",), ()),
    # folded into KV8 (with FLASH2_PF11: kv8_late; E144 / E170), layout v2 (E95 / E109)
    "FLASH2_KV8_KCARRY": (("1",), ("KV8", "FLASH2_PF11"), ()),
    "FLASH2_KV8_VLATE": (("2",), ("KV8", "FLASH2_PF11"), ()),
    "KV8_LAYOUT": (("2",), ("KV8",), ()),
    "KV8_SPLIT_VZERO": (("1",), (), ("KV8",)),   # the KV8 guard (E444) is always on; without KV8 it still selects the f16 clamp
    # folded into FLASH2_SX11 (E462 / E477)
    "FLASH2_SXH": (("1",), ("FLASH2_SX11",), ()),
    # folded defaults (no recipe set them off)
    "FLASH2_SPLITK_SELF": (("1",), (), ()),
    "IQ4_Q8": (("1",), (), ()),
    "IQ3_Q8": (("1",), (), ()),
    "IQ4_GEMM_DIRECT": (("1",), (), ()),
    # refuted: code deleted
    **{n: (OFF, (), ()) for n in (
        "ATTN_L17", "FLASH2_HALF256", "FLASH2_HOT11", "FLASH2_NG256", "FLASH2_PF11_COND", "FLASH2_PF11_EARLYPG",
        "FLASH2_PF11_KLATE", "FLASH2_PF11_V8", "FLASH2_PF11_VLATE", "FLASH2_PF11_VPIN", "FLASH2_VS_KV8LATE", "FLASH2_VS_PF",
        "GEMM256_LDS_PF", "GEMM_M256_MIN_N_ALL", "KV8_PARTIAL")},
}


def check(environ=None, who=None):
    env = os.environ if environ is None else environ
    bad = []
    for name, (ok, needs, scope) in RETIRED.items():
        v = env.get(name)
        if v is None or any(env.get(s) != "1" for s in scope):
            continue
        if v not in ok or any(env.get(p) != "1" for p in needs):
            want = " or ".join(repr(x) for x in ok) + (f" with {' and '.join(p + '=1' for p in needs)}" if needs else "")
            bad.append(f"{name}={v!r} (accepted: {want}, or unset)")
    if bad:
        sys.exit(f"{who or os.path.basename(sys.argv[0])}: retired switch set to something the kit no longer builds: "
                 + "; ".join(bad) + " (switch census 11-25; kit/retired.py)")
