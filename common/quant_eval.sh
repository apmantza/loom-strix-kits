#!/bin/bash
# Quality of a quant against a reference quant of the same model: llama.cpp's perplexity tool saves the reference's logits
# (--kl-divergence-base), then scores the candidate against them (--kl-divergence): mean KL divergence, same top token, and
# perplexity, the metrics model publishers report (e.g. Unsloth: mean KLD and same-top-token against their 8-bit quant).
# This judges the weights, not our kernels; a Loom package built from the winner still needs common/eq.sh (bit-exact against
# its own reference) and common/gates.sh.
# usage: quant_eval.sh REF.gguf CAND.gguf OUT_DIR [TEXT_FILE]
#   machine env: LLAMACPP_DIR (a llama.cpp build that reads this architecture's GGUFs); NGL (GPU layers, default 99);
#   CTX (default 512), CHUNKS (default 40: CTX x CHUNKS tokens scored). TEXT_FILE default: every markdown file of this repository.
set -u -o pipefail
H=$(cd "$(dirname "$0")" && pwd)
REF=$(realpath "${1:?usage: quant_eval.sh REF.gguf CAND.gguf OUT_DIR [TEXT]}"); CAND=$(realpath "${2:?candidate gguf}"); OUT=$(realpath -m "${3:?out dir}")
f=${LSK_ENV:-$HOME/.config/loom-strix-kits/env}; [ -f "$f" ] && { set -a; . "$f"; set +a; }
PP=${LLAMACPP_DIR:?set LLAMACPP_DIR (machine env)}/llama-perplexity
[ -x "$PP" ] || { echo "no llama-perplexity in $LLAMACPP_DIR"; exit 1; }
mkdir -p "$OUT"
TEXT=${4:-$OUT/text.txt}
if [ -z "${4:-}" ]; then find "$H/.." -name "*.md" -not -path "*/.git/*" -print0 | sort -z | xargs -0 cat > "$TEXT"; fi
[ -s "$TEXT" ] || { echo "empty text $TEXT"; exit 1; }
args=(-f "$TEXT" -c "${CTX:-512}" --chunks "${CHUNKS:-40}" -ngl "${NGL:-99}")
"$PP" -m "$REF" "${args[@]}" --kl-divergence-base "$OUT/ref.kld" > "$OUT/ref.log" 2>&1 || { echo "reference run failed ($OUT/ref.log)"; tail -3 "$OUT/ref.log"; exit 1; }
"$PP" -m "$CAND" "${args[@]}" --kl-divergence-base "$OUT/ref.kld" --kl-divergence > "$OUT/cand.log" 2>&1 || { echo "candidate run failed ($OUT/cand.log)"; tail -3 "$OUT/cand.log"; exit 1; }
python3 - "$OUT" "$REF" "$CAND" <<'PY'
import json, re, sys
out, ref, cand = sys.argv[1:4]
t = open(f"{out}/cand.log", errors="replace").read()
def num(label):
    m = re.search(r"^" + re.escape(label) + r"\s*:\s*([-0-9.]+)(?:\s*±\s*([0-9.]+))?", t, re.M)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None) if m else (None, None)
kld, kld_e = num("Mean    KLD"); top, top_e = num("Same top p")
ppl = re.findall(r"Final estimate: PPL = ([0-9.]+)", open(f"{out}/ref.log", errors="replace").read())
s = {"reference": ref, "candidate": cand, "mean_kld": kld, "mean_kld_err": kld_e, "same_top_pct": top, "same_top_err": top_e,
     "reference_ppl": float(ppl[-1]) if ppl else None}
if kld is None or top is None:
    print(f"could not read KLD / same-top lines from {out}/cand.log"); sys.exit(1)
json.dump(s, open(f"{out}/summary.json", "w"), indent=1)
print(f"mean KLD {kld:.5f}" + (f" ± {kld_e:.5f}" if kld_e else "") + f"   same top token {top:.2f} %" + (f" ± {top_e:.2f}" if top_e else "")
      + (f"   reference PPL {s['reference_ppl']:.3f}" if ppl else "") + f"   ({out}/summary.json)")
PY
