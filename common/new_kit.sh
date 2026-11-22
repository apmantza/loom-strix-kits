#!/bin/bash
# Scaffold a new kit: kits/NAME/ with kit.env, pins.json, lanes.json (the standard lane map; its model block from
# common/make_lanes.py once the weights are pinned), NOTES.md, README.md and the fetch.sh wrapper. Refuses an existing kit.
# usage: new_kit.sh NAME MODEL_ID
# Then: fill pins.json (common/fetch.py downloads and verifies it), write config.json (the generator's port config: kit/
# reads it), run common/make_lanes.py on the weights (--write kits/NAME/lanes.json), and keep NOTES.md as you learn the model.
set -u -o pipefail
LSK=$(cd "$(dirname "$0")/.." && pwd)
N=${1:?usage: new_kit.sh NAME MODEL_ID}; M=${2:?usage: new_kit.sh NAME MODEL_ID}
[[ $N =~ ^[a-z0-9][a-z0-9.-]*$ ]] || { echo "kit name: lowercase letters, digits, '.', '-'"; exit 1; }
K=$LSK/kits/$N
[ -e "$K" ] && { echo "kits/$N exists: refusing to overwrite"; exit 1; }
mkdir -p "$K"
cat > "$K/kit.env" <<EOF
# $N kit manifest (read by common/*.sh). Paths are relative to WORK (default: this directory; fetch.sh fills it).
KIT_MODEL_ID=$M
KIT_CONFIG=\${KIT_CONFIG:-config.json}   # the generator's port config, relative to this directory or absolute
KIT_TOKENIZER=tok/tokenizer.json
KIT_REF_GGUF=                           # a GGUF of the same model for the gates' llama.cpp reference outputs
# the recipe: generator switches (kit/*.py read them; unset = unchanged program). Start empty; add one switch per measured win.
KIT_RECIPE=""
KIT_BUILD_ARGS=\${KIT_BUILD_ARGS:-"--model-id $M --rows 4 --prefill 4096"}
KIT_SERVE_ARGS="--device=amdgpu://?completion_spin_ns=4294967295 --pool_backing=fixed --workers=8"
KIT_LANES=lanes.json
KIT_PROFILE_TOKENS=81000
EOF
cat > "$K/pins.json" <<EOF
{
 "_doc": "What this kit was built and measured with. common/fetch.py downloads and verifies these; common/fresh.py compares them with upstream.",
 "hrx_loom": "$(python3 -c 'import json,urllib.request; print(json.load(urllib.request.urlopen("https://pypi.org/pypi/hrx-loom/json", timeout=20))["info"]["version"])' 2>/dev/null || echo TODO)",
 "files": [
  {"repo": "TODO/repo", "revision": "TODO full commit sha", "path": "TODO.gguf", "sha256": "TODO", "dest": "models"},
  {"repo": "TODO/base-model", "revision": "TODO", "path": "tokenizer.json", "sha256": "TODO", "dest": "tok"}
 ]
}
EOF
python3 - "$LSK/kits/qwen38-27b/lanes.json" "$K/lanes.json" "$N" <<'PY'
import json, sys
src, dst, name = sys.argv[1:4]
d = json.load(open(src))
out = {"_doc": d["_doc"], "hardware": d["hardware"],
       "model": {"name": name, "_src": "TODO: common/make_lanes.py WEIGHTS [--config config.json] [--experts N --top-k K] [--loop F:L:T] --write this file"},
       "prefill_marker": d.get("prefill_marker"), "_prefill_marker_src": "the prefill attention kernel's name: submissions running it are prefill epochs (adapt to this kit's kernels)",
       "prefill_min_tokens": d.get("prefill_min_tokens"), "lanes": d["lanes"]}
json.dump(out, open(dst, "w"), indent=1); open(dst, "a").write("\n")
PY
cat > "$K/NOTES.md" <<EOF
# $N: model knowledge for optimisation work

What is specific to this model. The method is in skills/loom-strix-optimise/; device facts in its devices/ files.

## Model and serving shape
TODO: layers and their kinds, heads, formats, speculative decoding; the launch shapes serving actually runs (read them from
a served profile: common/events_ab.sh, then the workgroup counts per kernel).

## Lanes and the kernels behind them
TODO: adapt lanes.json's regexes to this kit's kernel names; one row per lane with its bound (bytes / issue / latency).

## Where the time is
TODO: common/events_summary.py kits/$N/lanes.json RUN after the first served profile.
EOF
cat > "$K/README.md" <<EOF
# $M on Strix Halo (Loom)

> **Work in progress.** New kit: nothing measured yet.

Weights: pins.json (common/fetch.py $N). Build: common/build.sh $N pkg. Serve: common/serve.sh $N pkg.
EOF
cat > "$K/fetch.sh" <<EOF
#!/bin/bash
# Download this kit's pinned files (pins.json) and verify every sha256.
exec python3 "\$(dirname "\$0")/../../common/fetch.py" $N "\$@"
EOF
chmod +x "$K/fetch.sh"
echo "created kits/$N: $(ls "$K" | tr '\n' ' ')"
echo "next: pins.json, config.json, common/make_lanes.py <weights> --write kits/$N/lanes.json"
