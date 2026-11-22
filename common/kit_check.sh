#!/bin/bash
# The gate for a kit's weights (run after fetch, before trusting a scoreboard, and whenever pins or a variant change):
#   1. the local files match pins.json (common/fetch.py --verify-only);
#   2. upstream freshness (common/fresh.py: hrx-loom and every pinned file against upstream; reported, not fatal);
#   3. the floors are not stale: common/make_lanes.py on the pinned model weights must reproduce lanes.json's decode bytes and
#      multiply-adds within 1 % (a hand-derived floor once counted an MTP layer no served program reads: 2 % too high).
# usage: [KIT_VARIANT=name] kit_check.sh KIT [MAKE_LANES_ARGS...]   (the model's flags, e.g. --kv kv8, --experts/--top-k, --loop;
#        the kit can record them in kit.env as KIT_LANES_ARGS)      exit 1 when 1 or 3 fails
set -u -o pipefail
H=$(cd "$(dirname "$0")" && pwd)
. "$H/env.sh"; lsk_load "${1:?usage: kit_check.sh KIT [MAKE_LANES_ARGS...]}"; shift
PINS=$KD/pins.json; [ -n "${KIT_VARIANT:-}" ] && PINS=$KD/variants/$KIT_VARIANT/pins.json
rc=0
echo "== pins ($PINS)"
"$PYTHON" "$H/fetch.py" "$KIT" --work "$WORK" --verify-only || { echo "KIT CHECK: local files do not match the pins"; rc=1; }
echo "== upstream"
"$PYTHON" "$H/fresh.py" "$KIT" ${VENV:+--venv "$VENV"} || echo "(freshness is reported, not fatal: re-measure on a newer SDK or weights before trusting old numbers)"
echo "== floors"
W=$("$PYTHON" -c "import json,sys; f=json.load(open(sys.argv[1]))['files'][0]; print(f['dest'] + '/' + f['path'])" "$PINS")
read -ra LA <<< "${KIT_LANES_ARGS:-}"
"$PYTHON" "$H/make_lanes.py" "$WORK/$W" "${LA[@]}" "$@" > "$STATE/kit_check.lanes.json" 2> "$STATE/kit_check.lanes.err" || { echo "KIT CHECK: make_lanes failed ($STATE/kit_check.lanes.err)"; exit 1; }
"$PYTHON" - "$KD/$KIT_LANES" "$STATE/kit_check.lanes.json" <<'PY' || rc=1
import json, sys
have = json.load(open(sys.argv[1]))["model"]; want = json.load(open(sys.argv[2]))
bad = []
for k in ("decode_weight_bytes", "linear_params", "kv_bytes_per_pos_layer", "attn_layers"):
    h, w = have.get(k), want.get(k)
    ok = h is not None and w is not None and abs(h - w) <= 0.01 * max(abs(w), 1)
    print(f"  {k:24s} lanes.json {h}  from the weights {w}  {'ok' if ok else 'STALE'}")
    bad += [] if ok else [k]
if bad:
    print("KIT CHECK: lanes.json is stale for " + ", ".join(bad) + ": regenerate with common/make_lanes.py --write"); sys.exit(1)
PY
echo "kit_check $KIT$([ -n "${KIT_VARIANT:-}" ] && echo " ($KIT_VARIANT)"): $([ $rc = 0 ] && echo PASS || echo FAIL)"; exit $rc
