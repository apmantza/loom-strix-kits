#!/bin/bash
# Exactness of a candidate against a reference, one server at a time: serve each with the kit's flags, save greedy outputs and
# draft counters (compare_outputs.py --long), and diff texts AND draft counters (exit 1 on either: a drafter-side bug once left
# every output identical while acceptance fell to 0). REF is a saved .json or a reference package dir.
# usage: eq.sh KIT CANDIDATE_PKG REF OUT_DIR
set -u -o pipefail
. "$(dirname "$0")/env.sh"; lsk_load "${1:?usage: eq.sh KIT PKG REF OUT_DIR}"
C=$(lsk_pkg "${2:?candidate}"); REF=${3:?reference .json or package}; OUT=$(realpath -m "${4:?out dir}"); mkdir -p "$OUT"
H=$LSK/common
trap 'bash $H/serve_as.sh $KIT --restore' EXIT
save() { bash "$H/serve_as.sh" "$KIT" "$1" || die "eq STOP: serve $1"
  "$PYTHON" "$H/compare_outputs.py" --long --save "$2" "http://127.0.0.1:$PORT" "$KIT_MODEL_ID" > "$2.log" 2>&1 || die "eq STOP: save $1 ($2.log)"; }
if [ -d "$REF" ]; then r=$(lsk_pkg "$REF"); save "$r" "$OUT/ref-$(basename "$r").json"; REF=$OUT/ref-$(basename "$r").json; fi
[ -f "$REF" ] || die "no reference $REF"
save "$C" "$OUT/$(basename "$C").json"
"$PYTHON" "$H/compare_outputs.py" --diff "$REF" "$OUT/$(basename "$C").json"; d=$?
echo "eq: $(basename "$C") vs $(basename "$REF"): $(case $d in 0) echo EXACT;; 3) echo "CONTROL FAILED (no verdict)";; *) echo DIFFERS;; esac) (exit $d)"; exit $d
