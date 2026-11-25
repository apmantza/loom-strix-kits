#!/bin/bash
# cool_wait.sh: the thermal start gate shared by agent_ab and events_ab.
# Recurrence: HRX-like E485 (q194's 8 events_ab starts ran back to back with no cooling: every start heat-soaked at 2.15 GHz while
# agent_ab halves started at 55 C; the two tools measured different clocks) and E368 (a hot first half biased every bench).
set -u
H=$(cd "$(dirname "$0")" && pwd); D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
fail=0
seq_stub() {   # a temperature command that returns the next value of a sequence on each call
  printf '%s\n' "$@" > "$D/seq"; : > "$D/n"
  cat > "$D/temp.sh" <<'EOF'
#!/bin/bash
d=$(dirname "$0"); n=$(wc -l < "$d/n"); echo x >> "$d/n"; sed -n "$((n + 1))p" "$d/seq"
EOF
  chmod +x "$D/temp.sh"; }
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

# 1. hot, then cooling: waits until <= T_START and reports the cool reading
seq_stub 70 62 54 50
out=$(COOL_TEMP_CMD="$D/temp.sh" COOL_STEP=0 T_START=55 bash "$H/cool_wait.sh" t1 "$D/t1.txt"); rc=$?
check "waits for the cool reading" '[ $rc -eq 0 ] && [[ "$out" == *"t1 start: edge 54 C"* ]] && [ "$(wc -l < "$D/n")" -eq 3 ]'
check "writes the start line to OUTFILE" 'grep -q "t1 start: edge 54 C" "$D/t1.txt"'

# 2. already cool: no wait
seq_stub 40
out=$(COOL_TEMP_CMD="$D/temp.sh" COOL_STEP=0 bash "$H/cool_wait.sh" t2); rc=$?
check "cool start passes at once" '[ $rc -eq 0 ] && [[ "$out" == *"edge 40 C after 0 s"* ]]'

# 3. no reading: fail closed (exit 2), never a silent start
seq_stub ""
out=$(COOL_TEMP_CMD="$D/temp.sh" COOL_STEP=0 bash "$H/cool_wait.sh" t3); rc=$?
check "unreadable temperature stops (exit 2)" '[ $rc -eq 2 ] && [[ "$out" == *"STOP"* ]]'
out=$(COOL_TEMP_CMD="false" COOL_STEP=0 bash "$H/cool_wait.sh" t3b); rc=$?
check "failing temperature command stops (exit 2)" '[ $rc -eq 2 ]'

# 4. never cools: starts at COOL_MAX with a HOT note
yes 80 | head -50 > "$D/seq"; : > "$D/n"
out=$(COOL_TEMP_CMD="$D/temp.sh" COOL_STEP=1 COOL_MAX=1 bash "$H/cool_wait.sh" t4); rc=$?
check "timeout starts HOT and says so" '[ $rc -eq 0 ] && [[ "$out" == *"starts HOT (80 C > 55 C"* ]] && [[ "$out" == *"t4 start: edge 80 C"* ]]'

exit $fail
