# Sourced by every common/ script: loads the machine env and a kit's manifest, and holds the server helpers.
#   machine env: $LSK_ENV, else ~/.config/loom-strix-kits/env (optional; see common/env.example)
#   kit manifest: kits/<kit>/kit.env (KIT_MODEL_ID, KIT_CONFIG, KIT_TOKENIZER, KIT_RECIPE, KIT_BUILD_ARGS, KIT_SERVE_ARGS, KIT_REF_GGUF)
# Paths in a manifest are relative to WORK: where packages, models/ and tok/ live (default: the kit's own directory).
LSK=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
die() { echo "$*" >&2; exit 1; }

lsk_load() {   # KIT
  KIT=${1:?kit name (a directory under kits/)}; KD=$LSK/kits/$KIT
  [ -f "$KD/kit.env" ] || die "no kit manifest $KD/kit.env"
  local e=${LSK_ENV:-$HOME/.config/loom-strix-kits/env}
  set -a; [ -f "$e" ] && . "$e"; . "$KD/kit.env"; set +a
  # a quant variant (kits/<kit>/variants/<name>/: its own pins.json and config.json) replaces the kit's weights and config
  if [ -n "${KIT_VARIANT:-}" ]; then
    [ -d "$KD/variants/$KIT_VARIANT" ] || die "no variant $KD/variants/$KIT_VARIANT"
    [ -f "$KD/variants/$KIT_VARIANT/config.json" ] && KIT_CONFIG=variants/$KIT_VARIANT/config.json
  fi
  WORK=${WORK:-$KD}; PORT=${PORT:-8160}; PYTHON=${PYTHON:-python3}
  KIT_SRC=${KIT_SRC:-$LSK/kit}; STATE=${STATE:-$WORK/state}
  LOOM_SERVE=${LOOM_SERVE:-${VENV:+$VENV/bin/}loom-serve}
  mkdir -p "$STATE"; cd "$WORK" || die "no WORK dir $WORK"
}

lsk_pkg() {   # PKG (absolute, or relative to WORK) -> absolute dir with a model.json
  local p; p=$(realpath -m "$1"); [ -f "$p/model.json" ] || die "no $p/model.json"; echo "$p"
}

lsk_served() {   # the model.json the process on $PORT serves (its --model resolved against its cwd); exit 1 if none
  local pid m
  pid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  [ -n "$pid" ] || return 1
  m=$(tr '\0' '\n' < /proc/$pid/cmdline | sed -n 's/^--model=//p')
  [ -n "$m" ] || return 1
  (cd /proc/$pid/cwd && realpath "$m")
}

lsk_up() {   # [SECONDS]: wait until $PORT answers /v1/models (default 60 s)
  local t=$((SECONDS + ${1:-60}))
  while :; do curl -s -m 5 127.0.0.1:$PORT/v1/models | grep -q '"object":"list"' && return 0; [ $SECONDS -lt $t ] || return 1; sleep 2; done
}

lsk_measuring() {   # a measurement server of ours is alive (its pid file in $STATE)
  local f; for f in "$STATE"/*.pid; do [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && { echo "$(basename "$f" .pid) pid $(cat "$f")"; return 0; }; done; return 1
}

lsk_stop() {   # stop our servers (pid files), then any managed server (LSK_STOP_HOOK), and wait for the port
  local f i p c want
  # LSK_GUARD (machine env, optional): a command that must succeed before anything is stopped. The stop hook alone fenced only the
  # managed server: a refused hook still came after the pid-file kills, and a dry run killed another run's profiling servers (E438).
  if [ -n "${LSK_GUARD:-}" ] && ! eval "$LSK_GUARD" >/dev/null; then echo "lsk_stop: LSK_GUARD refused; nothing stopped" >&2; return 1; fi
  # a pid file names a server only while that pid's command line runs our server program ($LOOM_SERVE's name, the check loom-skills
  # stop.sh makes); a stale file's reused pid is reported and left (11-27 review G: a stale state/*.pid killed an unrelated process)
  want=$(basename "${LOOM_SERVE:-loom-serve}")
  for f in "$STATE"/*.pid; do
    [ -f "$f" ] || continue; p=$(cat "$f"); { c=$(tr '\0' ' ' < "/proc/$p/cmdline"); } 2>/dev/null
    if [ -n "$c" ] && [[ "$c" == *"$want"* ]]; then kill "$p" 2>/dev/null
    elif [ -n "$c" ]; then echo "lsk_stop: $f names pid $p ($(cat /proc/$p/comm 2>/dev/null)), not a $want: stale, left running" >&2; fi
    rm -f "$f"
  done
  [ -n "${LSK_STOP_HOOK:-}" ] && eval "$LSK_STOP_HOOK" >/dev/null 2>&1
  for i in $(seq 1 60); do ss -ltn | grep -q ":$PORT " || return 0; sleep 1; done
  echo "lsk_stop: port $PORT still busy" >&2; return 1
}

lsk_restore() {   # hand the port back: stop ours, then LSK_RESTORE_HOOK (e.g. restart a managed live server); no hook = leave it free
  # exit 0 only when the port answers within LSK_RESTORE_WAIT s (default 300) after the hook; every failure prints one line
  # 'LIVE NOT RESTORED: <reason>' on stderr. Callers print their own success line only on exit 0 (`lsk_restore && echo restored`).
  local e w=${LSK_RESTORE_WAIT:-300} log=$STATE/restore_hook.log
  e=$(lsk_stop 2>&1 >/dev/null) || { echo "LIVE NOT RESTORED: the stop before it failed: $(echo "${e:-lsk_stop exit 1}" | grep -v '^ *$' | paste -sd'|' | cut -c1-400)" >&2; return 1; }
  [ -z "$e" ] || echo "$e" >&2   # a successful stop's notes (a stale pid file left running)
  [ -n "${LSK_RESTORE_HOOK:-}" ] || return 0
  eval "$LSK_RESTORE_HOOK" > "$log" 2>&1; e=$?
  [ $e -eq 0 ] || { echo "LIVE NOT RESTORED: restore hook exit $e: $(grep -v '^ *$' "$log" | tail -1 | cut -c1-300)" >&2; return 1; }
  lsk_up "$w" || { echo "LIVE NOT RESTORED: nothing answers on $PORT $w s after the restore hook (its output: $(grep -v '^ *$' "$log" | tail -1 | cut -c1-300))" >&2; return 1; }
}

lsk_serve_bg() {   # NAME PKG [extra loom-serve args]: serve PKG in the background with the kit's flags; pid in $STATE/NAME.pid
  local name=$1 p; p=$(lsk_pkg "$2"); shift 2
  [ "$(ulimit -l)" = unlimited ] || ulimit -l unlimited 2>/dev/null || die "memlock is $(ulimit -l) KiB; loom-serve needs unlimited"
  local -a sa; read -ra sa <<< "$KIT_SERVE_ARGS"   # word split without globbing (the device URL has a '?')
  (cd "$WORK" && exec "$LOOM_SERVE" --listen=127.0.0.1:$PORT --model="$p/model.json" "${sa[@]}" "$@" > "$STATE/$name.log" 2>&1) &
  echo $! > "$STATE/$name.pid"
  lsk_up || die "$p did not come up ($STATE/$name.log)"
  [ "$(lsk_served)" = "$p/model.json" ] || die ":$PORT serves '$(lsk_served)', not $p/model.json"
}
