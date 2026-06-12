#!/bin/bash
# GPU watchdog (DermNet-only policy): repeatedly kill the auto-respawning DermNet
# inference job (`... --mode infer --reuse`, matched by 'dermnet' in its cmdline),
# so it cannot steal GPU from the train_v2 run. Leaves ALL other processes alone.
# Self-exits once both training ranks are gone.

PATTERN="dermnet"                     # case-insensitive match against full cmdline
PROTECTED="2336054 2336055"           # the two DDP training ranks (never touch)
LAUNCHER="2335986"                    # parent torchrun launcher (never touch)
TRAIN_LOG="/mnt/tp/minh/nlp_261/capstone-dialect-transfer/results/train_v2.log"
LOG="/mnt/tp/minh/nlp_261/capstone-dialect-transfer/results/gpu_watchdog.log"
SELF=$$

is_protected() {
  local pid="$1"
  for p in $PROTECTED $LAUNCHER $SELF; do
    [ "$pid" = "$p" ] && return 0
  done
  return 1
}

# kill_dermnet <signal> : signal every process whose cmdline matches PATTERN,
# excluding protected PIDs and our own tooling (the watchdog / pgrep itself).
kill_dermnet() {
  local sig="$1"
  local pid cmd
  for pid in $(pgrep -fi "$PATTERN" 2>/dev/null); do
    [ -z "$pid" ] && continue
    is_protected "$pid" && continue
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null)
    [ -z "$cmd" ] && continue
    case "$cmd" in
      *gpu_watchdog*|*pgrep*) continue ;;     # never match our own machinery
    esac
    echo "$cmd" | grep -qi "$PATTERN" || continue
    [ "$sig" = "TERM" ] && echo "[$(date '+%F %T')] KILL($sig) dermnet pid=$pid cmd=[$cmd]" >> "$LOG"
    kill -"$sig" "$pid" 2>/dev/null
  done
}

echo "[$(date '+%F %T')] watchdog START (DermNet-only) pid=$SELF pattern='$PATTERN' protecting={$PROTECTED} launcher=$LAUNCHER" >> "$LOG"

while true; do
  kill_dermnet TERM
  sleep 3
  kill_dermnet KILL          # escalate for anything that ignored SIGTERM

  # liveness of the protected training run
  alive=0
  for p in $PROTECTED; do
    kill -0 "$p" 2>/dev/null && alive=$((alive+1))
  done

  step=$(grep -aoE '[0-9]+/6630' "$TRAIN_LOG" 2>/dev/null | tail -1)
  echo "[$(date '+%F %T')] alive=$alive/2 step=${step:-?}" >> "$LOG"

  if [ "$alive" -eq 0 ]; then
    echo "[$(date '+%F %T')] both training ranks gone -> training finished. watchdog EXIT." >> "$LOG"
    break
  fi

  sleep 12
done
