#!/usr/bin/env bash
#
# run-conversation.sh — adaptive multi-turn opencode conversation
#
# Usage: run-conversation.sh <prompt-file>
#
# Prompt file format:
#   - One prompt per line, separated by "---" lines
#   - "===" separates grow prompts from post-compaction prompts
#
# Required env vars: SCRIPT_DIR, LOG_DIR, MAX_MODEL_LEN
#
set -euo pipefail

PROMPT_FILE="$1"
label=$(basename "$PROMPT_FILE" .txt)

CONV_LOG="$LOG_DIR/conversation-${label}-${PHASE}.log"
STATS_FILE="$LOG_DIR/conv-stats-${label}-${PHASE}.txt"
THRESHOLD=$(( MAX_MODEL_LEN * 9 / 10 ))

# Isolate each conversation's SQLite database to avoid WAL lock contention
# when multiple opencode instances run in parallel.
export XDG_DATA_HOME="$LOG_DIR/opencode-data-${label}"
mkdir -p "$XDG_DATA_HOME"

log() { echo "[$(date '+%H:%M:%S')] [$label] $*"; }

# Parse prompt file into grow and post arrays.
grow_prompts=()
post_prompts=()
in_post=false
current=""
while IFS= read -r line; do
  if [ "$line" = "===" ]; then
    [ -n "$current" ] && grow_prompts+=("$current")
    current=""
    in_post=true
  elif [ "$line" = "---" ]; then
    if $in_post; then
      [ -n "$current" ] && post_prompts+=("$current")
    else
      [ -n "$current" ] && grow_prompts+=("$current")
    fi
    current=""
  else
    [ -n "$current" ] && current="$current $line" || current="$line"
  fi
done < "$PROMPT_FILE"
# Flush last prompt.
if $in_post; then
  [ -n "$current" ] && post_prompts+=("$current")
else
  [ -n "$current" ] && grow_prompts+=("$current")
fi

total=$(( ${#grow_prompts[@]} + ${#post_prompts[@]} ))
log "Starting (up to $total turns, compaction at $THRESHOLD tokens)"

turn=0
max_tokens=0
compacted=false
first_turn_ttft_ms=""

# Phase 1: grow context until 90% threshold.
for prompt in "${grow_prompts[@]}"; do
  turn=$((turn + 1))

  if [ "$turn" -eq 1 ]; then
    log "Turn $turn (grow)"
    t_start=$(date +%s%3N)
    (cd "$SCRIPT_DIR" && opencode run "$prompt") >> "$CONV_LOG" 2>&1
    t_end=$(date +%s%3N)
    first_turn_ttft_ms=$(( t_end - t_start ))
    log "First-turn wall time: ${first_turn_ttft_ms}ms"
  else
    log "Turn $turn (grow)"
    (cd "$SCRIPT_DIR" && opencode run -c "$prompt") >> "$CONV_LOG" 2>&1
  fi

  # Estimate token count from conversation log size (~4 chars/token).
  log_bytes=$(wc -c < "$CONV_LOG" 2>/dev/null || echo 0)
  est_tokens=$(( log_bytes / 4 ))
  [ "$est_tokens" -gt "$max_tokens" ] && max_tokens="$est_tokens"

  if [ "$est_tokens" -ge "$THRESHOLD" ]; then
    log ">>> Estimated ~${est_tokens} tokens, switching to post-compaction"
    compacted=true
    break
  fi
done

# Phase 2: post-compaction turns.
for prompt in "${post_prompts[@]}"; do
  turn=$((turn + 1))
  log "Turn $turn (post-compaction)"
  (cd "$SCRIPT_DIR" && opencode run -c "$prompt") >> "$CONV_LOG" 2>&1
done

# Write per-conversation stats.
{
  echo "label=$label"
  echo "turns=$turn"
  echo "max_tokens=$max_tokens"
  echo "compacted=$compacted"
  [ -n "$first_turn_ttft_ms" ] && echo "first_turn_ttft_ms=$first_turn_ttft_ms"
} > "$STATS_FILE"

log "Complete ($turn turns)"
