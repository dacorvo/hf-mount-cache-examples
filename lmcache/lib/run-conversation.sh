#!/usr/bin/env bash
#
# run-conversation.sh — single-turn hermes session against the local vLLM.
#
# Usage: run-conversation.sh <prompt-file>
#
# Prompt file format: lines forming one prompt; "---"/"===" markers split
# multiple prompts. Only the FIRST prompt is sent — this runner is
# single-turn (illustrative blunt prefix-cache test, not multi-turn agent).
#
# Required env: SCRIPT_DIR, LOG_DIR, MODEL, VLLM_URL, PHASE
#
set -uo pipefail

PROMPT_FILE="$1"
label=$(basename "$PROMPT_FILE" .txt)

CONV_LOG="$LOG_DIR/conversation-${label}-${PHASE}.log"
STATS_FILE="$LOG_DIR/conv-stats-${label}-${PHASE}.txt"
HERMES_BIN="${HERMES_BIN:-$HOME/.local/bin/hermes}"

# Per-conversation HERMES_HOME overlay so the user's real ~/.hermes is
# untouched. We copy the user's hermes config tree (skills, soul, snapshot)
# then rewrite config.yaml's `default:` to the profile's MODEL so hermes
# talks to the model vLLM is actually serving.
HERMES_DIR="$LOG_DIR/hermes-${label}"
rm -rf "$HERMES_DIR"
cp -r "$HOME/.hermes" "$HERMES_DIR"
sed -i "s|^  default: .*|  default: ${MODEL}|" "$HERMES_DIR/config.yaml"
sed -i "s|^  base_url: .*|  base_url: ${VLLM_URL}/v1|" "$HERMES_DIR/config.yaml"
# Hermes hard-floors context at 64K (both for the main model and an
# auxiliary "compression" model — both default to the same backend). Claim
# 64K for both so init passes; vLLM still enforces the real MAX_MODEL_LEN
# and our short prompts stay well under it.
HERMES_MIN_CONTEXT=65536
if ! grep -q '^  context_length:' "$HERMES_DIR/config.yaml"; then
  sed -i "/^  key_env:/a\\  context_length: ${HERMES_MIN_CONTEXT}" "$HERMES_DIR/config.yaml"
fi
if ! grep -q '^auxiliary:' "$HERMES_DIR/config.yaml"; then
  cat >> "$HERMES_DIR/config.yaml" <<EOF
auxiliary:
  compression:
    context_length: ${HERMES_MIN_CONTEXT}
EOF
fi

log() { echo "[$(date '+%H:%M:%S')] [$label] $*"; }

# Extract the first prompt: everything up to the first --- or === separator.
first_prompt=$(awk '/^(---|===)$/ { exit } { print }' "$PROMPT_FILE" | tr '\n' ' ' | sed 's/  */ /g; s/ $//')

if [ -z "$first_prompt" ]; then
  log "ERROR: no prompt parsed from $PROMPT_FILE"
  exit 1
fi

log "Starting (single turn, prompt=${first_prompt:0:80}...)"

t0=$(date +%s%3N)
HERMES_HOME="$HERMES_DIR" OPENAI_API_KEY=dummy timeout 1200 "$HERMES_BIN" chat \
  -q "$first_prompt" -Q --yolo --accept-hooks \
  > "$CONV_LOG" 2>&1
rc=$?
t1=$(date +%s%3N)
elapsed_ms=$(( t1 - t0 ))

if [ $rc -ne 0 ]; then
  log "WARNING: hermes exited with rc=$rc after ${elapsed_ms}ms"
fi

{
  echo "label=$label"
  echo "first_turn_ttft_ms=$elapsed_ms"
} > "$STATS_FILE"

log "Complete (${elapsed_ms}ms, rc=$rc)"
