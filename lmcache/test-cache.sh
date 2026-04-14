#!/usr/bin/env bash
#
# CLI for LMCache + vLLM cache integration tests.
#
# Orchestrates vLLM, hf-mount, and opencode conversations via
# process-compose for proper lifecycle management.
#
# Usage:
#   ./test-cache.sh [--profile <name>] <command>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../hf-mount" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/../.venv}"
PC_BIN="${PC_BIN:-$SCRIPT_DIR/../bin/process-compose}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -d "$VENV_DIR" ]] || { echo "ERROR: Venv not found at $VENV_DIR — run the root setup.sh first" >&2; exit 1; }
  source "$VENV_DIR/bin/activate"
fi

[ -x "$PC_BIN" ] || { echo "ERROR: process-compose not found at $PC_BIN — run the root setup.sh first" >&2; exit 1; }

# ── Profile loading ───────────────────────────────────────────────────

PROFILE="${PROFILE:-qwen2.5-7b-tp1}"
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

PROFILE_FILE="$SCRIPT_DIR/profiles/${PROFILE}.sh"
[ -f "$PROFILE_FILE" ] || { echo "ERROR: Profile not found: $PROFILE_FILE" >&2; exit 1; }
source "$PROFILE_FILE"

# ── Derived variables ─────────────────────────────────────────────────

export MOUNT_POINT="${MOUNT_POINT:-/tmp/hf-mount-lmcache}"
export BUCKET="${BUCKET:-dacorvo/lm-cache}"
export HF_MOUNT_CACHE_DIR="${HF_MOUNT_CACHE_DIR:-/tmp/hf-mount-cache-${PROFILE_NAME}}"
export LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/hf-mount-local-cache-${PROFILE_NAME}}"
export LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/$PROFILE_NAME}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_URL="http://localhost:$VLLM_PORT"
export HF_MOUNT_BIN="${HF_MOUNT_BIN:-$REPO_ROOT/target/release/hf-mount-nfs}"
export LMCACHE_BUCKET_PATH="$MOUNT_POINT/$PROFILE_NAME"
export LMCACHE_LOCAL_PATH="$LOCAL_CACHE_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
fi

# Export profile vars for generate-phase.sh.
export MODEL VLLM_PORT MAX_MODEL_LEN GPU_MEMORY_UTIL TOOL_CALL_PARSER
export SCRIPT_DIR LOG_DIR VLLM_URL MOUNT_POINT PROFILE_NAME
export HF_MOUNT_BIN BUCKET HF_MOUNT_CACHE_DIR
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
# Build extra vLLM args from profile.
[ -n "${TOOL_PARSER_PLUGIN:-}" ] && VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS --tool-parser-plugin $TOOL_PARSER_PLUGIN"
[ -n "${CHAT_TEMPLATE:-}" ] && VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS --chat-template $CHAT_TEMPLATE"
[ "${TP_SIZE:-1}" -gt 1 ] && VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS --tensor-parallel-size $TP_SIZE"
export VLLM_EXTRA_ARGS

source "$SCRIPT_DIR/lib/helpers.sh"

mkdir -p "$LOG_DIR"
SESSION_LOG="$LOG_DIR/session.log"

# ── opencode.json generation ──────────────────────────────────────────

generate_opencode_json() {
  cat > "$SCRIPT_DIR/opencode.json" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local vLLM ($MODEL_SHORT)",
      "options": {
        "baseURL": "http://localhost:${VLLM_PORT}/v1"
      },
      "models": {
        "$MODEL": {
          "name": "$MODEL_SHORT (cache test)",
          "options": {
            "max_tokens": $MAX_TOKENS
          },
          "limit": {
            "context": $MAX_MODEL_LEN,
            "output": $MAX_TOKENS
          }
        }
      }
    }
  },
  "model": "local/$MODEL"
}
EOF
}

# ── LMCache config generation ─────────────────────────────────────────

write_lmcache_config() {
  local gds_path="$1"
  local output="$2"
  cat > "$output" <<EOF
chunk_size: 256
local_cpu: true
max_local_cpu_size: 5.0
gds_path: "$gds_path"
cufile_buffer_size: 512
EOF
}

write_lmcache_config_cpuonly() {
  local output="$1"
  cat > "$output" <<EOF
chunk_size: 256
local_cpu: true
max_local_cpu_size: 5.0
EOF
}

# ── Run a phase via process-compose ───────────────────────────────────

run_phase() {
  local phase="$1"
  local prompt_set="$2"  # warmup or consume or dev
  shift 2
  local gen_flags=("$@")  # --mount, --mount-overlay, etc.

  local prompt_dir="$SCRIPT_DIR/prompts/$prompt_set"
  local yaml="$LOG_DIR/process-compose-${phase}.yaml"

  log "====== Phase: $phase (profile: $PROFILE_NAME) ======"
  generate_opencode_json
  clear_conv_stats

  export PHASE="$phase"
  export CACHE_DIR="${CACHE_DIR_OVERRIDE:-$MOUNT_POINT}"

  bash "$SCRIPT_DIR/lib/generate-phase.sh" "${gen_flags[@]}" "$phase" "$prompt_dir" "$yaml"

  local t0 t1
  t0=$(date +%s)
  "$PC_BIN" up -f "$yaml" --tui=false
  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  log "Phase $phase complete. Elapsed: ${elapsed}s"

  # Update the summary with elapsed time (process-compose can't track this).
  local summary="$LOG_DIR/summary-${phase}.txt"
  if [ -f "$summary" ]; then
    sed -i "s/^elapsed=.*/elapsed=${elapsed}s/" "$summary" 2>/dev/null || true
    grep -q "^elapsed=" "$summary" || echo "elapsed=${elapsed}s" >> "$summary"
  fi
}

# ── Phase commands ────────────────────────────────────────────────────

cmd_baseline() {
  local cfg="$LOG_DIR/lmcache_config_baseline.yaml"
  write_lmcache_config_cpuonly "$cfg"
  export LMCACHE_CONFIG_FILE="$cfg"
  export CACHE_DIR_OVERRIDE="none"
  run_phase "baseline" "consume"
}

cmd_local_cold() {
  rm -rf "$LMCACHE_LOCAL_PATH"
  mkdir -p "$LMCACHE_LOCAL_PATH"
  local cfg="$LOG_DIR/lmcache_config_local.yaml"
  write_lmcache_config "$LMCACHE_LOCAL_PATH" "$cfg"
  export LMCACHE_CONFIG_FILE="$cfg"
  export CACHE_DIR_OVERRIDE="$LMCACHE_LOCAL_PATH"
  run_phase "local-cold" "consume"
}

cmd_local_warmup() {
  rm -rf "$LMCACHE_LOCAL_PATH"
  mkdir -p "$LMCACHE_LOCAL_PATH"
  local cfg="$LOG_DIR/lmcache_config_local.yaml"
  write_lmcache_config "$LMCACHE_LOCAL_PATH" "$cfg"
  export LMCACHE_CONFIG_FILE="$cfg"
  export CACHE_DIR_OVERRIDE="$LMCACHE_LOCAL_PATH"
  run_phase "local-warmup" "warmup"
}

cmd_local_warm() {
  local file_count
  file_count=$(cache_file_count "$LMCACHE_LOCAL_PATH")
  [ "$file_count" -eq 0 ] && die "No cache files in $LMCACHE_LOCAL_PATH — run 'local-warmup' first"
  local cfg="$LOG_DIR/lmcache_config_local.yaml"
  [ -f "$cfg" ] || write_lmcache_config "$LMCACHE_LOCAL_PATH" "$cfg"
  export LMCACHE_CONFIG_FILE="$cfg"
  export CACHE_DIR_OVERRIDE="$LMCACHE_LOCAL_PATH"
  run_phase "local-warm" "consume"
}

cmd_bucket_warmup() {
  mkdir -p "$LMCACHE_BUCKET_PATH" 2>/dev/null || true
  local cfg="$LOG_DIR/lmcache_config_bucket.yaml"
  write_lmcache_config "$LMCACHE_BUCKET_PATH" "$cfg"
  export LMCACHE_CONFIG_FILE="$cfg"
  export CACHE_DIR_OVERRIDE="$LMCACHE_BUCKET_PATH"
  run_phase "bucket-warmup" "warmup" --mount
}

cmd_bucket_rw() {
  local cfg="$LOG_DIR/lmcache_config_bucket.yaml"
  [ -f "$cfg" ] || write_lmcache_config "$LMCACHE_BUCKET_PATH" "$cfg"
  export LMCACHE_CONFIG_FILE="$cfg"
  export CACHE_DIR_OVERRIDE="$LMCACHE_BUCKET_PATH"
  run_phase "bucket-rw" "consume" --mount
}

cmd_bucket_overlay() {
  local cfg="$LOG_DIR/lmcache_config_bucket.yaml"
  [ -f "$cfg" ] || write_lmcache_config "$LMCACHE_BUCKET_PATH" "$cfg"
  export LMCACHE_CONFIG_FILE="$cfg"
  export CACHE_DIR_OVERRIDE="$LMCACHE_BUCKET_PATH"
  run_phase "bucket-overlay" "consume" --mount-overlay
}

# ── Utility commands ──────────────────────────────────────────────────

cmd_status() {
  local show_all=false
  [[ "${1:-}" == "--all" ]] && show_all=true

  echo "=== Results ==="
  local -a summary_dirs
  if $show_all; then
    for d in "$SCRIPT_DIR/logs"/*/; do
      [ -d "$d" ] && summary_dirs+=("$d")
    done
  else
    summary_dirs=("$LOG_DIR/")
  fi

  local found=false
  for dir in "${summary_dirs[@]}"; do
    local profile_label
    profile_label=$(basename "$dir")
    for summary in "$dir"summary-*.txt; do
      [ -f "$summary" ] || continue
      found=true
      local phase="" profile="" timestamp="" cache_dir="" cache_files="" cache_size=""
      local elapsed="" total_turns="" max_tokens="" num_requests=""
      local prompt_tokens="" prompt_tokens_cached="" generation_tokens=""
      local prefix_cache_queries="" prefix_cache_hits=""
      local external_cache_queries="" external_cache_hits=""
      local avg_first_ttft_ms="" avg_ttft_ms="" avg_e2e_ms=""
      eval "$(cat "$summary")"
      if $show_all; then
        echo "--- $profile_label / $phase ---"
      else
        echo "--- $phase ---"
      fi
      [ -n "$elapsed" ] && echo "  Elapsed:        $elapsed"
      [ -n "$num_requests" ] && echo "  Requests:       $num_requests"
      [ -n "$total_turns" ] && echo "  Turns:          $total_turns"
      [ -n "$max_tokens" ] && [ "$max_tokens" != "0" ] && echo "  Max tokens:     $max_tokens"
      [ -n "$prompt_tokens" ] && echo "  Prompt tokens:  $prompt_tokens"
      [ -n "$prompt_tokens_cached" ] && echo "  Cached tokens:  $prompt_tokens_cached"
      [ -n "$generation_tokens" ] && echo "  Gen tokens:     $generation_tokens"
      if [ -n "$prefix_cache_hits" ] && [ -n "$prefix_cache_queries" ] && [ "$prefix_cache_queries" != "0.0" ]; then
        echo "  Prefix cache:   $(awk "BEGIN {printf \"%.1f\", ($prefix_cache_hits / $prefix_cache_queries) * 100}")%"
      fi
      if [ -n "$external_cache_hits" ] && [ -n "$external_cache_queries" ] && [ "$external_cache_queries" != "0.0" ]; then
        echo "  External cache: $(awk "BEGIN {printf \"%.1f\", ($external_cache_hits / $external_cache_queries) * 100}")%"
      fi
      [ -n "$avg_first_ttft_ms" ] && echo "  1st turn TTFT:  ${avg_first_ttft_ms}ms (wall clock)"
      [ -n "$avg_ttft_ms" ] && echo "  Avg TTFT:       ${avg_ttft_ms}ms"
      [ -n "$avg_e2e_ms" ] && echo "  Avg e2e:        ${avg_e2e_ms}ms"
      [ "$cache_dir" != "none" ] && [ -n "$cache_files" ] && echo "  Cache files:    $cache_files ($cache_size)"
      echo ""
    done
  done
  $found || echo "(no summaries — run a phase first)"
}

cmd_teardown() {
  pkill -f "vllm serve.*$VLLM_PORT" 2>/dev/null || true
  pkill -f "hf-mount-nfs.*$MOUNT_POINT" 2>/dev/null || true
  pkill -f "opencode run" 2>/dev/null || true
  if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
    sudo umount "$MOUNT_POINT" 2>/dev/null || true
  fi
  log "Teardown complete."
}

cmd_clear_bucket() {
  log "Listing files in bucket $BUCKET..."
  local file_list
  file_list=$(uv run --with "huggingface_hub>=1.0" python -c "
from huggingface_hub import HfApi
api = HfApi()
files = [f.path for f in api.list_bucket_tree('$BUCKET', recursive=True) if hasattr(f, 'size')]
for f in files:
    print(f)
" 2>/dev/null)

  local count
  count=$(echo "$file_list" | grep -c '[^[:space:]]' 2>/dev/null || echo "0")
  if [ "$count" -eq 0 ]; then
    log "Bucket is already empty."
    return 0
  fi

  log "Found $count files in bucket $BUCKET"
  echo "$file_list" | head -10
  [ "$count" -gt 10 ] && echo "  ... and $((count - 10)) more"

  log "Deleting all files..."
  uv run --with "huggingface_hub>=1.0" python -c "
from huggingface_hub import HfApi
api = HfApi()
files = [f.path for f in api.list_bucket_tree('$BUCKET', recursive=True) if hasattr(f, 'size')]
if files:
    api.batch_bucket_files('$BUCKET', delete=files)
    print(f'Deleted {len(files)} files.')
"
  log "Bucket $BUCKET cleared."
}

# ── Batch commands ────────────────────────────────────────────────────

cmd_run_all() {
  for phase in baseline local-cold local-warmup local-warm bucket-warmup bucket-rw bucket-overlay; do
    "cmd_${phase//-/_}"
  done
  log "====== All phases complete (profile: $PROFILE_NAME) ======"
  cmd_status
}

cmd_run_suite() {
  local -a profiles=("$@")
  if [ ${#profiles[@]} -eq 0 ]; then
    for f in "$SCRIPT_DIR/profiles"/*.sh; do
      profiles+=("$(basename "$f" .sh)")
    done
  fi

  for profile in "${profiles[@]}"; do
    log "============================================"
    log "  SUITE: profile=$profile"
    log "============================================"
    "$SCRIPT_DIR/test-cache.sh" --profile "$profile" run-all
  done

  "$SCRIPT_DIR/test-cache.sh" status --all
}

# ── Main dispatch ───────────────────────────────────────────────────────

case "${1:-help}" in
  baseline)         cmd_baseline ;;
  local-cold)       cmd_local_cold ;;
  local-warmup)     cmd_local_warmup ;;
  local-warm)       cmd_local_warm ;;
  bucket-warmup)    cmd_bucket_warmup ;;
  bucket-rw)        cmd_bucket_rw ;;
  bucket-overlay)   cmd_bucket_overlay ;;
  status)           shift; cmd_status "$@" ;;
  teardown)         cmd_teardown ;;
  clear-bucket)     cmd_clear_bucket ;;
  run-all)          cmd_run_all ;;
  run-suite)        shift; cmd_run_suite "$@" ;;
  help|--help|-h|*)
    cat <<EOF
Usage: $(basename "$0") [--profile <name>] <command>

Profiles (default: qwen2.5-7b-tp1):
$(for f in "$SCRIPT_DIR/profiles"/*.sh; do
    name=$(basename "$f" .sh)
    desc=$(head -1 "$f" | sed 's/^# *//')
    printf "  %-14s %s\n" "$name" "$desc"
  done)

Phases (6 parallel conversations each, orchestrated by process-compose):

  baseline         CPU-only LMCache, no disk. Pure prefix cache reference.
  local-cold       No mount. Cold local disk reference timing.
  local-warmup     No mount. Populate local disk cache (warmup prompts).
  local-warm       No mount. Consume from warm local disk cache.

  bucket-warmup    Read-write mount. Populate cache in the bucket.
  bucket-rw        Read-write mount. Consume cache from the bucket.
  bucket-overlay   Overlay mount. Consume cache from the bucket.

Batch:
  run-all          Run all 7 phases for the current profile.
  run-suite [p..]  Run all phases for each profile (default: all profiles).

Utilities:
  status [--all]   Show results for current profile (or all profiles).
  teardown         Kill all vLLM, hf-mount, and opencode processes.
  clear-bucket     Delete all files from the HF bucket.

Tip: attach to a running phase with: process-compose attach

Environment:
  PROFILE        Profile name              (default: qwen2.5-7b-tp1)
  MOUNT_POINT    Mount directory           (default: /tmp/hf-mount-lmcache)
  BUCKET         HuggingFace bucket ID     (default: dacorvo/lm-cache)
  VLLM_PORT      vLLM API port             (default: 8000)
  LOG_DIR        Log directory             (default: lmcache/logs/<profile>)
EOF
    ;;
esac
