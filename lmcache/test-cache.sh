#!/usr/bin/env bash
#
# CLI for LMCache + vLLM cache integration tests.
#
# Phases:
#   local-cold       — no mount, cold local disk (reference timing)
#   local-warmup     — no mount, populate local disk cache
#   local-warm       — no mount, consume from warm local disk
#   bucket-warmup    — read-write mount, populate cache in bucket
#   bucket-rw        — read-write mount, consume cache from bucket
#   bucket-overlay   — overlay mount, consume cache from bucket
#
# Usage:
#   ./test-cache.sh [--profile <name>] <command>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../hf-mount" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/../.venv}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -d "$VENV_DIR" ]] || { echo "ERROR: Venv not found at $VENV_DIR — run the root setup.sh first" >&2; exit 1; }
  source "$VENV_DIR/bin/activate"
fi

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

MOUNT_POINT="${MOUNT_POINT:-/tmp/hf-mount-lmcache}"
BUCKET="${BUCKET:-dacorvo/lm-cache}"
CACHE_DIR="${CACHE_DIR:-/tmp/hf-mount-cache-${PROFILE_NAME}}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/hf-mount-local-cache-${PROFILE_NAME}}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs/$PROFILE_NAME}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="http://localhost:$VLLM_PORT"
HF_MOUNT_BIN="${HF_MOUNT_BIN:-$REPO_ROOT/target/release/hf-mount-nfs}"

# LMCache gds_path is namespaced by profile within the mount/local dir.
LMCACHE_BUCKET_PATH="$MOUNT_POINT/$PROFILE_NAME"
LMCACHE_LOCAL_PATH="$LOCAL_CACHE_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
fi

# ── Source libraries ──────────────────────────────────────────────────

mkdir -p "$LOG_DIR"
SESSION_LOG="$LOG_DIR/session.log"

source "$SCRIPT_DIR/lib/helpers.sh"
source "$SCRIPT_DIR/lib/vllm.sh"
source "$SCRIPT_DIR/lib/hf-mount.sh"
source "$SCRIPT_DIR/lib/conversations.sh"

# ── Signal handler ────────────────────────────────────────────────────

cleanup() {
  echo ""
  echo "[$(date '+%H:%M:%S')] ==> Caught signal — cleaning up..."
  stop_vllm
  stop_hf_mount
  exit 130
}
trap cleanup INT TERM

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
  log "LMCache config: $output (gds_path=$gds_path)"
}

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
  log "Generated opencode.json for $MODEL (context=$MAX_MODEL_LEN)"
}

# ── Phase commands ────────────────────────────────────────────────────

cmd_bucket_warmup() {
  log "=== bucket-warmup (profile: $PROFILE_NAME) ==="
  stop_vllm
  stop_hf_mount

  start_hf_mount
  mkdir -p "$LMCACHE_BUCKET_PATH"

  local lmcache_cfg="$LOG_DIR/lmcache_config_bucket.yaml"
  write_lmcache_config "$LMCACHE_BUCKET_PATH" "$lmcache_cfg"
  start_vllm "$lmcache_cfg"
  generate_opencode_json

  local before
  before=$(cache_file_count "$LMCACHE_BUCKET_PATH")
  log "Cache files before warmup: $before"

  local t0 t1
  t0=$(date +%s)
  conversations_warmup
  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  local after
  after=$(cache_file_count "$LMCACHE_BUCKET_PATH")
  log "Warmup complete: $before -> $after cache files ($(cache_total_size "$LMCACHE_BUCKET_PATH") on mount)"
  log "  Elapsed: ${elapsed}s"
  if [ "$after" -gt "$before" ]; then
    log "SUCCESS: new KV cache chunks written through read-write mount"
  else
    log "WARNING: no new cache files appeared — check $LOG_DIR/vllm.log"
  fi
  print_cache_stats "bucket-warmup"
  save_summary "bucket-warmup" "$elapsed" "$LMCACHE_BUCKET_PATH"

  stop_vllm
  stop_hf_mount
}

cmd_bucket_rw() {
  log "=== bucket-rw (profile: $PROFILE_NAME) ==="
  stop_vllm
  stop_hf_mount

  start_hf_mount
  mkdir -p "$LMCACHE_BUCKET_PATH"

  local file_count
  file_count=$(cache_file_count "$LMCACHE_BUCKET_PATH")
  log "Cache files visible from bucket: $file_count"
  [ "$file_count" -eq 0 ] && die "No cache files visible — run 'bucket-warmup' first"

  local lmcache_cfg="$LOG_DIR/lmcache_config_bucket.yaml"
  [ -f "$lmcache_cfg" ] || write_lmcache_config "$LMCACHE_BUCKET_PATH" "$lmcache_cfg"
  start_vllm "$lmcache_cfg"
  generate_opencode_json

  local t0 t1
  t0=$(date +%s)
  conversations_consume
  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  log "Bucket-rw complete. Elapsed: ${elapsed}s"
  print_cache_stats "bucket-rw"
  save_summary "bucket-rw" "$elapsed" "$LMCACHE_BUCKET_PATH"

  stop_vllm
  stop_hf_mount
}

cmd_bucket_overlay() {
  log "=== bucket-overlay (profile: $PROFILE_NAME) ==="
  stop_vllm
  stop_hf_mount

  start_hf_mount --overlay
  mkdir -p "$LMCACHE_BUCKET_PATH"

  local file_count
  file_count=$(cache_file_count "$LMCACHE_BUCKET_PATH")
  log "Cache files visible via overlay: $file_count"
  [ "$file_count" -eq 0 ] && die "No cache files visible — run 'bucket-warmup' first"

  local lmcache_cfg="$LOG_DIR/lmcache_config_bucket.yaml"
  [ -f "$lmcache_cfg" ] || write_lmcache_config "$LMCACHE_BUCKET_PATH" "$lmcache_cfg"
  start_vllm "$lmcache_cfg"
  generate_opencode_json

  local t0 t1
  t0=$(date +%s)
  conversations_consume
  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  log "Bucket-overlay complete — new writes stayed local. Elapsed: ${elapsed}s"
  print_cache_stats "bucket-overlay"
  save_summary "bucket-overlay" "$elapsed" "$LMCACHE_BUCKET_PATH"

  stop_vllm
  stop_hf_mount
}

cmd_baseline() {
  log "=== baseline (profile: $PROFILE_NAME) ==="
  stop_vllm
  stop_hf_mount

  local lmcache_cfg="$LOG_DIR/lmcache_config_baseline.yaml"
  cat > "$lmcache_cfg" <<EOF
chunk_size: 256
local_cpu: true
max_local_cpu_size: 5.0
EOF
  start_vllm "$lmcache_cfg"
  generate_opencode_json

  local t0 t1
  t0=$(date +%s)
  conversations_consume
  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  log "Baseline complete. Elapsed: ${elapsed}s"
  print_cache_stats "baseline"
  save_summary "baseline" "$elapsed"

  stop_vllm
  stop_hf_mount
}

cmd_local_cold() {
  log "=== local-cold (profile: $PROFILE_NAME) ==="
  stop_vllm
  stop_hf_mount

  rm -rf "$LMCACHE_LOCAL_PATH"
  mkdir -p "$LMCACHE_LOCAL_PATH" "$LOG_DIR"

  local lmcache_cfg="$LOG_DIR/lmcache_config_local.yaml"
  write_lmcache_config "$LMCACHE_LOCAL_PATH" "$lmcache_cfg"
  start_vllm "$lmcache_cfg"
  generate_opencode_json

  local t0 t1
  t0=$(date +%s)
  conversations_consume
  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  log "Local-cold complete. Elapsed: ${elapsed}s"
  log "  Cache files: $(cache_file_count "$LMCACHE_LOCAL_PATH") in $LMCACHE_LOCAL_PATH"
  print_cache_stats "local-cold"
  save_summary "local-cold" "$elapsed" "$LMCACHE_LOCAL_PATH"

  stop_vllm
  stop_hf_mount
}

cmd_local_warmup() {
  log "=== local-warmup (profile: $PROFILE_NAME) ==="
  stop_vllm
  stop_hf_mount

  rm -rf "$LMCACHE_LOCAL_PATH"
  mkdir -p "$LMCACHE_LOCAL_PATH" "$LOG_DIR"

  local lmcache_cfg="$LOG_DIR/lmcache_config_local.yaml"
  write_lmcache_config "$LMCACHE_LOCAL_PATH" "$lmcache_cfg"
  start_vllm "$lmcache_cfg"
  generate_opencode_json

  local t0 t1
  t0=$(date +%s)
  conversations_warmup
  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  log "Local-warmup complete. Elapsed: ${elapsed}s"
  log "  Cache files: $(cache_file_count "$LMCACHE_LOCAL_PATH") in $LMCACHE_LOCAL_PATH"
  print_cache_stats "local-warmup"
  save_summary "local-warmup" "$elapsed" "$LMCACHE_LOCAL_PATH"

  stop_vllm
  stop_hf_mount
}

cmd_local_warm() {
  log "=== local-warm (profile: $PROFILE_NAME) ==="
  stop_vllm
  stop_hf_mount

  local file_count
  file_count=$(cache_file_count "$LMCACHE_LOCAL_PATH")
  [ "$file_count" -eq 0 ] && die "No cache files in $LMCACHE_LOCAL_PATH — run 'local-warmup' first"
  log "Found $file_count cache files in $LMCACHE_LOCAL_PATH"

  local lmcache_cfg="$LOG_DIR/lmcache_config_local.yaml"
  [ -f "$lmcache_cfg" ] || die "LMCache config not found — run 'local-warmup' first"
  start_vllm "$lmcache_cfg"
  generate_opencode_json

  local t0 t1
  t0=$(date +%s)
  conversations_consume
  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  log "Local-warm complete. Elapsed: ${elapsed}s"
  print_cache_stats "local-warm"
  save_summary "local-warm" "$elapsed" "$LMCACHE_LOCAL_PATH"

  stop_vllm
  stop_hf_mount
}

# ── Utility commands ──────────────────────────────────────────────────

cmd_status() {
  local show_all=false
  [[ "${1:-}" == "--all" ]] && show_all=true

  echo "=== Processes ==="
  if [ -f "$LOG_DIR/vllm.pid" ] && kill -0 "$(cat "$LOG_DIR/vllm.pid")" 2>/dev/null; then
    echo "vLLM:     PID $(cat "$LOG_DIR/vllm.pid") (running)"
  else
    echo "vLLM:     not running"
  fi
  if [ -f "$LOG_DIR/hf-mount.pid" ] && kill -0 "$(cat "$LOG_DIR/hf-mount.pid")" 2>/dev/null; then
    echo "hf-mount: PID $(cat "$LOG_DIR/hf-mount.pid") (running)"
  else
    echo "hf-mount: not running"
  fi
  if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
    echo "Mount:    ACTIVE at $MOUNT_POINT"
  else
    echo "Mount:    not mounted"
  fi
  echo ""

  # Collect summary dirs to scan.
  local -a summary_dirs
  if $show_all; then
    for d in "$SCRIPT_DIR/logs"/*/; do
      [ -d "$d" ] && summary_dirs+=("$d")
    done
  else
    summary_dirs=("$LOG_DIR/")
  fi

  echo "=== Results ==="
  local found=false
  for dir in "${summary_dirs[@]}"; do
    local profile_label
    profile_label=$(basename "$dir")
    for summary in "$dir"summary-*.txt; do
      [ -f "$summary" ] || continue
      found=true
      local phase="" profile="" timestamp="" cache_dir="" cache_files="" cache_size=""
      local elapsed="" avg_prompt_tps="" avg_generation_tps=""
      local prefix_cache_hit="" external_cache_hit=""
      eval "$(cat "$summary")"
      if $show_all; then
        echo "--- $profile_label / $phase ---"
      else
        echo "--- $phase ---"
      fi
      echo "  Timestamp:      $timestamp"
      echo "  Cache dir:      $cache_dir"
      echo "  Cache files:    $cache_files cache files ($cache_size)"
      [ -n "$elapsed" ] && echo "  Elapsed:        $elapsed"
      [ -n "$avg_prompt_tps" ] && echo "  Prompt:         $avg_prompt_tps tok/s"
      [ -n "$avg_generation_tps" ] && echo "  Generation:     $avg_generation_tps tok/s"
      [ -n "$prefix_cache_hit" ] && echo "  Prefix cache:   ${prefix_cache_hit}%"
      [ -n "$external_cache_hit" ] && echo "  External cache: ${external_cache_hit}%"
      echo ""
    done
  done
  $found || echo "(no summaries — run a phase first)"
}

cmd_logs() {
  local target="${1:-all}"
  case "$target" in
    vllm)
      [ -f "$LOG_DIR/vllm.log" ] || die "No vLLM log at $LOG_DIR/vllm.log"
      tail -f "$LOG_DIR/vllm.log"
      ;;
    hf-mount|mount)
      [ -f "$LOG_DIR/hf-mount.log" ] || die "No hf-mount log at $LOG_DIR/hf-mount.log"
      tail -f "$LOG_DIR/hf-mount.log"
      ;;
    *)
      log "Tailing all logs (Ctrl-C to stop)..."
      tail -f "$LOG_DIR"/*.log 2>/dev/null || die "No log files in $LOG_DIR"
      ;;
  esac
}

cmd_teardown() {
  stop_vllm
  stop_hf_mount
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
    log "====== Phase: $phase (profile: $PROFILE_NAME) ======"
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
  logs)             shift; cmd_logs "$@" ;;
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

Phases (3 conversations each, adaptive turns until 90% context + 3 post-compaction):

  baseline         CPU-only LMCache, no disk. Pure prefix cache reference.
  local-cold       No mount. Cold local disk reference timing.
  local-warmup     No mount. Populate local disk cache (warmup prompts).
  local-warm       No mount. Consume from warm local disk cache.

  bucket-warmup    Read-write mount. Populate cache in the bucket.
  bucket-rw        Read-write mount. Consume cache from the bucket.
  bucket-overlay   Overlay mount. Consume cache from the bucket.

Batch:
  run-all          Run all 6 phases for the current profile.
  run-suite [p..]  Run all phases for each profile (default: all profiles).

Utilities:
  status [--all]   Show results for current profile (or all profiles).
  logs [vllm|mount|all]   Tail log files (default: all).
  teardown         Stop vLLM, unmount, stop hf-mount.
  clear-bucket     Delete all files from the HF bucket.

Environment:
  PROFILE        Profile name              (default: qwen2.5-7b-tp1)
  MOUNT_POINT    Mount directory           (default: /tmp/hf-mount-lmcache)
  BUCKET         HuggingFace bucket ID     (default: dacorvo/lm-cache)
  VLLM_PORT      vLLM API port             (default: 8000)
  LOG_DIR        Log directory             (default: lmcache/logs/<profile>)
EOF
    ;;
esac
