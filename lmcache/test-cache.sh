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
# Each phase runs three independent multi-turn conversations (5 turns each)
# on different topics to exercise varied KV cache prefixes.
#
# Usage:
#   ./test-cache.sh <command>
#
set -euo pipefail

cleanup() {
  echo ""
  echo "[$(date '+%H:%M:%S')] ==> Caught signal — cleaning up..."
  stop_vllm
  stop_hf_mount
  exit 130
}
trap cleanup INT TERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../hf-mount" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/../.venv}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -d "$VENV_DIR" ]] || { echo "ERROR: Venv not found at $VENV_DIR — run the root setup.sh first" >&2; exit 1; }
  source "$VENV_DIR/bin/activate"
fi
MOUNT_POINT="${MOUNT_POINT:-/tmp/hf-mount-lmcache}"
BUCKET="${BUCKET:-dacorvo/lm-cache}"
CACHE_DIR="${CACHE_DIR:-/tmp/hf-mount-cache}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
VLLM_URL="http://localhost:$VLLM_PORT"

if [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
fi

HF_MOUNT_BIN="${HF_MOUNT_BIN:-$REPO_ROOT/target/release/hf-mount-nfs}"

# ── Low-level helpers ──────────────────────────────────────────────────

# Ensure LOG_DIR exists for all helpers that write to it.
mkdir -p "$LOG_DIR"

SESSION_LOG="$LOG_DIR/session.log"

log() {
  local msg="[$(date '+%H:%M:%S')] ==> $*"
  echo "$msg"
  echo "$msg" >> "$SESSION_LOG"
}
die() {
  log "ERROR: $*"
  exit 1
}

cache_file_count() {
  find "$MOUNT_POINT" -name "*.kvcache.safetensors" -type f 2>/dev/null | wc -l
}

cache_total_size() {
  du -sh "$MOUNT_POINT" 2>/dev/null | cut -f1
}

# Query a single prometheus metric from vLLM's /metrics endpoint.
# Returns the numeric value or "n/a" if not found.
prom_metric() {
  local name="$1"
  curl -sf "$VLLM_URL/metrics" 2>/dev/null \
    | grep "^${name} " \
    | awk '{print $2}' \
    || echo "n/a"
}

# Print LMCache hit/miss summary from prometheus metrics.
print_cache_stats() {
  local label="${1:-}"
  local retrieve_hit_rate store_reqs retrieve_reqs hit_tokens
  retrieve_hit_rate=$(prom_metric "lmcache:retrieve_hit_rate")
  lookup_hit_rate=$(prom_metric "lmcache:lookup_hit_rate")
  hit_tokens=$(prom_metric "lmcache:num_hit_tokens")
  store_reqs=$(prom_metric "lmcache:num_store_requests")
  retrieve_reqs=$(prom_metric "lmcache:num_retrieve_requests")

  log "--- LMCache stats${label:+ ($label)} ---"
  log "  Retrieve hit rate: $retrieve_hit_rate"
  log "  Lookup hit rate:   $lookup_hit_rate"
  log "  Hit tokens:        $hit_tokens"
  log "  Store requests:    $store_reqs"
  log "  Retrieve requests: $retrieve_reqs"

  # Also extract per-request hits from the vLLM log (most recent entries).
  if [ -f "$LOG_DIR/vllm.log" ]; then
    local hits
    hits=$(grep -c "LMCache hit tokens" "$LOG_DIR/vllm.log" 2>/dev/null || echo "0")
    log "  Log lines with hit info: $hits"
  fi
}

save_summary() {
  local phase="$1"
  local elapsed="${2:-}"
  local cache_dir="${3:-$MOUNT_POINT}"
  local summary="$LOG_DIR/summary-${phase}.txt"
  local pt_count pt_size

  pt_count=$(find "$cache_dir" -name "*.kvcache.safetensors" -type f 2>/dev/null | wc -l)
  pt_size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)

  # Filter vLLM log by current PID to avoid mixing stats across phases.
  local vllm_pid=""
  [ -f "$LOG_DIR/vllm.pid" ] && vllm_pid=$(cat "$LOG_DIR/vllm.pid")

  {
    echo "phase=$phase"
    echo "timestamp='$(date '+%Y-%m-%d %H:%M:%S')'"
    echo "cache_dir=$cache_dir"
    echo "cache_files=$pt_count"
    echo "cache_size=${pt_size:-0}"
    [ -n "$elapsed" ] && echo "elapsed=${elapsed}s"
    # Throughput from vLLM engine stats (filtered by current PID)
    if [ -f "$LOG_DIR/vllm.log" ] && [ -n "$vllm_pid" ]; then
      grep "pid=$vllm_pid" "$LOG_DIR/vllm.log" \
        | grep "Avg prompt throughput" \
        | sed 's/.*prompt throughput: \([0-9.]*\) tokens\/s, Avg generation throughput: \([0-9.]*\) tokens\/s.*Prefix cache hit rate: \([0-9.]*\)%, External prefix cache hit rate: \([0-9.]*\)%.*/\1 \2 \3 \4/' \
        | awk '{tp+=$1; tg+=$2; hr=$3; ehr=$4; n++}
          END {
            if (n>0) {
              printf "avg_prompt_tps=%.1f\n", tp/n
              printf "avg_generation_tps=%.1f\n", tg/n
              printf "prefix_cache_hit=%.1f\n", hr
              printf "external_cache_hit=%.1f\n", ehr
            }
          }'
    fi
  } > "$summary"
  log "Summary saved to $summary"
}

stop_vllm() {
  if [ -f "$LOG_DIR/vllm.pid" ]; then
    kill "$(cat "$LOG_DIR/vllm.pid")" 2>/dev/null && log "vLLM stopped" || log "vLLM already stopped"
    rm -f "$LOG_DIR/vllm.pid"
    sleep 3
  fi
}

stop_hf_mount() {
  if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
    sudo umount "$MOUNT_POINT" 2>/dev/null || true
  fi
  if [ -f "$LOG_DIR/hf-mount.pid" ]; then
    local pid
    pid="$(cat "$LOG_DIR/hf-mount.pid")"
    for _ in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill "$pid" 2>/dev/null || true
    rm -f "$LOG_DIR/hf-mount.pid"
  fi
  log "hf-mount stopped"
}

start_hf_mount() {
  [ -x "$HF_MOUNT_BIN" ] || die "Binary not found: $HF_MOUNT_BIN"
  [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set"

  mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$LOG_DIR"

  if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
    sudo umount "$MOUNT_POINT" 2>/dev/null || true
    sleep 1
  fi
  if [ -f "$LOG_DIR/hf-mount.pid" ]; then
    kill "$(cat "$LOG_DIR/hf-mount.pid")" 2>/dev/null || true
    rm -f "$LOG_DIR/hf-mount.pid"
  fi

  log "Mounting $BUCKET at $MOUNT_POINT (flags: ${*:-(none)})"
  RUST_LOG=hf_mount=info \
    "$HF_MOUNT_BIN" \
    --hf-token "$HF_TOKEN" \
    --cache-dir "$CACHE_DIR" \
    "$@" \
    bucket "$BUCKET" "$MOUNT_POINT" \
    > >(tee -a "$LOG_DIR/hf-mount.log") 2>&1 &

  local pid=$!
  echo "$pid" > "$LOG_DIR/hf-mount.pid"

  for i in $(seq 1 30); do
    if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
      log "Mount ready after ${i}s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      cat "$LOG_DIR/hf-mount.log"
      die "hf-mount exited unexpectedly"
    fi
    sleep 1
  done
  die "Mount not ready after 30s"
}

start_vllm() {
  local lmcache_cfg="${1:-$LOG_DIR/lmcache_config_bucket.yaml}"
  [ -f "$lmcache_cfg" ] || lmcache_cfg="$SCRIPT_DIR/lmcache_config_bucket.yaml"

  log "Starting vLLM on port $VLLM_PORT (model: $MODEL)"
  LMCACHE_CONFIG_FILE="$lmcache_cfg" \
  PYTHONHASHSEED=0 \
    vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > >(tee -a "$LOG_DIR/vllm.log") 2>&1 &

  local pid=$!
  echo "$pid" > "$LOG_DIR/vllm.pid"
  log "vLLM PID $pid (log: $LOG_DIR/vllm.log)"

  log "Waiting for vLLM to load model..."
  for i in $(seq 1 600); do
    if curl -sf "$VLLM_URL/v1/models" >/dev/null 2>&1; then
      log "vLLM ready after ${i}s"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -20 "$LOG_DIR/vllm.log"
      die "vLLM exited unexpectedly"
    fi
    if (( i % 30 == 0 )); then printf "  ... %ds\n" "$i"; fi
    sleep 1
  done
  die "vLLM not ready after 600s"
}

require_mount() {
  grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null \
    || die "Nothing mounted at $MOUNT_POINT — run 'bucket-warmup' first"
}

require_vllm() {
  curl -sf "$VLLM_URL/v1/models" >/dev/null 2>&1 \
    || die "vLLM not responding on $VLLM_URL — run 'bucket-warmup' first"
}

# ── Conversation runner ────────────────────────────────────────────────
#
# run_conversation <label> <prompt_1> <prompt_2> ... <prompt_N>
#
# Starts a new opencode session with prompt_1, then continues with
# prompt_2..N via `opencode run -c`. Reports per-turn cache file counts
# when a mount is active.
#
run_conversation() {
  local label="$1"; shift
  local turn=1
  local total=$#
  local conv_log="$LOG_DIR/conversation-${label}.log"

  log "--- Conversation: $label ($total turns) --- (log: $conv_log)"
  echo ""

  # First turn: new session.
  local prompt="$1"; shift
  log "  [$label] Turn $turn/$total"
  log "  Prompt: $prompt"
  (cd "$SCRIPT_DIR" && opencode run "$prompt") 2>&1 | tee -a "$conv_log"
  echo ""
  if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
    log "  Cache files: $(cache_file_count)"
  fi
  turn=$((turn + 1))

  # Remaining turns: continue session.
  for prompt in "$@"; do
    log "  [$label] Turn $turn/$total"
    log "  Continue: $prompt"
    (cd "$SCRIPT_DIR" && opencode run -c "$prompt") 2>&1 | tee -a "$conv_log"
    echo ""
    if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
      log "  Cache files: $(cache_file_count)"
    fi
    turn=$((turn + 1))
  done

  log "--- $label complete ---"
  echo ""
}

# ── Conversation definitions ───────────────────────────────────────────
#
# Two sets of conversations, each with 3 topics x 5 turns.
# Warmup and consume use different prompts so that consume does not get
# artificially high cache hit rates from identical token sequences.
# Cache hits in consume come from shared system-prompt prefixes and
# overlapping file reads, which is the realistic scenario.

conversations_warmup() {
  run_conversation "overlay-impl" \
    "Read src/setup.rs and explain what the --overlay flag does" \
    "How does overlay mode interact with the advanced_writes flag?" \
    "Now read src/virtual_fs/mod.rs and describe the overlay_root helper" \
    "Explain the staging_path method and how it differs in overlay mode" \
    "Summarize the full overlay data flow: from CLI flag to file read/write"

  run_conversation "test-coverage" \
    "Read src/virtual_fs/tests.rs and list all test functions related to overlay" \
    "Pick the three most important overlay tests and explain what each one verifies" \
    "Are there any edge cases in the overlay code that lack test coverage?" \
    "Read src/test_mocks.rs and explain how the mock filesystem supports overlay testing" \
    "Suggest one new integration test that would improve confidence in overlay mode"

  run_conversation "documentation" \
    "Read the README.md and summarize what it says about mounting buckets" \
    "Does the README mention the --overlay flag? What is missing?" \
    "Read src/setup.rs and draft a short usage section for overlay mode" \
    "What caveats should the documentation mention about overlay writes and persistence?" \
    "Write a concise FAQ entry: when should a user choose --overlay vs --read-only?"
}

conversations_consume() {
  run_conversation "nfs-backend" \
    "Read src/bin/hf-mount-nfs.rs and explain how the NFS server is started" \
    "How does NFS mode differ from FUSE mode in this project?" \
    "Read src/setup.rs and describe the is_nfs code paths" \
    "What NFS-specific limitations should a user be aware of?" \
    "Summarize the trade-offs between NFS and FUSE for overlay mounts"

  run_conversation "write-pipeline" \
    "Read src/xet.rs and explain the StagingDir and XetSessions structures" \
    "How does the flush manager decide when to upload staged files?" \
    "Read src/virtual_fs/mod.rs and trace what happens when a file is created and written" \
    "How does the write pipeline change when overlay mode is active?" \
    "What happens to staged files when the mount is unmounted gracefully?"

  run_conversation "error-handling" \
    "Read src/virtual_fs/mod.rs and find all places that return EPERM" \
    "Which operations are blocked in overlay mode and why?" \
    "What happens if a user tries to delete a remote file through an overlay mount?" \
    "Read the rename implementation and explain how overlay guards work" \
    "Suggest improvements to the error messages for overlay permission denials"
}

# ── bucket-warmup (read-write mount, populate bucket) ──────────────────

cmd_bucket_warmup() {
  # ── Step 1: Mount read-write and start vLLM ────────────────────────
  log "Stopping any running services..."
  stop_vllm
  stop_hf_mount

  log "Mounting bucket read-write..."
  start_hf_mount  # no --overlay → read-write

  # Generate LMCache config pointing at the mount point.
  local lmcache_cfg="$LOG_DIR/lmcache_config_bucket.yaml"
  log "Writing LMCache config to $lmcache_cfg"
  cat > "$lmcache_cfg" <<EOF
chunk_size: 256
local_cpu: true
max_local_cpu_size: 5.0
gds_path: "$MOUNT_POINT"
cufile_buffer_size: 512
EOF

  start_vllm "$lmcache_cfg"

  # ── Step 2: Run conversations ──────────────────────────────────────
  local before
  before=$(cache_file_count)
  log "Cache files before warmup: $before"
  echo ""

  conversations_warmup

  local after
  after=$(cache_file_count)
  log "Warmup complete: $before -> $after cache files ($(cache_total_size) on mount)"
  if [ "$after" -gt "$before" ]; then
    log "SUCCESS: new KV cache chunks written through read-write mount"
    log "         hf-mount will upload them to the bucket on flush/unmount."
  else
    log "WARNING: no new cache files appeared — check $LOG_DIR/vllm.log for LMCache errors"
  fi
  print_cache_stats "bucket-warmup"
  save_summary "bucket-warmup"

  stop_vllm
  stop_hf_mount
}

# ── bucket-rw (read-write mount, consume from bucket) ──────────────────

cmd_bucket_rw() {
  # ── Step 1: Tear down the warmup phase ─────────────────────────────
  log "Stopping previous phase..."
  stop_vllm

  log "Unmounting (flushing uploads to bucket)..."
  stop_hf_mount

  # ── Step 2: Remount read-write (fresh start) ──────────────────────
  log "Remounting read-write..."
  start_hf_mount  # no --overlay → read-write

  local file_count
  file_count=$(cache_file_count)
  log "Cache files visible from bucket: $file_count"
  if [ "$file_count" -eq 0 ]; then
    die "No cache files visible — upload may have failed (check $LOG_DIR/hf-mount.log)"
  fi

  # ── Step 3: Start vLLM and run conversations ───────────────────────
  local lmcache_cfg="$LOG_DIR/lmcache_config_bucket.yaml"
  [ -f "$lmcache_cfg" ] || die "LMCache config not found — run 'bucket-warmup' first"
  start_vllm "$lmcache_cfg"

  echo ""
  log "Running conversations (read-write mount — shared cache)..."
  echo ""

  local t0 t1
  t0=$(date +%s)

  conversations_consume

  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  echo ""
  log "Bucket-rw complete."
  log "  Elapsed: ${elapsed}s"
  print_cache_stats "bucket-rw"
  save_summary "bucket-rw" "$elapsed"
  log "Compare with: ./test-cache.sh local-cold"

  stop_vllm
  stop_hf_mount
}

# ── bucket-overlay (overlay mount, consume from bucket) ────────────────

cmd_bucket_overlay() {
  # ── Step 1: Tear down previous phase ───────────────────────────────
  log "Stopping previous phase..."
  stop_vllm

  log "Unmounting (flushing uploads to bucket)..."
  stop_hf_mount

  # ── Step 2: Remount with --overlay ─────────────────────────────────
  log "Remounting with --overlay..."
  start_hf_mount --overlay

  local file_count
  file_count=$(cache_file_count)
  log "Cache files visible via overlay: $file_count"
  if [ "$file_count" -eq 0 ]; then
    die "No cache files visible — upload may have failed (check $LOG_DIR/hf-mount.log)"
  fi

  # ── Step 3: Start vLLM and run conversations ───────────────────────
  local lmcache_cfg="$LOG_DIR/lmcache_config_bucket.yaml"
  [ -f "$lmcache_cfg" ] || die "LMCache config not found — run 'bucket-warmup' first"
  start_vllm "$lmcache_cfg"

  echo ""
  log "Running conversations (overlay mount — cache from bucket, writes stay local)..."
  echo ""

  local t0 t1
  t0=$(date +%s)

  conversations_consume

  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  echo ""
  log "Bucket-overlay complete — new writes stayed local."
  log "  Elapsed: ${elapsed}s"
  print_cache_stats "bucket-overlay"
  save_summary "bucket-overlay" "$elapsed"
  log "Compare with: ./test-cache.sh local-cold"

  stop_vllm
  stop_hf_mount
}

# ── local-cold (no mount, cold local disk) ─────────────────────────────

cmd_local_cold() {
  LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/hf-mount-local-cache}"

  log "Stopping any running services..."
  stop_vllm
  stop_hf_mount

  log "Preparing local cache directory $LOCAL_CACHE_DIR (no mount)..."
  rm -rf "$LOCAL_CACHE_DIR"
  mkdir -p "$LOCAL_CACHE_DIR" "$LOG_DIR"

  local lmcache_cfg="$LOG_DIR/lmcache_config_local.yaml"
  cat > "$lmcache_cfg" <<EOF
chunk_size: 256
local_cpu: true
max_local_cpu_size: 5.0
gds_path: "$LOCAL_CACHE_DIR"
cufile_buffer_size: 512
EOF

  start_vllm "$lmcache_cfg"

  echo ""
  log "Running conversations (cold local disk, no mount)..."
  echo ""

  local t0 t1
  t0=$(date +%s)

  conversations_consume

  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))
  local pt_count
  pt_count=$(find "$LOCAL_CACHE_DIR" -name "*.kvcache.safetensors" -type f 2>/dev/null | wc -l)

  echo ""
  log "Local-cold complete."
  log "  Elapsed:     ${elapsed}s"
  log "  Cache files: $pt_count cache files in $LOCAL_CACHE_DIR"
  print_cache_stats "local-cold"
  save_summary "local-cold" "$elapsed" "$LOCAL_CACHE_DIR"
  log "Compare with: ./test-cache.sh bucket-rw"

  stop_vllm
  stop_hf_mount
}

# ── local-warmup (no mount, populate local disk cache) ─────────────────

cmd_local_warmup() {
  LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/hf-mount-local-cache}"

  log "Stopping any running services..."
  stop_vllm
  stop_hf_mount

  log "Preparing local warmup directory $LOCAL_CACHE_DIR (no mount)..."
  rm -rf "$LOCAL_CACHE_DIR"
  mkdir -p "$LOCAL_CACHE_DIR" "$LOG_DIR"

  local lmcache_cfg="$LOG_DIR/lmcache_config_local.yaml"
  cat > "$lmcache_cfg" <<EOF
chunk_size: 256
local_cpu: true
max_local_cpu_size: 5.0
gds_path: "$LOCAL_CACHE_DIR"
cufile_buffer_size: 512
EOF

  start_vllm "$lmcache_cfg"

  echo ""
  log "Running warmup conversations (plain local disk, no mount)..."
  echo ""

  local t0 t1
  t0=$(date +%s)

  conversations_warmup

  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))
  local pt_count
  pt_count=$(find "$LOCAL_CACHE_DIR" -name "*.kvcache.safetensors" -type f 2>/dev/null | wc -l)

  echo ""
  log "Local-warmup complete."
  log "  Elapsed:     ${elapsed}s"
  log "  Cache files: $pt_count cache files in $LOCAL_CACHE_DIR"
  print_cache_stats "local-warmup"
  save_summary "local-warmup" "$elapsed" "$LOCAL_CACHE_DIR"

  stop_vllm
  stop_hf_mount
}

# ── local-warm (no mount, consume from warm local disk) ────────────────

cmd_local_warm() {
  LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/hf-mount-local-cache}"

  log "Stopping any running services..."
  stop_vllm
  stop_hf_mount

  local pt_count
  pt_count=$(find "$LOCAL_CACHE_DIR" -name "*.kvcache.safetensors" -type f 2>/dev/null | wc -l)
  if [ "$pt_count" -eq 0 ]; then
    die "No cache files in $LOCAL_CACHE_DIR — run 'local-warmup' first"
  fi
  log "Found $pt_count cache files in $LOCAL_CACHE_DIR"

  local lmcache_cfg="$LOG_DIR/lmcache_config_local.yaml"
  [ -f "$lmcache_cfg" ] || die "LMCache config not found — run 'local-warmup' first"
  start_vllm "$lmcache_cfg"

  echo ""
  log "Running conversations (warm local disk, no mount)..."
  echo ""

  local t0 t1
  t0=$(date +%s)

  conversations_consume

  t1=$(date +%s)
  local elapsed=$(( t1 - t0 ))

  echo ""
  log "Local-warm complete."
  log "  Elapsed: ${elapsed}s"
  print_cache_stats "local-warm"
  save_summary "local-warm" "$elapsed" "$LOCAL_CACHE_DIR"

  stop_vllm
  stop_hf_mount
}

# ── status ──────────────────────────────────────────────────────────────

cmd_status() {
  local summary="$LOG_DIR/summary.txt"

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

  echo "=== Results ==="
  local found=false
  for summary in "$LOG_DIR"/summary-*.txt; do
    [ -f "$summary" ] || continue
    found=true
    local phase="" timestamp="" cache_dir="" cache_files="" cache_size=""
    local elapsed="" avg_prompt_tps="" avg_generation_tps=""
    local prefix_cache_hit="" external_cache_hit=""
    eval "$(cat "$summary")"
    echo "--- $phase ---"
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
  $found || echo "(no summaries — run a phase first)"
}

# ── logs ────────────────────────────────────────────────────────────────

cmd_logs() {
  local target="${2:-all}"
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

# ── teardown ────────────────────────────────────────────────────────────

cmd_teardown() {
  stop_vllm
  stop_hf_mount

  echo ""
  log "Teardown complete."
  local count
  count=$(find "$MOUNT_POINT" -name "*.kvcache.safetensors" -type f 2>/dev/null | wc -l)
  if [ "$count" -gt 0 ]; then
    log "Note: $count cache files remain in $MOUNT_POINT"
  fi
}

# ── clear-bucket ───────────────────────────────────────────────────────

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
  echo ""

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

# ── Main dispatch ───────────────────────────────────────────────────────

case "${1:-help}" in
  local-cold)       cmd_local_cold ;;
  local-warmup)     cmd_local_warmup ;;
  local-warm)       cmd_local_warm ;;
  bucket-warmup)    cmd_bucket_warmup ;;
  bucket-rw)        cmd_bucket_rw ;;
  bucket-overlay)   cmd_bucket_overlay ;;
  status)           cmd_status ;;
  logs)             cmd_logs "$@" ;;
  teardown)         cmd_teardown ;;
  clear-bucket)     cmd_clear_bucket ;;
  help|--help|-h|*)
    cat <<EOF
Usage: $(basename "$0") <command>

Each phase runs 3 multi-turn conversations (5 turns each) on different
topics. Warmup and consume use different conversations to avoid biased
cache hit rates.

  local-cold       No mount. Cold local disk reference timing.
  local-warmup     No mount. Populate local disk cache (warmup prompts).
  local-warm       No mount. Consume from warm local disk cache.

  bucket-warmup    Read-write mount. Populate cache in the bucket.
  bucket-rw        Read-write mount. Consume cache from the bucket.
  bucket-overlay   Overlay mount. Consume cache from the bucket.

Utilities:
  status        Show process state and last run summary.
  logs [vllm|mount|all]   Tail log files (default: all).
  teardown      Stop vLLM, unmount, stop hf-mount.
  clear-bucket  Delete all files from the HF bucket.

Environment:
  MOUNT_POINT    Mount directory           (default: /tmp/hf-mount-lmcache)
  BUCKET         HuggingFace bucket ID     (default: dacorvo/lm-cache)
  LOCAL_CACHE_DIR   Local cache directory     (default: /tmp/hf-mount-local-cache)
  VLLM_PORT      vLLM API port             (default: 8000)
  MODEL          HuggingFace model ID      (default: Qwen/Qwen2.5-Coder-7B-Instruct)
  LOG_DIR        Log directory             (default: lmcache/logs/)
EOF
    ;;
esac
