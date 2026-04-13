# helpers.sh — logging, metrics, and summary helpers
#
# Expected globals: LOG_DIR, SESSION_LOG, VLLM_URL, MOUNT_POINT

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
  local dir="${1:-$MOUNT_POINT}"
  find "$dir" -name "*.kvcache.safetensors" -type f 2>/dev/null | wc -l
}

cache_total_size() {
  local dir="${1:-$MOUNT_POINT}"
  du -sh "$dir" 2>/dev/null | cut -f1
}

# Query a single prometheus metric from vLLM's /metrics endpoint.
prom_metric() {
  local name="$1"
  curl -sf "$VLLM_URL/metrics" 2>/dev/null \
    | grep "^${name} " \
    | awk '{print $2}' \
    || echo "n/a"
}

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

  local file_count file_size
  file_count=$(cache_file_count "$cache_dir")
  file_size=$(cache_total_size "$cache_dir")

  # Filter vLLM log by current PID to avoid mixing stats across phases.
  local vllm_pid=""
  [ -f "$LOG_DIR/vllm.pid" ] && vllm_pid=$(cat "$LOG_DIR/vllm.pid")

  {
    echo "phase=$phase"
    echo "profile=${PROFILE_NAME:-unknown}"
    echo "timestamp='$(date '+%Y-%m-%d %H:%M:%S')'"
    echo "cache_dir=$cache_dir"
    echo "cache_files=$file_count"
    echo "cache_size=${file_size:-0}"
    [ -n "$elapsed" ] && echo "elapsed=${elapsed}s"
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
