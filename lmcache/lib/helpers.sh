# helpers.sh — logging, metrics, and summary helpers
#
# Expected globals: LOG_DIR, SESSION_LOG, VLLM_URL, MOUNT_POINT

log() {
  local msg="[$(date '+%H:%M:%S')] ==> $*"
  echo "$msg"
  [ -n "${SESSION_LOG:-}" ] && echo "$msg" >> "$SESSION_LOG" || true
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
    | grep "^${name}" \
    | awk '{print $NF}' \
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

clear_conv_stats() {
  rm -f "$LOG_DIR"/conv-stats-*-"${PHASE}".txt
}

save_summary() {
  local phase="$1"
  local elapsed="${2:-}"
  local cache_dir="${3:-$MOUNT_POINT}"
  local summary="$LOG_DIR/summary-${phase}.txt"

  local file_count=0 file_size="0"
  if [ "$cache_dir" != "none" ]; then
    file_count=$(cache_file_count "$cache_dir")
    file_size=$(cache_total_size "$cache_dir")
  fi

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
    # Aggregate per-conversation stats (turns, max_tokens, first-turn TTFT).
    local total_turns=0 overall_max_tokens=0
    local ttft_sum_first=0 ttft_count_first=0
    for sf in "$LOG_DIR"/conv-stats-*-"${phase}".txt; do
      [ -f "$sf" ] || continue
      local turns=0 max_tokens=0 first_turn_ttft_ms=""
      eval "$(cat "$sf")"
      total_turns=$((total_turns + turns))
      [ "$max_tokens" -gt "$overall_max_tokens" ] && overall_max_tokens="$max_tokens"
      if [ -n "$first_turn_ttft_ms" ] && [ "$first_turn_ttft_ms" != "n/a" ]; then
        ttft_sum_first=$((ttft_sum_first + first_turn_ttft_ms))
        ttft_count_first=$((ttft_count_first + 1))
      fi
    done
    echo "total_turns=$total_turns"
    # "Total tokens" is logged by EngineCore, not APIServer. Find the
    # EngineCore PID that corresponds to our APIServer session.
    local engine_pid=""
    local vllm_log="$LOG_DIR/vllm-${phase}.log"
    if [ -f "$vllm_log" ] && [ -n "$vllm_pid" ]; then
      engine_pid=$(grep -o "EngineCore pid=[0-9]*" "$vllm_log" \
        | tail -1 | grep -o "[0-9]*")
    fi
    if [ -n "$engine_pid" ]; then
      overall_max_tokens=$(grep "pid=$engine_pid" "$vllm_log" \
        | grep -o "Total tokens [0-9]*" \
        | awk '{if($3>m)m=$3} END{print m+0}')
    fi
    echo "max_tokens=$overall_max_tokens"
    if [ "$ttft_count_first" -gt 0 ]; then
      echo "avg_first_ttft_ms=$((ttft_sum_first / ttft_count_first))"
    fi
    # ── Prometheus metrics (must be captured before vLLM stops) ──
    local pm
    # Tokens
    pm=$(prom_metric "vllm:prompt_tokens_total"); [ "$pm" != "n/a" ] && echo "prompt_tokens=$pm"
    pm=$(prom_metric "vllm:prompt_tokens_cached_total"); [ "$pm" != "n/a" ] && echo "prompt_tokens_cached=$pm"
    pm=$(prom_metric "vllm:generation_tokens_total"); [ "$pm" != "n/a" ] && echo "generation_tokens=$pm"
    # Prefix cache
    pm=$(prom_metric "vllm:prefix_cache_queries_total"); [ "$pm" != "n/a" ] && echo "prefix_cache_queries=$pm"
    pm=$(prom_metric "vllm:prefix_cache_hits_total"); [ "$pm" != "n/a" ] && echo "prefix_cache_hits=$pm"
    # External (LMCache) cache
    pm=$(prom_metric "vllm:external_prefix_cache_queries_total"); [ "$pm" != "n/a" ] && echo "external_cache_queries=$pm"
    pm=$(prom_metric "vllm:external_prefix_cache_hits_total"); [ "$pm" != "n/a" ] && echo "external_cache_hits=$pm"
    # Requests
    local req_count
    req_count=$(prom_metric "vllm:time_to_first_token_seconds_count")
    [ "$req_count" != "n/a" ] && echo "num_requests=$req_count"
    # TTFT
    local ttft_sum
    ttft_sum=$(prom_metric "vllm:time_to_first_token_seconds_sum")
    if [ "$ttft_sum" != "n/a" ] && [ "$req_count" != "n/a" ] && [ "$req_count" != "0.0" ]; then
      echo "avg_ttft_ms=$(awk "BEGIN {printf \"%.0f\", ($ttft_sum / $req_count) * 1000}")"
    fi
    # E2E latency
    local e2e_sum e2e_count
    e2e_sum=$(prom_metric "vllm:e2e_request_latency_seconds_sum")
    e2e_count=$(prom_metric "vllm:e2e_request_latency_seconds_count")
    if [ "$e2e_sum" != "n/a" ] && [ "$e2e_count" != "n/a" ] && [ "$e2e_count" != "0.0" ]; then
      echo "avg_e2e_ms=$(awk "BEGIN {printf \"%.0f\", ($e2e_sum / $e2e_count) * 1000}")"
    fi
  } > "$summary"
  log "Summary saved to $summary"
}
