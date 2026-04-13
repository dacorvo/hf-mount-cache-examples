# vllm.sh — start/stop vLLM with LMCache and TP support
#
# Expected globals: MODEL, VLLM_PORT, VLLM_URL, MAX_MODEL_LEN,
#   GPU_MEMORY_UTIL, TP_SIZE, LOG_DIR

start_vllm() {
  local lmcache_cfg="${1:-$LOG_DIR/lmcache_config_bucket.yaml}"
  [ -f "$lmcache_cfg" ] || lmcache_cfg="$SCRIPT_DIR/lmcache_config_bucket.yaml"

  local tp_args=()
  if [ "${TP_SIZE:-1}" -gt 1 ]; then
    tp_args=(--tensor-parallel-size "$TP_SIZE")
  fi

  log "Starting vLLM on port $VLLM_PORT (model: $MODEL, TP=${TP_SIZE:-1})"
  LMCACHE_CONFIG_FILE="$lmcache_cfg" \
  PYTHONHASHSEED=0 \
    vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL:-0.90}" \
    "${tp_args[@]}" \
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

stop_vllm() {
  if [ -f "$LOG_DIR/vllm.pid" ]; then
    kill "$(cat "$LOG_DIR/vllm.pid")" 2>/dev/null && log "vLLM stopped" || log "vLLM already stopped"
    rm -f "$LOG_DIR/vllm.pid"
    sleep 3
  fi
}

require_vllm() {
  curl -sf "$VLLM_URL/v1/models" >/dev/null 2>&1 \
    || die "vLLM not responding on $VLLM_URL"
}
