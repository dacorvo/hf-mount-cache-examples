# vllm.sh — start/stop vLLM with LMCache and TP support
#
# Expected globals: MODEL, VLLM_PORT, VLLM_URL, MAX_MODEL_LEN,
#   GPU_MEMORY_UTIL, TP_SIZE, TOOL_CALL_PARSER, TOOL_PARSER_PLUGIN,
#   CHAT_TEMPLATE, LOG_DIR

start_vllm() {
  local lmcache_cfg="${1:-$LOG_DIR/lmcache_config_bucket.yaml}"
  [ -f "$lmcache_cfg" ] || lmcache_cfg="$SCRIPT_DIR/lmcache_config_bucket.yaml"

  local -a extra_args=()
  if [ "${TP_SIZE:-1}" -gt 1 ]; then
    extra_args+=(--tensor-parallel-size "$TP_SIZE")
  fi
  if [ -n "${TOOL_PARSER_PLUGIN:-}" ]; then
    extra_args+=(--tool-parser-plugin "$TOOL_PARSER_PLUGIN")
  fi
  if [ -n "${CHAT_TEMPLATE:-}" ]; then
    extra_args+=(--chat-template "$CHAT_TEMPLATE")
  fi

  log "Starting vLLM on port $VLLM_PORT (model: $MODEL, TP=${TP_SIZE:-1}, parser=${TOOL_CALL_PARSER:-hermes})"
  LMCACHE_CONFIG_FILE="$lmcache_cfg" \
  PYTHONHASHSEED=0 \
    vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL:-0.90}" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER:-hermes}" \
    "${extra_args[@]}" \
    >> "$LOG_DIR/vllm.log" 2>&1 &

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
