#!/usr/bin/env bash
#
# generate-phase.sh — generate a process-compose YAML for a test phase
#
# Usage: generate-phase.sh [--mount|--mount-overlay] <phase> <prompt-dir> <output-yaml>
#
# Required env vars: MODEL, VLLM_PORT, MAX_MODEL_LEN, GPU_MEMORY_UTIL,
#   TOOL_CALL_PARSER, VLLM_EXTRA_ARGS, LMCACHE_CONFIG_FILE,
#   SCRIPT_DIR, LOG_DIR, VLLM_URL, VIRTUAL_ENV, HF_TOKEN,
#   MOUNT_POINT, PROFILE_NAME, CACHE_DIR
#
# For mount modes: HF_MOUNT_BIN, BUCKET, HF_MOUNT_CACHE_DIR
#
set -euo pipefail

# Parse flags.
MOUNT_MODE=""
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --mount) MOUNT_MODE="rw"; shift ;;
    --mount-overlay) MOUNT_MODE="overlay"; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

PHASE="$1"
PROMPT_DIR="$2"
OUTPUT="$3"

# Discover prompt files.
prompt_files=()
for f in "$PROMPT_DIR"/*.txt; do
  [ -f "$f" ] && prompt_files+=("$f")
done
if [ ${#prompt_files[@]} -eq 0 ]; then
  echo "ERROR: no prompt files found in $PROMPT_DIR" >&2
  exit 1
fi

# ── Header ────────────────────────────────────────────────────────────

cat > "$OUTPUT" << 'YAMLEOF'
version: "0.5"
log_location: "${LOG_DIR}/session.log"
log_level: info
YAMLEOF

cat >> "$OUTPUT" << YAMLEOF

environment:
  - "SCRIPT_DIR=${SCRIPT_DIR}"
  - "LOG_DIR=${LOG_DIR}"
  - "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  - "VLLM_URL=${VLLM_URL}"
  - "VIRTUAL_ENV=${VIRTUAL_ENV}"
  - "PATH=${VIRTUAL_ENV}/bin:\${PATH}"
  - "HF_TOKEN=${HF_TOKEN}"

processes:
YAMLEOF

# ── hf-mount (optional) ──────────────────────────────────────────────

vllm_depends=""
if [ -n "$MOUNT_MODE" ]; then
  mount_flags=""
  [ "$MOUNT_MODE" = "overlay" ] && mount_flags="--overlay"

  cat >> "$OUTPUT" << YAMLEOF

  hf-mount:
    command: >
      ${HF_MOUNT_BIN}
      --hf-token ${HF_TOKEN}
      --cache-dir ${HF_MOUNT_CACHE_DIR}
      ${mount_flags}
      bucket ${BUCKET} ${MOUNT_POINT}
    environment:
      - "RUST_LOG=hf_mount=info"
    log_location: "${LOG_DIR}/hf-mount.log"
    ready_log_line: "listening on"
    shutdown:
      signal: 15
      timeout_seconds: 30
YAMLEOF
  vllm_depends="
    depends_on:
      hf-mount:
        condition: process_log_ready"
fi

# ── vLLM ──────────────────────────────────────────────────────────────

cat >> "$OUTPUT" << YAMLEOF

  vllm:
    command: "vllm serve ${MODEL} --port ${VLLM_PORT} --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization ${GPU_MEMORY_UTIL} --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' ${VLLM_EXTRA_ARGS:-} --enable-auto-tool-choice --tool-call-parser ${TOOL_CALL_PARSER}"
    environment:
      - "LMCACHE_CONFIG_FILE=${LMCACHE_CONFIG_FILE}"
      - "PYTHONHASHSEED=0"
    log_location: "${LOG_DIR}/vllm.log"
    readiness_probe:
      http_get:
        host: 127.0.0.1
        port: ${VLLM_PORT}
        path: /v1/models
      period_seconds: 5
      failure_threshold: 120
    shutdown:
      signal: 15
      timeout_seconds: 15${vllm_depends}
YAMLEOF

# ── Conversations ─────────────────────────────────────────────────────

depends_list=""
for f in "${prompt_files[@]}"; do
  name=$(basename "$f" .txt)
  cat >> "$OUTPUT" << YAMLEOF

  conv-${name}:
    command: "${SCRIPT_DIR}/lib/run-conversation.sh ${f}"
    working_dir: "${SCRIPT_DIR}"
    log_location: "${LOG_DIR}/conv-${name}.log"
    depends_on:
      vllm:
        condition: process_healthy
YAMLEOF
  depends_list="${depends_list}      conv-${name}:\n        condition: process_completed\n"
done

# ── Summary ───────────────────────────────────────────────────────────

cat >> "$OUTPUT" << YAMLEOF

  summary:
    command: >
      bash -c '
        source ${SCRIPT_DIR}/lib/helpers.sh;
        VLLM_URL=${VLLM_URL};
        LOG_DIR=${LOG_DIR};
        MOUNT_POINT=${MOUNT_POINT};
        PROFILE_NAME=${PROFILE_NAME};
        save_summary "${PHASE}" "" "${CACHE_DIR}"
      '
    depends_on:
$(echo -e "$depends_list")
    availability:
      exit_on_end: true
YAMLEOF

echo "Generated: $OUTPUT (${#prompt_files[@]} conversations, mount=$MOUNT_MODE)"
