# hf-mount.sh — mount/unmount HF buckets
#
# Expected globals: HF_MOUNT_BIN, HF_TOKEN, MOUNT_POINT, CACHE_DIR,
#   BUCKET, LOG_DIR

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

require_mount() {
  grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null \
    || die "Nothing mounted at $MOUNT_POINT — run 'bucket-warmup' first"
}
