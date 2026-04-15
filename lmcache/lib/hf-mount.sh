# hf-mount.sh — mount/unmount HF buckets
#
# Expected globals: HF_MOUNT_BIN, HF_TOKEN, MOUNT_POINT, CACHE_DIR,
#   BUCKET, LOG_DIR

start_hf_mount() {
  [ -x "$HF_MOUNT_BIN" ] || die "Binary not found: $HF_MOUNT_BIN"
  [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set"

  mkdir -p "$MOUNT_POINT" "$CACHE_DIR" "$LOG_DIR"

  # If a previous hf-mount is still running, stop it via SIGTERM.
  # NEVER call umount — it corrupts NFS state and requires a reboot.
  if [ -f "$LOG_DIR/hf-mount.pid" ]; then
    local old_pid
    old_pid="$(cat "$LOG_DIR/hf-mount.pid")"
    if kill -0 "$old_pid" 2>/dev/null; then
      log "Stopping previous hf-mount (pid $old_pid) via SIGTERM"
      kill "$old_pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "$old_pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$old_pid" 2>/dev/null; then
        log "WARNING: hf-mount pid $old_pid still alive after 30s"
      fi
    fi
    rm -f "$LOG_DIR/hf-mount.pid"
  fi

  log "Mounting $BUCKET at $MOUNT_POINT (flags: ${*:-(none)})"
  RUST_LOG=hf_mount=info \
    "$HF_MOUNT_BIN" \
    --hf-token "$HF_TOKEN" \
    --cache-dir "$CACHE_DIR" \
    "$@" \
    bucket "$BUCKET" "$MOUNT_POINT" \
    >> "$LOG_DIR/hf-mount.log" 2>&1 &

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
  # Stop hf-mount via SIGTERM — it handles its own unmount cleanly.
  # NEVER call umount — it corrupts NFS state and requires a reboot.
  if [ -f "$LOG_DIR/hf-mount.pid" ]; then
    local pid
    pid="$(cat "$LOG_DIR/hf-mount.pid")"
    if kill -0 "$pid" 2>/dev/null; then
      log "Sending SIGTERM to hf-mount (pid $pid)"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        log "WARNING: hf-mount pid $pid still alive after 30s"
      fi
    fi
    rm -f "$LOG_DIR/hf-mount.pid"
  fi
  log "hf-mount stopped"
}

require_mount() {
  grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null \
    || die "Nothing mounted at $MOUNT_POINT — run 'bucket-warmup' first"
}
