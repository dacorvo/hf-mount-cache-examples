#!/usr/bin/env bash
#
# CLI for hf-mount + torch.compile cache integration test.
#
# Phases:
#   warmup   — mount RW, compile shapes A and B, artifacts upload to bucket
#   consume  — mount overlay, rerun A and B (cache hit), then C (recompile, local-only)
#   teardown — stop hf-mount, leave caches in place
#
# Usage:
#   ./run.sh <phase>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

command -v uv >/dev/null || { echo "ERROR: uv not found — run ./setup.sh first" >&2; exit 1; }

# ── Configuration ────────────────────────────────────────────────────

export MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
export DTYPE="${DTYPE:-bfloat16}"
# Single-GPU only — device_map="auto" + accelerate hooks break torch.compile's
# fullgraph=True requirement (hooks call torch.compiler.disable).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BUCKET="${BUCKET:-dacorvo/torch-compile-cache}"

MOUNT_POINT="/tmp/hf-mount-torch-compile"
HF_MOUNT_CACHE_DIR="/tmp/hf-mount-cache-torch-compile"
LOG_DIR="$SCRIPT_DIR/logs"
export TORCHINDUCTOR_CACHE_DIR="$MOUNT_POINT/inductor"

# Shape sets — BxC where C is prefill_chunk_size. Distinct chunk sizes
# produce distinct compiled prefill kernels (one per chunk shape); decode
# kernels are shape-flexible across cache_len, so we only vary chunk size.
#   - SHAPES_WARMUP: compiled during phase 1 (mount RW), uploaded to bucket
#   - SHAPES_RECOMPILE: new chunk size during phase 2, must recompile (local-only under overlay)
SHAPES_WARMUP=("1x64" "1x128")
SHAPES_RECOMPILE=("1x256")

if [ -z "${HF_TOKEN:-}" ]; then
  if [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
  fi
fi

mkdir -p "$LOG_DIR"

log()  { echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ── hf-mount lifecycle ───────────────────────────────────────────────

# Lifecycle goes through the `hf-mount` wrapper — `hf-mount start` daemonizes
# the backend, `hf-mount stop` performs a coordinated unmount. NEVER call
# umount on the mount point and NEVER kill the backend (hf-mount-nfs /
# hf-mount-fuse) directly: both leave a phantom NFS mount that hangs the
# system and requires a reboot. See AGENTS.md.
start_hf_mount() {
  local mode="$1"  # rw | overlay
  command -v hf-mount >/dev/null || die "hf-mount not found on PATH — run ./setup.sh or 'brew install hf-mount'"
  [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set"

  local extra_arg=""
  case "$mode" in
    # --advanced-writes: staging files + async batched flush. Without it,
    # every close() is a synchronous upload — kills Inductor (tens of
    # thousands of small file writes during compile).
    rw)      extra_arg="--advanced-writes" ;;
    overlay) extra_arg="--overlay" ;;  # implies --advanced-writes; never pushes to remote
    *)       die "unknown mount mode: $mode" ;;
  esac

  mkdir -p "$MOUNT_POINT" "$HF_MOUNT_CACHE_DIR"

  # If a previous daemon is still attached to this mount point, stop it
  # through the wrapper. NEVER `kill` the backend or `umount` the path.
  if grep -q " $MOUNT_POINT " /proc/mounts 2>/dev/null; then
    log "Previous mount detected at $MOUNT_POINT — stopping via hf-mount wrapper"
    hf-mount stop "$MOUNT_POINT" >> "$LOG_DIR/hf-mount.log" 2>&1 || true
  fi

  log "Starting hf-mount daemon: $BUCKET at $MOUNT_POINT (mode=$mode)"
  RUST_LOG=hf_mount=info \
    hf-mount start -- \
      --hf-token "$HF_TOKEN" \
      --cache-dir "$HF_MOUNT_CACHE_DIR" \
      $extra_arg \
      bucket "$BUCKET" "$MOUNT_POINT" \
      >> "$LOG_DIR/hf-mount.log" 2>&1

  for i in $(seq 1 30); do
    if grep -q " $MOUNT_POINT " /proc/mounts 2>/dev/null; then
      log "Mount ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  die "Mount not ready after 30s — check $LOG_DIR/hf-mount.log"
}

stop_hf_mount() {
  # Always go through the wrapper. NEVER `kill` the backend; that leaves a
  # phantom NFS mount that requires reboot to recover from.
  if grep -q " $MOUNT_POINT " /proc/mounts 2>/dev/null; then
    log "Stopping hf-mount daemon for $MOUNT_POINT (coordinated unmount)"
    hf-mount stop "$MOUNT_POINT" >> "$LOG_DIR/hf-mount.log" 2>&1 || \
      log "WARNING: hf-mount stop reported an error — check $LOG_DIR/hf-mount.log"
  fi
}

# ── Phases ───────────────────────────────────────────────────────────

cmd_warmup() {
  log "====== Phase: warmup (RW mount, populate bucket) ======"

  start_hf_mount rw
  trap 'stop_hf_mount' EXIT

  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

  local args=(--model "$MODEL" --dtype "$DTYPE" --output "$LOG_DIR/results-warmup.json" --phase warmup)
  for s in "${SHAPES_WARMUP[@]}"; do args+=(--shape "$s"); done

  uv run "$SCRIPT_DIR/compile_run.py" "${args[@]}"

  log "Letting hf-mount flush queued uploads..."
  sleep 5

  stop_hf_mount
  trap - EXIT
}

cmd_consume() {
  log "====== Phase: consume (overlay mount, cache hits + recompile) ======"

  # Wipe local hf-mount cache so cache hits must come from the bucket, not
  # leftover chunks. Keep the mount point empty for a clean overlay.
  log "Clearing local hf-mount cache + mount point"
  rm -rf "$HF_MOUNT_CACHE_DIR"
  rm -rf "$MOUNT_POINT"

  start_hf_mount overlay
  trap 'stop_hf_mount' EXIT

  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

  # Re-run warmup shapes (expect cache hits via bucket) + recompile shapes.
  local args=(--model "$MODEL" --dtype "$DTYPE" --output "$LOG_DIR/results-consume.json" --phase consume)
  for s in "${SHAPES_WARMUP[@]}"; do args+=(--shape "$s"); done
  for s in "${SHAPES_RECOMPILE[@]}"; do args+=(--shape "$s"); done

  uv run "$SCRIPT_DIR/compile_run.py" "${args[@]}"

  stop_hf_mount
  trap - EXIT
}

cmd_teardown() {
  stop_hf_mount
  log "Teardown complete. Caches preserved in $MOUNT_POINT and $HF_MOUNT_CACHE_DIR."
}

cmd_clear_bucket() {
  log "Clearing bucket $BUCKET"
  uv run --quiet --with "huggingface_hub>=1.0" python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
files = [f.path for f in api.list_bucket_tree("$BUCKET", recursive=True) if hasattr(f, "size")]
if files:
    api.batch_bucket_files("$BUCKET", delete=files)
    print(f"Deleted {len(files)} files.")
else:
    print("Bucket already empty.")
EOF
}

cmd_run_all() {
  cmd_warmup
  cmd_consume
}

# ── Dispatch ─────────────────────────────────────────────────────────

case "${1:-help}" in
  warmup)       cmd_warmup ;;
  consume)      cmd_consume ;;
  teardown)     cmd_teardown ;;
  clear-bucket) cmd_clear_bucket ;;
  run-all)      cmd_run_all ;;
  help|--help|-h|*)
    cat <<EOF
Usage: $(basename "$0") <command>

Phases:
  warmup        Mount $BUCKET RW, compile shapes ${SHAPES_WARMUP[*]}, unmount.
                Artifacts are uploaded to the bucket.
  consume       Mount $BUCKET overlay, rerun warmup shapes (cache hits via
                bucket), then compile new shapes ${SHAPES_RECOMPILE[*]} (recompile,
                stays local). Bucket is NOT updated.
  run-all       warmup + consume.

Utilities:
  teardown      Stop hf-mount via the wrapper (NEVER call umount on NFS).
  clear-bucket  Delete every file in $BUCKET.

Environment:
  MODEL                $MODEL
  DTYPE                $DTYPE
  BUCKET               $BUCKET
EOF
    ;;
esac
