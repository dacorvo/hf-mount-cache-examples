#!/usr/bin/env bash
#
# CLI for hf-mount + torch.compile cache integration test.
#
# Phases:
#   warmup   — mount RW, compile shapes A and B, artifacts upload to bucket
#   consume  — mount overlay, rerun A and B (cache hit), then C (recompile, local-only)
#   verify   — confirm bucket unchanged after consume + local artifacts present
#   teardown — stop hf-mount, leave caches in place
#
# Usage:
#   ./test-compile.sh <phase>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/../.venv}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -d "$VENV_DIR" ]] || { echo "ERROR: Venv not found at $VENV_DIR — run the root setup.sh first" >&2; exit 1; }
  source "$VENV_DIR/bin/activate"
fi

# ── Configuration ────────────────────────────────────────────────────

export MODEL="${MODEL:-google/gemma-4-E4B-it}"
export DTYPE="${DTYPE:-bfloat16}"
# Single-GPU only — device_map="auto" + accelerate hooks break torch.compile's
# fullgraph=True requirement (hooks call torch.compiler.disable).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BUCKET="${BUCKET:-dacorvo/torch-compile-cache}"
export MOUNT_POINT="${MOUNT_POINT:-/tmp/hf-mount-torch-compile}"
export HF_MOUNT_CACHE_DIR="${HF_MOUNT_CACHE_DIR:-/tmp/hf-mount-cache-torch-compile}"
export HF_MOUNT_BIN="${HF_MOUNT_BIN:-$(command -v hf-mount || echo "$HOME/.local/bin/hf-mount")}"
export LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
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
  [ -x "$HF_MOUNT_BIN" ] || die "Binary not found at $HF_MOUNT_BIN — run ./setup.sh first"
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
    "$HF_MOUNT_BIN" stop "$MOUNT_POINT" >> "$LOG_DIR/hf-mount.log" 2>&1 || true
  fi

  log "Starting hf-mount daemon: $BUCKET at $MOUNT_POINT (mode=$mode)"
  RUST_LOG=hf_mount=info \
    "$HF_MOUNT_BIN" start -- \
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
    "$HF_MOUNT_BIN" stop "$MOUNT_POINT" >> "$LOG_DIR/hf-mount.log" 2>&1 || \
      log "WARNING: hf-mount stop reported an error — check $LOG_DIR/hf-mount.log"
  fi
}

# ── Bucket inspection ────────────────────────────────────────────────

list_bucket_files() {
  uv run --quiet --with "huggingface_hub>=1.0" python - <<EOF
import os
from huggingface_hub import HfApi
api = HfApi()
files = sorted([f.path for f in api.list_bucket_tree("$BUCKET", recursive=True) if hasattr(f, "size")])
for f in files:
    print(f)
EOF
}

snapshot_bucket() {
  local out="$1"
  list_bucket_files > "$out" 2>/dev/null || true
  wc -l < "$out" | tr -d ' '
}

# ── Phases ───────────────────────────────────────────────────────────

cmd_warmup() {
  log "====== Phase: warmup (RW mount, populate bucket) ======"
  local before="$LOG_DIR/bucket-before-warmup.txt"
  local before_n
  before_n=$(snapshot_bucket "$before") || true
  log "Bucket files before warmup: $before_n"

  start_hf_mount rw
  trap 'stop_hf_mount' EXIT

  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

  local args=(--model "$MODEL" --dtype "$DTYPE" --output "$LOG_DIR/results-warmup.json" --phase warmup)
  for s in "${SHAPES_WARMUP[@]}"; do args+=(--shape "$s"); done

  python3 "$SCRIPT_DIR/compile_run.py" "${args[@]}"

  log "Letting hf-mount flush queued uploads..."
  sleep 5

  stop_hf_mount
  trap - EXIT

  local after="$LOG_DIR/bucket-after-warmup.txt"
  local after_n
  after_n=$(snapshot_bucket "$after")
  log "Bucket files after warmup: $after_n (delta: $((after_n - before_n)))"
}

cmd_consume() {
  log "====== Phase: consume (overlay mount, verify cache hits + recompile) ======"

  # Wipe local hf-mount cache so cache hits must come from the bucket, not
  # leftover chunks. Keep the mount point empty for a clean overlay.
  log "Clearing local hf-mount cache + mount point"
  rm -rf "$HF_MOUNT_CACHE_DIR"
  rm -rf "$MOUNT_POINT"

  local before="$LOG_DIR/bucket-before-consume.txt"
  local before_n
  before_n=$(snapshot_bucket "$before")
  log "Bucket files before consume: $before_n"

  start_hf_mount overlay
  trap 'stop_hf_mount' EXIT

  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

  # Re-run warmup shapes (expect cache hits via bucket) + recompile shapes.
  local args=(--model "$MODEL" --dtype "$DTYPE" --output "$LOG_DIR/results-consume.json" --phase consume)
  for s in "${SHAPES_WARMUP[@]}"; do args+=(--shape "$s"); done
  for s in "${SHAPES_RECOMPILE[@]}"; do args+=(--shape "$s"); done

  python3 "$SCRIPT_DIR/compile_run.py" "${args[@]}"

  stop_hf_mount
  trap - EXIT

  # In overlay mode the mount point IS the local layer. Once the daemon
  # is gone and the mount is released, the directory holds only files
  # that were written locally during the phase (recompile artifacts).
  # Wait briefly for the kernel to drop the stale NFS mount.
  log "Snapshotting overlay-local artifacts at $MOUNT_POINT..."
  for _ in $(seq 1 10); do
    if find "$MOUNT_POINT" -maxdepth 1 -type d >/dev/null 2>&1; then break; fi
    sleep 1
  done
  find "$MOUNT_POINT" -type f 2>/dev/null | sort > "$LOG_DIR/overlay-local-files.txt" || true

  local after="$LOG_DIR/bucket-after-consume.txt"
  local after_n
  after_n=$(snapshot_bucket "$after")
  log "Bucket files after consume: $after_n (delta: $((after_n - before_n)))"
}

cmd_verify() {
  log "====== Phase: verify ======"
  local pass=true

  # 1. Compare bucket file lists from before/after consume — they MUST match.
  if [ -f "$LOG_DIR/bucket-before-consume.txt" ] && [ -f "$LOG_DIR/bucket-after-consume.txt" ]; then
    if diff -q "$LOG_DIR/bucket-before-consume.txt" "$LOG_DIR/bucket-after-consume.txt" >/dev/null; then
      log "PASS: bucket file list unchanged across consume phase"
    else
      log "FAIL: bucket changed during consume phase:"
      diff "$LOG_DIR/bucket-before-consume.txt" "$LOG_DIR/bucket-after-consume.txt" | head -40
      pass=false
    fi
  else
    log "SKIP: bucket-before/after-consume snapshots missing — run 'consume' first"
    pass=false
  fi

  # 2. Verify warmup shapes were cache hits in consume.
  #
  # Signal: cache_files_added. A cache hit writes far fewer files than a
  # recompile (metadata only vs. Triton kernels, FX graphs, etc.).
  # Rather than a fixed threshold we compare each hit shape against the
  # recompile shape from the same run — a hit must add < 5% of the
  # recompile file count. This is self-calibrating across platforms.
  if [ -f "$LOG_DIR/results-warmup.json" ] && [ -f "$LOG_DIR/results-consume.json" ]; then
    python3 - "$LOG_DIR/results-warmup.json" "$LOG_DIR/results-consume.json" <<'PY' || pass=false
import json, sys
warm = json.load(open(sys.argv[1]))
cons = json.load(open(sys.argv[2]))
warm_by = {s["shape"]: s for s in warm["shapes"]}
# Use the recompile shape (not in warmup) as the baseline.
recompile_added = max(
    (s["cache_files_added"] for s in cons["shapes"] if s["shape"] not in warm_by),
    default=0,
)
if recompile_added == 0:
    print("  ERROR: no recompile shape found — cannot calibrate hit threshold")
    sys.exit(1)
ok = True
print(f"  Warmup cache: {warm['cache_files_total']} files, {warm['cache_size_total']}")
print(f"  Consume cache (post-run): {cons['cache_files_total']} files, {cons['cache_size_total']}")
print(f"  Recompile baseline: {recompile_added} files")
print(f"  {'shape':<8} {'warm_first':>11} {'cons_first':>11} {'cons_2nd':>10} {'files_added':>12}  result")
for s in cons["shapes"]:
    shape = s["shape"]
    added = s["cache_files_added"]
    if shape in warm_by:
        w = warm_by[shape]["first_call_s"]
        c = s["first_call_s"]
        is_hit = added < recompile_added * 0.05
        verdict = "HIT " if is_hit else "MISS"
        if not is_hit:
            ok = False
        print(f"  {shape:<8} {w:>10.2f}s {c:>10.2f}s {s['second_call_s']:>9.3f}s {added:>12}  {verdict}")
    else:
        verdict = "RECOMPILE" if added > 0 else "(no new files?)"
        if added == 0:
            ok = False
        print(f"  {shape:<8} {'-':>11} {s['first_call_s']:>10.2f}s {s['second_call_s']:>9.3f}s {added:>12}  {verdict}")
sys.exit(0 if ok else 1)
PY
    [ "$pass" = "true" ] && log "PASS: cache hits + recompile signatures look correct"
  else
    log "SKIP: results JSON missing — run 'warmup' and 'consume' first"
    pass=false
  fi

  # 3. Verify overlay-local artifacts exist at the mount point after unmount.
  if [ -f "$LOG_DIR/overlay-local-files.txt" ]; then
    local local_n
    local_n=$(wc -l < "$LOG_DIR/overlay-local-files.txt" | tr -d ' ')
    if [ "$local_n" -gt 0 ]; then
      log "PASS: overlay-local layer holds $local_n files at $MOUNT_POINT"
      log "      (head:)"
      head -5 "$LOG_DIR/overlay-local-files.txt" | sed 's/^/        /'
    else
      log "FAIL: no overlay-local files captured"
      pass=false
    fi
  fi

  echo
  if [ "$pass" = "true" ]; then
    log "====== VERIFY: ALL CHECKS PASSED ======"
  else
    log "====== VERIFY: SOME CHECKS FAILED ======"
    return 1
  fi
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

cmd_clean_local() {
  stop_hf_mount
  log "Removing $MOUNT_POINT and $HF_MOUNT_CACHE_DIR"
  rm -rf "$MOUNT_POINT" "$HF_MOUNT_CACHE_DIR"
}

cmd_run_all() {
  cmd_warmup
  cmd_consume
  cmd_verify
}

# ── Dispatch ─────────────────────────────────────────────────────────

case "${1:-help}" in
  warmup)       cmd_warmup ;;
  consume)      cmd_consume ;;
  verify)       cmd_verify ;;
  teardown)     cmd_teardown ;;
  clean-local)  cmd_clean_local ;;
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
  verify        Check that the bucket file list is unchanged across the
                consume phase, that warmup shapes were cache hits, and that
                overlay-local artifacts exist at the mount point.
  run-all       warmup + consume + verify.

Utilities:
  teardown      SIGTERM hf-mount (NEVER call umount on NFS).
  clean-local   teardown + remove mount point and local cache directory.
  clear-bucket  Delete every file in $BUCKET.

Environment:
  MODEL                $MODEL
  BUCKET               $BUCKET
  MOUNT_POINT          $MOUNT_POINT
  HF_MOUNT_CACHE_DIR   $HF_MOUNT_CACHE_DIR
  HF_MOUNT_BIN         $HF_MOUNT_BIN
  LOG_DIR              $LOG_DIR
EOF
    ;;
esac
