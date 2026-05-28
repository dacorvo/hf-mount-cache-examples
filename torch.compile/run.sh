#!/usr/bin/env bash
#
# CLI for hf-mount + torch.compile cache integration test.
#
# Phases:
#   warmup   — mount RW, compile SHAPES, artifacts upload to bucket
#   consume  — mount overlay, run SHAPES (cache hit if warmup ran first)
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
# Single-GPU only — device_map="auto" + accelerate hooks break torch.compile's
# fullgraph=True requirement (hooks call torch.compiler.disable).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BUCKET="${BUCKET:-dacorvo/torch-compile-cache}"

MOUNT_POINT="/tmp/hf-mount-torch-compile"
LOG_DIR="$SCRIPT_DIR/logs"

# Inductor artifacts are not portable across GPU compute capabilities or CPU
# architectures, so the bucket subpath has to be hardware-specific — otherwise
# two hosts sharing the bucket would clobber each other. nvidia-smi may be
# absent (macOS / CPU-only host), in which case we fall back to uname -m.
HW_TAG="cpu-$(uname -m)"
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
  [ -n "$CUDA_CAP" ] && HW_TAG="cuda-sm$CUDA_CAP"
fi
export TORCHINDUCTOR_CACHE_DIR="$MOUNT_POINT/inductor/$HW_TAG"

# Shapes — BxC where C is prefill_chunk_size. The same shape list is used
# by warmup (compiles it, uploads to bucket) and consume (runs it under the
# overlay mount, expects a cache hit if warmup ran first).
SHAPES=("1x64")

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

# Cross-platform check for an active mount at $MOUNT_POINT.
#   Linux: /proc/mounts is the source of truth.
#   macOS: no /proc, and the kernel resolves /tmp -> /private/tmp, so we have
#          to compare the resolved path against `mount` output.
is_mounted() {
  if [ -r /proc/mounts ]; then
    grep -q " $MOUNT_POINT " /proc/mounts
  else
    local resolved
    resolved="$(cd "$MOUNT_POINT" 2>/dev/null && pwd -P)" || return 1
    mount | grep -q " on $resolved "
  fi
}

# Stop any hf-mount daemon currently attached to $MOUNT_POINT. MUST run before
# any rm/ls/find/stat on the mount path — those operations block forever on a
# live NFS mount whose daemon we're about to replace. Always go through the
# wrapper; NEVER umount the path or kill the backend directly (both leave a
# phantom mount that requires reboot — see AGENTS.md).
ensure_unmounted() {
  if is_mounted; then
    log "Previous mount detected at $MOUNT_POINT — stopping via hf-mount wrapper"
    hf-mount stop "$MOUNT_POINT" >> "$LOG_DIR/hf-mount.log" 2>&1 || \
      die "hf-mount stop failed — check $LOG_DIR/hf-mount.log"
  fi
}

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

  mkdir -p "$MOUNT_POINT"

  log "Starting hf-mount daemon: $BUCKET at $MOUNT_POINT (mode=$mode)"
  RUST_LOG=hf_mount=info \
    hf-mount start -- \
      --hf-token "$HF_TOKEN" \
      $extra_arg \
      bucket "$BUCKET" "$MOUNT_POINT" \
      >> "$LOG_DIR/hf-mount.log" 2>&1

  for i in $(seq 1 30); do
    if is_mounted; then
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
  if is_mounted; then
    log "Stopping hf-mount daemon for $MOUNT_POINT (coordinated unmount)"
    hf-mount stop "$MOUNT_POINT" >> "$LOG_DIR/hf-mount.log" 2>&1 || \
      log "WARNING: hf-mount stop reported an error — check $LOG_DIR/hf-mount.log"
  fi
}

# ── Phases ───────────────────────────────────────────────────────────

cmd_warmup() {
  log "====== Phase: warmup (RW mount, populate bucket) ======"

  # Stop any stale daemon FIRST — rm/ls on a live NFS mount hangs forever.
  ensure_unmounted
  # Start from a clean mount point so the RW mount has nothing stale under it.
  rm -rf "$MOUNT_POINT"

  start_hf_mount rw
  trap 'stop_hf_mount' EXIT

  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

  local args=(--model "$MODEL" --output "$LOG_DIR/results-warmup.json" --phase warmup)
  for s in "${SHAPES[@]}"; do args+=(--shape "$s"); done

  uv run "$SCRIPT_DIR/compile_run.py" "${args[@]}"

  log "Letting hf-mount flush queued uploads..."
  sleep 5

  stop_hf_mount
  trap - EXIT
}

cmd_consume() {
  log "====== Phase: consume (overlay mount, cache hit if bucket warmed) ======"

  # Stop any stale daemon FIRST — rm/ls on a live NFS mount hangs forever.
  ensure_unmounted
  # Keep the mount point empty for a clean overlay layer.
  log "Clearing mount point"
  rm -rf "$MOUNT_POINT"

  start_hf_mount overlay
  trap 'stop_hf_mount' EXIT

  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

  local args=(--model "$MODEL" --output "$LOG_DIR/results-consume.json" --phase consume)
  for s in "${SHAPES[@]}"; do args+=(--shape "$s"); done

  uv run "$SCRIPT_DIR/compile_run.py" "${args[@]}"

  stop_hf_mount
  trap - EXIT
}

# Baseline: run compile_run.py with NO hf-mount and a fresh local Inductor
# cache dir. Measures the raw cold-compile cost the bucket-backed flow has
# to beat.
cmd_vanilla() {
  log "====== Phase: vanilla (no hf-mount, fresh local Inductor cache) ======"

  local vanilla_cache="/tmp/inductor-vanilla"
  rm -rf "$vanilla_cache"
  mkdir -p "$vanilla_cache"

  local args=(--model "$MODEL" --output "$LOG_DIR/results-vanilla.json" --phase vanilla)
  for s in "${SHAPES[@]}"; do args+=(--shape "$s"); done

  TORCHINDUCTOR_CACHE_DIR="$vanilla_cache" \
    uv run "$SCRIPT_DIR/compile_run.py" "${args[@]}"
}

cmd_teardown() {
  stop_hf_mount
  log "Teardown complete. Mount point preserved at $MOUNT_POINT."
}

cmd_clear_bucket() {
  # Scope deletion to this host's hardware subtree — Inductor caches for other
  # GPUs / archs live under sibling prefixes and must not be touched.
  local prefix="inductor/$HW_TAG"
  log "Clearing $BUCKET under $prefix/"
  uv run --quiet --with "huggingface_hub>=1.0" python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
files = [f.path for f in api.list_bucket_tree("$BUCKET", prefix="$prefix", recursive=True) if hasattr(f, "size")]
if files:
    api.batch_bucket_files("$BUCKET", delete=files)
    print(f"Deleted {len(files)} files.")
else:
    print("Nothing under $prefix/ — bucket subtree already empty.")
EOF
}

cmd_run_all() {
  # Start with a clean logs/ dir so one run-all == one self-contained log set.
  # Single-phase invocations (./run.sh warmup, etc.) still append to whatever
  # is there, so manual sequences keep their history.
  rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

  log "====== Baseline: vanilla compile (no hf-mount) ======"
  cmd_vanilla

  log "====== With hf-mount: warmup populates the bucket, then consume ======"
  cmd_warmup
  cmd_consume

  log "====== Summary: first_call_s, vanilla compile vs hf-mount cache hit ======"
  uv run --quiet python - <<EOF
import json
vanilla = json.load(open("$LOG_DIR/results-vanilla.json"))
warm = json.load(open("$LOG_DIR/results-consume.json"))
print()
print(f"  {'shape':<10} {'vanilla_s':>11} {'cached_s':>11}  {'speedup':>10}")
print(f"  {'-'*10:<10} {'-'*11:>11} {'-'*11:>11}  {'-'*10:>10}")
for v, w in zip(vanilla["shapes"], warm["shapes"]):
    speedup = v["first_call_s"] / w["first_call_s"] if w["first_call_s"] > 0 else float("inf")
    print(f"  {v['shape']:<10} {v['first_call_s']:>11.2f} {w['first_call_s']:>11.2f}  {speedup:>9.1f}x")
print()
EOF
}

# ── Dispatch ─────────────────────────────────────────────────────────

case "${1:-help}" in
  vanilla)      cmd_vanilla ;;
  warmup)       cmd_warmup ;;
  consume)      cmd_consume ;;
  teardown)     cmd_teardown ;;
  clear-bucket) cmd_clear_bucket ;;
  run-all)      cmd_run_all ;;
  help|--help|-h|*)
    cat <<EOF
Usage: $(basename "$0") <command>

Phases:
  vanilla       Run shapes ${SHAPES[*]} with no hf-mount, against a fresh
                local Inductor cache. Baseline cold-compile cost.
  warmup        Mount $BUCKET RW, compile shapes ${SHAPES[*]}, unmount.
                Artifacts are uploaded to the bucket.
  consume       Mount $BUCKET overlay, run shapes ${SHAPES[*]}. If warmup ran
                first (or another producer populated the bucket subtree),
                first_call_s drops to a cache hit.
  run-all       Full benchmark: vanilla + warmup + consume. Prints a
                per-shape vanilla-vs-cached first_call_s summary from
                results-vanilla.json and results-consume.json.
                Does NOT touch the bucket beyond what warmup writes.

Utilities:
  teardown      Stop hf-mount via the wrapper (NEVER call umount on NFS).
  clear-bucket  Delete this host's HW_TAG subtree from $BUCKET.

Environment:
  MODEL                $MODEL
  BUCKET               $BUCKET
EOF
    ;;
esac
