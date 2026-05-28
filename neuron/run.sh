#!/usr/bin/env bash
#
# CLI for hf-mount + Neuron NEFF-cache integration test.
#
# Phases:
#   warmup   — mount RW, compile SHAPES on Neuron, NEFFs upload to bucket
#   consume  — mount overlay, run SHAPES (cache hit if warmup ran first)
#   teardown — stop hf-mount, leave caches in place
#
# Usage:
#   VENV=/path/to/neuron-venv ./run.sh <phase>
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

command -v uv >/dev/null || { echo "ERROR: uv not found — run ./setup.sh first" >&2; exit 1; }

# ── Configuration ────────────────────────────────────────────────────

export MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
export BUCKET="${BUCKET:-dacorvo/neuron-compile-cache}"

# Pre-built venv with the Neuron SDK (torch_neuronx, neuronx_cc, neuronxcc)
# plus transformers>=5.6 + accelerate. See the AWS Neuron docs for the install
# workflow.
VENV="${VENV:?VENV must point to a Python venv with torch_neuronx installed}"
[ -x "$VENV/bin/python" ] || { echo "ERROR: $VENV/bin/python not executable" >&2; exit 1; }

MOUNT_POINT="/tmp/hf-mount-neuron"
LOG_DIR="$SCRIPT_DIR/logs"

# NEFFs are tied to the compiler version (different neuronxcc versions emit
# incompatible cache keys), so the bucket subpath has to encode neuronxcc's
# version. Detected from the venv's `neuronxcc.__version__`.
NXCC_VERSION="$("$VENV/bin/python" -c 'import neuronxcc; print(neuronxcc.__version__)')"
[ -n "$NXCC_VERSION" ] || { echo "ERROR: could not import neuronxcc from $VENV" >&2; exit 1; }
HW_TAG="neuronxcc-$NXCC_VERSION"
export TORCH_NEURONX_NEFF_CACHE_DIR="$MOUNT_POINT/$HW_TAG"

# Shapes — BxC where C is prefill_chunk_size. Same shape list drives both
# warmup (compiles + uploads) and consume (overlay mount, expects cache hits
# if warmup ran first). Varying chunk_size forces a distinct compiled prefill
# graph per shape; decode is shape-flexible across cache_len.
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
#
# Rules from hf-mount-cache-examples/AGENTS.md — phantom mounts here brick
# the kernel until reboot:
#   ✅ DO use the `hf-mount` wrapper for start/stop/status.
#   ❌ NEVER `umount` the mount point (even with -l, -f, sudo).
#   ❌ NEVER `kill` the hf-mount-nfs / hf-mount-fuse backend directly.

# is_mounted — true if anything is mounted at $MOUNT_POINT. /proc/mounts is
# the source of truth on Linux (the only platform Neuron supports).
is_mounted() {
  grep -q " $MOUNT_POINT " /proc/mounts
}

# Stop any hf-mount daemon currently attached to $MOUNT_POINT BEFORE any
# rm / ls / find / stat / du on the path — those will block forever on a
# phantom NFS mount.
ensure_unmounted() {
  if is_mounted; then
    log "Previous mount detected at $MOUNT_POINT — stopping via hf-mount wrapper"
    hf-mount stop "$MOUNT_POINT" >> "$LOG_DIR/hf-mount.log" 2>&1 || \
      die "hf-mount stop failed — check $LOG_DIR/hf-mount.log"
  fi
}

start_hf_mount() {
  local mode="$1"  # rw | overlay
  command -v hf-mount >/dev/null || die "hf-mount not found on PATH — run ./setup.sh or 'brew install hf-mount'"
  [ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set"

  local extra_arg=""
  case "$mode" in
    # --advanced-writes: staging files + async batched flush. Without it,
    # every close() is a synchronous upload — NEFFs are written multiple
    # times per compile and synchronous uploads would dominate wall time.
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

  mkdir -p "$TORCH_NEURONX_NEFF_CACHE_DIR"

  local args=(--model "$MODEL" --output "$LOG_DIR/results-warmup.json" --phase warmup)
  for s in "${SHAPES[@]}"; do args+=(--shape "$s"); done

  "$VENV/bin/python" "$SCRIPT_DIR/compile_run.py" "${args[@]}"

  log "Letting hf-mount flush queued uploads..."
  sleep 5

  stop_hf_mount
  trap - EXIT
}

# cmd_consume [<label>] — optional <label> tags the output as
# results-consume-<label>.json so run-all can keep cold and warm side by side.
cmd_consume() {
  local label="${1:-}"
  local out_file="$LOG_DIR/results-consume${label:+-$label}.json"

  log "====== Phase: consume (overlay mount, cache hit if bucket warmed) ======"

  ensure_unmounted
  log "Clearing mount point"
  rm -rf "$MOUNT_POINT"

  start_hf_mount overlay
  trap 'stop_hf_mount' EXIT

  mkdir -p "$TORCH_NEURONX_NEFF_CACHE_DIR"

  local args=(--model "$MODEL" --output "$out_file" --phase consume)
  for s in "${SHAPES[@]}"; do args+=(--shape "$s"); done

  "$VENV/bin/python" "$SCRIPT_DIR/compile_run.py" "${args[@]}"

  stop_hf_mount
  trap - EXIT
}

cmd_baseline() {
  # Standard inference with no bucket access — fresh local NEFF cache,
  # no hf-mount. Measures the cold-compile cost on a plain machine.
  log "====== Phase: baseline (standard inference, no bucket) ======"

  local cache_dir="/tmp/neff-cache-baseline"
  rm -rf "$cache_dir"
  mkdir -p "$cache_dir"

  local args=(--model "$MODEL" --output "$LOG_DIR/results-baseline.json" --phase baseline)
  for s in "${SHAPES[@]}"; do args+=(--shape "$s"); done

  TORCH_NEURONX_NEFF_CACHE_DIR="$cache_dir" \
    "$VENV/bin/python" "$SCRIPT_DIR/compile_run.py" "${args[@]}"
}

cmd_teardown() {
  stop_hf_mount
  log "Teardown complete. Mount point preserved at $MOUNT_POINT."
}

cmd_clear_bucket() {
  # Scope deletion to this neuronxcc version's subtree — caches for other
  # compiler versions live under sibling prefixes and must not be touched.
  local prefix="$HW_TAG"
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
  rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

  log "====== Baseline: standard inference, no bucket ======"
  cmd_baseline

  log "====== Bucket path: warmup populates the bucket, then consume ======"
  cmd_warmup
  cmd_consume warm

  log "====== Summary: first_call_s, baseline vs warm-consume ======"
  uv run --quiet python - <<EOF
import json
base = json.load(open("$LOG_DIR/results-baseline.json"))
warm = json.load(open("$LOG_DIR/results-consume-warm.json"))
print()
print(f"  {'shape':<10} {'baseline_first_s':>18} {'warm_first_s':>14}  {'speedup':>10}")
print(f"  {'-'*10:<10} {'-'*18:>18} {'-'*14:>14}  {'-'*10:>10}")
for b, w in zip(base["shapes"], warm["shapes"]):
    speedup = b["first_call_s"] / w["first_call_s"] if w["first_call_s"] > 0 else float("inf")
    print(f"  {b['shape']:<10} {b['first_call_s']:>18.2f} {w['first_call_s']:>14.2f}  {speedup:>9.1f}x")
print()
EOF
}

# ── Dispatch ─────────────────────────────────────────────────────────

case "${1:-help}" in
  baseline)     cmd_baseline ;;
  warmup)       cmd_warmup ;;
  consume)      cmd_consume ;;
  teardown)     cmd_teardown ;;
  clear-bucket) cmd_clear_bucket ;;
  run-all)      cmd_run_all ;;
  help|--help|-h|*)
    cat <<EOF
Usage: $(basename "$0") <command>

Phases:
  baseline      Standard inference with no bucket access. Fresh local NEFF
                cache, no hf-mount. Captures the cold-compile baseline.
  warmup        Mount $BUCKET RW, compile shapes ${SHAPES[*]}, unmount.
                Newly produced NEFFs are uploaded to the bucket.
  consume       Mount $BUCKET overlay, run shapes ${SHAPES[*]}. If warmup ran
                first, first_call_s drops to a cache hit; otherwise it
                compiles cold and the new NEFFs stay local (bucket
                unchanged).
  run-all       baseline + warmup + warm consume. Prints baseline vs
                warm-consume summary from results-baseline.json and
                results-consume-warm.json.

Utilities:
  teardown      Stop hf-mount via the wrapper (NEVER call umount on NFS).
  clear-bucket  Delete every NEFF under $BUCKET/$HW_TAG/.

Environment:
  MODEL                $MODEL
  BUCKET               $BUCKET
  VENV                 $VENV
  TORCH_NEURONX_NEFF_CACHE_DIR  $TORCH_NEURONX_NEFF_CACHE_DIR
EOF
    ;;
esac
