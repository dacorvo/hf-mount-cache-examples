#!/usr/bin/env bash
#
# Sanity test: drive hf-mount through process-compose using the wrapper's
# start/stop subcommands. Verifies the host stays clean across mount modes.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_MOUNT_BIN="${HF_MOUNT_BIN:-$SCRIPT_DIR/../../hf-mount/target/release/hf-mount}"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}"
PC_BIN="${PC_BIN:-$SCRIPT_DIR/../../bin/process-compose}"

[ -x "$HF_MOUNT_BIN" ] || { echo "ERROR: hf-mount wrapper not found at $HF_MOUNT_BIN" >&2; exit 1; }
[ -x "$PC_BIN" ] || { echo "ERROR: process-compose not found at $PC_BIN" >&2; exit 1; }
[ -n "$HF_TOKEN" ] || { echo "ERROR: HF_TOKEN not set" >&2; exit 1; }

mkdir -p /tmp/sanity-mnt /tmp/sanity-cache

assert_unmounted() {
  local label="$1"
  if grep -q ' /tmp/sanity-mnt ' /proc/mounts; then
    echo "FAIL [$label]: /tmp/sanity-mnt still in /proc/mounts" >&2
    "$HF_MOUNT_BIN" status >&2 || true
    exit 1
  fi
  if "$HF_MOUNT_BIN" status 2>/dev/null | grep -q sanity-mnt; then
    echo "FAIL [$label]: hf-mount status still lists sanity-mnt daemon" >&2
    exit 1
  fi
  echo "  OK [$label]: nothing mounted at /tmp/sanity-mnt"
}

run_phase() {
  local label="$1"
  local yaml="$2"
  echo "=== Phase $label ==="
  "$PC_BIN" up -f "$yaml" --tui=false
  echo "  process-compose exited cleanly"
}

assert_unmounted "pre-check"
run_phase "A: read-only" "$SCRIPT_DIR/sanity-ro.yaml"
assert_unmounted "after RO"
run_phase "B: overlay" "$SCRIPT_DIR/sanity-overlay.yaml"
assert_unmounted "after overlay"

echo "=== SANITY PASSED ==="
