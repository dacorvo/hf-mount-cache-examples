#!/usr/bin/env bash
#
# Shared setup for all hf-mount cache integration tests.
#
# This script:
#   1. Initializes the hf-mount submodule and builds hf-mount-nfs
#   2. Creates a Python venv with uv and installs vLLM
#
# Individual test directories have their own setup.sh for extra deps.
# The lmcache test additionally requires hermes-agent to be installed
# (https://github.com/NousResearch/hermes-agent) — see lmcache/README.md.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/hf-mount"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"

log()  { echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" &>/dev/null || die "$1 not found – please install it first"; }

# ── 0. System dependencies ───────────────────────────────────────────

log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq nfs-common pkg-config libssl-dev python3-dev

if ! command -v process-compose &>/dev/null; then
  log "Installing process-compose..."
  curl -fsSL https://raw.githubusercontent.com/F1bonacc1/process-compose/main/scripts/get-pc.sh | sh
fi

# ── 1. hf-mount submodule + build ────────────────────────────────────

log "Initializing hf-mount submodule..."
(cd "$SCRIPT_DIR" && git submodule update --init --recursive)

log "Building hf-mount-nfs..."
need cargo
(cd "$REPO_ROOT" && cargo build --release --features nfs --bin hf-mount-nfs)
log "Binary: $REPO_ROOT/target/release/hf-mount-nfs"

# ── 2. Python venv with uv ───────────────────────────────────────────

if ! command -v uv &>/dev/null; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
need uv

log "Creating venv at $VENV_DIR..."
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

log "Installing vLLM..."
uv pip install vllm

# ── Done ──────────────────────────────────────────────────────────────

cat <<EOF

============================================================
  Shared setup complete
============================================================
  hf-mount-nfs: $REPO_ROOT/target/release/hf-mount-nfs
  Python venv:  $VENV_DIR

  Now run the test-specific setup:
    source $VENV_DIR/bin/activate
    cd lmcache && ./setup.sh        # lmcache also needs hermes-agent
    cd torch.compile && ./setup.sh
============================================================
EOF
