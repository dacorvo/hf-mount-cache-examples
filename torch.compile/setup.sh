#!/usr/bin/env bash
#
# Self-contained setup for the torch.compile + hf-mount integration test.
#
# Installs:
#   1. System deps via apt-get on Debian/Ubuntu (skipped on macOS, where
#      NFS is built in)
#   2. hf-mount via Homebrew (homebrew-core has bottles for macOS and Linux)
#   3. uv + a Python venv at ../.venv
#   4. torch + transformers + accelerate into the venv
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

log()  { echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ── 1. System dependencies (Linux only) ──────────────────────────────

if [[ "$(uname -s)" == "Linux" ]]; then
  command -v apt-get >/dev/null || die "apt-get not found — this script assumes a Debian/Ubuntu host on Linux."
  log "Installing system dependencies (nfs-common, python3-dev)..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq nfs-common python3-dev
else
  log "Non-Linux host ($(uname -s)) — skipping apt-get (NFS client is built in)."
fi

# ── 2. hf-mount via Homebrew ─────────────────────────────────────────

if command -v hf-mount &>/dev/null; then
  log "hf-mount already installed: $(command -v hf-mount)"
else
  command -v brew &>/dev/null || die "brew not found — install Homebrew first (https://brew.sh) then re-run."
  log "Installing hf-mount via Homebrew..."
  brew install hf-mount
fi

# ── 3. Python venv with uv ───────────────────────────────────────────

if ! command -v uv &>/dev/null; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv &>/dev/null || die "uv not found after install"

log "Creating venv at $VENV_DIR..."
uv venv "$VENV_DIR" --allow-existing
source "$VENV_DIR/bin/activate"

# ── 4. Python deps ───────────────────────────────────────────────────

log "Installing torch + transformers + accelerate..."
uv pip install "torch>=2.4" "transformers>=4.45" "accelerate"

cat <<EOF

============================================================
  torch.compile setup complete
============================================================
  hf-mount:    $(command -v hf-mount)
  Python venv: $VENV_DIR

  Run the test:
    source $VENV_DIR/bin/activate
    ./run.sh run-all
============================================================
EOF
