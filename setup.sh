#!/usr/bin/env bash
#
# Shared setup for all hf-mount cache integration tests.
#
# This script:
#   1. Installs hf-mount from the official release
#   2. Creates a Python venv with uv and installs vLLM
#
# Individual test directories have their own setup.sh for extra deps.
# The lmcache test additionally requires hermes-agent to be installed
# (https://github.com/NousResearch/hermes-agent) — see lmcache/README.md.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"

log()  { echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" &>/dev/null || die "$1 not found – please install it first"; }

# ── 0. System dependencies ───────────────────────────────────────────

log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq nfs-common python3-dev

if ! command -v process-compose &>/dev/null; then
  log "Installing process-compose..."
  curl -fsSL https://raw.githubusercontent.com/F1bonacc1/process-compose/main/scripts/get-pc.sh | sh
fi

# ── 1. Install hf-mount ─────────────────────────────────────────────

HF_MOUNT_INSTALL_DIR="${HF_MOUNT_INSTALL_DIR:-$HOME/.local/bin}"
HF_MOUNT_REPO="huggingface/hf-mount"

install_hf_mount() {
  local arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64)        arch="x86_64" ;;
    aarch64|arm64) arch="aarch64" ;;
    *) die "Unsupported architecture: $arch" ;;
  esac

  local base_url="https://github.com/${HF_MOUNT_REPO}/releases/latest/download"
  mkdir -p "$HF_MOUNT_INSTALL_DIR"

  for bin in hf-mount hf-mount-nfs hf-mount-fuse; do
    local asset="${bin}-${arch}-linux"
    log "  Downloading ${asset}..."
    curl -fSL "${base_url}/${asset}" -o "${HF_MOUNT_INSTALL_DIR}/${bin}"
    chmod +x "${HF_MOUNT_INSTALL_DIR}/${bin}"
  done
}

if command -v hf-mount &>/dev/null; then
  log "hf-mount already installed: $(command -v hf-mount)"
else
  log "Installing hf-mount from GitHub releases..."
  install_hf_mount
  export PATH="$HF_MOUNT_INSTALL_DIR:$PATH"
  log "hf-mount installed to $HF_MOUNT_INSTALL_DIR/"
fi

# ── 2. Python venv with uv ───────────────────────────────────────────

if ! command -v uv &>/dev/null; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
need uv

log "Creating venv at $VENV_DIR..."
uv venv "$VENV_DIR" --allow-existing
source "$VENV_DIR/bin/activate"

log "Installing vLLM..."
uv pip install vllm

# ── Done ──────────────────────────────────────────────────────────────

cat <<EOF

============================================================
  Shared setup complete
============================================================
  hf-mount:    $(command -v hf-mount || echo ~/.local/bin/hf-mount)
  Python venv: $VENV_DIR

  Now run the test-specific setup:
    source $VENV_DIR/bin/activate
    cd lmcache && ./setup.sh        # lmcache also needs hermes-agent
    cd torch.compile && ./setup.sh
============================================================
EOF
