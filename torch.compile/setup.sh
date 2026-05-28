#!/usr/bin/env bash
#
# Self-contained setup for the torch.compile + hf-mount integration test.
#
# Installs:
#   1. System deps (nfs-common, python3-dev)
#   2. hf-mount from the latest GitHub release
#   3. uv + a Python venv at ../.venv
#   4. torch + transformers + accelerate into the venv
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

log()  { echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ── 1. System dependencies ───────────────────────────────────────────

log "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq nfs-common python3-dev

# ── 2. hf-mount ──────────────────────────────────────────────────────

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
  hf-mount:    $(command -v hf-mount || echo $HF_MOUNT_INSTALL_DIR/hf-mount)
  Python venv: $VENV_DIR

  Run the test:
    source $VENV_DIR/bin/activate
    ./run.sh run-all
============================================================
EOF
