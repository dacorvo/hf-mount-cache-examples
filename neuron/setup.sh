#!/usr/bin/env bash
#
# Self-contained setup for the Neuron + hf-mount integration test.
#
# Installs:
#   1. System deps via apt-get on Debian/Ubuntu (nfs-common for the NFS client).
#   2. hf-mount via Homebrew (homebrew-core has bottles for Linux).
#   3. uv — used only for the clear-bucket helper's ephemeral HfApi env.
#
# Does NOT install the Neuron SDK. compile_run.py imports torch_neuronx,
# transformers, etc. from a pre-built venv (see VENV in run.sh). Building
# that venv from a Neuron SDK release is out of scope here — see the AWS
# Neuron docs.
#
set -euo pipefail

log()  { echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ── 1. System dependencies (Linux only) ──────────────────────────────

if [[ "$(uname -s)" == "Linux" ]]; then
  command -v apt-get >/dev/null || die "apt-get not found — this script assumes a Debian/Ubuntu host."
  log "Installing system dependencies (nfs-common)..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq nfs-common
else
  die "Neuron only ships for Linux on AWS Trn/Inf instances — this host ($(uname -s)) is not supported."
fi

# ── 2. hf-mount via Homebrew ─────────────────────────────────────────

if command -v hf-mount &>/dev/null; then
  log "hf-mount already installed: $(command -v hf-mount)"
else
  command -v brew &>/dev/null || die "brew not found — install Homebrew first (https://brew.sh) then re-run."
  log "Installing hf-mount via Homebrew..."
  brew install hf-mount
fi

# ── 3. uv ────────────────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv &>/dev/null || die "uv not found after install"

# ── 4. Neuron prerequisites (advisory) ───────────────────────────────

if ! command -v neuron-ls &>/dev/null; then
  log "WARNING: neuron-ls not on PATH — install aws-neuronx-tools before running ./run.sh."
fi

if [[ -n "${VENV:-}" && -x "$VENV/bin/python" ]]; then
  if "$VENV/bin/python" -c 'import torch_neuronx, neuronxcc' >/dev/null 2>&1; then
    log "Neuron venv looks good: $VENV"
  else
    log "WARNING: $VENV/bin/python does not import torch_neuronx + neuronxcc."
    log "         Build the venv from a Neuron binary drop before running ./run.sh."
  fi
else
  log "VENV is not set or not a venv — export VENV=/path/to/venv before running ./run.sh."
fi

cat <<EOF

============================================================
  neuron setup complete
============================================================
  hf-mount:    $(command -v hf-mount)
  uv:          $(command -v uv)
  VENV:        ${VENV:-(unset — required before ./run.sh)}

  Run the test:
    export VENV=/path/to/neuron/venv
    ./run.sh run-all
============================================================
EOF
