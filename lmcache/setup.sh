#!/usr/bin/env bash
#
# LMCache-specific setup. Run the root setup.sh first.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/../.venv}"

log()  { echo "==> $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$VENV_DIR" ]] || die "Venv not found at $VENV_DIR — run the root setup.sh first"
source "$VENV_DIR/bin/activate"

log "Installing LMCache..."
uv pip install lmcache

log "LMCache setup complete."
