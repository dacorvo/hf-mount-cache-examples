#!/usr/bin/env bash
# Run the post-trace pipeline for a phase-N run:
#   1. analyze_traces.py → analysis_v2.json
#   2. splice_correctness_phase2_5.py → splice_correctness.json
#   3. write a per-run summary.md that captures the headline numbers.
#
# Usage:
#   tserve/finalize_run.sh runs/phase2_6-2026-05-01-1700/

set -euo pipefail

RUN_DIR="${1:?usage: $0 <run_dir>}"
RUN_DIR="$(realpath "$RUN_DIR")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "=== analyzing traces ==="
uv run --script "$SCRIPT_DIR/analyze_traces.py" "$RUN_DIR" \
    --output "$RUN_DIR/analysis_v2.json" 2>&1 | tee "$RUN_DIR/analysis.log"

echo
echo "=== splice correctness measurement (4-bit) ==="
# Stop any running server to free GPUs
pkill -f "transformers serve" 2>/dev/null || true
sleep 5

PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run --script "$SCRIPT_DIR/splice_correctness_phase2_5.py" "$RUN_DIR" \
    --quantize-4bit --skip-existing 2>&1 | tee "$RUN_DIR/splice.log"

echo
echo "=== summary ==="
cat <<EOF > "$RUN_DIR/summary.md"
# $(basename "$RUN_DIR") summary

## Trace dir
\`$RUN_DIR\`

## Analysis (\`analysis_v2.json\`)
$(grep -E "^  total pairs|^  after dropping|^  longest-match|^  covered-tokens|^  most-popular|^  match content categories|^    [a-z-]+:" "$RUN_DIR/analysis.log" 2>/dev/null | head -40)

## Splice correctness (\`splice_correctness.json\`)
$(tail -10 "$RUN_DIR/splice.log" 2>/dev/null | head -30)
EOF

echo "wrote $RUN_DIR/summary.md"
echo
cat "$RUN_DIR/summary.md"
