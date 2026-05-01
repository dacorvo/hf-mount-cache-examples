#!/usr/bin/env bash
# Phase 2: drive ~10 overlapping tasks against lmcache-src to capture
# cross-session manifest data. Each task runs as a fresh OpenCode
# session against the local Gemma-4 E4B endpoint.
#
# Tasks are designed so they touch overlapping files (README, AGENTS,
# v1/, storage_backend/) — that's the substrate for cross-session
# tool-result reuse, which is the body-splice question we want to
# measure on real piloted-agent traces.
#
# All tasks run from /home/ubuntu/hf-mount-cache-examples/lmcache-src
# so OpenCode picks up the same AGENTS.md every time.

set -euo pipefail
TASKS_DIR="$(dirname "$0")/.."
LMCACHE_DIR="$TASKS_DIR/lmcache-src"
TRACE_DIR="$TASKS_DIR/runs/phase2-2026-05-01-1113"

cd "$LMCACHE_DIR"

declare -a TASKS=(
    "Read README.md and explain in two sentences what LMCache is."
    "Find every place that touches a prefix tree (any file). Just list filenames + line numbers, no code dump."
    "List the public API of the lmcache.v1 package by reading lmcache/v1/__init__.py."
    "What CLI commands does lmcache expose? Read lmcache/cli/ and list them with one-line descriptions."
    "Find every callsite of serialize_kv (or close variants) across the codebase. List file + line."
    "Read lmcache/connections.py and explain the connection pool in one paragraph."
    "Read lmcache/v1/cache_engine.py and explain the cache lifecycle in three bullets."
    "Find every place that registers a storage backend. List file + line."
    "Read benchmarks/ to find the main benchmark entry script and explain what it measures."
    "Read lmcache/storage_backend/__init__.py and explain how storage backends are dispatched."
)

mkdir -p "$TRACE_DIR/sessions"

for i in "${!TASKS[@]}"; do
    n=$(printf "%02d" "$((i + 1))")
    task="${TASKS[$i]}"
    echo "=== task $n: $task" | tee -a "$TRACE_DIR/sessions.log"
    timeout 600 opencode run --format json \
        --model "local/google/gemma-4-E4B-it" "$task" \
        > "$TRACE_DIR/sessions/task_${n}.opencode.json" \
        2> "$TRACE_DIR/sessions/task_${n}.opencode.err" || \
        echo "  (task $n failed or timed out)" | tee -a "$TRACE_DIR/sessions.log"
    echo "  done" | tee -a "$TRACE_DIR/sessions.log"
done

echo "all tasks complete; trace dumps in $TRACE_DIR"
