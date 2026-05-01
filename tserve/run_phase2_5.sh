#!/usr/bin/env bash
# Phase 2.5: natural-prompt deep-trajectory OpenCode sessions.
#
# Tasks are framed as realistic engineering questions a developer
# would actually ask about an unfamiliar codebase. No artificial
# scaffolding, no "for each X do Y", no offset/limit constraints.
# Whatever the agent does in response is what we measure.
#
# Server: chunked-SDPA bf16 Gemma-4 E4B on 4 A10G (handles 21k+
# context without OOM thanks to TSERVE_SDPA_CHUNK_Q=1024).

set -uo pipefail
TASKS_DIR="$(dirname "$0")/.."
LMCACHE_DIR="$TASKS_DIR/lmcache-src"
TRACE_DIR="$TASKS_DIR/runs/phase2_5-2026-05-01-1430"

cd "$LMCACHE_DIR"

declare -a TASKS=(
"Implement a new in-memory storage backend that mirrors the existing CPU backend in lmcache/storage_backend/. Outline what files I would need to create or modify and what the main methods would look like — no need to actually write the file."

"I'm seeing a regression where evictions sometimes leave stale shards on disk after a crash. Help me find which code paths could be responsible by reading the v1 eviction logic."

"Audit the connection-pool lifecycle in lmcache/connections.py for potential leaks or race conditions. Cite specific lines."

"I want to write integration tests for LMCacheEngine.put. Look at the existing tests and outline what new test cases I should add to cover edge cases."

"Walk me through how the vLLM integration hands off a sequence to LMCache. Trace the call chain from the entry point through three levels of method calls."

"Compare the v0 storage_backend implementations to the v1 ones. What changed structurally?"

"Find every place that touches the prefix tree and explain how it's used. Cite file:line for each."

"How does LMCache decide whether to keep a chunk on GPU or move it to CPU? Walk me through the policy."

"Trace what happens internally when a cache miss occurs in LMCache.v1, from the lookup call to the fallback prefill."

"Read the benchmarks/ directory. What does each benchmark measure?"

"I want to add a metric for the cache hit rate. Where should I instrument it? Show me the relevant places."

"Walk through how the Redis backend serializes K/V chunks for storage and what overhead that adds."

"How do I configure storage backend selection? Read the configuration code and explain."

"How does LMCache handle concurrent access to the same chunk from multiple readers? Find the locking or serialization logic."

"Find the entry point where vLLM hands off a sequence to LMCache, then trace 3 levels of method calls. List file:line at each step."
)

mkdir -p "$TRACE_DIR/sessions"

for i in "${!TASKS[@]}"; do
    n=$(printf "%02d" "$((i + 1))")
    task="${TASKS[$i]}"
    echo "=== task $n: ${task:0:80}..." | tee -a "$TRACE_DIR/sessions.log"
    timeout 1800 opencode run --format json \
        --model "local/google/gemma-4-E4B-it" "$task" \
        > "$TRACE_DIR/sessions/task_${n}.opencode.json" \
        2> "$TRACE_DIR/sessions/task_${n}.opencode.err" || \
        echo "  (task $n failed or timed out)" | tee -a "$TRACE_DIR/sessions.log"
    n_steps=$(grep -c '"type":"step_finish"' "$TRACE_DIR/sessions/task_${n}.opencode.json" 2>/dev/null || echo 0)
    n_tool=$(grep -c '"type":"tool_use"' "$TRACE_DIR/sessions/task_${n}.opencode.json" 2>/dev/null || echo 0)
    n_errs=$(grep -c '"type":"error"' "$TRACE_DIR/sessions/task_${n}.opencode.json" 2>/dev/null || echo 0)
    echo "  done (steps=$n_steps tool_calls=$n_tool errors=$n_errs)" | tee -a "$TRACE_DIR/sessions.log"
done

echo "all tasks complete; trace dumps in $TRACE_DIR"
