#!/usr/bin/env bash
# Phase 2.6: scale up the natural-prompt task suite for stronger
# statistics on body-match density. 30 sessions on lmcache-src,
# mix of investigative / exploratory framings, all natural.
#
# Designed to run overnight. ETA: ~3-4 h on chunked-SDPA bf16
# Gemma-4 E4B / 4×A10G.

set -uo pipefail
TASKS_DIR="$(dirname "$0")/.."
LMCACHE_DIR="$TASKS_DIR/lmcache-src"
TRACE_DIR="$TASKS_DIR/runs/phase2_6-2026-05-01-1700"

cd "$LMCACHE_DIR"

declare -a TASKS=(
"Read README.md and summarise LMCache's value proposition in 3 bullets."

"Find the entry point where vLLM hands off a sequence to LMCache and trace 3 levels of method calls. Cite file:line at each step."

"Audit lmcache/connections.py for connection leaks. Cite specific lines."

"Walk me through how the Redis storage backend serializes K/V chunks."

"How does LMCache decide whether to keep a chunk on GPU or move it to CPU?"

"Find every place that touches the prefix tree and explain its role."

"Compare the v0 and v1 storage_backend implementations. What changed?"

"Trace what happens when a cache miss occurs in v1, from lookup through fallback prefill."

"Read the benchmarks/ directory and explain what each benchmark measures."

"Where is the cache hit rate computed today, and where would I add a hit-rate metric?"

"How is storage backend selection configured? Walk through the config code."

"How does LMCache handle concurrent reader access to the same chunk?"

"Find every callsite of \`evict_lru\` (or close variants). Give file:line and a one-line note."

"Read lmcache/v1/cache_engine.py and explain the cache lifecycle in 4 bullets."

"Find every \`raise\` in lmcache/v1/cache_engine.py. Classify the recovery as retry, re-raise, or swallow with file:line."

"Find every \`async def\` in lmcache/. Cite file:line and one-line purpose."

"Read tests/v1/cache_controller/. What scenarios do the existing tests cover?"

"Find every place a storage backend is registered. Cite file:line."

"How does LMCache integrate with vLLM's \`forward\` callback chain? Trace the path."

"Read lmcache/cli/. What commands does the CLI expose? One line each."

"Locate where K/V tensors are serialised to disk. What format is used?"

"How does LMCache handle GPU OOM during a cache miss?"

"What's the difference between LMCacheEngine.put and LMCacheEngine.put_async?"

"Find the configuration parser. What are the top 5 most-used config keys based on lmcache/v1/?"

"Trace the lifecycle of a single K/V chunk from creation to eviction."

"What lock or synchronization primitive does the cache use to protect concurrent writes?"

"Read lmcache/observability.py and explain what it instruments."

"Read lmcache/integration/ — how many integrations are there and what do they each do?"

"Where in the codebase are tensors offloaded between GPU and CPU? List file:line and a one-line note."

"Find the main \`__init__.py\` of the lmcache package. What is exported at the top level?"
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
