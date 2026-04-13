# conversations.sh — adaptive conversation runner and topic definitions
#
# Expected globals: SCRIPT_DIR, MOUNT_POINT, LOG_DIR, MAX_MODEL_LEN, VLLM_URL
# Expected functions: log, cache_file_count

# Get the latest token count from the vLLM log for the most recent request.
last_token_count() {
  grep "Total tokens" "$LOG_DIR/vllm.log" 2>/dev/null \
    | tail -1 \
    | sed 's/.*Total tokens \([0-9]*\).*/\1/' \
    || echo "0"
}

# ── Adaptive conversation runner ──────────────────────────────────────
#
# run_conversation <label> <grow_prompt_1> ... -- <post_prompt_1> ...
#
# Phase 1 (grow): sends grow prompts one by one, checking token count
#   after each turn. Stops when context reaches 90% of MAX_MODEL_LEN.
# Phase 2 (post-compaction): sends up to 3 post-compaction prompts to
#   verify cache hits survive opencode's context compaction.
#
# The "--" separator divides grow prompts from post-compaction prompts.
# If 90% is never reached, all grow prompts run then post prompts follow.

run_conversation() {
  local label="$1"; shift
  local conv_log="$LOG_DIR/conversation-${label}.log"
  local threshold=$(( MAX_MODEL_LEN * 9 / 10 ))

  # Split prompts at "--" separator.
  local -a grow_prompts=()
  local -a post_prompts=()
  local past_separator=false
  for arg in "$@"; do
    if [ "$arg" = "--" ]; then
      past_separator=true
      continue
    fi
    if $past_separator; then
      post_prompts+=("$arg")
    else
      grow_prompts+=("$arg")
    fi
  done

  local total=$(( ${#grow_prompts[@]} + ${#post_prompts[@]} ))
  log "--- Conversation: $label (up to $total turns, compaction at $threshold tokens) --- (log: $conv_log)"
  echo ""

  local turn=0
  local compacted=false

  # Phase 1: grow context until 90% threshold.
  for i in "${!grow_prompts[@]}"; do
    local prompt="${grow_prompts[$i]}"
    turn=$((turn + 1))

    if [ "$turn" -eq 1 ]; then
      log "  [$label] Turn $turn (grow)"
      log "  Prompt: $prompt"
      (cd "$SCRIPT_DIR" && opencode run "$prompt") 2>&1 | tee -a "$conv_log"
    else
      log "  [$label] Turn $turn (grow)"
      log "  Continue: $prompt"
      (cd "$SCRIPT_DIR" && opencode run -c "$prompt") 2>&1 | tee -a "$conv_log"
    fi
    echo ""

    if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
      log "  Cache files: $(cache_file_count)"
    fi

    local tokens
    tokens=$(last_token_count)
    log "  Tokens: $tokens / $threshold ($(( tokens * 100 / MAX_MODEL_LEN ))%)"

    if [ "$tokens" -ge "$threshold" ]; then
      log "  >>> Reached 90% of context — switching to post-compaction prompts"
      compacted=true
      break
    fi
  done

  # Phase 2: post-compaction turns (always run, up to 3).
  local post_count=0
  for prompt in "${post_prompts[@]}"; do
    turn=$((turn + 1))
    post_count=$((post_count + 1))
    log "  [$label] Turn $turn (post-compaction $post_count/${#post_prompts[@]})"
    log "  Continue: $prompt"
    (cd "$SCRIPT_DIR" && opencode run -c "$prompt") 2>&1 | tee -a "$conv_log"
    echo ""

    if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
      log "  Cache files: $(cache_file_count)"
    fi

    local tokens
    tokens=$(last_token_count)
    log "  Tokens: $tokens / $threshold ($(( tokens * 100 / MAX_MODEL_LEN ))%)"
  done

  if $compacted; then
    log "--- $label complete (compaction triggered, $turn turns) ---"
  else
    log "--- $label complete ($turn turns, threshold not reached) ---"
  fi
  echo ""
}

# ── Conversation definitions ──────────────────────────────────────────
#
# Each conversation has two sections separated by "--":
#   - Grow prompts: generation-heavy (write code, docs, tests) to inflate
#     the context window toward 90% of MAX_MODEL_LEN.
#   - Post-compaction prompts: shorter follow-ups to verify cache hits
#     survive after opencode compacts the conversation history.
#
# Warmup and consume use different topics so that cache hits come from
# shared system-prompt prefixes, not identical conversations.

conversations_warmup() {
  run_conversation "overlay-codegen" \
    "Read src/setup.rs and src/virtual_fs/mod.rs. Write a complete Rust module that implements a standalone overlay filesystem library extracted from this codebase, with full doc comments." \
    "Add comprehensive error handling to the module you just wrote. Every public function should return a Result with a custom OverlayError enum. Write the full updated code." \
    "Write a complete test module for the overlay library with at least 10 unit tests covering: reads, writes, deletes, renames, directory operations, and conflict resolution." \
    "Read src/test_mocks.rs. Write a mock filesystem backend for the overlay library that implements all the traits needed for testing, with configurable failure injection." \
    "Write a CLI binary that uses the overlay library to mount a directory with overlay semantics. Include argument parsing with clap, signal handling, and graceful shutdown." \
    "Write a complete README.md for the overlay library with: overview, installation, usage examples, API reference, architecture diagram in ASCII, and contributing guide." \
    "Refactor all the code above to use async/await throughout. Write the full updated implementation of every module." \
    "Write a CHANGELOG.md and a migration guide for users moving from the old inline overlay code to the new library." \
    -- \
    "Summarize the key design decisions in the overlay library in 3 bullet points." \
    "What is the single most important test case you wrote and why?" \
    "List the public API surface of the overlay library: function signatures only."

  run_conversation "cache-design" \
    "Read src/xet.rs and src/virtual_fs/mod.rs. Design and write a complete Rust caching layer that sits between the virtual filesystem and the storage backend, with LRU eviction and configurable size limits." \
    "Write the full implementation of the cache eviction policy. Support LRU, LFU, and FIFO strategies with a trait-based design. Include all three implementations." \
    "Write a comprehensive benchmark suite for the caching layer using criterion. Include benchmarks for: sequential reads, random reads, cache hit/miss ratios, and eviction performance." \
    "Write integration tests that verify the caching layer works correctly with concurrent readers and writers. Use tokio for async test execution." \
    "Write a metrics and observability module for the caching layer. Export prometheus metrics for: hit rate, miss rate, eviction count, cache size, and latency histograms." \
    "Write a configuration system for the caching layer that reads from YAML, environment variables, and command-line flags, with validation and defaults." \
    "Refactor all the code above to use async/await throughout. Write the full updated implementation of every module." \
    "Write a design document (ADR) explaining the architecture decisions, trade-offs, and alternatives considered for the caching layer." \
    -- \
    "What are the three most likely failure modes of the caching layer?" \
    "Which eviction strategy would you recommend for KV cache chunks and why?" \
    "List all configuration parameters with their default values."

  run_conversation "nfs-rewrite" \
    "Read src/bin/hf-mount-nfs.rs. Write a complete rewrite of the NFS server startup code with better error handling, structured logging, and graceful shutdown." \
    "Write a connection pool manager for the NFS server that handles multiple concurrent client connections with configurable limits and timeouts." \
    "Write a complete health check system for the NFS server: readiness probe, liveness probe, and a /status HTTP endpoint that reports connection count, uptime, and error rates." \
    "Write a comprehensive test suite for the NFS server with both unit tests and integration tests. Mock the filesystem layer and test all NFS operations." \
    "Write a Dockerfile and docker-compose.yml for running the NFS server in a container with proper signal handling, health checks, and volume mounts." \
    "Write a Kubernetes deployment manifest (Deployment, Service, ConfigMap, PVC) for the NFS server with resource limits, anti-affinity, and rolling updates." \
    "Write a complete troubleshooting guide for the NFS server covering: mount failures, permission errors, performance issues, and network debugging." \
    "Write a performance tuning guide with benchmarks comparing different NFS configurations: read-ahead sizes, write-back policies, and thread pool sizes." \
    -- \
    "What is the most common NFS mount failure and how should the error message guide the user?" \
    "Summarize the Kubernetes deployment in 5 lines of pseudocode." \
    "List all the health check endpoints with their expected response format."
}

conversations_consume() {
  run_conversation "fuse-codegen" \
    "Read src/setup.rs and explain the FUSE mount setup. Then write a complete Rust implementation of a FUSE passthrough filesystem that mirrors a local directory with read-write support." \
    "Add overlay semantics to the passthrough filesystem: reads check local first then remote, writes always go local. Write the full updated implementation." \
    "Write a complete test harness for the FUSE filesystem with automated mount/unmount, file operation tests, and performance benchmarks." \
    "Write a user-space cache for the FUSE filesystem that caches file metadata and small file contents in memory. Include cache invalidation logic and size limits." \
    "Write a complete logging and tracing system for the FUSE filesystem using the tracing crate. Add span-based tracing for every filesystem operation." \
    "Write a systemd service file, install script, and man page for the FUSE filesystem. Include proper dependency ordering and socket activation support." \
    "Write a stress test that creates thousands of files, performs random reads and writes, and verifies data integrity after each operation." \
    "Write a security audit report for the FUSE implementation covering: path traversal, symlink attacks, race conditions, and resource exhaustion." \
    -- \
    "What is the single biggest security risk in the FUSE passthrough and how would you mitigate it?" \
    "Summarize the tracing spans you added: name and purpose of each." \
    "List the systemd service dependencies in order."

  run_conversation "api-codegen" \
    "Read src/hub_api.rs or any HTTP client code in the project. Write a complete Rust HTTP API client library for HuggingFace Hub with typed request/response models for all bucket operations." \
    "Add retry logic with exponential backoff, circuit breaker pattern, and request rate limiting to the API client. Write the full implementation." \
    "Write a complete mock HTTP server for testing the API client. Support configurable responses, latency simulation, and error injection." \
    "Write a CLI tool that uses the API client to manage HuggingFace buckets: list, create, delete, upload, download, and sync operations." \
    "Write comprehensive API documentation with examples for every endpoint. Include curl examples, Rust code examples, and error response documentation." \
    "Write a token management system for the API client: token refresh, secure storage, multiple account support, and environment variable fallback." \
    "Write a parallel upload/download manager that handles large files with multipart uploads, resumable transfers, and progress reporting." \
    "Write a complete integration test suite that runs against the mock server and verifies every API operation including edge cases and error handling." \
    -- \
    "What HTTP status codes does the retry logic handle and which ones should it not retry?" \
    "Summarize the CLI commands: name and one-line description for each." \
    "List the environment variables the token manager checks, in priority order."

  run_conversation "config-codegen" \
    "Read src/setup.rs and the CLI argument parsing code. Write a complete configuration management library in Rust that merges config from: YAML files, environment variables, CLI flags, and defaults." \
    "Write a validation layer for the configuration library with custom validators, dependency checks between fields, and helpful error messages." \
    "Write a configuration migration system that handles version upgrades: schema versioning, automatic migration, and backward compatibility." \
    "Write a live configuration reload system that watches config files for changes and applies updates without restarting. Use notify for file watching." \
    "Write a configuration documentation generator that produces Markdown docs from the config schema, including descriptions, defaults, and examples." \
    "Write a configuration diff and audit system that logs all configuration changes with timestamps, previous values, and the source of each change." \
    "Write a complete test suite covering: parsing, validation, migration, live reload, and edge cases like missing files and invalid values." \
    "Write a user guide for the configuration library with: getting started, advanced usage, migration from other config systems, and FAQ." \
    -- \
    "What happens if a YAML config file and an environment variable conflict? Which wins?" \
    "List all migration steps needed to go from config v1 to v2." \
    "Summarize the live reload mechanism in 3 sentences."
}
