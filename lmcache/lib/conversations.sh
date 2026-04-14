# conversations.sh — parallel conversation runner and topic definitions
#
# Expected globals: SCRIPT_DIR, MOUNT_POINT, LOG_DIR, MAX_MODEL_LEN, VLLM_URL
# Expected functions: log, cache_file_count, prom_metric

# ── Adaptive conversation runner ──────────────────────────────────────
#
# run_conversation <label> <grow_prompt_1> ... -- <post_prompt_1> ...
#
# Phase 1 (grow): sends grow prompts one by one, checking token count
#   after each turn. Stops when context reaches 90% of MAX_MODEL_LEN.
# Phase 2 (post-compaction): sends post-compaction prompts to verify
#   cache hits survive opencode's context compaction.
#
# The "--" separator divides grow prompts from post-compaction prompts.
# If 90% is never reached, all grow prompts run then post prompts follow.
#
# Writes per-conversation stats to $LOG_DIR/conv-stats-${label}.txt.
# Designed to run concurrently — uses wall-clock timing for first-turn
# TTFT and skips shared-state queries like prometheus diffs.

run_conversation() {
  local label="$1"; shift
  local conv_log="$LOG_DIR/conversation-${label}.log"
  local stats_file="$LOG_DIR/conv-stats-${label}.txt"
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
  log "--- Conversation: $label (up to $total turns, compaction at $threshold tokens) ---"

  local turn=0
  local max_tokens=0
  local compacted=false
  local first_turn_ttft_ms=""

  # Phase 1: grow context until 90% threshold.
  for i in "${!grow_prompts[@]}"; do
    local prompt="${grow_prompts[$i]}"
    turn=$((turn + 1))

    if [ "$turn" -eq 1 ]; then
      log "  [$label] Turn $turn (grow)"
      # Measure first-turn wall-clock time (includes network + prefill + first token).
      local t_start t_end
      t_start=$(date +%s%3N)
      (cd "$SCRIPT_DIR" && opencode run "$prompt") >> "$conv_log" 2>&1
      t_end=$(date +%s%3N)
      first_turn_ttft_ms=$(( t_end - t_start ))
      log "  [$label] First-turn wall time: ${first_turn_ttft_ms}ms"
    else
      log "  [$label] Turn $turn (grow)"
      (cd "$SCRIPT_DIR" && opencode run -c "$prompt") >> "$conv_log" 2>&1
    fi

    # Check token count from conversation log (grep opencode output for token info).
    # Under concurrency, the vLLM log is shared, so we use a rough estimate:
    # the conversation log size as a proxy for context growth.
    local log_tokens
    log_tokens=$(wc -c < "$conv_log" 2>/dev/null || echo 0)
    # Rough estimate: 4 chars per token for English text
    local est_tokens=$(( log_tokens / 4 ))
    [ "$est_tokens" -gt "$max_tokens" ] && max_tokens="$est_tokens"

    if [ "$est_tokens" -ge "$threshold" ]; then
      log "  [$label] >>> Estimated ~${est_tokens} tokens, switching to post-compaction"
      compacted=true
      break
    fi
  done

  # Phase 2: post-compaction turns.
  for prompt in "${post_prompts[@]}"; do
    turn=$((turn + 1))
    log "  [$label] Turn $turn (post-compaction)"
    (cd "$SCRIPT_DIR" && opencode run -c "$prompt") >> "$conv_log" 2>&1
  done

  # Write per-conversation stats.
  {
    echo "label=$label"
    echo "turns=$turn"
    echo "max_tokens=$max_tokens"
    echo "compacted=$compacted"
    [ -n "$first_turn_ttft_ms" ] && echo "first_turn_ttft_ms=$first_turn_ttft_ms"
  } > "$stats_file"

  log "--- $label complete ($turn turns) ---"
}

# ── Parallel wrapper ──────────────────────────────────────────────────
#
# run_conversations_parallel <func_name>
#
# Calls the given function, which backgrounds all run_conversation calls.
# Waits for all background jobs to complete.

run_conversations_parallel() {
  local func="$1"
  log "Starting conversations in parallel..."
  # Capture background PIDs from the conversation function.
  local pids_before
  pids_before=$(jobs -p 2>/dev/null | sort)
  $func
  local pids_after
  pids_after=$(jobs -p 2>/dev/null | sort)
  # Wait only for the new PIDs (the conversation jobs).
  local conv_pids
  conv_pids=$(comm -13 <(echo "$pids_before") <(echo "$pids_after"))
  if [ -n "$conv_pids" ]; then
    log "Waiting for $(echo "$conv_pids" | wc -w) conversations..."
    wait $conv_pids
  fi
  log "All conversations complete."
}

# ── Conversation definitions ──────────────────────────────────────────
#
# 6 conversations per set (warmup / consume) for better averaging.
# Each conversation runs as a background job for parallelism.
# Each has grow prompts (generation-heavy) separated by "--" from
# post-compaction prompts (short follow-ups).
#
# Warmup and consume use different topics so that cache hits come from
# shared system-prompt prefixes, not identical conversations.

conversations_warmup() {
  run_conversation "overlay-codegen" \
    "Read ../hf-mount/src/setup.rs and src/virtual_fs/mod.rs. Write a complete Rust module that implements a standalone overlay filesystem library extracted from this codebase, with full doc comments." \
    "Add comprehensive error handling to the module you just wrote. Every public function should return a Result with a custom OverlayError enum. Write the full updated code." \
    "Write a complete test module for the overlay library with at least 10 unit tests covering: reads, writes, deletes, renames, directory operations, and conflict resolution." \
    "Read ../hf-mount/src/test_mocks.rs. Write a mock filesystem backend for the overlay library that implements all the traits needed for testing, with configurable failure injection." \
    "Write a CLI binary that uses the overlay library to mount a directory with overlay semantics. Include argument parsing with clap, signal handling, and graceful shutdown." \
    "Write a complete README.md for the overlay library with: overview, installation, usage examples, API reference, architecture diagram in ASCII, and contributing guide." \
    "Refactor all the code above to use async/await throughout. Write the full updated implementation of every module." \
    "Write a CHANGELOG.md and a migration guide for users moving from the old inline overlay code to the new library." \
    -- \
    "Summarize the key design decisions in the overlay library in 3 bullet points." \
    "What is the single most important test case you wrote and why?" \
    "List the public API surface of the overlay library: function signatures only." &

  run_conversation "cache-design" \
    "Read ../hf-mount/src/xet.rs and src/virtual_fs/mod.rs. Design and write a complete Rust caching layer that sits between the virtual filesystem and the storage backend, with LRU eviction and configurable size limits." \
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
    "List all configuration parameters with their default values." &

  run_conversation "nfs-rewrite" \
    "Read ../hf-mount/src/bin/hf-mount-nfs.rs. Write a complete rewrite of the NFS server startup code with better error handling, structured logging, and graceful shutdown." \
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
    "List all the health check endpoints with their expected response format." &

  run_conversation "streaming-impl" \
    "Read ../hf-mount/src/virtual_fs/mod.rs and src/xet.rs. Write a complete streaming read implementation that serves file data as it is downloaded, without waiting for the full file to be cached locally." \
    "Write a prefetch scheduler that predicts which files will be read next based on access patterns. Implement the full prediction logic and prefetch queue." \
    "Write a bandwidth manager that throttles prefetch operations when active reads are in progress. Include configurable priority levels and bandwidth limits." \
    "Write a complete test suite for the streaming read path including: partial reads, concurrent reads of the same file, read-ahead cancellation, and error recovery." \
    "Write a metrics dashboard specification for the streaming system: what to measure, alert thresholds, and Grafana panel definitions in JSON." \
    "Write a retry and fallback system for failed streaming reads. Include circuit breakers, timeout handling, and automatic fallback to full-file download." \
    "Write a complete user guide for the streaming read feature: configuration options, performance tuning, and troubleshooting common issues." \
    "Refactor the streaming implementation to support multiple concurrent download backends (HTTP, S3, GCS) with a trait-based abstraction." \
    -- \
    "What happens if a prefetched file is evicted before it is read?" \
    "Describe the worst-case latency scenario for a streaming read." \
    "List all configurable parameters for the prefetch scheduler." &

  run_conversation "auth-system" \
    "Read any authentication or token handling code in the project. Write a complete token management module that handles: HF token refresh, expiry tracking, multi-account support, and secure storage using OS keyrings." \
    "Write an OAuth2 device flow implementation for CLI authentication. Include the full flow: device code request, user notification, polling, and token exchange." \
    "Write a credential cache that stores tokens encrypted at rest with a user-provided passphrase. Use AES-256-GCM for encryption." \
    "Write comprehensive tests for the auth system including: token refresh races, expired token handling, keyring failures, and encryption round-trips." \
    "Write a security hardening guide for the auth system covering: token storage best practices, rotation policies, and audit logging." \
    "Write a multi-tenant authentication proxy that routes requests to different HF accounts based on namespace prefixes in the mount path." \
    "Write an audit logging system that records all authentication events: login, token refresh, token expiry, and failed attempts." \
    "Write a complete integration test that simulates the full lifecycle: login, use, token refresh, and logout." \
    -- \
    "What is the most dangerous security mistake a user could make with this auth system?" \
    "List all the places where tokens are stored or transmitted." \
    "Describe the token refresh flow in 5 steps." &

  run_conversation "perf-benchmark" \
    "Read the project source code. Write a complete benchmarking framework in Rust using criterion that measures: mount latency, file read throughput, write throughput, metadata operation latency, and concurrent access scalability." \
    "Write a load generator that simulates realistic workloads: sequential large file reads, random small file reads, burst writes, and mixed read-write patterns." \
    "Write a comparison benchmark that runs the same workload against: local filesystem, NFS mount, FUSE mount, and overlay mount. Generate a markdown report with tables." \
    "Write a regression test that compares current performance against stored baselines and flags any degradation beyond configurable thresholds." \
    "Write a CI integration that runs the benchmarks on every PR and posts a performance summary as a PR comment." \
    "Write a statistical analysis module that computes confidence intervals, detects outliers, and generates publication-quality charts using plotters." \
    "Write a profiling integration that captures flamegraphs during benchmark runs and identifies the top 10 hotspots." \
    "Write a capacity planning tool that extrapolates benchmark results to predict performance at different scales (more files, larger files, more concurrent users)." \
    -- \
    "Which metric is most important for the overlay use case and why?" \
    "What is the minimum number of benchmark iterations needed for statistical significance?" \
    "List all environment variables that affect benchmark behavior." &
}

conversations_consume() {
  run_conversation "fuse-codegen" \
    "Read ../hf-mount/src/setup.rs and explain the FUSE mount setup. Then write a complete Rust implementation of a FUSE passthrough filesystem that mirrors a local directory with read-write support." \
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
    "List the systemd service dependencies in order." &

  run_conversation "api-codegen" \
    "Read ../hf-mount/src/hub_api.rs or any HTTP client code in the project. Write a complete Rust HTTP API client library for HuggingFace Hub with typed request/response models for all bucket operations." \
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
    "List the environment variables the token manager checks, in priority order." &

  run_conversation "config-codegen" \
    "Read ../hf-mount/src/setup.rs and the CLI argument parsing code. Write a complete configuration management library in Rust that merges config from: YAML files, environment variables, CLI flags, and defaults." \
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
    "Summarize the live reload mechanism in 3 sentences." &

  run_conversation "plugin-system" \
    "Read the project source code and identify extension points. Write a complete plugin system in Rust that allows loading dynamic libraries (.so) at runtime to extend the filesystem with custom backends." \
    "Write a plugin API with versioned traits: StoragePlugin, CachePlugin, and AuthPlugin. Include backward compatibility guarantees and a negotiation protocol." \
    "Write a plugin manager that handles: discovery (scan a directory), loading (dlopen), initialization, dependency resolution, and graceful unloading." \
    "Write three example plugins: a logging plugin that traces all filesystem operations, a compression plugin that transparently compresses cached files, and a quota plugin that enforces per-user storage limits." \
    "Write comprehensive tests for the plugin system including: version mismatch handling, plugin crashes, circular dependencies, and hot-reload." \
    "Write a plugin sandboxing system that restricts plugins to specific filesystem operations and prevents them from accessing files outside their designated scope." \
    "Write a plugin marketplace specification: manifest format, versioning, dependency declaration, and installation protocol." \
    "Write a complete developer guide for plugin authors: API reference, lifecycle hooks, debugging tips, and example project template." \
    -- \
    "What is the most dangerous thing a malicious plugin could do?" \
    "List the trait methods each plugin type must implement." \
    "Describe the plugin lifecycle from discovery to unloading." &

  run_conversation "migration-tool" \
    "Write a complete data migration tool in Rust that moves files between HuggingFace buckets with progress reporting, checksums, and resume support." \
    "Add parallel transfer support with configurable concurrency. Implement work-stealing across transfer threads for optimal throughput." \
    "Write a dry-run mode that shows what would be transferred without actually moving data. Include a diff report showing new, modified, and deleted files." \
    "Write a schedule system that can run migrations at specified times with retry logic and email/webhook notifications on completion or failure." \
    "Write an integration test suite that creates temporary buckets, migrates data, and verifies integrity." \
    "Write a conflict resolution system for migrations: detect conflicting files at the destination, offer merge strategies (overwrite, skip, rename), and generate conflict reports." \
    "Write a bandwidth throttling system for migrations that limits transfer speed to avoid saturating network links, with configurable schedules." \
    "Write a migration audit trail that records every file transferred, skipped, or failed, with checksums and timestamps, exportable as CSV." \
    -- \
    "What is the safest way to handle a migration that fails halfway through?" \
    "List all the CLI flags the migration tool accepts." \
    "Describe the resume mechanism: what state is persisted and where." &

  run_conversation "monitoring-stack" \
    "Write a complete monitoring stack for hf-mount deployments: a Prometheus exporter that collects mount health, I/O throughput, cache hit rates, and error counts from all running mounts on a machine." \
    "Write alerting rules for Prometheus that detect: mount failures, degraded performance, disk space exhaustion, and token expiry." \
    "Write a Grafana dashboard in JSON that visualizes: mount status, throughput over time, cache hit ratio, and error rate. Include variable selectors for filtering by mount point." \
    "Write a log aggregation pipeline using fluentd that collects hf-mount logs, parses structured fields, and forwards to Elasticsearch with proper index templates." \
    "Write runbooks for the three most common operational issues: mount won't start, slow reads, and cache corruption." \
    "Write an SLO definition for hf-mount availability and performance: target percentiles, error budgets, and burn rate alerts." \
    "Write a capacity planning model that predicts storage and bandwidth requirements based on mount count, file sizes, and access patterns." \
    "Write an incident response playbook for hf-mount outages: detection, triage, mitigation, and post-mortem template." \
    -- \
    "What single metric would you alert on first and at what threshold?" \
    "List all the Prometheus metrics the exporter exposes." \
    "Describe the escalation path for a mount failure alert." &
}
