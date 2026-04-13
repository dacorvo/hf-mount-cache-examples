# conversations.sh — conversation runner and topic definitions
#
# Expected globals: SCRIPT_DIR, MOUNT_POINT, LOG_DIR
# Expected functions: log, cache_file_count

run_conversation() {
  local label="$1"; shift
  local turn=1
  local total=$#
  local conv_log="$LOG_DIR/conversation-${label}.log"

  log "--- Conversation: $label ($total turns) --- (log: $conv_log)"
  echo ""

  # First turn: new session.
  local prompt="$1"; shift
  log "  [$label] Turn $turn/$total"
  log "  Prompt: $prompt"
  (cd "$SCRIPT_DIR" && opencode run "$prompt") 2>&1 | tee -a "$conv_log"
  echo ""
  if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
    log "  Cache files: $(cache_file_count)"
  fi
  turn=$((turn + 1))

  # Remaining turns: continue session.
  for prompt in "$@"; do
    log "  [$label] Turn $turn/$total"
    log "  Continue: $prompt"
    (cd "$SCRIPT_DIR" && opencode run -c "$prompt") 2>&1 | tee -a "$conv_log"
    echo ""
    if grep -q "$MOUNT_POINT" /proc/mounts 2>/dev/null; then
      log "  Cache files: $(cache_file_count)"
    fi
    turn=$((turn + 1))
  done

  log "--- $label complete ---"
  echo ""
}

# ── Conversation definitions ───────────────────────────────────────────
#
# Two sets of conversations, each with 3 topics x 5 turns.
# Warmup and consume use different prompts so that consume does not get
# artificially high cache hit rates from identical token sequences.
# Cache hits in consume come from shared system-prompt prefixes and
# overlapping file reads, which is the realistic scenario.

conversations_warmup() {
  run_conversation "overlay-impl" \
    "Read src/setup.rs and explain what the --overlay flag does" \
    "How does overlay mode interact with the advanced_writes flag?" \
    "Now read src/virtual_fs/mod.rs and describe the overlay_root helper" \
    "Explain the staging_path method and how it differs in overlay mode" \
    "Summarize the full overlay data flow: from CLI flag to file read/write"

  run_conversation "test-coverage" \
    "Read src/virtual_fs/tests.rs and list all test functions related to overlay" \
    "Pick the three most important overlay tests and explain what each one verifies" \
    "Are there any edge cases in the overlay code that lack test coverage?" \
    "Read src/test_mocks.rs and explain how the mock filesystem supports overlay testing" \
    "Suggest one new integration test that would improve confidence in overlay mode"

  run_conversation "documentation" \
    "Read the README.md and summarize what it says about mounting buckets" \
    "Does the README mention the --overlay flag? What is missing?" \
    "Read src/setup.rs and draft a short usage section for overlay mode" \
    "What caveats should the documentation mention about overlay writes and persistence?" \
    "Write a concise FAQ entry: when should a user choose --overlay vs --read-only?"
}

conversations_consume() {
  run_conversation "nfs-backend" \
    "Read src/bin/hf-mount-nfs.rs and explain how the NFS server is started" \
    "How does NFS mode differ from FUSE mode in this project?" \
    "Read src/setup.rs and describe the is_nfs code paths" \
    "What NFS-specific limitations should a user be aware of?" \
    "Summarize the trade-offs between NFS and FUSE for overlay mounts"

  run_conversation "write-pipeline" \
    "Read src/xet.rs and explain the StagingDir and XetSessions structures" \
    "How does the flush manager decide when to upload staged files?" \
    "Read src/virtual_fs/mod.rs and trace what happens when a file is created and written" \
    "How does the write pipeline change when overlay mode is active?" \
    "What happens to staged files when the mount is unmounted gracefully?"

  run_conversation "error-handling" \
    "Read src/virtual_fs/mod.rs and find all places that return EPERM" \
    "Which operations are blocked in overlay mode and why?" \
    "What happens if a user tries to delete a remote file through an overlay mount?" \
    "Read the rename implementation and explain how overlay guards work" \
    "Suggest improvements to the error messages for overlay permission denials"
}
