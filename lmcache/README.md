# LMCache + vLLM + hf-mount Integration Test

Verifies that LMCache's bucket-backed external prefix cache works
end-to-end through an HF Bucket mounted with `hf-mount`, and that
**overlay** mode lets a consumer reuse the bucket-stored prefix without
mutating the bucket.

## What it does

LMCache intercepts vLLM's KV-cache I/O and persists chunks to a
configured backend. Pointing its `gds_path` inside an `hf-mount`-mounted
bucket makes the chunks shared.

Three phases (mirroring `../torch.compile/`):

| Phase    | Mount    | Action                                                                 | Bucket effect            |
|----------|----------|------------------------------------------------------------------------|--------------------------|
| warmup   | rw       | One hermes turn → vLLM computes prefill → LMCache writes chunks        | Files uploaded           |
| consume  | overlay  | Cold-start vLLM → hermes turn with a *different* prompt → prefix hit   | Unchanged (writes local) |
| verify   | —        | Diff bucket file list, check `external_cache_hits`, overlay-local files| —                        |

The cache hit in `consume` comes from the **shared hermes system prompt
prefix** (~13K tokens of tool definitions, soul, skills) — identical
across runs. The user-message tail differs between warmup
(`prompts/warmup/auth-system.txt`) and consume
(`prompts/consume/api-codegen.txt`) so the visible cache hit is purely
prefix coverage, not a trivial whole-prompt match.

## Prerequisites

NVIDIA GPU with CUDA 12.1+, Rust toolchain, Python 3.10+. From the repo
root:

```bash
./setup.sh                       # builds hf-mount, venv with vLLM
source .venv/bin/activate
cd lmcache && ./setup.sh         # installs LMCache
```

Plus **hermes-agent** must be on PATH (typically `~/.local/bin/hermes`):
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).
The runner copies your `~/.hermes/` into a per-conversation overlay and
overrides `model.{default,base_url,context_length}` — your real config
is untouched.

`HF_TOKEN` must be exported (or in `~/.cache/huggingface/token`) and
must have write access to the target bucket.

## Quick start

```bash
source .venv/bin/activate
cd lmcache

# Optional: clear the bucket before a fresh run
./test-cache.sh clear-bucket

# End-to-end (~5 min on 1× A10 with Qwen2.5-Coder-7B)
./test-cache.sh run-all

# Or step through manually
./test-cache.sh warmup
./test-cache.sh consume
./test-cache.sh verify

# Show summary
./test-cache.sh status
```

## Profiles

`profiles/<name>.sh` defines a model, TP degree, and resource limits.
Today only one profile is exercised end-to-end:

| Profile          | Model                          | TP | GPUs   |
|------------------|--------------------------------|----|--------|
| `qwen2.5-7b-tp1` | Qwen/Qwen2.5-Coder-7B-Instruct | 1  | 1× A10 |

Pick it with `--profile qwen2.5-7b-tp1` (also the default).

Adding new profiles is a matter of dropping a `.sh` file in
`profiles/` and pointing it at any vLLM-compatible model with a
working tool-call parser (hermes sends tool definitions in every
request). Models with hybrid attention (Qwen3.5, Qwen3-Coder-Next) are
incompatible with LMCache regardless — see "Model compatibility"
below.

## Lifecycle (how the mount stays clean)

`process-compose` drives the YAML rendered from
[`templates/phase.yaml.j2`](templates/phase.yaml.j2) for each phase:

- `hf-mount` runs as `is_daemon: true` via `hf-mount start ... bucket ...`
- Readiness probe: `grep ' /tmp/hf-mount-lmcache ' /proc/mounts`
- Shutdown: `shutdown.command: hf-mount stop /tmp/hf-mount-lmcache`
  (the wrapper's coordinated unmount — never raw umount)
- `vllm` `depends_on: hf-mount: process_healthy`
- `conv-<name>` `depends_on: vllm: process_healthy`, runs
  `lib/run-conversation.sh`
- `summary` `depends_on: conv-<name>: process_completed`,
  `availability.exit_on_end: true` so finishing it tears the project
  down in dependency order

There's also `lmcache/sanity/` — two minimal process-compose YAMLs
(read-only and overlay) that exercise the mount lifecycle end-to-end
without vLLM. Useful for proving the `hf-mount start` / `hf-mount stop`
pattern in isolation when something looks off.

## How `verify` decides

Three checks, all must pass:

1. **Bucket file list unchanged across consume.** Snapshots taken by
   `cmd_warmup` / `cmd_consume` via `huggingface_hub.HfApi.list_bucket_tree`
   are diffed. Overlay mode must not propagate to remote.
2. **`external_cache_hits > 0`** in `summary-consume.txt` — read by the
   `prom_metric vllm:external_prefix_cache_hits_total`. Confirms
   LMCache served prefix tokens from the bucket.
3. **Overlay-local layer holds new chunks.** After unmount the upper
   layer of the overlay (the mount-point directory itself, on local
   disk) must contain the `.kvcache.safetensors` files written for the
   divergent user-message tail.

## Configuration

| Variable             | Default                                          |
|----------------------|--------------------------------------------------|
| `PROFILE`            | `qwen2.5-7b-tp1`                                 |
| `BUCKET`             | `dacorvo/lm-cache`                               |
| `MOUNT_POINT`        | `/tmp/hf-mount-lmcache`                          |
| `HF_MOUNT_CACHE_DIR` | `/tmp/hf-mount-cache-<profile>`                  |
| `HF_MOUNT_BIN`       | `../hf-mount/target/release/hf-mount` (wrapper)  |
| `VLLM_PORT`          | `8000`                                           |
| `LOG_DIR`            | `lmcache/logs/<profile>` (gitignored)            |

Model-specific knobs (`MODEL`, `TP_SIZE`, `MAX_MODEL_LEN`,
`TOOL_CALL_PARSER`, `TOOL_PARSER_PLUGIN`, `CHAT_TEMPLATE`) live in
profile files and shouldn't normally be overridden.

## Files

```
lmcache/
  test-cache.sh        # CLI: warmup / consume / verify / status / clear-bucket / teardown
  setup.sh             # uv pip install lmcache
  README.md            # this file
  lib/
    helpers.sh                            # logging + summary file aggregator
                                          # (cache_file_count, prom_metric, save_summary)
    generate-phase.py                     # renders templates/phase.yaml.j2 from profile
                                          # + LMCACHE_CONFIG_FILE + mount mode → YAML
    run-conversation.sh                   # the conv-<name> process body: copies
                                          # ~/.hermes/ into a per-conv overlay, rewrites
                                          # config.yaml's model.{default,base_url,
                                          # context_length}, runs `hermes chat -q ...`,
                                          # writes first_turn_ttft_ms to a stats file
    qwen2_5_coder_tool_parser.py          # vLLM tool-parser plugin used by the qwen2.5
    tool_chat_template_qwen2_5_coder.jinja  # profile (Qwen2.5-Coder emits tool calls in
                                          # a non-default format that vLLM needs to parse)
  templates/
    phase.yaml.j2        # the only template — rendered per phase by generate-phase.py.
                         # Defines four process-compose processes:
                         #   1. hf-mount  (is_daemon: true, exec readiness probe on
                         #                 /proc/mounts, shutdown.command: hf-mount stop)
                         #   2. vllm      (depends_on hf-mount: process_healthy, http
                         #                 readiness on /v1/models)
                         #   3. conv-<n>  (one per .txt in prompts/<set>/, runs
                         #                 lib/run-conversation.sh, depends_on vllm)
                         #   4. summary   (depends_on conv-*: process_completed,
                         #                 calls helpers.sh save_summary, exit_on_end)
  profiles/
    qwen2.5-7b-tp1.sh    # model + TP + GPU mem util + tool-parser plugin path
  prompts/
    warmup/<one>.txt     # the one prompt sent during warmup
    consume/<one>.txt    # the one prompt sent during consume (different from warmup so
                         # cache hits come from the shared hermes system prompt prefix
                         # only, not from accidental whole-prompt match)
  sanity/                # mount-lifecycle sanity test, no vLLM, no LMCache, no hermes —
                         # just process-compose + hf-mount, useful when something looks
                         # off in the wrapper-driven mount path
    sanity-ro.yaml         # process-compose YAML: mount dacorvo/lm-cache --read-only
                           # at /tmp/sanity-mnt, run `ls`, exit_on_end shuts it down
    sanity-overlay.yaml    # same but --overlay
    run-sanity.sh          # runs both YAMLs sequentially, asserts /proc/mounts is empty
                           # between and after the two phases (no zombie mounts)
  logs/<profile>/      # per-phase logs and summaries (gitignored):
                       #   process-compose-<phase>.yaml      — rendered YAML
                       #   session-<phase>.log               — process-compose stdout
                       #   vllm-<phase>.log                  — vLLM stdout
                       #   hf-mount-<phase>.log              — hf-mount stdout
                       #   conv-<name>-<phase>.log           — hermes stdout
                       #   conversation-<name>-<phase>.log   — hermes session output
                       #   conv-stats-<name>-<phase>.txt     — per-conv first_turn_ttft_ms
                       #   summary-<phase>.txt               — aggregated metrics
                       #   bucket-{before,after}-<phase>.txt — HF API listing snapshots
                       #   hermes-<name>/                    — per-conv HERMES_HOME overlay
```

## Model compatibility

LMCache uses vLLM's `--kv-transfer-config` to intercept KV cache
operations, which **disables the hybrid KV cache manager**. Models with
hybrid attention architectures (e.g. GatedDeltaNet + standard attention)
are incompatible.

| Architecture                | Example models                       | LMCache compatible |
|-----------------------------|--------------------------------------|--------------------|
| Standard attention (GQA/MHA)| Qwen2.5-Coder, Qwen3-Coder-30B, Llama-3.1 | Yes               |
| Hybrid (GatedDeltaNet)      | Qwen3.5, Qwen3-Coder-Next            | **No**            |

See [LMCache#2845](https://github.com/LMCache/LMCache/issues/2845) and
[vllm#36771](https://github.com/vllm-project/vllm/issues/36771) for
details.
