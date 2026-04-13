# LMCache + vLLM + hf-mount Integration Test

Integration test for hf-mount with vLLM and LMCache. Compares KV cache
performance across local disk, read-write bucket mounts, and overlay
mounts. Supports multiple models and tensor parallelism configurations
via profiles.

## Prerequisites

NVIDIA GPU(s) with CUDA 12.1+, Rust toolchain, Python 3.10+.

### Setup

From the repo root:

```bash
# Shared setup: submodule, build hf-mount, venv with vLLM, opencode
./setup.sh
source .venv/bin/activate

# LMCache-specific
cd lmcache && ./setup.sh
```

### HuggingFace token

```bash
export HF_TOKEN=hf_...
# or: huggingface-cli login
```

## Quick start

```bash
source .venv/bin/activate
cd lmcache

# Run all 6 phases with the default profile (qwen2.5-7b-tp1):
./test-cache.sh run-all

# Run all phases for all profiles:
./test-cache.sh run-suite

# Run a specific phase with a specific profile:
./test-cache.sh --profile qwen2.5-32b-tp4 bucket-overlay

# Show results:
./test-cache.sh status           # current profile
./test-cache.sh status --all     # all profiles
```

## Profiles

Each profile defines a model, tensor parallelism degree, and resource
limits. Profiles live in `profiles/`:

| Profile | Model | TP | GPUs |
|---------|-------|----|------|
| `qwen2.5-7b-tp1` (default) | Qwen2.5-Coder-7B-Instruct | 1 | 1x A10 |
| `qwen2.5-32b-tp4` | Qwen2.5-Coder-32B-Instruct | 4 | 4x A10 |

Select via `--profile <name>` or `PROFILE=<name>`.

## Phases

Each phase runs 3 multi-turn opencode conversations (5 turns each).
Warmup and consume use different topics to avoid biased cache hit rates.

| Phase | Mount | Cache state | Purpose |
|-------|-------|-------------|---------|
| `local-cold` | none | empty | Cold start reference |
| `local-warmup` | none | → warm | Populate local disk cache |
| `local-warm` | none | warm | Warm local cache baseline |
| `bucket-warmup` | read-write | → warm | Populate bucket cache |
| `bucket-rw` | read-write | warm | Shared cache via RW mount |
| `bucket-overlay` | overlay | warm | Shared cache via overlay |

## Results (qwen2.5-7b-tp1, single A10)

Qwen2.5-Coder-7B-Instruct, 3 conversations per phase (~10K token
system prompt from opencode):

| Phase | Elapsed | External cache hit |
|-------|---------|-------------------|
| local-cold | 92s | 0.0% |
| local-warmup | 112s | 0.0% |
| local-warm | 64s | 94.0% |
| bucket-warmup | (tbd) | 0.0% |
| bucket-rw | 120s | 94.0% |
| **bucket-overlay** | **75s** | **94.1%** |

**Key findings:**

- **Overlay is 37% faster than read-write** (75s vs 120s) for consumers.
  Read-write mounts flush new KV cache chunks back to the bucket on every
  store — overlay avoids this write-back I/O entirely.
- **94% external cache hit rate** across all warm phases — the shared
  system prompt prefix (~10K tokens) is cached on first use and reused
  across all subsequent conversations.
- **bucket-warmup vs local-warmup** shows the cost of writing cache
  chunks through hf-mount in real time versus writing locally and pushing
  to the bucket in one batch afterward.
- **Local warm disk is fastest** (64s) — the baseline for what's
  achievable without network I/O.

## Directory structure

```
lmcache/
  test-cache.sh           # main CLI
  setup.sh                # LMCache-specific deps
  lib/
    helpers.sh            # logging, metrics, summaries
    vllm.sh               # vLLM lifecycle (TP support)
    hf-mount.sh           # mount lifecycle
    conversations.sh      # conversation runner + topic definitions
  profiles/
    qwen2.5-7b-tp1.sh             # Qwen2.5-Coder-7B, TP=1
    qwen2.5-32b-tp4.sh            # Qwen2.5-Coder-32B, TP=4
  logs/<profile>/         # per-profile logs and summaries (gitignored)
  opencode.json           # generated at runtime (gitignored)
```

## Configuration

Environment variable overrides:

| Variable         | Default                             | Description                    |
|------------------|-------------------------------------|--------------------------------|
| `PROFILE`        | `qwen2.5-7b-tp1`                            | Profile name                   |
| `BUCKET`         | `dacorvo/lm-cache`                  | HuggingFace bucket ID          |
| `MOUNT_POINT`    | `/tmp/hf-mount-lmcache`             | Mount directory                |
| `VLLM_PORT`      | `8000`                              | vLLM API port                  |
| `LOG_DIR`        | `lmcache/logs/<profile>`            | Log files directory            |
| `HF_MOUNT_BIN`   | `hf-mount/target/release/hf-mount-nfs` | Path to hf-mount-nfs binary |

Model-specific settings (`MODEL`, `TP_SIZE`, `MAX_MODEL_LEN`, etc.) are
set by the profile and should not normally be overridden.
