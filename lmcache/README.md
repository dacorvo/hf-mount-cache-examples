# LMCache + vLLM + hf-mount Integration Test

Integration test for hf-mount with vLLM and LMCache. Compares KV cache
performance across local disk, read-write bucket mounts, and overlay
mounts.

## Prerequisites

NVIDIA GPU with 24 GB VRAM (A10 or comparable), CUDA 12.1+, Rust toolchain.

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

The token must have access to `Qwen/Qwen2.5-Coder-7B-Instruct`. Either:

```bash
export HF_TOKEN=hf_...
```

Or log in with the CLI (the scripts read `~/.cache/huggingface/token`):

```bash
huggingface-cli login
```

## Usage

```bash
# From the repo root, after running setup.sh and lmcache/setup.sh
source .venv/bin/activate
cd lmcache

# Local disk (no hf-mount)
./test-cache.sh local-cold       # cold start reference
./test-cache.sh local-warmup     # populate local disk cache
./test-cache.sh local-warm       # consume from warm local cache

# HF bucket via hf-mount
./test-cache.sh bucket-warmup    # populate cache in bucket (RW mount)
./test-cache.sh bucket-rw        # consume from bucket (RW mount)
./test-cache.sh bucket-overlay   # consume from bucket (overlay mount)

# Utilities
./test-cache.sh status           # process state + results from all phases
./test-cache.sh teardown         # stop everything
./test-cache.sh clear-bucket     # delete all files from the bucket
```

## Results

First round on a single A10 (24 GB) with Qwen2.5-Coder-7B-Instruct,
3 multi-turn opencode conversations per phase (5 turns each, ~10K token
system prompt):

| Phase | Elapsed | Prompt tok/s | Gen tok/s | Prefix cache | External cache |
|-------|---------|-------------|-----------|-------------|---------------|
| local-cold | 92s | 93.1 | 17.6 | 92.9% | 0.0% |
| local-warmup | 112s | 76.7 | 19.6 | 92.9% | 0.0% |
| local-warm | 64s | 11.0 | 13.4 | 91.9% | 94.0% |
| bucket-warmup | — | 83.8 | 19.4 | 92.4% | 0.0% |
| bucket-rw | 120s | 8.3 | 16.4 | 92.0% | 94.0% |
| **bucket-overlay** | **75s** | **12.2** | **13.6** | **92.0%** | **94.1%** |

**Key findings:**

- **Overlay is 37% faster than read-write** (75s vs 120s) for consumers.
  Read-write mounts flush new KV cache chunks back to the bucket on every
  store, adding I/O overhead that overlay avoids entirely.
- **94% external cache hit rate** across all warm phases — the shared
  system prompt prefix (~10K tokens) is cached on first use and reused
  across all subsequent conversations.
- **Local warm disk is fastest** (64s) — the baseline for what's
  achievable without network I/O.

## Test design

Each phase runs three independent multi-turn conversations (5 turns each).
Warmup and consume use **different** topics so that consume does not get
artificially high cache hit rates from identical token sequences.

**Warmup topics**: overlay implementation, test coverage, documentation.

**Consume topics**: NFS backend, write pipeline, error handling.

Cache hits come from shared system-prompt prefixes and overlapping file
reads — the realistic scenario where different users ask different
questions but share the same coding assistant configuration.

## How it works

### local-cold — cold start reference

Starts vLLM with LMCache pointed at an empty local directory. Runs
consume conversations. No prior cache exists, so all tokens are computed
from scratch. This is the baseline.

### local-warmup / local-warm — warm local disk

`local-warmup` populates LMCache on plain local disk with warmup
conversations. `local-warm` restarts vLLM and runs consume conversations
on the same directory. LMCache's GDS backend scans existing cache files
on startup and serves them as external cache hits.

### bucket-warmup — populate shared cache

Mounts the bucket **read-write** and starts vLLM with LMCache pointed at
the mount. Warmup conversations generate KV cache that LMCache writes as
safetensors files through the mount. hf-mount uploads them to the bucket.

### bucket-rw — shared cache (read-write mount)

Unmounts, remounts **read-write**, restarts vLLM cold. Consume
conversations trigger LMCache disk reads — cache hits come from the
bucket. New cache chunks are also written back through the mount.

### bucket-overlay — producer/consumer (overlay mount)

Same as bucket-rw but remounts with **`--overlay`**. Cache reads come
from the bucket (read-through), new writes stay local and are never
uploaded. This avoids write-back I/O overhead, making it significantly
faster for consumers. No write access to the bucket is required.

## Log files

All output goes to both the console and log files under `$LOG_DIR`:

| File | Content |
|------|---------|
| `session.log` | Timestamped script progress |
| `hf-mount.log` | hf-mount-nfs output |
| `vllm.log` | vLLM + LMCache output |
| `summary-{phase}.txt` | Per-phase results (parsed by `status`) |
| `conversation-{topic}.log` | Per-conversation opencode output |

## Configuration

Environment variable overrides:

| Variable         | Default                             | Description                    |
|------------------|-------------------------------------|--------------------------------|
| `MODEL`          | `Qwen/Qwen2.5-Coder-7B-Instruct`   | HuggingFace model ID           |
| `BUCKET`         | `dacorvo/lm-cache`                  | HuggingFace bucket ID          |
| `MOUNT_POINT`    | `/tmp/hf-mount-lmcache`             | Mount directory                |
| `CACHE_DIR`      | `/tmp/hf-mount-cache`               | hf-mount chunk cache directory |
| `LOCAL_CACHE_DIR`| `/tmp/hf-mount-local-cache`         | Local cache directory          |
| `VLLM_PORT`      | `8000`                              | vLLM API port                  |
| `MAX_MODEL_LEN`  | `32768`                             | vLLM max context length        |
| `LOG_DIR`        | `lmcache/logs/`                     | Log files directory            |
| `HF_MOUNT_BIN`   | `hf-mount/target/release/hf-mount-nfs` | Path to hf-mount-nfs binary |
