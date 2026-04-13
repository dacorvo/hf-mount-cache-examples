# hf-mount Cache Examples

Integration tests illustrating how
[hf-mount](https://github.com/huggingface/hf-mount) can accelerate ML
inference by sharing file-based caches through HuggingFace Buckets.

## The problem

ML inference stacks rely on expensive, deterministic file-based caches:

- **Compilation caches** — AWS Neuron (`~/.cache/neuron-compile-cache`),
  `torch.compile`, vLLM compiled graphs, JAX/XLA HLO programs, Triton
  kernels. These take minutes to hours to build on first run.
- **KV caches** — LMCache stores KV cache chunks as individual files
  with LRU eviction, enabling prefix cache hits on first request.

These caches are local to each machine. When a new instance spins up, it
starts cold — recompiling everything or re-processing prompts from scratch.

## Two approaches

### Read-write shared cache

The simplest approach: multiple instances mount the same bucket
**read-write**. Cache files produced by any instance are uploaded to the
bucket and become available to all others. This works well for a single
user sharing a cache across their own instances.

### Producer / consumer with overlay mode

For larger deployments, hf-mount's `--overlay` flag provides
**read-through, write-local** semantics:

- Remote bucket contents are readable on demand (lazy fetch).
- New writes persist locally without uploading to the bucket.
- Local files take precedence over remote ones on conflict.

**Cache producers** (few machines) mount the bucket **read-write** and
fill it directly. **Cache consumers** (many machines) mount with
**`--overlay`** — cached artifacts are fetched lazily from the bucket,
and local misses rebuild without polluting the shared cache. No write
access is required for consumers.

## What the tests verify

Each test runs four phases and compares timings and cache hit rates:

| Phase | Mount mode | Purpose |
|-------|-----------|---------|
| baseline | none | Reference timing on plain local disk |
| warmup | read-write | Populate the shared cache in the bucket |
| consume | read-write | Consume from bucket via shared read-write mount |
| consume-overlay | overlay | Consume from bucket via overlay (read-through, write-local) |

## Examples

Each subdirectory is a self-contained integration test exercising a
specific cache type.

| Directory | Cache type | Stack |
|-----------|-----------|-------|
| [lmcache/](lmcache/) | KV cache | vLLM + LMCache + opencode |

### Quick start

```bash
# Shared setup: build hf-mount, create venv with vLLM, install opencode
./setup.sh
source .venv/bin/activate

# Run a specific test (e.g. LMCache)
cd lmcache
./setup.sh                       # install LMCache
./test-cache.sh baseline         # reference timing (no mount)
./test-cache.sh warmup           # populate cache (read-write mount)
./test-cache.sh consume          # consume via read-write mount
./test-cache.sh consume-overlay  # consume via overlay mount
./test-cache.sh teardown         # stop everything
```

## Repository structure

```
├── setup.sh            # shared: submodule init, build hf-mount, venv, opencode
├── hf-mount/           # git submodule (huggingface/hf-mount)
└── lmcache/
    ├── setup.sh        # LMCache-specific deps
    ├── test-cache.sh   # test CLI (baseline / warmup / consume / consume-overlay / ...)
    ├── opencode.json   # opencode config for local vLLM server
    └── README.md       # detailed documentation
```

## Prerequisites

- NVIDIA GPU with 24 GB VRAM (A10 or comparable)
- CUDA 12.1+
- Rust toolchain (cargo)
- Python 3.10+
- HuggingFace token with access to the model and bucket
