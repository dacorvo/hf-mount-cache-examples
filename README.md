# hf-mount Cache Examples

Integration tests illustrating how
[hf-mount](https://github.com/huggingface/hf-mount) can accelerate ML
inference by sharing file-based caches through HuggingFace Buckets.

## The problem

ML inference stacks rely on expensive, deterministic file-based caches:

- **Compilation caches** — `torch.compile`/Inductor, AWS Neuron, vLLM
  compiled graphs, JAX/XLA HLO, Triton kernels. Minutes-to-hours to
  build on first run.
- **KV caches** — LMCache stores prefix-cache chunks as individual
  files with LRU eviction, enabling prefix-cache hits across requests.

These caches are local to each machine. When a new instance spins up,
it starts cold — recompiling everything or re-processing prompts from
scratch.

## Two approaches

### Read-write shared cache

Multiple instances mount the same bucket **read-write**. Cache files
produced by any instance propagate to the bucket and become available
to others. Simplest model; works well for a single user sharing a
cache across their own instances.

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

Both tests share the same three-phase shape:

| Phase     | Mount mode | Purpose                                                        |
|-----------|------------|----------------------------------------------------------------|
| `warmup`  | read-write | Populate the shared cache in the bucket                        |
| `consume` | overlay    | Read from bucket, recompute locally for misses (bucket pristine) |
| `verify`  | —          | Diff bucket file list, check cache-hit signatures, check overlay-local files |

## Examples

| Directory                                          | Cache type             | Stack                                  |
|----------------------------------------------------|------------------------|----------------------------------------|
| [`lmcache/`](lmcache/)                             | LMCache prefix KV cache| vLLM + LMCache + hermes-agent          |
| [`torch.compile/`](torch.compile/)                 | Inductor on-disk cache | PyTorch + transformers (CausalLM)      |

Each subdirectory has its own README detailing its specifics.

## Quick start

```bash
# Shared setup: hf-mount submodule + build, venv with vLLM
./setup.sh
source .venv/bin/activate

# Either example follows the same warmup / consume / verify pattern:
cd lmcache && ./setup.sh && ./test-cache.sh run-all
# or
cd torch.compile && ./setup.sh && ./test-compile.sh run-all
```

`lmcache/` additionally requires
[hermes-agent](https://github.com/NousResearch/hermes-agent) on PATH —
see [`lmcache/README.md`](lmcache/README.md).

## Repository structure

```
├── setup.sh           # shared: submodule init, build hf-mount, venv with vLLM
├── hf-mount/          # git submodule (huggingface/hf-mount)
├── lmcache/           # LMCache prefix-cache test
│   ├── setup.sh
│   ├── test-cache.sh  # CLI: warmup / consume / verify / status / clear-bucket / teardown
│   └── README.md
└── torch.compile/     # torch.compile Inductor cache test
    ├── setup.sh
    ├── test-compile.sh # CLI: warmup / consume / verify / teardown
    └── README.md
```

## Prerequisites

- NVIDIA GPU (16+ GB VRAM for `torch.compile`, 24+ GB for `lmcache`)
- CUDA 12.1+
- Rust toolchain (cargo)
- Python 3.10+
- HuggingFace token with access to the chosen bucket
