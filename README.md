# hf-mount Cache Examples

Integration tests illustrating how
[hf-mount](https://github.com/huggingface/hf-mount) can accelerate ML
inference by sharing compilation caches through HuggingFace Buckets.

## The problem

ML inference stacks rely on expensive, deterministic compilation caches:
`torch.compile`/Inductor, AWS Neuron, vLLM compiled graphs, JAX/XLA HLO,
Triton kernels. Minutes-to-hours to build on first run.

These caches are local to each machine. When a new instance spins up,
it starts cold, recompiling everything or re-processing from scratch.

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

## Examples

| Directory                                          | Cache type             | Stack                                  |
|----------------------------------------------------|------------------------|----------------------------------------|
| [`torch.compile/`](torch.compile/)                 | Inductor on-disk cache | PyTorch + transformers (CausalLM)      |

Each subdirectory has its own README detailing its specifics.
