# Findings — when bucket-backed caching pays off, and when it doesn't

## The per-file overhead floor

HF Storage Buckets (xet-backed) impose a per-file overhead — manifest
update, content-address lookup, xorb commit — that does not amortize
with file size. Measured on this setup:

| Direction | Rate |
|---|---|
| Producer-through-mount upload (hf-mount, `--advanced-writes`) | ~7 files/sec |
| Consumer-via-direct-API download (`huggingface_hub.sync_bucket`) | ~8 files/sec |
| Effective throughput at 75 KB avg | ~500 KB/s |
| File listing (`list_bucket_tree`) | ~5,000 files / 3.6 s |

It's the same ceiling regardless of access method, so it's not an
hf-mount property — it's a property of the bucket / xet storage layer.

## Cache value = produce_cost − fetch_cost

A cache only pays off when the cost to **fetch** a cached entry is
lower than the cost to **produce** it from scratch. The per-file floor
above sets a lower bound on fetch_cost. So a workload's cache profile
needs to land on the right side of that bound.

## torch.compile JIT cache is on the wrong side

Profiled cache for `google/gemma-4-E4B-it` + `mode="max-autotune-no-cudagraphs"` +
chunked prefill (one warmup shape, ~6 min local-SSD compile):

```
files       : 5,520
total size  : 414 MB
avg         : 75 KB   (skewed by one 148 MB .so)
median      :  8 KB
p90         : 38 KB
p99         : 190 KB

size buckets:
  <  1 KB :   506   ( 9.2%)
  1- 10 KB: 2,779   (50.3%)   ← majority
  10-100 KB: 2,106  (38.2%)
  100KB-1MB:  116   ( 2.1%)
  > 1 MB  :    13   ( 0.2%)
```

Per kernel, Inductor emits 6 IR siblings (`.source .ttir .ttgir .llir
.ptx .cubin`) plus per-op `.json` and `.best_config` — most of which
exist for debug/hygiene, not runtime correctness. The vast majority of
the cache is small and individually cheap to recompute; only the
single `.so` (36 % of total bytes) and the `.best_config` files
represent meaningful unique compute.

Net producer cost: running the warmup through an hf-mount RW mount
took **~30 minutes** to write what compiles in **~6 minutes** on local
SSD (5.3× overhead, dominated by per-file flush throughput). On the
consumer side, lazy-fetching even a *subset* still pays the per-file
cost on each touched file.

For this workload, fetch_cost ≥ produce_cost per file → the bucket
loses to local recompute. This isn't a tuning problem; it's structural.

By extension this rules out other dev-style compile caches with the
same shape: JAX persistent cache, standalone Triton autotune cache,
vLLM v0 compile cache — anything that emits many small intermediate
artifacts.

## What does land on the right side

Workloads where **each cached artifact is large and represents
significant unique compute**:

- **AWS Neuron NEFFs** — a few MB per file, 10+ minutes of compute per
  artifact. Cache fetch easily dominates.
- **TensorRT engines** — hundreds of MB per file, 5–30 min per build,
  per (model, GPU, batch) tuple.
- **Quantized model weights** (GPTQ / AWQ output) — multi-GB files,
  hours of calibration compute.
- **AOTInductor `.so`** — one file per compiled model, hundreds of MB,
  full compile pipeline behind it. *But*: if the consumer knows in
  advance which `.so` it wants, plain file transfer suffices and the
  bucket/overlay pattern adds no value.

The bucket+overlay pattern earns its keep specifically when:
1. Each artifact is **large** and **expensive** to produce
   (per-file overhead becomes negligible).
2. Consumers **don't know in advance** which subset they need
   (lazy fetch wins over up-front download).

The torch.compile JIT cache fails (1). AOT compile fails (2). NEFFs,
TensorRT engines, and per-config quantized weights satisfy both.

## Demo implication

The torch.compile example as written runs end-to-end and demonstrates
correct overlay semantics (bucket invariance across consume, recompile
isolation, etc.), but the timing numbers don't tell a compelling
producer/consumer story — the cache profile fights the bucket's
strengths. A demo that genuinely showcases the pattern needs an
artifact that satisfies both (1) and (2) above.

## Code-path appendix: why no Inductor flag reshapes the on-disk layout

Read against `pytorch@27f2e80e30f` (master).

There are two flags in `torch._inductor.config` that sound like they
could reduce the per-file overhead by bundling artifacts together:

- `bundle_triton_into_fx_graph_cache` (defaults to `True` in OSS)
- `bundled_autotune_remote_cache` (defaults to `None` in OSS, i.e. off)

Tracing each through the source shows neither affects what hits a
bucket-mounted `TORCHINDUCTOR_CACHE_DIR`.

### `bundle_triton_into_fx_graph_cache`

Read in `torch/_inductor/triton_bundler.py:117-131`
(`TritonBundler.is_enabled`).

- `TritonBundler.collect()` (triton_bundler.py:259-346) runs at FX
  cache-write time. It iterates `triton_cache_dir(device)` —
  **where Triton has already written the per-kernel files** —
  reads them, and packs them into the FX cache entry payload.
- `TritonBundler.read_and_emit()` (triton_bundler.py:348-413) runs on
  FX cache hit. It unpacks the bundled payload into the same
  `triton_cache_dir(device)`. Line 376 bails out if the directory is
  already non-empty.

The flag makes FX cache entries self-contained (so a cache hit can
reconstruct missing per-kernel files), but it **never suppresses the
original per-kernel writes**. Those writes come from Triton's runtime
before Inductor's bundler runs. On a bucket mount, the bucket sees
the per-kernel files first, then the bundled FX entry containing a
copy.

### `bundled_autotune_remote_cache`

Read in `torch/_inductor/runtime/autotune_cache.py:388-419`
(`_should_use_bundled_autotune_remote_cache`).

When True, `AutotuneCacheBundler.begin_compile`
(autotune_cache.py:498-557) creates a `RemoteBundledAutotuneCache` and
attempts a load. On miss, individual autotune results are collected
via `put()` (autotune_cache.py:586+) and bundled into one remote-cache
entry at `end_compile`.

Critical detail at autotune_cache.py:444 — the local autotune cache
is still populated alongside:

```python
local_cache.put(filename, data)
```

So the on-disk `.best_config` files appear regardless. The flag only
changes whether autotune results also go to a *remote* cache as one
bundled entry instead of N individual entries.

### Remote-cache backend dispatch (OSS)

`torch/_inductor/remote_cache.py:370-388` declares every remote-cache
class as a `RedisRemoteCache` subclass:

```python
class RemoteAutotuneCache(RedisRemoteCache): pass
class RemoteBundledAutotuneCache(RedisRemoteCache): pass
class RemoteFxGraphCache(RedisRemoteCache): pass
class RemoteAOTAutogradCache(RedisRemoteCache): pass
class RemoteDynamoPGOCache(RedisRemoteCache): pass
```

`create_cache()` (remote_cache.py:390-416) only has three modes:
`local_cache_cls` for tests, `fbcode` (Meta-internal), `oss_cache_cls`
which is always one of the Redis classes above. **There is no plugin
mechanism in OSS for a non-Redis remote backend.** Pointing Inductor's
remote cache at HF Buckets would require subclassing
`RemoteCacheBackend` and monkey-patching the `torch._inductor.remote_cache`
module — a development project, not a config knob.

### On-disk write path

`torch/_inductor/runtime/cache_dir_utils.py:35-42`:

```python
def triton_cache_dir(device: int) -> str:
    if (directory := os.getenv("TRITON_CACHE_DIR")) is not None:
        return directory
    return os.path.join(cache_dir(), "triton", str(device))
```

Triton's own runtime writes the `.cubin / .ttir / .ttgir / .source /
.ptx / .llir / .json` siblings here, independent of any Inductor
config flag.

### So: no flag combination changes what hits the bucket mount

For the file-based cache pointed at by `TORCHINDUCTOR_CACHE_DIR`, no
combination of `bundle_triton_into_fx_graph_cache` /
`bundled_autotune_remote_cache` / `fx_graph_remote_cache` /
`autotune_remote_cache` reduces the per-kernel-files-times-thousands
shape we measured. The remote-cache code path is a parallel channel
that routes to Redis in OSS and bypasses the file-based cache dir
entirely.

The realistic way to make Inductor cache through hf-mount cleanly is
not to mount the file-based cache dir on a bucket at all — it's to
implement a custom `RemoteCacheBackend` that uses
`huggingface_hub` to read/write a bucket, and wire it in for
`fx_graph_remote_cache` / `autotune_remote_cache`. Then each compile
emits **one bundled entry per FX-graph key** to the bucket instead of
thousands of per-kernel files. That changes the workload's shape to
the side of the cache-value math the bucket actually wins on.
