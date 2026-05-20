# Findings — when bucket-backed caching pays off, and when it doesn't

## The per-file overhead floor

HF Storage Buckets are xet-backed: file content is content-addressed
and shipped in batched xorbs, but each file still pays a metadata
roundtrip. Measured rates differ by access path:

| Access path | Rate |
|---|---|
| Producer-through-mount upload (hf-mount, `--advanced-writes`) | ~7 files/sec |
| Direct API upload (`HfApi.sync_bucket`, local → bucket) | ~51 files/sec |
| Direct API download (`HfApi.sync_bucket`, bucket → local, cold xet) | ~16 files/sec, **highly variable** |
| File listing (`list_bucket_tree`) | ~5,000 files / 3.6 s |

The per-file ceiling is a property of the **access path**, not the
bucket itself. Direct `sync_bucket` batches per-xorb and is ~6× faster
than the mount, which flushes each file individually under
`--advanced-writes`. Two operational caveats matter for cache
workloads:

1. **Download is asymmetric with upload.** Same 2,642-file llama
   payload: 51 s up, 166 s down (cold xet). xet uploads dedupe and
   batch aggressively; downloads fan out per-file metadata fetches.

2. **Download can hang.** Two back-to-back cold-xet fetches of the
   same bucket, ~10 minutes apart, same client code: 166 s vs "no
   file activity for 12 min, killed at 26 min". Bucket fetch
   reliability under hub load is a real concern — applications need
   their own retry/timeout layer.

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

The original "~30 min producer through hf-mount RW vs 6 min local SSD"
number we measured was a property of the **mount path**, not the
bucket. Using direct `sync_bucket` upload for the same payload, a
7,000-file cache uploads in ~140 s (51 files/sec). The bucket itself
isn't the structural bottleneck we first thought.

What does keep this workload on the wrong side of `produce − fetch`:

1. **Cold-consumer fetch is barely faster than fresh compile.**
   Llama-3.2-3B (2,642-file cache): fresh compile 192 s, cold-xet
   fetch 166 s. ~26 s saved — well within fetch-time variance.

2. **Fetch latency has a long tail.** Same llama fetch from the same
   bucket: 166 s in one run, indefinitely hung in another. You can't
   put that on the consumer-cold-start critical path without a
   fallback to fresh compile.

3. **"Ship a subset" doesn't work.** See the next section — stripping
   the per-kernel `triton/` tree breaks consumer cache hits even on
   local disk.

By extension this rules out other dev-style compile caches with the
same shape: JAX persistent cache, standalone Triton autotune cache,
vLLM v0 compile cache — anything that emits many small intermediate
artifacts.

## Stripping `triton/` breaks consumer cache hits

The asymmetric pattern that *should* have worked: compile locally,
strip the `triton/` subtree (per-kernel files) before sync, ship only
the FX-bundled entries to the bucket. For gemma this drops the cache
from 7,033 files (191 MB) to 1,055 files (120 MB) — 6.7× fewer files
to handle. The premise (see appendix below) was that
`TritonBundler.read_and_emit()` would unpack per-kernel files from
the bundled FX entries on cache hit, so the strip should be
transparent to the consumer.

It isn't. Measured on local disk (no bucket, no overlay, just a
populated `TORCHINDUCTOR_CACHE_DIR`):

| Cache state | first_call_s | files added | files at end |
|---|---|---|---|
| Warm local, full | 41 | 132 | 7,301 |
| Warm local, `triton/` stripped | **286** | **3,243** | **4,298** |

The stripped cache forces Inductor to recompile ~3,000 kernels —
286 s on the consumer's critical path, on plain local disk. Whatever
guard sits in front of `read_and_emit()`, it doesn't engage when the
on-disk `triton/` tree is absent.

The 174 s and 401 s overlay measurements we collected earlier with
the stripped bucket therefore weren't measuring bucket fetch — they
were mostly measuring this same consumer-side recompile, plus
~115 s of overlay/NFS overhead.

Practical consequence: the bucket can't be shipped selectively. Pay
for the full sync, or have the consumer recompile. No middle path.

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

The flag was intended to make FX cache entries self-contained, but it
**never suppresses the original per-kernel writes**. Those writes
come from Triton's runtime before Inductor's bundler runs. On a
bucket mount, the bucket sees the per-kernel files first, then the
bundled FX entry containing a copy.

> **Correction.** The original reading of this code claimed a cache
> hit can reconstruct missing per-kernel files from the bundle. The
> strip experiment (see "Stripping `triton/` breaks consumer cache
> hits" above) contradicts that: removing the on-disk `triton/` tree
> forces Inductor to recompile, not unpack from the bundle. Either
> the FX cache lookup doesn't engage when the kernel tree is missing,
> or `read_and_emit` is called but doesn't fully restore. Either way,
> the bundle is not a substitute for the on-disk per-kernel files
> from the consumer's perspective.

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
