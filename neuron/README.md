# Neuron + hf-mount Integration Test

Shares the AWS Neuron on-disk NEFF cache across machines through an HF
bucket mounted with `hf-mount`. Consumers mount the bucket with
`--overlay` so they read shared NEFFs but keep any new compilations local.

## What it does

`TORCH_NEURONX_NEFF_CACHE_DIR` redirects every NEFF artifact (the
compiled kernels produced by `neuronx-cc`) to a chosen directory.
Pointing it inside an `hf-mount` mount makes the cache shared.

| Phase    | Mount    | Action                                            | Bucket effect            |
|----------|----------|---------------------------------------------------|--------------------------|
| baseline | none     | Compile each shape into a fresh local cache       | None                     |
| warmup   | rw       | Compile each shape in `SHAPES`                    | NEFFs uploaded           |
| consume  | overlay  | Run each shape; cache hit if a prior warmup ran   | Unchanged (local writes) |

The benchmark compares two paths:

- **`baseline`** — standard inference with no bucket: fresh local NEFF
  cache, no `hf-mount`. The first call pays the full cold compile cost.
- **`warmup` + `consume`** — `warmup` mounts the bucket RW and pays the
  cold compile cost once (uploading NEFFs); subsequent `consume` runs
  mount the bucket as overlay and lazy-fetch the NEFFs, so the first
  call is a cache hit.

`run-all` runs `baseline`, then `warmup`, then `consume`, and prints a
per-shape `first_call_s` table comparing `baseline` to the
post-warmup `consume`.

Overlay is what makes consumer hosts safe: local writes (a cache-miss
NEFF, dispatch metadata, lock files) stay on the local filesystem layer
without touching the bucket, so a misconfigured consumer can't poison
the shared cache.

## Why NEFFs are a good fit for bucket sharing

NEFFs are coarse-grained: in the static-cache + `torch.compile(backend="neuron")`
path used here, a typical dense causal LM compiles into just **2 NEFFs**
— one prefill graph and one decode graph — totalling tens of MB. That's
orders of magnitude fewer files than Inductor's per-kernel JIT cache,
so the `hf-mount` per-file metadata overhead is negligible and the
bucket-sync path stays cheap.

## Running

Prerequisites:

- **A pre-built Neuron venv** containing `torch_neuronx`, `neuronx_cc`,
  `neuronxcc`, `transformers>=5.6`, and `accelerate`. This example does
  *not* install the Neuron SDK; see the AWS Neuron docs for the install
  workflow. Export `VENV=/path/to/that/venv` before invoking `run.sh`.
- **AWS Neuron hardware** (`trn1*`, `trn2*`, `inf2*`).
- **`HF_TOKEN`** exported (or `~/.cache/huggingface/token` populated via
  `hf auth login`), with write access to `$BUCKET`.
- **[Homebrew](https://brew.sh)** — `setup.sh` uses it to install
  `hf-mount` (bottled for Linux).

```bash
./setup.sh                      # one-time: install hf-mount + uv
export VENV=/path/to/neuron-venv
./run.sh run-all                # cold consume + warmup + warm consume, prints summary
```

`run-all` is the headline command: it runs `baseline` (no bucket) to
capture the cold-compile cost, then `warmup` + `consume` (overlay) to
capture the bucket-accelerated path, and prints a per-shape
`baseline_first_s` vs `warm_first_s` table.

Individual commands also work: `baseline`, `warmup`, `consume`,
`teardown`, `clear-bucket`. Each phase writes a JSON report under
`logs/` — `results-baseline.json`, `results-warmup.json`,
`results-consume.json` (or, under `run-all`,
`results-consume-warm.json`) — with per-shape first-call and
second-call timings plus `cache_neffs_added`.

## Configuration

| Variable             | Default                              |
|----------------------|--------------------------------------|
| `MODEL`              | `meta-llama/Llama-3.2-1B`            |
| `BUCKET`             | `dacorvo/neuron-compile-cache`       |
| `VENV`               | *(required)* — venv with the Neuron SDK |

The `SHAPES` list and the mount path (under `/tmp`) are at the top of
`run.sh`. The bucket is sharded by `neuronxcc-<version>` because NEFFs
are tied to the compiler version; `clear-bucket` only touches this
host's compiler-version subtree.

## Example results

On a `trn2.3xlarge` (1 Neuron device, 4 cores; `neuronxcc 2.0.253977`) with
`MODEL=meta-llama/Llama-3.2-1B` and `SHAPES=("1x64")`:

| Shape | `baseline_first_s` | `warm_first_s` | Speedup |
|-------|------------------:|---------------:|--------:|
| 1x64  | 575.88            | 25.07          | 23.0x   |

What the cache is doing:

- `baseline` (no bucket): compiled from scratch — 63 NEFFs written to a
  fresh local cache dir, `first_call_s = 575.88 s`.
- `warmup` (RW mount): same compile cost (`first_call_s = 663.16 s`)
  with the overhead of uploading 91 NEFFs / 28.9 MB to the bucket.
- `consume` (overlay): NEFFs lazy-fetched from the bucket on first
  reference; `first_call_s = 25.07 s` — 23× faster than a fresh compile.

Model load (`from_pretrained` + `.to("neuron")`) takes ~6 s in every
phase — it's the compile + NEFF-load portion of the first `generate()`
call that the bucket cache short-circuits.

## Caveats

This example exercises the warmup → consume cache-reuse flow under
favorable conditions. Two effects to know about before generalizing:

### 1. Coverage is per-`(model, neuronxcc-version, shape)`

A NEFF's cache key encodes the compiled graph plus the compiler version.
Bumping `neuronxcc` (binary drop refresh) invalidates the entire prior
subtree — the bucket simply gains a new `neuronxcc-<new-version>/`
sibling. The shape axis is what varies day to day: prefill graphs are
keyed on `prefill_chunk_size`, so a deployment that sweeps many chunk
sizes accumulates one prefill NEFF per chunk.

### 2. `torch.compile(backend="neuron")` only — not the eager path

The repo's "eager" mode (one NEFF per op via the dispatch backend,
`DynamicCache`) emits hundreds of small NEFFs per model rather than 2.
Bucket sync still works there, but the file-count profile starts to
resemble the Inductor caveats from the `torch.compile/` example. This
example uses the compile-mode path on purpose.
